"""增量材料核验：宿主复制未受影响的旧结论，模型显式检查新增原文的影响。"""
from copy import deepcopy
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.after_sales.semantic import PROMPT, RequirementAssessment, abstention, validate_response
from src.after_sales.service import _checksum
from src.errors import AgentError, ErrorCode

VERSION = 'case-evidence-transition-v1'


class RequirementTransition(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    requirement_id: str
    action: Literal['PRESERVE', 'REASSESS']
    assessment: RequirementAssessment | None = None
    reason: str = Field(min_length=1, max_length=300)

    @model_validator(mode='after')
    def complete_action(self):
        if (self.action == 'REASSESS') != (self.assessment is not None):
            raise ValueError('REASSESS requires assessment; PRESERVE must not rewrite it')
        if self.assessment and self.assessment.requirement_id != self.requirement_id:
            raise ValueError('Assessment target differs from transition')
        return self


class IncrementalResponse(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    requirements: list[RequirementTransition] = Field(max_length=32)
    withdrawal_confirmed: bool = False
    withdrawal_quote: str = Field(default='', max_length=2000)


DELTA_PROMPT = '''For each requirement return exactly one transition, not a full rewritten history.
PRESERVE copies the host's prior_assessments[requirement_id].assessment unchanged and has no assessment field.
It is allowed ONLY when preserve_allowed=true AND you have checked the full latest_request/new source_text
for new contradictions or relevant corrections, even if the incoming evidence has another type.
Prefer PRESERVE for unaffected prior conclusions. It is not an assertion of objective truth.
REASSESS supplies a full assessment. Use it when preserve_allowed=false, new material affects that requirement,
or you cannot confidently establish absence of an impact. For an otherwise preservable requirement,
REASSESS must cite incoming evidence that caused the change/uncertainty. Do not downgrade merely because
the latest message discusses another requirement. Give a short Chinese reason for the transition.
Missing/uncertain material is not permission to invent restrictions, withdrawal, or refund approval.
'''


def add_prior_context(inputs, evaluation, args):
    """只从政策引擎已核验的当前评估投影依赖；旧来源/时效失效不允许复用。"""
    current = {r['requirement_id']: r for r in evaluation.facts['evidence_requirements']}
    by_id = {e['evidence_id']: e for e in inputs['evidence']}
    incoming = by_id.get('incoming')
    prior = {}
    for requirement in inputs['requirements']:
        row = current[requirement['requirement_id']]
        semantic = row.get('semantic_assessment')
        if not semantic or semantic.get('error_code'):
            continue
        assessment = {k: semantic[k] for k in RequirementAssessment.model_fields}
        citations = {c['evidence_id'] for c in assessment['citations']}
        matching = {e['evidence_id'] for e in inputs['evidence'] if e['evidence_type'] == requirement['evidence_type']}
        invalid_sources = {key for key, value in row.get('rejected_evidence', {}).items()
                           if not value.startswith('SEMANTIC_') and value not in {'REVOKED', 'SUPERSEDED'}}
        reason = ('SOURCE_OR_TIME_INVALID' if invalid_sources or not citations <= set(by_id)
                  else 'WITHDRAWAL_AFFECTS_REQUIREMENT' if args.get('revoke_evidence_id') in matching | citations
                  else 'MATCHING_MATERIAL_ADDED' if incoming and incoming['evidence_type'] == requirement['evidence_type']
                  else 'EXPLICIT_REASSESSMENT' if not incoming
                  else 'UNCHANGED_DEPENDENCIES')
        prior[requirement['requirement_id']] = dict(assessment_id=semantic['assessment_id'], assessment=assessment,
            preserve_allowed=reason == 'UNCHANGED_DEPENDENCIES', reason_code=reason)
    inputs['prior_assessments'] = prior
    return inputs


def _report(inputs, result, decisions=None):
    prior = inputs.get('prior_assessments', {})
    decisions = {d.requirement_id: d for d in decisions or []}
    rows = {r['requirement_id']: r for r in result['requirements']}
    changes = []
    for req in inputs['requirements']:
        identity = req['requirement_id']
        old = prior.get(identity, {})
        decision = decisions.get(identity)
        changes.append(dict(requirement_id=identity, evidence_type=req['evidence_type'], description=req['description'],
            from_status=old.get('assessment', {}).get('status'), to_status=rows[identity]['status'],
            action=decision.action if decision else 'REASSESS',
            reason=decision.reason if decision else '核验未得到可靠结果，需要继续确认。',
            reason_code=old.get('reason_code', 'NO_VALID_PRIOR_ASSESSMENT'),
            prior_assessment_id=old.get('assessment_id')))
    body = dict(version=VERSION, case_id=inputs.get('case_id'), from_revision=inputs['case_revision'],
                input_hash=_checksum(inputs), changes=changes, error_code=result.get('error_code'))
    return {**body, 'report_id': _checksum(body)}


def validate_transition(result, inputs, report):
    body = {k: v for k, v in report.items() if k != 'report_id'}
    if (report.get('version') != VERSION or report.get('report_id') != _checksum(body)
            or report.get('input_hash') != _checksum(inputs) or report.get('case_id') != inputs.get('case_id')
            or report.get('from_revision') != inputs['case_revision']):
        raise ValueError('Transition report identity mismatch')
    results = {r['requirement_id']: r for r in result['requirements']}
    requirements = {r['requirement_id']: r for r in inputs['requirements']}
    if report.get('error_code') != result.get('error_code'):
        raise ValueError('Transition error mismatch')
    changes = report['changes']
    if len(changes) != len(results) or {c['requirement_id'] for c in changes} != set(results):
        raise ValueError('Transition coverage mismatch')
    for change in changes:
        old = inputs.get('prior_assessments', {}).get(change['requirement_id'], {})
        requirement = requirements[change['requirement_id']]
        row = results[change['requirement_id']]
        if (change['action'] not in {'PRESERVE', 'REASSESS'} or not isinstance(change.get('reason'), str)
                or not change['reason'].strip() or len(change['reason']) > 300
                or change.get('evidence_type') != requirement['evidence_type']
                or change.get('description') != requirement['description']
                or change.get('prior_assessment_id') != old.get('assessment_id')
                or change.get('reason_code') != old.get('reason_code', 'NO_VALID_PRIOR_ASSESSMENT')):
            raise ValueError('Transition metadata mismatch')
        if change['to_status'] != row['status'] or change['from_status'] != old.get('assessment', {}).get('status'):
            raise ValueError('Transition status mismatch')
        if change['action'] == 'PRESERVE' and (not old.get('preserve_allowed') or row != old.get('assessment')):
            raise ValueError('Preserved assessment changed or dependencies invalidated')


def assess_incremental(inputs, *, model, runtime):
    # 已有的来源、撤回和充分性规范仍适用，只替换模型输出形状。
    prompt = (PROMPT + DELTA_PROMPT
              + '\nSchema:\n' + json.dumps(IncrementalResponse.model_json_schema(), ensure_ascii=False, separators=(',', ':'))
              + '\nEvidence input:\n' + json.dumps(inputs, ensure_ascii=False, separators=(',', ':')))
    def unavailable(code):
        result = abstention(inputs, code)
        return {**result, 'transition_report': _report(inputs, result)}
    if len(prompt) > min(24000, runtime.limits.max_context_chars):
        return unavailable('SEMANTIC_CONTEXT_LIMIT')
    if not inputs['requirements'] and not inputs.get('requested_withdrawal'):
        result = dict(requirements=[], withdrawal_confirmed=False, withdrawal_quote='')
        return {**result, 'transition_report': _report(inputs, result)}
    try:
        if hasattr(model, 'json_mode'):
            model = model.json_mode()
        response = runtime.model_call(model, prompt, purpose='evidence_verification', allow_tools=False, isolated=True)
        delta = IncrementalResponse.model_validate_json(response.content)
        ids = [r.requirement_id for r in delta.requirements]
        if len(ids) != len(set(ids)) or set(ids) != {r['requirement_id'] for r in inputs['requirements']}:
            raise ValueError('Incremental response must cover all requirements once')
        rows = []
        for transition in delta.requirements:
            old = inputs.get('prior_assessments', {}).get(transition.requirement_id, {})
            if transition.action == 'PRESERVE':
                if not old.get('preserve_allowed'):
                    raise ValueError('Invalid preservation')
                rows.append(deepcopy(old['assessment']))
            else:
                row = transition.assessment.model_dump()
                if old.get('preserve_allowed') and not any(c['evidence_id'] == 'incoming' for c in row['citations']):
                    raise ValueError('Changed prior conclusion needs an incoming impact citation')
                rows.append(row)
        result = validate_response(json.dumps(dict(requirements=rows, withdrawal_confirmed=delta.withdrawal_confirmed,
                                      withdrawal_quote=delta.withdrawal_quote), ensure_ascii=False), inputs)
        report = _report(inputs, result, delta.requirements)
        validate_transition(result, inputs, report)
        return {**result, 'transition_report': report}
    except (ValueError, TypeError):
        return unavailable('SEMANTIC_PROTOCOL_INVALID')
    except AgentError as error:
        if error.code not in {c.value for c in ErrorCode if c.name.startswith('MODEL_')}:
            raise
        return unavailable('SEMANTIC_MODEL_UNAVAILABLE')


def committed_transition(prepared, case, assessment_id):
    report = prepared.get('transition_report')
    if report is None:
        return None
    body = {k: deepcopy(v) for k, v in report.items() if k != 'report_id'}
    body.update(case_id=case.case_id, to_revision=case.revision, assessment_id=assessment_id,
                assessment_report_id=report['report_id'], commit_status='COMMITTED',
                policy_outcome=(case.latest_decision or {}).get('outcome'))
    return {**body, 'report_id': _checksum(body)}
