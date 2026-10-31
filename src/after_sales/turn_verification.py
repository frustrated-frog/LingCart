"""用独立 HUMAN 需求核对材料归属；修复反馈不反向修改独立解释。"""
import json

from src.errors import AgentError


class CaseTurnMismatch(AgentError):
    def __init__(self, message, *, fields, question, instruction=None):
        super().__init__('REQUEST_SEMANTICS_MISMATCH', message, retryable=False)
        self.feedback = {'issue': message, 'fields': fields, 'instruction': instruction or
                         'Correct only the inconsistent interpretation using HUMAN evidence. '
                         'Do not invent restrictions, facts or authorization.'}
        self.question = question

    def repair_text(self):
        return json.dumps(self.feedback, ensure_ascii=False, separators=(',', ':'))


def trusted_case_scope(cases):
    return [dict(case_id=c['case_id'], order_id=c['order_id'], revision=c['revision'],
                 requirements=[{k: r.get(k) for k in ('requirement_id', 'description', 'evidence_type')}
                               for r in c.get('requirements', [])],
                 pending_question={k: (c.get('pending_clarification') or {}).get(k)
                                   for k in ('requirement_id', 'question')}) for c in cases.values()]


def material_protocol_errors(payload, issues):
    """只识别材料协议错误，不把其他目标或权限错误伪装成材料澄清。"""
    objectives = payload.get('objectives', []) if isinstance(payload, dict) else []
    fields = {'evidence_updates', 'evidence_text', 'evidence_readiness_claims', 'requirement_id',
              'clarification_id', 'revoke_evidence_id'}
    messages = ('Batch evidence updates use', 'Each policy requirement may be updated',
                'A readiness update cannot add or revoke evidence')
    for issue in issues:
        loc = issue['loc']
        if (len(loc) < 2 or loc[0] != 'objectives' or not isinstance(loc[1], int)
                or loc[1] >= len(objectives) or not isinstance(objectives[loc[1]], dict)
                or objectives[loc[1]].get('kind') != 'AFTER_SALES_CONTINUE'):
            return None
        if not ((len(loc) > 2 and loc[2] in fields)
                or (len(loc) == 2 and any(message in issue['msg'] for message in messages))):
            return None
    return [{'path': '.'.join(map(str, issue['loc'])), 'message': issue['msg']} for issue in issues] if issues else None


def verify_material_scope(context, objectives, rows, cases, request):
    facts = [row for row in rows if row.material_scope is not None and row.source_quote in request]
    for row in facts:
        scope = row.material_scope
        case = cases.get(scope.case_id)
        if not case or scope.requirement_id not in {r['requirement_id'] for r in case['requirements']}:
            raise CaseTurnMismatch('独立材料归属不属于当前会话政策', fields={'material_scope': scope.model_dump()},
                                   question='这份材料对应哪个案件和材料要求？')

    covered = set()
    for objective in objectives:
        if objective.kind != 'AFTER_SALES_CONTINUE':
            continue
        updates = [dict(requirement_id=u.requirement_id, disposition=u.disposition, quote=u.source_quote)
                   for u in objective.evidence_updates]
        requirement_by_type = {r['evidence_type']: r['requirement_id'] for r in cases[objective.case_id]['requirements']}
        updates.extend(dict(requirement_id=requirement_by_type.get(c.evidence_type), disposition='DEFERRED', quote=c.source_quote)
                       for c in objective.evidence_readiness_claims)
        if objective.evidence_text:
            updates.append(dict(requirement_id=objective.requirement_id, disposition='PROVIDED', quote=objective.evidence_text))
        for update in updates:
            matches = [row for row in facts if row.material_scope.case_id == objective.case_id
                       and row.material_scope.disposition == update['disposition']
                       and (row.source_quote in update['quote'] or update['quote'] in row.source_quote)]
            implicit = (objective.evidence_text and not objective.evidence_updates
                        and (objective._argument_provenance.get('requirement_id') or {}).get('binding_mode') == 'IMPLICIT')
            if not implicit and any(row.material_scope.requirement_id == update['requirement_id'] for row in matches):
                matches = [row for row in matches if row.material_scope.requirement_id == update['requirement_id']]
            targets = {row.material_scope.requirement_id for row in matches}
            if len(targets) != 1:
                raise CaseTurnMismatch('材料归属缺失或包含多个要求',
                    fields={'source_quote': update['quote'], 'independent_targets': sorted(targets)},
                    instruction=('Split materials into evidence_updates using policy IDs and verbatim quotes; '
                                 'objective.source_quote covers all update quotes. Omit legacy fields. '
                                 'No invented facts or authorization.') if len(targets) > 1 else None,
                    question='请说明这段材料对应哪一项要求；若涉及多项，请分别说明。')
            target = next(iter(targets))
            if target != update['requirement_id']:
                binding = objective._argument_provenance.get('requirement_id') or {}
                if (objective.evidence_text and not objective.evidence_updates
                        and binding.get('resolver') == 'pending_clarification_ledger'
                        and binding.get('binding_mode') == 'IMPLICIT'):
                    old = objective.requirement_id
                    objective.requirement_id = target
                    objective.clarification_id = None
                    objective._argument_provenance.pop('clarification_id', None)
                    objective._argument_provenance['requirement_id'] = dict(
                        resolver='independent_material_scope', resolver_version='1', normalized_value=target,
                        source_quote=update['quote'])
                    context.record('case_material_target_reconciled', case_id=objective.case_id,
                                   from_requirement_id=old, to_requirement_id=target, source_quote=update['quote'])
                else:
                    raise CaseTurnMismatch('显式材料目标与独立归属不一致',
                        fields={'source_quote': update['quote'], 'proposed_requirement_id': update['requirement_id'],
                                'independent_requirement_id': target},
                        question='这段材料具体对应哪一项要求？现有两个归属判断不一致。')
            covered.update(row.requirement_id for row in matches)
    continued_cases = {o.case_id for o in objectives if o.kind == 'AFTER_SALES_CONTINUE'}
    missing = [row for row in facts if row.material_scope.case_id in continued_cases and row.requirement_id not in covered]
    if missing:
        raise CaseTurnMismatch('本轮材料更新未完整覆盖独立事实',
            fields={'missing_updates': [dict(**row.material_scope.model_dump(), source_quote=row.source_quote) for row in missing]},
            question='本轮有材料尚未明确归属，请分别说明要补充或延期的材料。')
    context.record('case_material_scope_verified', updates=[dict(**row.material_scope.model_dump(), source_quote=row.source_quote)
                                                          for row in facts if row.requirement_id in covered])
