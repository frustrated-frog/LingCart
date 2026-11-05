"""区分等待授权、明确禁令和撤回授权；两次解释一致后编译到既有边界。"""
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.after_sales.turn_verification import CaseTurnMismatch
from src.request_ir import locate_source_span


class ControlIntent(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    kind: Literal['REQUIRE_CONFIRMATION', 'PROHIBIT', 'RELEASE_PROHIBITION', 'REVOKE_APPROVAL']
    scope: Literal['CASE', 'SESSION']
    case_id: str | None = Field(None, min_length=1, max_length=64)
    restriction: Literal['READ_ONLY', 'NO_PAYMENT', 'NO_REFUND', 'NO_SUBMISSION'] | None = None

    @model_validator(mode='after')
    def coherent(self):
        if (self.scope == 'CASE') != bool(self.case_id):
            raise ValueError('CASE needs case_id; SESSION has no case_id')
        if (self.kind in {'PROHIBIT', 'RELEASE_PROHIBITION'}) != bool(self.restriction):
            raise ValueError('Only prohibition/release names a restriction')
        return self


class ControlChange(ControlIntent):
    source_quote: str = Field(min_length=1, max_length=500)


def enabled(context=None):
    from src.runtime import current_runtime
    context = context or current_runtime()
    return bool(context and getattr(context.mechanisms.get('planning'), 'control_intents', False))


INSTRUCTION = '''Existing cases use control_intents:[{kind,scope:CASE|SESSION,case_id?,restriction?,source_quote}]; no execution_boundaries.
CASE needs a listed case_id; SESSION omits it. Include AFTER_SALES_CONTINUE.
REQUIRE_CONFIRMATION: assess/propose, not consent. Missing consent or asking first is not a ban.
PROHIBIT: explicit persistent ban. RELEASE_PROHIBITION: explicit release. Both need restriction
READ_ONLY|NO_PAYMENT|NO_REFUND|NO_SUBMISSION. A new goal cannot release a ban.
REVOKE_APPROVAL withdraws old approval; future assessment remains allowed. No kind grants authorization.
Draft/pause/resume: AFTER_SALES_CONTINUE, never cancel/confirm. Keep explicit material_changes.
Unclear desired outcome: CLARIFY plus clear material updates. Case goal mode is not a ban.
'''

SEMANTIC_INSTRUCTION = '''Represent control language as a separate EXECUTION_BOUNDARY requirement with control_intent:
{kind:REQUIRE_CONFIRMATION|PROHIBIT|RELEASE_PROHIBITION|REVOKE_APPROVAL,scope:CASE|SESSION,case_id?,restriction?};
the requirement's top-level restriction must be null. CASE requires the exact trusted case_id, SESSION omits it.
Only PROHIBIT/RELEASE_PROHIBITION carry a nested restriction (READ_ONLY|NO_PAYMENT|NO_REFUND|NO_SUBMISSION).
REQUIRE_CONFIRMATION means the user has not authorized effects but assessment and proposing confirmation may continue.
A statement that this message is not consent, or asking to be consulted, is not an explicit ban. Never equate missing consent
with PROHIBIT. REVOKE_APPROVAL withdraws old approval, not future assessment. PROHIBIT remains active until an explicit
release: retract the old prohibition requirement and add RELEASE_PROHIBITION for the exact scope/effect. A new goal or
REQUIRE_CONFIRMATION alone cannot retract a prohibition. Preserve old facts and outcomes. No control grants authorization.
'''


def _key(value):
    return (value.get('case_id'), value.get('restriction'))


def _validate_scope(intent, cases):
    if intent.scope == 'CASE' and intent.case_id not in cases:
        raise CaseTurnMismatch('控制意图不属于当前会话案件', fields={'case_id': intent.case_id}, question='请明确要调整哪个案件。')


def restrictive_controls(context, contract, cases):
    """独立解释得到的明确禁令先收窄；解除只在两次解释一致后执行。"""
    if not enabled(context): return
    from src.execution_boundaries import current_boundaries
    active = current_boundaries(context)
    for row in contract['requirements']:
        raw = row['requirement'].get('control_intent')
        if not raw: continue
        intent = ControlIntent.model_validate(raw)
        _validate_scope(intent, cases)
        if intent.kind != 'PROHIBIT': continue
        boundary = dict(restriction=intent.restriction, origin='control', source=deepcopy(row['source']),
                        requirement_id=row['requirement']['requirement_id'])
        if intent.case_id:
            boundary.update(case_id=intent.case_id, order_id=cases[intent.case_id]['order_id'])
        if boundary not in active: active.append(boundary)
    context.snapshot['execution_boundaries'] = active


def verify_controls(context, understanding, rows, contract, cases, request):
    if not enabled(context): return
    changed = {r['requirement_id'] for r in contract['revisions'] if r['operation'] in {'ADD', 'REPLACE'}}
    expected = [row.control_intent for row in rows if row.requirement_id in changed and row.control_intent]
    actual = [ControlIntent.model_validate(c.model_dump(exclude={'source_quote'})) for c in understanding.control_intents]
    for intent in [*actual, *expected]: _validate_scope(intent, cases)
    targets = {o.case_id for o in understanding.objectives if o.kind == 'AFTER_SALES_CONTINUE'}
    if any(c.scope == 'CASE' and c.case_id not in targets for c in actual):
        raise CaseTurnMismatch('控制范围未绑定已解析的案件目标', fields={}, question='请明确这项控制针对哪个案件。')
    def identity(intent): return (intent.kind, intent.scope, intent.case_id, intent.restriction)
    if len(set(map(identity, actual))) != len(actual) or sorted(map(identity, actual), key=str) != sorted(map(identity, expected), key=str):
        raise CaseTurnMismatch('控制意图与独立解释不一致', fields={'independent_controls':[c.model_dump() for c in expected]},
                               question='请明确是继续核验并等待确认，还是禁止后续提交。')
    releases = {_key(c.model_dump()) for c in expected if c.kind == 'RELEASE_PROHIBITION'}
    previous = context.snapshot.get('after_sales_previous_semantics') or {}
    current_prohibitions = {_key(row.control_intent.model_dump()) for row in rows
                            if row.control_intent and row.control_intent.kind == 'PROHIBIT'}
    current_prohibitions |= {(None, row.restriction) for row in rows if row.restriction}
    for old in previous.get('requirements', []):
        requirement = old['requirement']
        control = requirement.get('control_intent') or {}
        key = (_key(control) if control.get('kind') == 'PROHIBIT'
               else (None, requirement['restriction']) if requirement.get('restriction') else None)
        if key and key not in current_prohibitions and key not in releases:
            raise CaseTurnMismatch('旧禁令不能由缺少授权或目标修订隐式解除', fields={}, question='请明确是否解除原有禁令及其范围。')
    if any(row.control_intent and row.control_intent.kind == 'PROHIBIT' and _key(row.control_intent.model_dump()) in releases for row in rows):
        raise CaseTurnMismatch('解除限制仍保留了同范围禁令', fields={}, question='请明确要解除哪项原有限制。')
    if understanding.execution_boundaries or any(row.restriction for row in rows if row.requirement_id in changed):
        raise CaseTurnMismatch('当前控制协议不能混入全局限制旧字段', fields={'protocol':'control_intents'},
                               question='请明确本轮操作对哪个案件生效。')


def commit_controls(context, understanding, request):
    if not enabled(context): return
    from src.execution_boundaries import current_boundaries
    from src.after_sales.context import current_case_context
    cases = {c['case_id']: c for c in current_case_context() if c.get('case_id')}
    active = current_boundaries(context)
    accepted = []
    revoked = deepcopy(context.snapshot.get('revoked_approvals', (context.snapshot.get('conversation') or {}).get('revoked_approvals', [])))
    pending = (context.snapshot.get('conversation') or {}).get('pending_action') or {}
    for control in understanding.control_intents:
        _validate_scope(control, cases)
        source = locate_source_span(control.source_quote, request).model_dump(mode='json')
        if control.kind == 'RELEASE_PROHIBITION':
            active = [b for b in active if _key(b) != _key(control.model_dump())]
        item = dict(control=control.model_dump(exclude={'source_quote'}), source=source,
                    order_id=cases[control.case_id]['order_id'] if control.case_id else None)
        if control.kind == 'REVOKE_APPROVAL' and pending.get('action_id') and matches(item, pending):
            if pending['action_id'] not in revoked: revoked.append(pending['action_id'])
        accepted.append(item)
        context.record('case_control_interpreted', **item, status='VERIFIED',
                       effect={'REQUIRE_CONFIRMATION':'ALLOW_ASSESSMENT_REQUIRE_CONFIRMATION', 'PROHIBIT':'PERSIST_RESTRICTION',
                               'RELEASE_PROHIBITION':'RELEASE_NAMED_RESTRICTION', 'REVOKE_APPROVAL':'INVALIDATE_OLD_APPROVAL'}[control.kind])
    context.checkpoint(execution_boundaries=active, accepted_controls=accepted, revoked_approvals=revoked)


def matches(item, arguments):
    from src.execution_boundaries import scope_matches
    control = item['control']
    return control['scope'] == 'SESSION' or scope_matches(control['case_id'], item['order_id'], arguments)


def approval_revoked(context, approval):
    if not context or not approval: return False
    revoked = context.snapshot.get('revoked_approvals', (context.snapshot.get('conversation') or {}).get('revoked_approvals', []))
    return approval.get('action_id') in revoked


def confirmation_revoked_this_turn(context, arguments):
    return bool(context and any(item['control']['kind'] == 'REVOKE_APPROVAL' and matches(item, arguments)
                               for item in context.snapshot.get('accepted_controls', [])))
