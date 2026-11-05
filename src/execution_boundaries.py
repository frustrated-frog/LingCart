"""持久化的用户执行限制；只收窄权限，不能替代确认。"""
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.errors import AgentError
from src.request_ir import SourceSpan, locate_source_span


class BoundaryChange(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    restriction: Literal['READ_ONLY', 'NO_PAYMENT', 'NO_REFUND', 'NO_SUBMISSION']
    operation: Literal['DENY', 'RETRACT'] = 'DENY'
    source_quote: str = Field(min_length=1, max_length=500)


class ExecutionBoundary(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    restriction: Literal['READ_ONLY', 'NO_PAYMENT', 'NO_REFUND', 'NO_SUBMISSION']
    source: SourceSpan
    origin: Literal['dialogue', 'adaptive', 'control']
    requirement_id: str | None = None
    case_id: str | None = None
    order_id: int | None = None

    @model_validator(mode='after')
    def scope_identity(self):
        if bool(self.case_id) != (self.order_id is not None):
            raise ValueError('Case-scoped boundary requires case and order identities')
        return self


def current_boundaries(context):
    rows = context.snapshot.get('execution_boundaries',
        (context.snapshot.get('conversation') or {}).get('execution_boundaries', []))
    try:
        if not isinstance(rows, list):
            raise ValueError('Boundaries must be a list')
        for row in rows:
            ExecutionBoundary.model_validate(row)
    except (ValueError, TypeError) as error:
        raise AgentError('EXECUTION_BOUNDARY_INVALID', '已保存的执行限制无法校验，不能继续执行', retryable=False) from error
    return deepcopy(rows)


def apply_dialogue_boundaries(context, changes, request, history=None):
    if not context or not changes:
        return
    active = current_boundaries(context)
    seen = set()
    for raw in changes:
        change = BoundaryChange.model_validate(raw)
        if change.restriction in seen:
            raise AgentError('EXECUTION_BOUNDARY_CONFLICT', '同一轮不能重复声明或冲突修改同一执行限制', retryable=False)
        seen.add(change.restriction)
        source = locate_source_span(change.source_quote, request)  # 解除限制也必须是本轮原文。
        active = [row for row in active if row['restriction'] != change.restriction or row.get('case_id')]
        if change.operation == 'DENY':
            active.append(dict(restriction=change.restriction, source=source.model_dump(), origin='dialogue'))
    previous = context.snapshot.get('request_semantics') or (context.snapshot.get('conversation') or {}).get('request_semantics')
    if previous:
        from src.request_semantics import apply_extraction, validate_contract
        prior_rows = validate_contract(previous)
        by_restriction = {BoundaryChange.model_validate(c).restriction: BoundaryChange.model_validate(c) for c in changes}
        revisions = []
        for row in prior_rows:
            change = by_restriction.get(row.restriction)
            if change:
                revisions.append(dict(operation='RETRACT', previous_id=row.requirement_id, source_quote=change.source_quote))
            else:
                revisions.append(dict(operation='PRESERVE', previous_id=row.requirement_id))
        # 新限制由 dialogue 来源持久化；旧语义契约删除对应项，避免下轮 PRESERVE 复活已解除限制。
        if any(row['operation'] == 'RETRACT' for row in revisions):
            context.snapshot['request_semantics'] = apply_extraction(
                {'changes': revisions}, request, history, previous=previous)
    context.snapshot['execution_boundaries'] = active
    context.record('execution_boundaries_updated', boundaries=active, changes=[BoundaryChange.model_validate(c).model_dump() for c in changes])


def apply_semantic_boundaries(context, contract, *, restrict_only=False):
    from src.request_semantics import validate_contract
    validate_contract(contract)
    active = [row for row in current_boundaries(context) if restrict_only or row.get('origin') != 'adaptive']
    for row in contract['requirements']:
        restriction = row['requirement']['restriction']
        if restriction:
            boundary = dict(restriction=restriction, source=deepcopy(row['source']), origin='adaptive',
                            requirement_id=row['requirement']['requirement_id'])
            if boundary not in active:
                active.append(boundary)
    context.snapshot['execution_boundaries'] = active
    context.record('execution_boundaries_updated', boundaries=active, semantics_sha256=contract['sha256'],
                   restrict_only=restrict_only)


def require_allowed(context, name, spec, arguments=None):
    """在账本重放、preflight 和副作用执行之前检查，已有审批不能越过限制。"""
    if context is None:
        return
    from src.after_sales.control_intent import approval_revoked
    from src.runtime import ReplayPolicy
    if spec.replay_policy == ReplayPolicy.REQUIRES_APPROVAL and approval_revoked(context, context.approval):
        raise AgentError('APPROVAL_REVOKED', '这份旧确认已被用户撤回，需要重新核验并单独确认', retryable=False)
    from src.runtime import BusinessEffect, ToolEffect
    if arguments and set(spec.business_effects) & {BusinessEffect.REFUND, BusinessEffect.FINAL_SUBMISSION}:
        from src.after_sales.goals import current_goals
        for goal in current_goals(context).values():
            if goal['mode'] != 'CONTINUE' and (arguments.get('case_id') == goal['case_id']
                    or arguments.get('order_id') == goal['order_id']):
                context.record('case_goal_execution_blocked', tool=name, case_id=goal['case_id'], mode=goal['mode'])
                raise AgentError('CASE_GOAL_HELD', '当前案件目标不允许提交或退款，继续处理也不等于确认授权', retryable=False)
    for row in current_boundaries(context):
        if row.get('case_id') and (not arguments or (arguments.get('case_id') != row['case_id']
                                                   and arguments.get('order_id') != row['order_id'])):
            continue
        restriction = row['restriction']
        blocked = (
            (restriction == 'READ_ONLY' and spec.effect == ToolEffect.WRITE)
            or (restriction == 'NO_PAYMENT' and BusinessEffect.PAYMENT in spec.business_effects)
            or (restriction == 'NO_REFUND' and BusinessEffect.REFUND in spec.business_effects)
            or (restriction == 'NO_SUBMISSION' and bool(set(spec.business_effects) & {BusinessEffect.PAYMENT, BusinessEffect.REFUND, BusinessEffect.FINAL_SUBMISSION}))
        )
        if blocked:
            context.record('execution_boundary_blocked', tool=name, restriction=restriction, source=row['source'])
            raise AgentError('EXECUTION_BOUNDARY_VIOLATION', '当前用户要求禁止此操作：' + row['source']['text'], retryable=False)


def action_restricted(context, action):
    """旧确认恢复展示前也遵守新限制；检查不执行工具。"""
    if not context or not action:
        return False
    from src import tools
    from src.runtime import tool_spec
    name = action.get('tool')
    function = getattr(tools, name, None) if isinstance(name, str) else None
    if function is None:
        return False  # 旧格式确认仍由原有审批校验负责。
    try:
        require_allowed(context, name, tool_spec(name, function), {**action.get('args', {}), **action})
    except AgentError as error:
        if error.code in {'EXECUTION_BOUNDARY_VIOLATION', 'CASE_GOAL_HELD', 'APPROVAL_REVOKED'}:
            return True
        raise
    return False
