"""持久化的用户执行限制；只收窄权限，不能替代确认。"""
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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
    origin: Literal['dialogue', 'adaptive']
    requirement_id: str | None = None


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
        active = [row for row in active if row['restriction'] != change.restriction]
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


def apply_semantic_boundaries(context, contract):
    from src.request_semantics import validate_contract
    validate_contract(contract)
    active = [row for row in current_boundaries(context) if row.get('origin') != 'adaptive']
    for row in contract['requirements']:
        restriction = row['requirement']['restriction']
        if restriction:
            active.append(dict(restriction=restriction, source=deepcopy(row['source']), origin='adaptive',
                               requirement_id=row['requirement']['requirement_id']))
    context.snapshot['execution_boundaries'] = active
    context.record('execution_boundaries_updated', boundaries=active, semantics_sha256=contract['sha256'])


def require_allowed(context, name, spec):
    """在账本重放、preflight 和副作用执行之前检查，已有审批不能越过限制。"""
    if context is None:
        return
    from src.runtime import ToolEffect
    for row in current_boundaries(context):
        restriction = row['restriction']
        blocked = (
            (restriction == 'READ_ONLY' and spec.effect == ToolEffect.WRITE)
            or (restriction == 'NO_PAYMENT' and (name == 'process_payment' or spec.approval_action == 'PROCESS_PAYMENT'))
            or (restriction == 'NO_REFUND' and (name == 'initiate_refund' or spec.approval_action == 'INITIATE_REFUND'))
            or (restriction == 'NO_SUBMISSION' and spec.effect == ToolEffect.WRITE and spec.approval_action is not None)
        )
        if blocked:
            context.record('execution_boundary_blocked', tool=name, restriction=restriction, source=row['source'])
            raise AgentError('EXECUTION_BOUNDARY_VIOLATION', '当前用户要求禁止此操作：' + row['source']['text'], retryable=False)
