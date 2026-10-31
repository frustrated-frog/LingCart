"""持久化的失败身份与有界恢复策略；不授予工具或副作用权限。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

from src.constraint_bindings import fingerprint
from src.errors import AgentError


@dataclass(frozen=True)
class FailureSignature:
    stage: str
    code: str
    target_sha256: str
    basis_sha256: str
    version: str = "failure-signature/v1"

    @property
    def sha256(self):
        return fingerprint(asdict(self))


def recovery_basis(state, plan=None, *, product_id=None):
    basis = dict(request_sha256=fingerprint(state['request']), goal_revision=state.get('goal_revision', 0),
                 goal_sha256=fingerprint(state.get('goal')), binding_revision=state.get('binding_revision', 0),
                 bindings_sha256=fingerprint(state.get('constraint_bindings', {})))
    if product_id is not None and plan is not None:
        # 相同查询重读、步骤编号和时间戳不算新证据；商品事实改变才允许重试。
        products = {}
        for step in plan.steps:
            if (step.tool not in {'search_products', 'discover_products', 'compare_products'}
                    or step.status not in {'SUCCEEDED', 'REUSED'} or not isinstance(step.result, dict)):
                continue
            for product in step.result.get('products', []):
                if product.get('product_id') == product_id:
                    products[product_id] = product
        basis['product_evidence_sha256'] = fingerprint(products)
    return basis


def recovery_target(decision):
    if decision is None:
        return {"protocol": "decision"}
    payload = decision.model_dump(exclude_none=True)
    # 改写理由、笔记或规划文案不是修正执行参数。
    for field in ('reason', 'working_memory', 'next_steps'):
        payload.pop(field, None)
    return payload


def signature(stage, code, target, basis):
    return FailureSignature(stage, code, fingerprint(target), fingerprint(basis))


def previous_failure(context, failure):
    return context.snapshot.get('failure_ledger', {}).get(failure.sha256)


def _action(code):
    if code in {'MODEL_FORMAT_ERROR', 'AUDIT_FORMAT_ERROR'}:
        return 'REPAIR_PROTOCOL'
    if code in {'UNKNOWN_TOOL', 'INVALID_TOOL_ARGUMENTS'}:
        return 'REPAIR_ARGUMENTS'
    if code in {'REQUEST_AUDIT_REJECTED', 'REQUEST_IR_COMPILE_ERROR'}:
        return 'REVISE_INTERPRETATION'
    if 'BINDING' in code:
        return 'REVISE_BINDING'
    return 'CHANGE_CANDIDATE_OR_EVIDENCE'


def register_failure(context, failure, *, attempt_id, max_recoveries=2):
    """Atomically persist the event and counter; replay of one attempt is idempotent."""
    ledger = context.snapshot.setdefault('failure_ledger', {})
    old = ledger.get(failure.sha256)
    attempt = fingerprint(str(attempt_id))
    if old and (attempt in old['attempts'] or old['blocked']):
        return deepcopy(old)
    if not old and len(ledger) >= 64:
        raise AgentError('FAILURE_LEDGER_BUDGET_EXHAUSTED', '失败恢复账本已达到运行上限', retryable=False)
    attempts = [*(old or {}).get('attempts', []), attempt]
    occurrence = len(attempts)
    row = dict(signature=asdict(failure), signature_sha256=failure.sha256,
               attempts=attempts, occurrence=occurrence, max_recoveries=max_recoveries,
               blocked=occurrence > max_recoveries,
               recovery_action='STOP' if occurrence > max_recoveries else _action(failure.code))
    ledger[failure.sha256] = row
    context.record('failure_registered', **{key: deepcopy(value) for key, value in row.items() if key != 'attempts'})
    return deepcopy(row)


def feedback(row):
    return {key: deepcopy(row[key]) for key in ('signature_sha256', 'occurrence', 'max_recoveries', 'blocked', 'recovery_action')}
