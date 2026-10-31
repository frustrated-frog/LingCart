"""售后复用独立需求契约：核对业务结果，事实和未知材料不会成为伪目标。"""
from collections import Counter
from copy import deepcopy

from src.errors import AgentError
from src.request_semantics import validate_contract
from src.request_semantics_runtime import extract_contract


def verify_intent(context, understanding, request, history):
    objectives = [o for o in understanding.objectives if o.kind in {'AFTER_SALES_CASE', 'AFTER_SALES_CONTINUE'}]
    if not objectives or context is None:
        return
    if any((o.kind == 'AFTER_SALES_CASE' and (not o.order_id or not o.reason_code))
           or (o.kind == 'AFTER_SALES_CONTINUE' and not o.case_id) for o in objectives):
        return  # 缺失受理参数由既有入口澄清，不产生写入。
    if 'after_sales_previous_semantics' not in context.snapshot:
        previous = context.snapshot.get('request_semantics') or (context.snapshot.get('conversation') or {}).get('request_semantics')
        context.snapshot['after_sales_previous_semantics'] = deepcopy(previous)
    # 同一轮规划修复保持提取输入不变，复用独立结果，不把刚接受的输出当成上一轮输入。
    previous = context.snapshot['after_sales_previous_semantics']
    contract = extract_contract(context, request, history, previous=previous)
    from src.execution_boundaries import apply_semantic_boundaries
    apply_semantic_boundaries(context, contract, restrict_only=True)
    rows = validate_contract(contract)
    from src.after_sales.context import current_case_context
    cases = {case['case_id']: case for case in current_case_context()}
    expected = Counter()
    for objective in objectives:
        if objective.kind == 'AFTER_SALES_CASE':
            expected[(objective.order_id, objective.reason_code)] += 1
        else:
            case = cases.get(objective.case_id)
            if not case or not case.get('reason_code'):
                raise AgentError('CASE_CONTEXT_CONFLICT', '继续案件缺少当前会话身份', retryable=False)
            expected[(case['order_id'], case['reason_code'])] += 1
    actual = Counter((row.outcome.order_id, row.outcome.reason_code) for row in rows
                     if row.kind == 'OUTCOME' and row.outcome.kind == 'after_sales_case')
    if actual != expected:
        raise AgentError('REQUEST_SEMANTICS_MISMATCH', '独立售后需求与案件目标不一致，尚未执行写入', retryable=False)
    restrictions = {row.restriction for row in rows if row.restriction}
    for change in understanding.execution_boundaries:
        if (change.operation == 'DENY') != (change.restriction in restrictions):
            raise AgentError('REQUEST_SEMANTICS_MISMATCH', '执行限制与独立需求不一致', retryable=False)
    # 新提取的限制只有完整语义接受后才允许释放。
    apply_semantic_boundaries(context, contract)
    context.snapshot['request_semantics'] = deepcopy(contract)
    context.snapshot['after_sales_intent_contract'] = deepcopy(contract)
    context.record('after_sales_intent_verified', contract_sha256=contract['sha256'],
                   outcomes=[dict(order_id=order, reason_code=reason) for order, reason in expected],
                   facts=[row.requirement_id for row in rows if row.kind in {'USER_FACT', 'UNKNOWN'}])
