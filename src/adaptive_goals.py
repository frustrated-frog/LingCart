"""Evidence contracts for model-authored outcomes, never an execution plan."""
from copy import deepcopy

from src.task_plan import _facts, verify_observation


def new_ledger(goal):
    return [dict(objective_id=o['id'], status='PENDING', evidence_refs=[], failures=[])
            for o in goal['objectives']]


def record_outcome(state, objective_id, step):
    row = next(row for row in state['objective_ledger'] if row['objective_id'] == objective_id)
    row.update(status='VERIFIED' if step.status == 'SUCCEEDED' else 'INVALIDATED',
               evidence_refs=[step.step_id], failures=(step.result or {}).get('failures', [])
               or ([step.error['code']] if step.error else []))


def _claim_error(state, objective, step, indexed):
    if step is None or step.status not in {'SUCCEEDED', 'REUSED'} or step.result is None:
        return 'EVIDENCE_NOT_SUCCESSFUL'
    if objective['kind'] == 'recommend':
        if (step.tool != 'select_product' or step.args.get('objective_id') != objective['id']
                or step.args.get('goal_revision') != state['goal_revision']
                or not step.result.get('selected_product') or step.result.get('failures')):
            return 'RECOMMENDATION_NOT_VERIFIED'
        return None
    if step.step_id not in state.get('read_evidence_refs', []):
        return 'EVIDENCE_NOT_READ'
    # Archive reads preserve historical data; they are not independent business evidence.
    source = step
    while source.tool == 'read_evidence':
        source = indexed.get(source.args.get('reference'))
        if source is None or source.status not in {'SUCCEEDED', 'REUSED'}:
            return 'EVIDENCE_NOT_SUCCESSFUL'
    allowed = ({'get_order_status', 'get_order_details', 'track_shipment'}
               if objective['kind'] == 'get_order_status' else {objective['kind']})
    if source.tool not in allowed or objective['kind'] == 'unresolved':
        return 'EVIDENCE_KIND_MISMATCH'
    if objective['order_id'] is not None and source.args.get('order_id') != objective['order_id']:
        return 'EVIDENCE_TARGET_MISMATCH'
    if objective['question'] is not None and source.args.get('question') != objective['question']:
        return 'EVIDENCE_TARGET_MISMATCH'
    if objective['kind'] == 'compare_products':
        expected = set(objective['product_ids'] or source.args['product_ids'])
        actual = {p['product_id'] for p in step.result.get('products', [])}
        if expected != actual:
            return 'EVIDENCE_TARGET_MISMATCH'
    verify_observation(source)
    return None


def check_claims(state, plan, claims):
    """Check each submitted claim against its typed outcome and actual evidence."""
    indexed = {s.step_id: s for s in plan.steps}
    objectives = {o['id']: o for o in state['goal']['objectives']}
    submitted = {c.objective_id: c for c in claims}
    failures = [dict(objective_id=key, code='UNKNOWN_OBJECTIVE') for key in submitted if key not in objectives]
    for row in state['objective_ledger']:
        identity = row['objective_id']
        claim = submitted.get(identity)
        if claim is None:
            failures.append(dict(objective_id=identity, code='MISSING_COMPLETION_CLAIM'))
            continue
        errors = [_claim_error(state, objectives[identity], indexed.get(ref), indexed) for ref in claim.evidence_refs]
        if objectives[identity]['kind'] == 'recommend' and len(claim.evidence_refs) != 1:
            errors.append('AMBIGUOUS_RECOMMENDATION')
        errors = sorted({error for error in errors if error})
        row.update(status='INVALIDATED' if errors else 'VERIFIED', evidence_refs=list(claim.evidence_refs), failures=errors)
        failures.extend(dict(objective_id=identity, code=error) for error in errors)
    return failures


def verification(state, *, complete=False):
    rows = deepcopy(state.get('objective_ledger', []))
    return dict(all_goals_satisfied=complete and bool(rows) and all(r['status'] == 'VERIFIED' for r in rows),
                side_effects_executed=False, goal_revision=state['goal_revision'], objectives=rows,
                incomplete_objectives=[r['objective_id'] for r in rows if r['status'] != 'VERIFIED'])


def recommendation_text(step):
    result = step.result
    product = result['selected_product']
    return (f"本地演示目录推荐：{product['name']}（{product['product_id']}），¥{product['price']:.2f}，"
            f"核验库存 {product['stock']}。\n选择依据：{result['rationale']}\n"
            f"证据：{', '.join(result['evidence_refs'])}；核验时间：{result['observation']['timestamp']}。")


def outcome_report(state, plan, *, complete=False):
    indexed = {s.step_id: s for s in plan.steps}
    objectives = {o['id']: o for o in (state.get('goal') or {}).get('objectives', [])}
    parts, missing = [], []
    for row in state.get('objective_ledger', []):
        objective = objectives[row['objective_id']]
        if row['status'] != 'VERIFIED':
            missing.append(objective['description'])
            continue
        step = indexed[row['evidence_refs'][-1]]
        while step.tool == 'read_evidence':
            step = indexed[step.args['reference']]
        if objective['kind'] == 'recommend':
            # A selection remains provisional until the compound completion refresh.
            fact = recommendation_text(step) if complete else '已选出候选，任务完成前仍需重新核验价格、库存及约束。'
        else:
            fact = _facts(step)
            if isinstance(step.result, dict) and step.result.get('observation'):
                fact += f"（查询时间：{step.result['observation']['timestamp']}）"
        parts.append(f"{objective['description']}：\n{fact}\n证据步骤：{step.step_id}。")
    if missing:
        parts.append('尚未完成：' + '；'.join(missing) + '。')
    return '\n\n'.join(parts)
