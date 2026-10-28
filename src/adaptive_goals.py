"""Evidence contracts for model-authored outcomes, never an execution plan."""
from copy import deepcopy

from src.task_plan import _facts, verify_observation


def new_ledger(goal):
    return [dict(objective_id=o['id'], status='PENDING', evidence_refs=[], failures=[],
                 target_count=o.get('quantity', 1) if o['kind'] == 'recommend' else 1,
                 selected_product_ids=[])
            for o in goal['objectives']]


def record_outcome(state, objective_id, step):
    row = next(row for row in state['objective_ledger'] if row['objective_id'] == objective_id)
    if step.status == 'SUCCEEDED':
        product_id = ((step.result or {}).get('selected_product') or {}).get('product_id')
        if product_id and product_id not in row.setdefault('selected_product_ids', []):
            row['selected_product_ids'].append(product_id)
            row.setdefault('evidence_refs', []).append(step.step_id)
        target = int(row.get('target_count', 1))
        row.update(status='VERIFIED' if len(row['selected_product_ids']) >= target else 'PENDING', failures=[])
        return
    failures = (step.result or {}).get('failures', []) or ([step.error['code']] if step.error else [])
    row.update(status='PENDING' if row.get('evidence_refs') else 'INVALIDATED', failures=failures)


def _claim_error(state, objective, step, indexed):
    if step is not None and objective['question'] is not None and step.tool == 'answer_faq' and step.args.get('question') != objective['question']:
        return 'EVIDENCE_TARGET_MISMATCH'
    if step is None or step.status not in {'SUCCEEDED', 'REUSED'} or step.result is None:
        return 'EVIDENCE_NOT_SUCCESSFUL'
    if objective['kind'] == 'recommend':
        if (step.tool != 'select_product' or step.args.get('objective_id') != objective['id']
                or step.args.get('goal_revision') != state['goal_revision']
                or not step.result.get('selected_product') or step.result.get('failures')):
            return 'RECOMMENDATION_NOT_VERIFIED'
        from src.constraint_bindings import fingerprint
        if (step.result.get('binding_revision', 0) != state.get('binding_revision', 0)
                or (state.get('constraint_bindings') and
                    step.result.get('bindings_sha256') != fingerprint(state['constraint_bindings']))):
            return 'RECOMMENDATION_BINDING_STALE'
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
    try:
        verify_observation(source)
    except Exception as error:
        from src.errors import AgentError
        if not isinstance(error, AgentError):
            raise
        return error.code
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
        refs = list(claim.evidence_refs)
        if objectives[identity]['kind'] == 'recommend':
            selections = [indexed[ref] for ref in refs if ref in indexed and indexed[ref].tool == 'select_product'
                          and _claim_error(state, objectives[identity], indexed[ref], indexed) is None]
            by_product = {}
            for selection in selections:
                product_id = selection.result['selected_product']['product_id']
                by_product[product_id] = selection
            selection_refs = {selection.step_id for selection in selections}
            support = [ref for ref in refs if ref not in selection_refs]
            valid_support = all(ref in indexed and indexed[ref].status in {'SUCCEEDED', 'REUSED'}
                and ref in state.get('read_evidence_refs', []) and isinstance(indexed[ref].result, dict)
                and any(p.get('product_id') in by_product for p in indexed[ref].result.get('products', []))
                for ref in support)
            errors = [_claim_error(state, objectives[identity], indexed.get(ref), indexed)
                      for ref in refs if ref in selection_refs]
            if not valid_support:
                errors.extend(_claim_error(state, objectives[identity], indexed.get(ref), indexed)
                              for ref in support)
            target = int(objectives[identity].get('quantity', 1))
            existing_errors = [error for error in errors if error]
            if not existing_errors and len(by_product) != target:
                errors.append('AMBIGUOUS_RECOMMENDATION' if target == 1
                              else 'RECOMMENDATION_CARDINALITY_MISMATCH')
            refs = [selection.step_id for selection in by_product.values()]
        else:
            errors = [_claim_error(state, objectives[identity], indexed.get(ref), indexed) for ref in refs]
        errors = sorted({error for error in errors if error})
        row.update(status='INVALIDATED' if errors else 'VERIFIED', evidence_refs=refs, failures=errors,
                   selected_product_ids=[indexed[ref].result['selected_product']['product_id'] for ref in refs]
                   if objectives[identity]['kind'] == 'recommend' and not errors else row.get('selected_product_ids', []))
        failures.extend(dict(objective_id=identity, code=error) for error in errors)
    return failures


def verification(state, *, complete=False):
    rows = deepcopy(state.get('objective_ledger', []))
    return dict(all_goals_satisfied=complete and bool(rows) and all(r['status'] == 'VERIFIED' for r in rows),
                side_effects_executed=False, goal_revision=state['goal_revision'], objectives=rows,
                delivery_committed=bool(complete),
                incomplete_objectives=[r['objective_id'] for r in rows
                                       if not complete or r['status'] != 'VERIFIED'])


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
        if objective['kind'] == 'recommend':
            # A selection remains provisional until the compound completion refresh.
            steps = [indexed[reference] for reference in row['evidence_refs']]
            fact = ('\n'.join(recommendation_text(step) for step in steps) if complete
                    else f"已选出 {len(steps)}/{row.get('target_count', 1)} 个候选，任务完成前仍需重新核验价格、库存及约束。")
            evidence = '、'.join(step.step_id for step in steps)
        else:
            step = indexed[row['evidence_refs'][-1]]
            while step.tool == 'read_evidence':
                step = indexed[step.args['reference']]
            fact = _facts(step)
            if isinstance(step.result, dict) and step.result.get('observation'):
                fact += f"（查询时间：{step.result['observation']['timestamp']}）"
            evidence = step.step_id
        parts.append(f"{objective['description']}：\n{fact}\n证据步骤：{evidence}。")
    if missing:
        parts.append('尚未完成：' + '；'.join(missing) + '。')
    return '\n\n'.join(parts)
