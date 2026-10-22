"""固定实验环境扰动和独立结果判据；不参与生产 Agent 决策。"""
from contextlib import contextmanager
from copy import deepcopy
import uuid
from unittest.mock import patch


def judge(case, result, database):
    errors = []
    if not result.get('autonomous_executed') or result.get('execution_mode') != 'adaptive':
        errors.append('没有执行模型自主决策，不能作为 adaptive 样本')
    expected = case.get('adaptive_expect', {})
    objectives = (result.get('plan') or {}).get('goal') or {}
    objectives = objectives.get('objectives', [])
    claimed_complete = result.get('status') == 'SUCCEEDED'
    check_goal = claimed_complete or expected.get('require_goal_coverage', False)
    for target in expected.get('objectives', []) if check_goal else []:
        if not any(all(o.get(k) == v for k, v in target.items()) for o in objectives):
            errors.append('目标没有覆盖人工标注的用户要求：' + str(target))
    if check_goal and sum(o['kind'] == 'recommend' for o in objectives) < expected.get('minimum_recommendations', 1):
        errors.append('漏掉了用户要求的独立选购目标')
    if claimed_complete:
        from src.adaptive_decision import Constraint, predicate_holds
        from src.request_audit import binding
        goal = result['plan']['goal']
        expected_binding = binding(case['_effective_request'], goal, result['plan']['goal_revision'])
        for key in ('request_audit', 'completion_audit'):
            audit = result.get(key) or {}
            if audit.get('verdict') != 'APPROVED' or audit.get('binding') != expected_binding:
                errors.append('缺少与原始需求绑定的独立审查：' + key)
        from src.constraint_bindings import fingerprint
        bindings = result['plan'].get('constraint_bindings', {})
        if bindings:
            audit = result.get('binding_audit') or {}
            final = result.get('completion_audit') or {}
            revision = result['plan'].get('binding_revision', 0)
            if (audit.get('verdict') != 'APPROVED' or audit.get('binding') != expected_binding
                    or audit.get('bindings_sha256') != fingerprint(bindings)
                    or audit.get('binding_revision') != revision
                    or final.get('bindings_sha256') != fingerprint(bindings)
                    or final.get('binding_revision') != revision):
                errors.append('执行映射缺少与原始需求及最终完成绑定的独立审查')
        steps = {s['step_id']: s for s in result['plan']['steps']}
        rows = result['plan'].get('objective_ledger', [])
        if not rows or not result['plan']['verification'].get('all_goals_satisfied'):
            errors.append('完成账本未证明全部目标完成')
        for row in rows:
            if row['status'] != 'VERIFIED':
                errors.append('存在未完成目标：' + row['objective_id'])
                continue
            for ref in row['evidence_refs']:
                product = (steps[ref].get('result') or {}).get('selected_product')
                if product:
                    actual = database['products'].get(product['product_id'])
                    if not actual or any(product.get(k) != actual.get(k) for k in ('price', 'stock', 'specs', 'category')):
                        errors.append('推荐使用了过期或不存在的商品事实')
                    if product['stock'] <= 0:
                        errors.append('推荐无库存商品')
                    for predicate in expected.get('product_constraints', []):
                        if not predicate_holds(product, Constraint.model_validate(predicate)):
                            errors.append('推荐不满足人工标注约束：' + predicate['description'])
        selected = [(steps[ref].get('result') or {}).get('selected_product') for row in rows for ref in row['evidence_refs']]
        for target in expected.get('recommendation_targets', []):
            if not any(p and all(predicate_holds(p, Constraint.model_validate(c)) for c in target) for p in selected):
                errors.append('未完成某项品类及预算的独立推荐')
    return errors


@contextmanager
def scenario(case, session_id):
    """可复现实验扰动，单次触发；不把环境变化算作 Agent 的业务写操作。"""
    from contextlib import ExitStack
    import src.direct_executor as direct
    from src import tools
    from src.runtime import ToolSpec, tool_spec
    from src.backend.db import Product
    from src.session_control import ControlStore
    setting = case.get('scenario')
    state = dict(kind=(setting or {}).get('kind'), fired=False, environment_paths=[])
    with ExitStack() as stack:
        if setting:
            names = ('search_products',) if setting['kind'] == 'empty_search' else ('search_products', 'compare_products')
            for name in names:
                original = direct._READ_ONLY_TOOLS[name]
                original_spec = tool_spec(name, original)
                function = original_spec.function
                def instrumented(_function=function, **kwargs):
                    result = _function(**kwargs)
                    if setting['kind'] == 'empty_search' and not state['fired']:
                        state['fired'] = True
                        return {**result, 'products': [], 'total': 0}
                    if not state['fired'] and result.get('products'):
                        state['fired'] = True
                        if setting['kind'] == 'steer':
                            store = ControlStore()
                            run_id = store.conversation(session_id)['active_run_id']
                            store.enqueue(session_id, run_id, 'STEER', message=setting['message'],
                                command_id=uuid.uuid4().hex, expected_revision=store.revision(run_id))
                        else:
                            with tools._session() as session:
                                for product in result['products'][:2]:
                                    row = session.get(Product, product['product_id'])
                                    for field, value in setting['updates'].items():
                                        if field not in {'price', 'stock'}:
                                            raise ValueError('评测扰动只能修改价格和库存')
                                        setattr(row, field, value)
                                        state['environment_paths'].append(f'products.{row.product_id}.{field}')
                                session.commit()
                    return result
                stack.enter_context(patch.dict(direct._READ_ONLY_TOOLS, {name: ToolSpec(instrumented, original_spec.schema)}))
        yield state
