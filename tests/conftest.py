"""模型协议测试的边界替身；新语义链路另用独立响应测试。"""
import pytest


@pytest.fixture(autouse=True)
def no_unconfigured_semantic_model(monkeypatch):
    from src.errors import AgentError
    def unavailable(**_):
        raise AgentError('MODEL_UNAVAILABLE', '测试未配置语义模型替身', retryable=False)
    monkeypatch.setattr('src.request_semantics_runtime.get_llm', unavailable)


@pytest.fixture
def scripted_semantic_contract(monkeypatch):
    """既有脚本固定需求语义，专测工具/审查/恢复；不度量独立提取准确率。"""
    from src.request_semantics import apply_extraction, affected_outcomes, compare_goal
    from src.adaptive_decision import ShoppingGoal
    def prepare(context, state, goal, history):
        normalized = ShoppingGoal.model_validate(goal).model_dump()
        rows = []
        for objective in normalized['objectives']:
            rows.append(dict(requirement_id=objective['id'], kind='OUTCOME', description=objective['description'],
                source_quote=objective['source_quote'], outcome={key: objective[key] for key in
                    ('kind', 'quantity', 'order_id', 'question', 'product_ids')}))
        for index, condition in enumerate(normalized['hard_constraints']):
            rows.append(dict(requirement_id=f'global_{index}', kind='CONDITION', description=condition['description'],
                source_quote=condition['source_quote'], predicate=condition,
                targets=[o['id'] for o in normalized['objectives'] if o['kind'] == 'recommend']))
        for objective in normalized['objectives']:
            for index, condition in enumerate(objective['hard_constraints']):
                rows.append(dict(requirement_id=f"{objective['id']}_c{index}", kind='CONDITION', description=condition['description'],
                    source_quote=condition['source_quote'], predicate=condition, targets=[objective['id']]))
        contract = apply_extraction({'changes': [dict(operation='ADD', requirement=row) for row in rows]}, state['request'], history)
        mapping = compare_goal(contract, goal, state['request'], history)
        prior = state.get('previous_semantics')
        impact = affected_outcomes(prior, contract) if prior else dict(preserved=[], changed=[], retracted=[], added=sorted(mapping))
        return dict(contract=contract, objective_mapping=mapping, impact=impact)
    monkeypatch.setattr('src.request_semantics_runtime.prepare_goal', prepare)
