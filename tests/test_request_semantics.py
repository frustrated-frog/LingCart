"""需求契约回归：不调用公网模型，不度量语言理解准确率。"""
from copy import deepcopy

from langchain_core.messages import HumanMessage
import pytest

from src.request_semantics import apply_extraction, affected_outcomes, compare_goal
from src.errors import AgentError


REQUEST = '推荐两款耳机，每款500元以内，再推荐一个300元以内的键盘'


def outcome(identity, quote, quantity):
    return dict(requirement_id=identity, kind='OUTCOME', description=quote,
                source_quote=quote, outcome=dict(kind='recommend', quantity=quantity))


def condition(identity, quote, target, value):
    return dict(requirement_id=identity, kind='CONDITION', description=quote,
                source_quote=quote, targets=[target],
                predicate=dict(description=quote, source_quote=quote, field='price', op='lte', value=value))


def initial():
    rows = [outcome('headphones', '推荐两款耳机', 2), condition('headphone_budget', '每款500元以内', 'headphones', 500),
            outcome('keyboard', '推荐一个300元以内的键盘', 1), condition('keyboard_budget', '300元以内', 'keyboard', 300)]
    return apply_extraction(dict(changes=[dict(operation='ADD', requirement=row) for row in rows]), REQUEST)


def goal():
    return dict(summary='耳机及键盘推荐', objectives=[
        dict(id='ear', kind='recommend', description='耳机', source_quote='推荐两款耳机', quantity=2,
             hard_constraints=[dict(description='预算', source_quote='每款500元以内', field='price', op='lte', value=500)]),
        dict(id='key', kind='recommend', description='键盘', source_quote='推荐一个300元以内的键盘', quantity=1,
             hard_constraints=[dict(description='预算', source_quote='300元以内', field='price', op='lte', value=300)]),
    ])


def revision():
    return dict(changes=[
        dict(operation='REPLACE', previous_id='headphones', source_quote='耳机改一款',
             requirement=outcome('headphones', '耳机改一款', 1)),
        dict(operation='REPLACE', previous_id='headphone_budget', source_quote='不超过400元',
             requirement=condition('headphone_budget', '不超过400元', 'headphones', 400)),
        dict(operation='PRESERVE', previous_id='keyboard'),
        dict(operation='PRESERVE', previous_id='keyboard_budget'),
        dict(operation='ADD', requirement=dict(requirement_id='no_purchase', kind='PROHIBITION',
             description='先不购买', source_quote='先别购买', restriction='READ_ONLY')),
    ])


def test_bidirectional_compile_matches_distinct_scopes_and_cardinality():
    assert compare_goal(initial(), goal(), REQUEST) == {'headphones': 'ear', 'keyboard': 'key'}


@pytest.mark.parametrize('mutation', ['quantity', 'scope', 'missing', 'invented', 'negation'])
def test_compiler_catches_two_model_semantic_disagreement(mutation):
    candidate = goal()
    if mutation == 'quantity':
        candidate['objectives'][0]['quantity'] = 1
    elif mutation == 'scope':
        candidate['hard_constraints'] = candidate['objectives'][1].pop('hard_constraints')
    elif mutation == 'missing':
        candidate['objectives'].pop()
    elif mutation == 'invented':
        candidate['objectives'][1]['hard_constraints'].append(dict(description='库存', source_quote='键盘', field='stock', op='gte', value=1))
    else:
        candidate['objectives'][0]['hard_constraints'][0]['op'] = 'gte'
    with pytest.raises(AgentError, match='需求语义契约'):
        compare_goal(initial(), candidate, REQUEST)


def test_revision_changes_only_headphones_and_preserves_keyboard():
    request = '耳机改一款，不超过400元，键盘不变，先别购买'
    prior = initial()
    revised = apply_extraction(revision(), request, [HumanMessage(content=REQUEST)], previous=prior)
    assert affected_outcomes(prior, revised) == dict(preserved=['keyboard'], changed=['headphones'], added=[], retracted=[])
    rows = {item['requirement']['requirement_id']: item['requirement'] for item in revised['requirements']}
    assert rows['keyboard']['outcome']['quantity'] == 1
    assert rows['keyboard_budget']['predicate']['value'] == 300
    assert rows['no_purchase']['restriction'] == 'READ_ONLY'


def test_omitted_previous_requirement_is_not_silently_retracted():
    changes = revision()
    changes['changes'].pop(3)
    with pytest.raises(AgentError, match='未声明'):
        apply_extraction(changes, '耳机改一款，不超过400元，先别购买', [HumanMessage(content=REQUEST)], previous=initial())


def test_steering_cannot_use_old_text_to_authorize_a_new_change():
    changes = revision()
    changes['changes'][0]['source_quote'] = '推荐两款耳机'
    latest = '耳机改一款，不超过400元，先别购买'
    with pytest.raises(AgentError, match='HUMAN'):
        apply_extraction(changes, REQUEST + '\n' + latest, previous=initial(), change_text=latest)


def test_retraction_cannot_leave_a_dangling_condition_scope():
    changes = dict(changes=[dict(operation='RETRACT', previous_id='headphones', source_quote='不要耳机了'),
                           *[dict(operation='PRESERVE', previous_id=key)
                             for key in ('headphone_budget', 'keyboard', 'keyboard_budget')]])
    with pytest.raises(AgentError, match='已撤销'):
        apply_extraction(changes, '不要耳机了', [HumanMessage(content=REQUEST)], previous=initial())


def test_preserve_does_not_accept_an_edited_requirement():
    changes = revision()
    changes['changes'][2]['requirement'] = outcome('keyboard', '键盘', 2)
    with pytest.raises(ValueError, match='Only ADD/REPLACE'):
        apply_extraction(changes, '耳机改一款，不超过400元，先别购买', [HumanMessage(content=REQUEST)], previous=initial())


def test_saved_contract_tampering_is_rejected_before_compilation():
    contract = deepcopy(initial())
    contract['requirements'][0]['requirement']['outcome']['quantity'] = 1
    with pytest.raises(AgentError, match='指纹'):
        compare_goal(contract, goal(), REQUEST)


def test_constraint_quote_can_be_narrower_without_changing_meaning():
    candidate = goal()
    candidate['objectives'][0]['hard_constraints'][0]['source_quote'] = '500元以内'
    assert compare_goal(initial(), candidate, REQUEST)['headphones'] == 'ear'


def test_rehashed_source_offsets_cannot_mislead_the_inspector():
    from src.constraint_bindings import fingerprint
    contract = initial()
    contract['requirements'][0]['source']['start'] += 1
    contract['requirements'][0]['source']['end'] += 1
    contract['sha256'] = fingerprint({k: v for k, v in contract.items() if k != 'sha256'})
    with pytest.raises(AgentError, match='来源发生变化'):
        compare_goal(contract, goal(), REQUEST)
