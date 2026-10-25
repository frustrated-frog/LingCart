from copy import deepcopy
from pathlib import Path
import json
import pytest

from src.pg_credit import edge_credit
from src.pg_hypotheses import EvolutionProposal, hypothesis_identity
from src.pg_learning import EvolutionSpec, ModelGraphRefiner, evolve
from src.procedural_graph import ProceduralGraph
from tests.test_pg_evidence import sample as evidence_sample
from tests.test_pg_learning import report as make_report


def graph():
    return ProceduralGraph.from_path(Path('research/graphs/pg-seed-v2.json'))


def sample(current, *, passed=True, progress=True, repetition=1):
    value = evidence_sample(passed=passed)
    value.update(case_id='A01', repetition=repetition, split='dev', model_attempts=4,
                 safety_violations=0, protocol_violations=0)
    events = value['turns'][0]['result']['events']
    events[0].update(graph_sha256=current.fingerprint, retrieved_edge_ids=['goal-search'], edge_ids=['goal-search'],
                     applicability={'evaluations': [{'edge_id': 'goal-search', 'status': 'TRUE'}]})
    events.insert(3, {'type': 'pg_decision_reviewed', 'sequence': 3.5, 'decision_key': 'decision-1',
        'review_id': 'r1', 'status': 'ALIGNED', 'matched_edge_ids': ['goal-search'], 'actual_action': 'search_products'})
    events.append({'type': 'pg_action_outcome', 'sequence': 8, 'outcome_id': 'o1', 'review_id': 'r1',
        'goal_revision': 0, 'schema_version': 2, 'learning_eligible': progress,
        'progress_status': 'PROGRESSED' if progress else 'NO_PROGRESS',
        'information_gain': {'status': 'NEW_INFORMATION' if progress else 'NO_NEW_INFORMATION',
                             'newly_verified_objective_ids': ['choice'] if passed and progress else []}})
    return value


def report(current, samples):
    data = make_report(current, ('A01',))
    data['samples'] = samples
    data['manifest']['repeats'] = len(samples)
    return data


def proposal(current, *, text='改善空结果后的恢复路径'):
    return dict(hypothesis=dict(problem=text, failure_mode='UNRESOLVED_GOAL', expected_change='RECOVERY',
        target_edges=['search-retry'], source_samples=['A01:1'], evidence_ids=[]),
        patch=dict(reason=text, source_cases=['A01'], add_edges=[dict(id='search-retry', source='search_products',
            target='search_products', relation='TRIGGERS', condition='观察为空', guidance=text,
            pitfalls='不能把一次空结果当作目录无候选', applicability={
                'all_of': [{'fact': 'trajectory.latest_outcome', 'operator': 'EQ', 'value': 'EMPTY'}], 'any_of': []})]))


def test_failed_task_keeps_useful_edge_and_success_does_not_reward_a_repeat():
    current = graph()
    credit = edge_credit(current, report(current, [sample(current, passed=False),
        sample(current, passed=True, progress=False, repetition=2)]), train_cases=('A01',))
    edge = next(row for row in credit['edges'] if row['edge_id'] == 'goal-search')
    assert edge['support'] == edge['counter'] == 1 and edge['classification'] == 'MIXED'
    assert edge['contrasts'][0]['failed']['counts'] == {'SUPPORT': 1}
    assert edge['contrasts'][0]['successful']['counts'] == {'COUNTER': 1}
    assert credit['causal_claim'] is False
    assert next(row for row in credit['edges'] if row['edge_id'] == 'search-inspect')['classification'] == 'UNKNOWN'


@pytest.mark.parametrize('missing', ['condition', 'exposure', 'adoption', 'version', 'graph', 'outcome', 'review'])
def test_incomplete_evidence_chain_abstains(missing):
    current = graph()
    value = sample(current)
    events = value['turns'][0]['result']['events']
    if missing == 'condition': events[0]['applicability']['evaluations'][0]['status'] = 'UNKNOWN'
    elif missing == 'exposure': events[2]['pg_status'] = 'OMITTED'
    elif missing == 'adoption': events[3]['status'] = 'DIVERGED'
    elif missing == 'version': events[-1]['schema_version'] = 1
    elif missing == 'graph': events[0]['graph_sha256'] = 'wrong'
    elif missing == 'review': events.pop(3)
    else: events.pop()
    row = next(row for row in edge_credit(current, report(current, [value]), train_cases=('A01',))['edges']
               if row['edge_id'] == 'goal-search')
    assert row['classification'] == 'UNKNOWN'
    assert row['support'] == row['counter'] == 0


def test_credit_refuses_test_validation_and_wrong_graph_reports():
    current = graph()
    for field, value in [('split', 'test'), ('case_id', 'validation-case')]:
        task = sample(current)
        task[field] = value
        with pytest.raises(ValueError): edge_credit(current, report(current, [task]), train_cases=('A01',))
    data = report(current, [sample(current)])
    data['manifest']['harness_profile']['mechanisms'][0]['configuration']['graph_sha256'] = 'other'
    with pytest.raises(ValueError): edge_credit(current, data, train_cases=('A01',))


def test_hypothesis_identity_survives_paraphrase_and_rejects_foreign_evidence():
    current = graph()
    credit = edge_credit(current, report(current, [sample(current)]), train_cases=('A01',))
    left = EvolutionProposal.model_validate_json(json.dumps(proposal(current)))
    right = EvolutionProposal.model_validate_json(json.dumps(proposal(current, text='换一种描述表达同一恢复策略')))
    assert hypothesis_identity(current, left, credit) == hypothesis_identity(current, right, credit)
    wrong = proposal(current)
    wrong['hypothesis']['source_samples'] = ['A02:1']
    with pytest.raises(ValueError): hypothesis_identity(current, EvolutionProposal.model_validate_json(json.dumps(wrong)), credit)
    wrong = proposal(current)
    wrong['patch']['delete_edges'] = ['goal-search']
    with pytest.raises(ValueError, match='全部'):
        hypothesis_identity(current, EvolutionProposal.model_validate_json(json.dumps(wrong)), credit)


def test_evolution_memory_blocks_paraphrased_rejection_and_hides_validation_cases(tmp_path):
    current = graph()
    seen, rollouts = [], []
    class Refiner:
        def propose(self, g, data, memory):
            assert all(s['case_id'] == 'A01' for s in data['samples'])
            seen.append(deepcopy(memory))
            return {'proposal': json.dumps(proposal(g, text='恢复策略' + str(len(seen)))), 'error': None}
    def evaluator(g, cases, label):
        rollouts.append(label)
        data = make_report(g, cases, passed=g.version == 0)
        data['manifest']['harness_profile']['mechanisms'][0]['version'] = '4'
        if cases == ('A01',): data['samples'] = [sample(g)]
        return data
    spec = EvolutionSpec(experiment_id='credit-test', dataset='eval/adaptive_tasks.json', train_cases=('A01',),
        validation_cases=('A02',), repeats=1, iterations=2, evolution_policy='credit-memory-v1',
        online_profile='adaptive-pg-context-v4')
    retained, result = evolve(current, spec, evaluator=evaluator, refiner=Refiner(), output_dir=tmp_path / 'evolve',
                              execution_source='test_harness')
    assert retained == current
    assert [row['reason'] for row in result['history']] == ['VALIDATION_REGRESSION', 'REPEATED_REJECTED_HYPOTHESIS']
    assert 'iteration-02-validation' not in rollouts
    assert seen[1][0]['blocks_retry'] is True and 'A02' not in json.dumps(seen[1])
    assert result['history'][0]['validation_changes'][0]['case_id'] == 'A02'
    assert result['history'][1]['memory_entry']['blocks_retry'] is True  # 继承已有拒绝，不伪装成新的验证结论。
    assert (tmp_path / 'evolve/iteration-01-credit.json').exists()
    assert (tmp_path / 'evolve/iteration-02-memory-input.json').exists()


def test_incomparable_validation_does_not_blacklist_the_same_structure(tmp_path):
    current = graph()
    class Refiner:
        def propose(self, g, data, memory):
            assert all(not item['blocks_retry'] for item in memory)
            return {'proposal': json.dumps(proposal(g)), 'error': None}
    def evaluator(g, cases, label):
        data = make_report(g, cases)
        data['manifest']['harness_profile']['mechanisms'][0]['version'] = '4'
        if cases == ('A01',): data['samples'] = [sample(g)]
        if label == 'iteration-01-validation': data['providers'] = ['unavailable-primary']
        return data
    spec = EvolutionSpec(experiment_id='credit-test', dataset='eval/adaptive_tasks.json', train_cases=('A01',),
        validation_cases=('A02',), repeats=1, iterations=2, evolution_policy='credit-memory-v1',
        online_profile='adaptive-pg-context-v4')
    retained, result = evolve(current, spec, evaluator=evaluator, refiner=Refiner(), output_dir=tmp_path / 'evolve',
                              execution_source='test_harness')
    assert [row['reason'] for row in result['history']] == ['INCOMPARABLE_VALIDATION', 'VALIDATION_NON_REGRESSION']
    assert retained.version == 1
    assert result['history'][0]['memory_entry']['blocks_retry'] is False


def test_refiner_receives_credit_and_structured_memory_through_gateway(monkeypatch):
    from langchain_core.messages import AIMessage
    current = graph()
    data = report(current, [sample(current)])
    data['edge_credit'] = edge_credit(current, data, train_cases=('A01',))
    memory = [{'hypothesis_id': 'previous', 'blocks_retry': True, 'validation_summary': {'regressed_samples': 1}}]
    captured = []
    class Model:
        def invoke(self, messages):
            captured.extend(messages)
            return AIMessage(content=json.dumps(proposal(current)))
    monkeypatch.setattr('src.pg_learning.get_llm', lambda **_: Model())
    result = ModelGraphRefiner(evolution_policy='credit-memory-v1').propose(current, data, memory)
    assert result['error'] is None
    payload = json.loads(captured[-1].content)
    assert payload['edge_credit'] == data['edge_credit']
    assert payload['hypothesis_memory'] == memory and 'rejected_history' not in payload
    assert '输出严格 JSON EvolutionProposal' in captured[0].content
    assert '输出严格 JSON GraphPatch' not in captured[0].content


def test_control_and_candidate_freeze_the_same_online_mechanism_and_budget():
    root = Path('research/experiments/pg-credit-memory-v1')
    left = json.loads((root / 'baseline-evolution-spec.json').read_text())
    right = json.loads((root / 'evolution-spec.json').read_text())
    for key in ('experiment_id', 'evolution_policy'):
        left.pop(key)
        right.pop(key)
    assert left == right
    assert left['online_profile'] == 'adaptive-pg-context-v4'


@pytest.mark.parametrize('invalid,reason', [('source', 'HYPOTHESIS_SOURCE_MISMATCH'),
    ('evidence', 'HYPOTHESIS_EVIDENCE_MISMATCH'), ('target', 'HYPOTHESIS_TARGET_MISMATCH')])
def test_invalid_hypothesis_preserves_proposal_without_validation_or_blacklisting(tmp_path, invalid, reason):
    current = graph()
    bad = proposal(current)
    if invalid == 'source': bad['patch']['source_cases'] = ['A01:1']
    if invalid == 'evidence': bad['hypothesis']['evidence_ids'] = ['validation-evidence']
    if invalid == 'target': bad['hypothesis']['target_edges'] = ['goal-search']
    class Refiner:
        def propose(self, *args): return {'proposal': json.dumps(bad), 'error': None}
    labels = []
    def evaluator(g, cases, label):
        labels.append(label)
        data = make_report(g, cases)
        data['manifest']['harness_profile']['mechanisms'][0]['version'] = '4'
        if cases == ('A01',): data['samples'] = [sample(g)]
        return data
    spec = EvolutionSpec(experiment_id='credit-test', dataset='eval/adaptive_tasks.json', train_cases=('A01',),
        validation_cases=('A02',), repeats=1, iterations=1, evolution_policy='credit-memory-v1',
        online_profile='adaptive-pg-context-v4')
    retained, result = evolve(current, spec, evaluator=evaluator, refiner=Refiner(), output_dir=tmp_path / 'evolve',
                              execution_source='test_harness')
    assert retained == current and result['history'][0]['reason'] == reason
    assert result['history'][0]['hypothesis']['problem'] == bad['hypothesis']['problem']
    assert not result['hypothesis_memory'] and labels == ['seed-validation', 'iteration-01-training']
