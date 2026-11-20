from copy import deepcopy
import json

import pytest
from langchain_core.messages import AIMessage

from src import pg_sim
from src.pg_learning import EvolutionSpec, ModelGraphRefiner, evolve
from src.pg_experiment import plan
from src.procedural_graph import canonical
from tests.test_pg_credit import graph, proposal
from tests.test_pg_learning import report


def spec(**updates):
    return EvolutionSpec(experiment_id='sim-test', dataset='eval/adaptive_tasks.json',
        train_cases=('A01',), validation_cases=('A02',), repeats=1, iterations=1,
        evolution_policy='sim-v1', online_profile='adaptive-pg-context-v4', **updates)


def evaluator(current, cases, label):
    value = report(current, cases, passed='candidate-training' in label or cases == ('A02',))
    value['manifest']['harness_profile']['mechanisms'][0]['version'] = '4'
    return value


class Refiner:
    def __init__(self, response=None):
        self.response, self.inputs = response, []

    def propose(self, current, report, memory):
        self.inputs.append(deepcopy(report['self_improvement_memory']))
        result = self.response or {**proposal(current), 'predicted_fixes': ['A01:1']}
        return dict(proposal=canonical(result), error=None)


def initial(tmp_path, **kwargs):
    return evolve(graph(), spec(), evaluator=kwargs.pop('evaluator', evaluator), refiner=kwargs.pop('refiner', Refiner()),
                  output_dir=tmp_path / 'source', execution_source='test_harness', **kwargs)


def test_forecast_is_frozen_before_measurement_and_validation_details_stay_out(tmp_path):
    folder = tmp_path / 'source'
    calls = []
    def evaluate(current, cases, label):
        calls.append(label)
        if label.endswith('candidate-training'):
            assert json.loads((folder / 'iteration-01-sim-forecast.json').read_text())['predicted_fixes'] == ['A01:1']
        return evaluator(current, cases, label)
    retained, result = initial(tmp_path, evaluator=evaluate)
    attempt = result['sim_memory']['problems'][0]['attempts'][0]
    assert calls == ['seed-validation', 'iteration-01-training', 'iteration-01-candidate-training', 'iteration-01-validation']
    assert attempt['fixed'] == ['A01:1'] and attempt['missed'] == []
    assert attempt['observation_status'] == 'MEASURED'
    assert attempt['patch']['add_edges'][0]['id'] == 'search-retry'
    assert 'A02' not in canonical(result['sim_memory'])
    assert result['sim_memory']['retained_sha256'] == retained.fingerprint
    assert pg_sim.SIMMemory.model_validate_json(canonical(result['sim_memory'])).fingerprint == result['sim_memory_sha256']


@pytest.mark.parametrize('mode', ['none', 'problems', 'full'])
def test_continuation_carries_graph_and_history_but_conditions_change_only_visible_memory(tmp_path, mode):
    retained, source = initial(tmp_path)
    memory = pg_sim.SIMMemory.model_validate_json(canonical(source['sim_memory']))
    response = {**proposal(retained), 'predicted_fixes': ['A01:1']}
    response['patch']['delete_edges'] = ['search-retry']
    response['patch']['add_edges'][0]['guidance'] = '新环境下先补充工具观察再选择恢复方向'
    if mode != 'none':
        response['problem_id'] = memory.problems[0].problem_id
    refiner = Refiner(response)
    def changed(current, cases, label):
        data = evaluator(current, cases, label)
        data['manifest']['purpose_models'] = {'planner': 'new-model'}
        return data
    _, result = evolve(retained, spec(sim_memory_mode=mode), evaluator=changed, refiner=refiner,
        output_dir=tmp_path / 'target', execution_source='test_harness', sim_source=memory)
    assert result['history'][0]['status'] == 'ACCEPTED'
    visible = refiner.inputs[0]['problems']
    if mode == 'none':
        assert visible == []
    elif mode == 'problems':
        assert visible[0]['problem_id'] == memory.problems[0].problem_id and 'attempts' not in visible[0]
    else:
        assert visible[0]['attempts'][0]['environment_changed'] is True
        assert visible[0]['attempts'][0]['fixed'] == ['A01:1']
    assert len(result['sim_memory']['problems'][0]['attempts']) == 2
    assert len(memory.problems[0].attempts) == 1


def test_rejected_edit_preserves_missed_prediction_and_does_not_poison_other_environment(tmp_path):
    def failed(current, cases, label):
        value = evaluator(current, cases, label)
        if label in {'iteration-01-validation', 'iteration-01-candidate-training'}:
            value['samples'][0]['passed'] = False
        return value
    retained, result = initial(tmp_path, evaluator=failed)
    assert retained == graph()
    attempt = result['sim_memory']['problems'][0]['attempts'][0]
    assert attempt['status'] == 'REJECTED' and attempt['reason'] == 'VALIDATION_REGRESSION'
    assert attempt['fixed'] == [] and attempt['missed'] == ['A01:1']
    assert 'blocks_retry' not in canonical(result['sim_memory'])


@pytest.mark.parametrize('kind', ['validation_prediction', 'unknown_problem', 'duplicate_prediction'])
def test_bad_predictions_never_reach_candidate_evaluation(tmp_path, kind):
    response = proposal(graph())
    response['predicted_fixes'] = ['A02:1'] if kind == 'validation_prediction' else ['A01:1', 'A01:1']
    if kind == 'unknown_problem':
        response.update(problem_id='sim-' + 'a' * 20, predicted_fixes=['A01:1'])
    calls = []
    def evaluate(current, cases, label):
        calls.append(label)
        return evaluator(current, cases, label)
    _, result = initial(tmp_path, evaluator=evaluate, refiner=Refiner(response))
    assert result['history'][0]['status'] == 'REJECTED' and result['sim_memory']['problems'] == []
    assert calls == ['seed-validation', 'iteration-01-training']


def test_inheritance_checks_source_graph_and_split_and_snapshot_integrity(tmp_path):
    retained, result = initial(tmp_path)
    memory = pg_sim.SIMMemory.model_validate_json(canonical(result['sim_memory']))
    for current, source in [(graph(), 'test_harness'), (retained, 'live_provider')]:
        with pytest.raises(ValueError, match='身份'):
            pg_sim.inherit(memory, graph=current, execution_source=source)
    context = memory.problems[0].attempts[0].environment
    with pytest.raises(ValueError, match='验证'):
        pg_sim.check_partition(memory, context, ('A01',))
    altered = deepcopy(result['sim_memory'])
    altered['problems'][0]['attempts'][0]['missed'] = ['A02:1']
    with pytest.raises(ValueError):
        pg_sim.SIMMemory.model_validate_json(canonical(altered))


def test_sim_refiner_reads_the_declared_memory_through_existing_gateway(monkeypatch):
    current = graph()
    data = evaluator(current, ('A01',), 'training')
    from src.pg_credit import edge_credit
    data['edge_credit'] = edge_credit(current, data, train_cases=('A01',))
    data['self_improvement_memory'] = {'mode': 'problems', 'problems': [{'problem_id': 'prior'}]}
    seen = []
    class Model:
        def invoke(self, messages):
            seen.append(messages)
            return AIMessage(content=canonical({**proposal(current), 'predicted_fixes': ['A01:1']}))
    monkeypatch.setattr('src.pg_learning.get_llm', lambda **_: Model())
    response = ModelGraphRefiner(evolution_policy='sim-v1').propose(current, data, [])
    assert response['error'] is None
    assert json.loads(seen[0][-1].content)['self_improvement_memory'] == data['self_improvement_memory']
    assert 'predicted_fixes' in seen[0][0].content


def test_sim_budget_counts_candidate_training_in_every_condition():
    plans = [plan('smoke', 'sim-v1', mode) for mode in ('none', 'problems', 'full')]
    assert all(p['max_samples'] == 4 and p['max_model_attempts'] == 34 and p['groups']['test'] == [] for p in plans)
    assert len({p['max_token_units'] for p in plans}) == 1
    assert plan('smoke', 'credit-memory-v1')['max_samples'] == 3


def test_persisted_source_is_copied_and_idempotent_before_configuration_checks(tmp_path, monkeypatch):
    from src.pg_jobs import JobStore, StartExperiment, LabError
    retained, result = initial(tmp_path)
    store = JobStore(tmp_path / 'jobs')
    source = store.create(StartExperiment(request_id='sim-source-job', preset='smoke', evolution_policy='sim-v1'), launch=False)
    store.update(source['job_id'], status='COMPLETED', execution_source='test_harness',
                 evolution=result, retained_graph=retained.model_dump(mode='json'))
    request = StartExperiment(request_id='sim-target-job', preset='smoke', evolution_policy='sim-v1',
                              sim_source_job_id=source['job_id'], sim_memory_mode='problems')
    target = store.create(request, launch=False)
    assert target['sim_source'] == result['sim_memory']
    assert target['frozen']['seed_sha256'] == retained.fingerprint
    assert target['frozen']['sim_source_sha256'] == result['sim_memory_sha256']
    monkeypatch.setattr('src.llm.get_model_status', lambda: {'configured': False})
    assert store.create(request)['job_id'] == target['job_id']
    for update in ({'sim_memory_mode': 'none'}, {'sim_source_job_id': None}):
        with pytest.raises(LabError, match='更换'):
            store.create(request.model_copy(update=update), launch=False)
    # 修改返回对象不改变数据库快照；完成的来源也不能被正常 update 改写。
    target['sim_source']['problems'].clear()
    assert store.get(target['job_id'])['sim_source']['problems']
    assert not store.update(source['job_id'], evolution={})
    with pytest.raises(LabError, match='来源'):
        store._sim_source(source['job_id'], launch=True)


@pytest.mark.parametrize('invalid', ['unfinished', 'hash', 'graph'])
def test_invalid_job_sources_are_rejected(tmp_path, invalid):
    from src.pg_jobs import JobStore, StartExperiment, LabError
    retained, result = initial(tmp_path)
    store = JobStore(tmp_path / 'jobs')
    source = store.create(StartExperiment(request_id='source-for-validation', preset='smoke', evolution_policy='sim-v1'), launch=False)
    if invalid != 'unfinished':
        if invalid == 'hash':
            result['sim_memory_sha256'] = 'bad'
        store.update(source['job_id'], status='COMPLETED', execution_source='test_harness', evolution=result,
                     retained_graph=(graph() if invalid == 'graph' else retained).model_dump(mode='json'))
    with pytest.raises(LabError):
        store.create(StartExperiment(request_id='invalid-source-test', preset='smoke', evolution_policy='sim-v1',
                                     sim_source_job_id=source['job_id']), launch=False)


def test_sim_strategy_cannot_persist_business_facts(tmp_path):
    response = proposal(graph())
    response['hypothesis']['problem'] = '订单 123 的恢复策略'
    _, result = initial(tmp_path, refiner=Refiner(response))
    assert result['history'][0]['status'] == 'REJECTED'
    assert result['sim_memory']['problems'] == []


def test_sim_cli_dry_run_checks_online_graph_and_never_calls_provider(capsys):
    from eval.run_pg_evolution import main
    args = ['--spec', 'research/experiments/pg-sim-v1/evolution-spec.json']
    with pytest.raises(SystemExit):
        main(args)
    assert main([*args, '--base-graph', 'research/graphs/pg-seed-v2.json']) == 0
    assert json.loads(capsys.readouterr().out)['model_calls'] == 0


def test_worker_continuation_keeps_fixed_prediction_even_when_validation_rejects(tmp_path):
    import tempfile
    from pathlib import Path
    from eval.run_pg_lab import run_job
    from eval.run_live_eval import summarize, environment_manifest
    from src.pg_jobs import JobStore, StartExperiment
    from src.pg_experiment import ROOT
    from src.harness import get_runner
    from tests.test_pg_experiment import test_worker_three_arms_are_after_learning_and_test_never_reaches_refiner
    test_worker_three_arms_are_after_learning_and_test_never_reaches_refiner(tmp_path, True, 'sim-v1')
    store = JobStore(tmp_path)
    source = store.list()[0]
    job = store.create(StartExperiment(request_id='sim-worker-continuation', preset='smoke', evolution_policy='sim-v1',
        sim_source_job_id=source['job_id']), launch=False)
    assert job['seed_graph'] == source['retained_graph']
    calls = []
    class ContinuedRefiner:
        def propose(self, current, training, rejected):
            assert len(training['self_improvement_memory']['problems']) == 1 and rejected == []
            response = proposal(current)
            response['patch'].update(source_cases=['PG01'], delete_edges=['search-retry'])
            response['patch']['add_edges'][0]['guidance'] = '补充观察后再选择新的恢复方向'
            response['hypothesis']['source_samples'] = ['PG01:1']
            response.update(problem_id=source['evolution']['sim_memory']['problems'][0]['problem_id'], predicted_fixes=['PG01:1'])
            return dict(proposal=canonical(response), error=None, events=[])
    def evaluate(dataset, **kwargs):
        calls.append(tuple(kwargs['case_ids']))
        provenance = get_runner(kwargs['harness_profile']).provenance()
        context = next(m for m in provenance['mechanisms'] if m['slot'] == 'context')
        manifest = dict(environment_manifest(dataset, 'adaptive', kwargs['dataset_path']), suite='adaptive', split='dev',
            repeats=1, limits={'max_model_calls': 8}, purpose_models={'planner': 'scripted-next-model'}, harness_profile=provenance)
        passed = len(calls) in {1, 3}
        sample = dict(case_id=kwargs['case_ids'][0], split='dev', repetition=1, passed=passed, adaptive_verified=True,
            model_attempts=4, tool_calls=1, safety_violations=0, protocol_violations=0, latency_ms=1,
            turns=[{'failures': [] if passed else ['脚本判据未通过'], 'message': '测试替身', 'result': {'events': [
                {'type': 'mechanism_applied', 'slot': 'context', 'mechanism_id': context['id'],
                 'graph_sha256': context['configuration']['graph_sha256']}]}}])
        kwargs['on_sample'](sample)
        return dict(status='COMPLETED', execution_source='test_harness', live_verified=False, providers=['scripted-next-model'],
                    manifest=manifest, samples=[sample], metrics=summarize([sample], 1))
    with tempfile.TemporaryDirectory(dir=ROOT / 'research/graphs') as graphs:
        run_job(job['job_id'], store=store, evaluator=evaluate, refiner=ContinuedRefiner(),
                execution_source='test_harness', graph_root=Path(graphs))
    result = store.get(job['job_id'])
    assert result['status'] == 'COMPLETED', result
    assert result['completed_samples'] == 4 and calls == [('PG07',), ('PG01',), ('PG01',), ('PG07',)]
    attempts = result['evolution']['sim_memory']['problems'][0]['attempts']
    assert len(attempts) == 2 and attempts[-1]['fixed'] == ['PG01:1']
    assert attempts[-1]['reason'] == 'VALIDATION_REGRESSION' and attempts[-1]['status'] == 'REJECTED'
    assert result['retained_graph'] == source['retained_graph']
    assert len(store.get(source['job_id'])['evolution']['sim_memory']['problems'][0]['attempts']) == 1
