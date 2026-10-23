from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
import tempfile
import time
from pathlib import Path

import pytest

from src.pg_experiment import challenge_dataset, plan, behavior_metrics, ROOT
from src.pg_jobs import JobStore, StartExperiment, LabError


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    import src.harness.registry as registry
    monkeypatch.setattr(registry, '_REGISTRY', registry.build_registry())


def test_challenge_partitions_cover_distinct_families_and_bounded_cost():
    dataset = challenge_dataset()
    assert len({c['scenario_family'] for c in dataset['cases']}) == 18
    small, full = plan('smoke'), plan('full')
    assert small['max_samples'] == 6 and full['max_samples'] == 180
    assert small['max_model_attempts'] == 50
    assert small['max_token_units'] == 361500
    assert small['groups'] == {'train': ['PG01'], 'validation': ['PG07'], 'test': ['PG13']}
    assert not set(small['groups']['test']) & set(small['spec']['train_cases'] + small['spec']['validation_cases'])
    assert {c['category'] for c in dataset['cases'] if c['learning_partition'] == 'test'} == {
        '空结果恢复', '工具选择', '行动依赖', '多目标覆盖', '环境变化', '组合迁移'}


def test_recovery_and_repeated_empty_calls_have_explicit_denominators():
    event = dict(type='tool_result', tool='search_products', args={'query': '耳机'}, result={'products': []})
    sample = dict(case_id='PG01', repetition=1, passed=True, model_attempts=3, tool_calls=3,
        safety_violations=0, protocol_violations=0, latency_ms=5,
        turns=[{'result': {'events': [event, deepcopy(event), {**event, 'args': {'query': ''}}],
                           'evaluation_environment': {'fired': True}}, 'usage': None}])
    metrics = behavior_metrics({'samples': [sample]})
    assert metrics['repeated_ineffective_calls'] == 1
    assert metrics['recovery_rate'] == 1 and metrics['pg_exposure_rate'] == 0
    assert metrics['pg_localization_abstentions'] == 0
    assert metrics['pg_shadow_disagreements'] == 0
    assert metrics['tokens']['total_tokens'] is None
    assert behavior_metrics({'samples': []})['recovery_rate'] is None


def test_persistent_idempotency_concurrent_requests_and_cancellation(tmp_path):
    store = JobStore(tmp_path)
    request = StartExperiment(request_id='same-request', preset='smoke')
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = list(pool.map(lambda _: store.create(request, launch=False), range(2)))
    assert jobs[0]['job_id'] == jobs[1]['job_id']
    job_id = jobs[0]['job_id']
    with pytest.raises(LabError, match='不能更换'):
        store.create(StartExperiment(request_id='same-request', preset='full'), launch=False)
    with pytest.raises(LabError, match='已有实验'):
        store.create(StartExperiment(request_id='different-request', preset='smoke'), launch=False)
    assert store.claim(job_id) and not store.claim(job_id)
    store.cancel(job_id)
    assert JobStore(tmp_path).get(job_id)['cancel_requested']
    store.update(job_id, status='CANCELLED')
    assert not store.update(job_id, status='RUNNING')


def test_full_experiment_requires_explicit_cost_confirmation(tmp_path):
    store = JobStore(tmp_path)
    with pytest.raises(LabError, match='明确确认'):
        store.create(StartExperiment(request_id='full-without-confirm', preset='full'), launch=False)
    job = store.create(StartExperiment(request_id='full-confirmed', preset='full', confirm_full_cost=True), launch=False)
    assert job['frozen']['max_samples'] == 180


def test_dead_workers_are_interrupted_without_relaunch(tmp_path):
    store = JobStore(tmp_path)
    request = StartExperiment(request_id='dead-request', preset='smoke')
    job = store.create(request, launch=False)
    with store.connect() as db:
        db.execute('UPDATE jobs SET heartbeat=?', (time.time() - 100,))
    assert store.get(job['job_id'])['status'] == 'INTERRUPTED'
    assert store.create(request, launch=False)['status'] == 'INTERRUPTED'
    with pytest.raises(LabError):
        store.directory('../escape')


def test_worker_keeps_partial_samples_on_cancel_and_refuses_duplicate(tmp_path):
    from eval.run_pg_lab import run_job
    store = JobStore(tmp_path)
    job = store.create(StartExperiment(request_id='cancel-worker', preset='smoke'), launch=False)
    calls = []
    def evaluator(dataset, **kwargs):
        calls.append(kwargs['split'])
        store.cancel(job['job_id'])
        kwargs['on_sample']({'case_id': 'PG07', 'repetition': 1, 'passed': False})
        pytest.fail('取消应在样本落盘后终止')
    with tempfile.TemporaryDirectory(dir=ROOT / 'research/graphs') as graphs:
        run_job(job['job_id'], store=store, evaluator=evaluator, execution_source='test_harness', graph_root=Path(graphs))
        run_job(job['job_id'], store=store, evaluator=evaluator, graph_root=Path(graphs))
    saved = store.get(job['job_id'])
    assert calls == ['dev'] and saved['status'] == 'CANCELLED'
    assert saved['completed_samples'] == 1
    assert list(store.directory(job['job_id']).glob('samples/**/*.json'))


def test_worker_stops_when_persisted_usage_crosses_frozen_budget(tmp_path):
    from eval.run_pg_lab import run_job
    store = JobStore(tmp_path)
    job = store.create(StartExperiment(request_id='budget-worker', preset='smoke'), launch=False)
    def evaluator(dataset, **kwargs):
        kwargs['on_sample']({'case_id': 'PG07', 'repetition': 1, 'passed': False,
            'model_attempts': job['frozen']['max_model_attempts'] + 1,
            'turns': [{'result': {'events': [], 'budget': {'consumed': {
                'prompt_tokens': 1, 'completion_tokens': 1}}}}]})
        pytest.fail('预算超限后不能继续执行')
    with tempfile.TemporaryDirectory(dir=ROOT / 'research/graphs') as graphs:
        run_job(job['job_id'], store=store, evaluator=evaluator,
                execution_source='test_harness', graph_root=Path(graphs))
    saved = store.get(job['job_id'])
    assert saved['status'] == 'FAILED' and saved['error'] == 'PG_BUDGET_EXCEEDED'
    assert saved['budget_usage']['model_attempts'] == job['frozen']['max_model_attempts'] + 1
    assert list(store.directory(job['job_id']).glob('samples/**/*.json'))


@pytest.mark.parametrize('accept', [False, True])
def test_worker_three_arms_are_after_learning_and_test_never_reaches_refiner(tmp_path, accept):
    from eval.run_pg_lab import run_job
    from src.harness import get_runner
    from eval.run_live_eval import summarize, environment_manifest
    store = JobStore(tmp_path)
    job = store.create(StartExperiment(request_id=f'complete-worker-{accept}', preset='smoke'), launch=False)
    order, refinements = [], []
    class Refiner:
        def propose(self, graph, report, rejected):
            refinements.append([s['case_id'] for s in report['samples']])
            order.append('refine')
            from tests.test_pg_learning import proposal
            patch = proposal()
            patch['source_cases'] = ['PG01']
            return dict(proposal=json.dumps(patch) if accept else None,
                        error=None if accept else 'SCRIPTED_REJECTION', events=[])
    def evaluator(dataset, **kwargs):
        order.append(kwargs['split'])
        provenance = get_runner(kwargs['harness_profile']).provenance()
        context = next(m for m in provenance['mechanisms'] if m['slot'] == 'context')
        manifest = dict(environment_manifest(dataset, 'adaptive', kwargs['dataset_path']),
            suite='adaptive', split=kwargs['split'], repeats=kwargs['repeats'], limits={'max_model_calls': 16},
            purpose_models={'planner': 'scripted'}, harness_profile=provenance,
            selected_cases=[{'id': c, 'split': kwargs['split'], 'turns': 1} for c in kwargs['case_ids']])
        samples = []
        for case in kwargs['case_ids']:
            sample = dict(case_id=case, split=kwargs['split'], repetition=1, passed=True, adaptive_verified=True,
                model_attempts=4, tool_calls=1, safety_violations=0, protocol_violations=0, latency_ms=1,
                turns=[{'failures': [], 'message': '脚本夹具', 'result': {'events': [{'type': 'mechanism_applied',
                    'slot': 'context', 'mechanism_id': context['id'],
                    'graph_sha256': context.get('configuration', {}).get('graph_sha256')}]}}])
            samples.append(sample)
            kwargs['on_sample'](sample)
        return dict(status='COMPLETED', execution_source='test_harness', live_verified=False, providers=['scripted'],
                    manifest=manifest, samples=samples, metrics=summarize(samples, 1))
    with tempfile.TemporaryDirectory(dir=ROOT / 'research/graphs') as graphs:
        run_job(job['job_id'], store=store, evaluator=evaluator, refiner=Refiner(),
                execution_source='test_harness', graph_root=Path(graphs))
    saved = store.get(job['job_id'])
    assert saved['status'] == 'COMPLETED', saved
    assert order == ['dev', 'dev', 'refine', *(['dev'] if accept else []), 'test', 'test', 'test']
    assert refinements == [job['frozen']['groups']['train']]
    assert saved['comparison']['conclusion'] == 'SMOKE_ONLY'
    assert saved['comparison']['evidence_valid'] is False
    assert saved['comparison']['failures'] == []
    assert saved['history'][0]['status'] == ('ACCEPTED' if accept else 'REJECTED')
    assert set(saved['arms']) == {'baseline', 'seed', 'evolved'}


def test_empty_search_fault_is_once_and_does_not_mutate_business_data(monkeypatch):
    from eval.adaptive_checks import scenario
    from src.direct_executor import _READ_ONLY_TOOLS
    from src.runtime import ToolSpec, tool_spec
    original = _READ_ONLY_TOOLS['search_products']
    spec = tool_spec('search_products', original)
    product = {'product_id': 'P1101'}
    monkeypatch.setitem(_READ_ONLY_TOOLS, 'search_products', ToolSpec(lambda **kwargs: {'products': [product], 'total': 1}, spec.schema))
    with scenario({'scenario': {'kind': 'empty_search'}}, 'test') as state:
        function = tool_spec('search_products', _READ_ONLY_TOOLS['search_products']).function
        assert function()['products'] == []
        assert function()['products'] == [product]
        assert state['fired'] and not state['environment_paths']
        assert state['tool_overrides'][0]['original_result']['products'] == [product]


def test_spawn_failure_is_durable_and_cannot_rebill_retry(tmp_path, monkeypatch):
    import src.pg_jobs as jobs
    import src.llm
    monkeypatch.setattr(src.llm, 'get_model_status', lambda: {'configured': True})
    calls = []
    def failed_spawn(*args, **kwargs):
        calls.append(args)
        raise OSError('sensitive external error')
    monkeypatch.setattr(jobs.subprocess, 'Popen', failed_spawn)
    store = JobStore(tmp_path)
    request = StartExperiment(request_id='spawn-failure', preset='smoke')
    first = store.create(request)
    assert first['error'] == 'PG_WORKER_START_FAILED' and first['status'] == 'FAILED'
    assert store.create(request)['job_id'] == first['job_id']
    assert sum('-m' in call[0] for call in calls) == 1 and 'sensitive' not in json.dumps(first)


def test_fault_judging_requires_physical_source_and_rejects_arbitrary_rewrites():
    from eval.adaptive_checks import observed_with_faults
    raw = {'tool': 'search_products', 'args': {'query': '耳机', 'limit': 4},
           'result': {'products': [{'product_id': 'P1101'}], 'total': 1}}
    injected = {'products': [], 'total': 0}
    environment = {'kind': 'empty_search', 'fired': True, 'tool_overrides': [{
        'tool': 'search_products', 'args': {'query': '耳机'}, 'original_result': raw['result'],
        'injected_result': injected}]}
    checked = observed_with_faults([raw], environment)
    assert checked[0]['result'] == injected and raw['result']['total'] == 1
    with pytest.raises(ValueError, match='物理工具证据'):
        observed_with_faults([], environment)
    environment['tool_overrides'][0]['injected_result']['products'] = [{'product_id': 'P9999'}]
    with pytest.raises(ValueError, match='声明范围'):
        observed_with_faults([raw], environment)


def test_empty_fault_keeps_real_backend_trace_judge_valid():
    from eval import run_task_eval as offline
    from eval.adaptive_checks import scenario, observed_with_faults
    from src.direct_executor import _READ_ONLY_TOOLS
    from src.runtime import tool_spec
    runtime = offline._runtime()
    with offline._isolated_db(runtime, {}):
        before = offline._snapshot(runtime, runtime['tools']._engine)
        with offline._observe_tools(runtime) as observed:
            with scenario({'scenario': {'kind': 'empty_search'}}, 'fault-source') as environment:
                result = tool_spec('search_products', _READ_ONLY_TOOLS['search_products']).function(category='耳机')
        assert result['products'] == [] and observed[0]['result']['products']
        trace = {'status': 'SUCCEEDED', 'reply': '本次检索为空', 'run_id': 'fault-test',
                 'events': [{'type': 'terminal', 'run_id': 'fault-test', 'sequence': 1}], 'trace': [
            {'tool': 'search_products', 'args': {'category': '耳机'}, 'result': result}]}
        checked = observed_with_faults(observed, environment)
        assert offline.judge_turn({'statuses': ['SUCCEEDED']}, trace, before, before, checked, 0) == []
        assert any('physical backend evidence' in f for f in
                   offline.judge_turn({'statuses': ['SUCCEEDED']}, trace, before, before, observed, 0))
