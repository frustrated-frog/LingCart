"""PG 实验任务的独立进程执行器；UI 启动后按冻结计划顺序运行。"""
import argparse
from copy import deepcopy
import threading

from eval.run_live_eval import evaluate
from eval.run_pg_evolution import graph_profile
from src.harness import get_registry, get_runner
from src.pg_experiment import (ROOT, DATASET, SEED, ARMS, challenge_dataset, behavior_metrics,
                               test_contract, comparison)
from src.pg_jobs import JobStore, identity
from src.pg_evidence import project_sample_evidence
from src.pg_learning import EvolutionSpec, ModelGraphRefiner, evolve, write_artifact
from src.procedural_graph import ProceduralGraph


class ExperimentStopped(Exception):
    pass


class ExperimentChanged(Exception):
    pass


class ExperimentBudgetExceeded(Exception):
    pass


def run_job(job_id, *, store=None, evaluator=evaluate, refiner=None, execution_source='live_provider', graph_root=None):
    store = store or JobStore()
    if not store.claim(job_id):
        return
    done = threading.Event()
    def pulse():
        while not done.wait(10):
            store.heartbeat(job_id)
    thread = threading.Thread(target=pulse, daemon=True)
    thread.start()
    try:
        job = store.get(job_id)
        output = store.directory(job_id)
        output.mkdir(exist_ok=False)
        frozen = job['frozen']
        spec = EvolutionSpec.model_validate_json(__import__('json').dumps(frozen['spec']))
        dataset = challenge_dataset()
        spec.validate_dataset(dataset)
        graph = ProceduralGraph.from_path(SEED)
        graphs = (graph_root or ROOT / 'research/graphs/pg-lab') / job_id
        write_artifact(output / 'frozen.json', dict(plan=frozen, identity=job['identity'], dataset=dataset,
                                                  seed=graph.model_dump(mode='json')))
        profiles, history, arms, reports = {}, [], {}, {}
        completed = 0
        learning_reports = []
        learning_cost = {'rollout_model_attempts': 0, 'refiner_model_attempts': 0,
                         'rollout_tokens': None, 'refiner_token_units': 0}
        budget_usage = {'model_attempts': 0, 'token_units': 0}
        store.update(job_id, execution_source=execution_source)

        def check_budget():
            if (budget_usage['model_attempts'] > frozen['cost_guard']['max_model_attempts']
                    or budget_usage['token_units'] > frozen['cost_guard']['max_token_units']):
                raise ExperimentBudgetExceeded()

        def check():
            current = store.get(job_id)
            if current['cancel_requested'] or current['status'] != 'RUNNING':
                raise ExperimentStopped()
            if job['identity'] != identity():
                raise ExperimentChanged()

        def profile(current):
            if current.fingerprint not in profiles:
                path = graphs / f'graph-{current.fingerprint}.json'
                write_artifact(path, current.model_dump(mode='json'))
                selected = graph_profile(current, path)
                get_registry().register_profile(selected)
                write_artifact(graphs / f'profile-{current.fingerprint}.json', selected.model_dump(mode='json'))
                profiles[current.fingerprint] = selected
            return profiles[current.fingerprint].profile_id

        def execute(current, cases, label, *, split='dev', profile_id=None):
            nonlocal completed
            check()
            selected = profile_id or profile(current)
            stage_count = 0
            store.update(job_id, stage=label, stage_completed=0, stage_total=len(cases) * spec.repeats)
            def on_sample(sample):
                nonlocal completed, stage_count
                persisted = {**sample, 'pg_evidence': project_sample_evidence(sample)}
                write_artifact(output / 'samples' / label / f'{sample["case_id"]}-{sample["repetition"]}.json', persisted)
                budget_usage['model_attempts'] += sample.get('model_attempts', 0)
                for turn in sample.get('turns', []):
                    consumed = ((turn.get('result') or {}).get('budget') or {}).get('consumed') or {}
                    budget_usage['token_units'] += consumed.get('prompt_tokens', 0) + consumed.get('completion_tokens', 0)
                completed += 1
                stage_count += 1
                store.update(job_id, completed_samples=completed, stage_completed=stage_count,
                             budget_usage=budget_usage.copy())
                check_budget()
                check()
            report = evaluator(dataset, repeats=spec.repeats, split=split, case_ids=cases, execution_mode='adaptive',
                harness_profile=selected, dataset_path=DATASET, max_model_calls_per_run=spec.max_model_calls_per_run,
                max_prompt_tokens_per_run=spec.max_prompt_tokens_per_run,
                max_completion_tokens_per_run=spec.max_completion_tokens_per_run,
                max_seconds_per_run=spec.max_seconds_per_run,
                max_provider_attempts_per_call=spec.max_provider_attempts_per_call,
                on_sample=on_sample)
            write_artifact(output / f'{label}.json', report)
            if report.get('status') != 'COMPLETED':
                raise RuntimeError('PG_INCOMPLETE_STAGE')
            if split == 'dev':
                learning_reports.append(report)
                learning_cost['rollout_model_attempts'] = sum(r['metrics']['model_attempts'] for r in learning_reports)
                tokens = [r['metrics']['tokens']['total_tokens'] for r in learning_reports]
                learning_cost['rollout_tokens'] = sum(tokens) if all(t is not None for t in tokens) else None
                store.update(job_id, learning_cost=learning_cost)
            return report

        delegate = refiner or ModelGraphRefiner(max_prompt_tokens=spec.max_refiner_prompt_tokens,
                                                max_completion_tokens=spec.max_refiner_completion_tokens,
                                                max_provider_attempts=spec.max_provider_attempts_per_call)
        class Refiner:
            def propose(self, current, report, rejected):
                check()
                store.update(job_id, stage='refinement', stage_completed=0, stage_total=1)
                proposal = delegate.propose(current, report, rejected)
                events = proposal.get('events') or []
                attempts = sum(e.get('type') in {'model_started', 'model_retry'} for e in events)
                consumed = proposal.get('budget') or {}
                token_units = consumed.get('prompt_tokens', 0) + consumed.get('completion_tokens', 0)
                learning_cost['refiner_model_attempts'] += attempts
                learning_cost['refiner_token_units'] += token_units
                budget_usage['model_attempts'] += attempts
                budget_usage['token_units'] += token_units
                store.update(job_id, learning_cost=learning_cost, budget_usage=budget_usage.copy())
                check_budget()
                return proposal

        def on_selection(record):
            history.append(record)
            store.update(job_id, history=deepcopy(history))

        retained, evolution = evolve(graph, spec, evaluator=execute, refiner=Refiner(), output_dir=graphs,
            execution_source=execution_source, on_progress=on_selection)
        store.update(job_id, retained_graph=retained.model_dump(mode='json'), evolution=evolution)
        identifiers = dict(baseline='adaptive-v1', seed=profile(graph), evolved=profile(retained))
        provenance = {arm: get_runner(pid).provenance() for arm, pid in identifiers.items()}
        contracts = {arm: test_contract(provenance[arm], provenance[base], spec.repeats)
                     if identifiers[arm] != identifiers[base] else None
                     for arm, base in (('seed', 'baseline'), ('evolved', 'seed'))}
        # 在读取 test 轨迹之前冻结图与两个对照契约；之后不再调用 Refiner。
        write_artifact(output / 'test-contracts.json', contracts)
        store.update(job_id, contracts=contracts)
        for arm in ARMS:
            report = execute(None, frozen['groups']['test'], 'test-' + arm, split='test', profile_id=identifiers[arm])
            reports[arm] = report
            arms[arm] = dict(profile=provenance[arm], metrics=behavior_metrics(report),
                execution_source=report['execution_source'], live_verified=report.get('live_verified', False),
                samples=[dict(case_id=s['case_id'], repetition=s['repetition'], passed=s['passed'],
                              failures=[f for t in s['turns'] for f in t['failures']]) for s in report['samples']])
            store.update(job_id, arms=deepcopy(arms))
        result = comparison(reports, frozen, contracts)
        write_artifact(output / 'comparison.json', result)
        check()
        store.update(job_id, status='COMPLETED', stage='completed', comparison=result)
    except ExperimentStopped:
        store.update(job_id, status='CANCELLED', error='PG_CANCELLED')
    except ExperimentChanged:
        store.update(job_id, status='FAILED', error='PG_IDENTITY_CHANGED')
    except ExperimentBudgetExceeded:
        store.update(job_id, status='FAILED', error='PG_BUDGET_EXCEEDED')
    except Exception as error:
        # 原始已完成样本仍在磁盘；Provider 异常文本不能进入状态接口。
        store.update(job_id, status='FAILED', error='PG_EXPERIMENT_FAILED', error_type=type(error).__name__)
    finally:
        done.set()
        thread.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id', required=True)
    run_job(parser.parse_args().job_id)


if __name__ == '__main__':
    main()
