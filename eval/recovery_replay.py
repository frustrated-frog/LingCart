"""仅隔离评测使用的失败点冻结与多策略回放，不是生产 Run 的恢复入口。"""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from datetime import date, datetime
import json
import uuid

from sqlalchemy import select

from eval import run_task_eval as offline
from eval.adaptive_checks import judge, scenario
from eval.recovery_faults import recovery_fault
from src.context_engine import HookDecision
from src.harness import get_registry
from src.procedural_graph import fingerprint
from src.rrsi_policy import failure_kind
from src.runtime import BudgetLimits, RunRuntime, runtime_scope
from src.runtime_state import new_runtime_snapshot, seal_runtime_snapshot
from src.task_plan import TaskPlan, TaskStep
from src.trace_projector import project_trace


class Captured(BaseException):
    def __init__(self, point):
        self.point = point


def dump_database(engine):
    from src.backend.db import Base
    with engine.connect() as connection:
        return {table.name: sorted([offline._json(dict(row._mapping)) for row in connection.execute(select(table))],
                                   key=lambda row: json.dumps(row, sort_keys=True))
                for table in Base.metadata.sorted_tables}


@contextmanager
def restore_database(payload):
    from src import tools
    from src.backend.db import Base, get_engine
    original = tools._engine, tools._SessionLocal
    engine = get_engine('sqlite:///:memory:')
    try:
        tools.configure_db(engine)
        tables = Base.metadata.sorted_tables
        if set(payload) != {t.name for t in tables}:
            raise ValueError('REPLAY_DATABASE_SCHEMA_MISMATCH')
        with engine.begin() as connection:
            for table in reversed(tables):
                connection.execute(table.delete())
            for table in tables:
                for raw in payload[table.name]:
                    row = deepcopy(raw)
                    for column in table.columns:
                        if row.get(column.name) is not None and column.type.python_type in {date, datetime}:
                            row[column.name] = column.type.python_type.fromisoformat(row[column.name])
                    connection.execute(table.insert().values(**row))
        if fingerprint(dump_database(engine)) != fingerprint(payload):
            raise ValueError('REPLAY_DATABASE_RESTORE_MISMATCH')
        yield engine
    finally:
        tools._engine, tools._SessionLocal = original
        engine.dispose()


def read_only(_payload):
    return HookDecision(allowed=_payload['policy']['effect'] == 'READ', reason='回放只允许只读工具')


def capture_case(case, limits, *, source_manifest):
    """采集第一次已提交的可恢复失败；未遇到失败时明确返回 NOT_CAPTURED。"""
    if len(case['turns']) != 1 or case.get('scenario', {}).get('kind') == 'steer':
        raise ValueError('REPLAY_CASE_UNSUPPORTED')
    runtime = offline._runtime()
    turn = case['turns'][0]
    history = offline._history(runtime, turn.get('history', []))
    runner = get_registry().resolve('adaptive-recovery-control-v1')
    context = RunRuntime('capture-' + uuid.uuid4().hex, limits)
    context.hooks.register('before_tool', 'replay-read-only', read_only)
    with offline._isolated_db(runtime, case.get('fixture', {})) as engine:
        with scenario(case, context.run_id) as environment:
            def capture(payload):
                state = payload['adaptive']
                plan = TaskPlan(state['plan']['goals'], [TaskStep(**s) for s in state['plan']['steps']], state['plan'].get('verification', {}))
                if not state.get('goal') or state.get('pending') or not failure_kind(state, plan):
                    return
                data = dict(schema_version=1, case=case, history=turn.get('history', []),
                    origin_run_id=context.run_id, origin_profile=runner.provenance(),
                    source_manifest=source_manifest, limits=limits.__dict__,
                    adaptive=deepcopy(state), models=deepcopy(context.models),
                    budget=deepcopy(context.counters), prefix_events=deepcopy(context.events),
                    environment=deepcopy(environment), database=dump_database(engine))
                data['origin_sha256'] = fingerprint(data)
                raise Captured(data)
            context.hooks.register('recovery_checkpoint', 'capture-failure', capture)
            try:
                with runtime_scope(context):
                    result = runner(turn['message'], history)
            except Captured as stopped:
                return dict(status='CAPTURED', point=stopped.point)
    return dict(status='NOT_CAPTURED', reason='NO_RECOVERABLE_CHECKPOINT', result=result)


def replay(point, profile_id, *, source_manifest, repetition=1):
    if point.get('schema_version') != 1 or point.get('origin_sha256') != fingerprint({k:v for k,v in point.items() if k != 'origin_sha256'}):
        raise ValueError('REPLAY_POINT_CORRUPT')
    for field in ('source_tree_sha256', 'dataset_sha256', 'dependency_lock_sha256', 'purpose_models'):
        if point['source_manifest'].get(field) != source_manifest.get(field):
            raise ValueError('REPLAY_ENVIRONMENT_CHANGED')
    if point['adaptive'].get('pending') or not point['adaptive'].get('goal'):
        raise ValueError('REPLAY_POINT_NOT_COMMITTED')
    case = point['case']
    runtime = offline._runtime()
    run_id = 'replay-' + uuid.uuid4().hex
    snapshot = new_runtime_snapshot(run_id, conversation={}, history=point['history'])
    # 这是明确分叉的新研究 Run；不伪装成原 Run 恢复，也不复制原 Profile 的选择事件。
    snapshot.update(adaptive=deepcopy(point['adaptive']), budget=deepcopy(point['budget']),
        models=deepcopy(point['models']), goal_revision=point['adaptive']['goal_revision'], phase='PLAN', current_node='PLAN')
    context = RunRuntime(run_id, BudgetLimits(**point['limits']), snapshot=seal_runtime_snapshot(snapshot, run_id))
    context.hooks.register('before_tool', 'replay-read-only', read_only)
    context.record('recovery_replay_origin', origin_run_id=point['origin_run_id'], origin_sha256=point['origin_sha256'],
                   database_sha256=fingerprint(point['database']), prefix_budget=point['budget'])
    runner = get_registry().resolve(profile_id)
    with restore_database(point['database']) as engine:
        before = offline._snapshot(runtime, engine)
        before_full = dump_database(engine)
        kind = point['environment'].get('kind')
        fault_state = dict(kind=kind, fired=False)
        faults = (recovery_fault(kind, fault_state, remaining_faults=point['environment']['remaining_faults'])
                  if kind in {'tool_timeout', 'repeated_tool_failure', 'invalid_arguments'} else nullcontext())
        with faults, offline._observe_tools(runtime) as observed, runtime_scope(context):
            result = runner(point['adaptive']['request'], offline._history(runtime, point['history']))
        after = offline._snapshot(runtime, engine)
        mutations = offline._diff(before_full, dump_database(engine))
        attempts = sum(e['type'] in {'model_started','model_retry'} for e in result['events'])
        failures = offline.judge_turn(case['turns'][0]['expect'], result, before, after, observed, attempts)
        failures += judge({**case, '_effective_request': point['adaptive']['request']}, result, after)
        if mutations:
            failures.append('REPLAY_DATABASE_MUTATED')
    integrity = project_trace(result['events'])['integrity']['complete'] is True
    if not integrity:
        failures.append('REPLAY_TRACE_INVALID')
    delta = {key: context.counters[key] - point['budget'].get(key, 0)
             for key in ('model_calls','tool_calls','prompt_tokens','completion_tokens')}
    prefix_steps = len(point['adaptive']['plan']['steps'])
    from src.recovery_evidence import recovery_trace
    sample = dict(case_id=case['id'], repetition=repetition, passed=not failures,
        turns=[dict(result=result)])
    return dict(profile_id=profile_id, case_id=case['id'], split=case['split'], repetition=repetition,
        origin_sha256=point['origin_sha256'], database_sha256=fingerprint(point['database']),
        prefix_budget=point['budget'], suffix_budget=delta, suffix_usage=result.get('usage'),
        suffix_steps=max(0, len((result.get('plan') or {}).get('steps', [])) - prefix_steps),
        passed=not failures, failures=failures, integrity=integrity, changed_db_paths=mutations,
        environment=fault_state, result=result, observed_tools=observed, recovery=recovery_trace(sample),
        searches=deepcopy(context.snapshot.get('adaptive', {}).get('recovery_searches', [])),
        suffix_models={key:deepcopy(value) for key,value in context.models.items()
                       if int(key) > point['adaptive']['model_cursor']})
