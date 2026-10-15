"""用真实文件DB验证会话控制的执行边界与重启语义。"""
import threading
import time

import pytest

from src import tools
from src.backend.db import get_engine, get_session_factory
from src.backend.seed import seed
from src.durable import run_conversation, resume_run
from src.errors import AgentError
from src.orchestrator import run_orchestrator
from src.runtime import BudgetLimits, current_runtime, ToolExecutor, TraceRecorder
from src.session_control import ControlStore, SessionController
from src.trace_projector import project_trace


@pytest.fixture
def database(tmp_path):
    engine = get_engine(f'sqlite:///{tmp_path / "controls.db"}')
    tools.configure_db(engine)
    with get_session_factory(engine)() as session:
        seed(session)
    yield
    engine.dispose()


def wait_result(run_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        run = ControlStore().run(run_id)
        if run['result'] is not None:
            return run['result']
        time.sleep(.01)
    raise AssertionError('任务未收敛')


def test_steer_changes_target_and_only_one_terminal(database):
    store = ControlStore()
    def runner(message, **kwargs):
        context = current_runtime()
        if '运行中补充要求' not in message:
            store.enqueue('s', context.run_id, 'STEER', message='改查订单 1003', command_id='change', expected_revision=0)
        return run_orchestrator('查询订单 1003 的状态' if '运行中补充要求' in message else '查询订单 1002 的状态', **kwargs)
    result = run_conversation('查询订单 1002 的状态', 's', runner=runner)
    assert result['trace'][-1]['result']['order_id'] == 1003
    assert len([e for e in result['events'] if e['type'] == 'terminal']) == 1
    assert store.commands('s', result['run_id'])[0]['status'] == 'APPLIED'
    assert project_trace(result['events'])['integrity']['complete'] is True


def test_async_adaptive_profile_and_explicit_budget_reach_runtime(database):
    observed = []
    def runner(message, **kwargs):
        observed.append(current_runtime().limits)
        return run_orchestrator(message, **kwargs)
    controller = SessionController()
    first = controller.start('adaptive-profile', '查询订单1002', execution_mode='adaptive', runner=runner)
    assert wait_result(first['run_id'])['status'] == 'SUCCEEDED'
    assert observed[-1] == BudgetLimits.adaptive()
    explicit = BudgetLimits(max_model_calls=1, max_seconds=5)
    second = controller.start('explicit-profile', '查询订单1002', execution_mode='adaptive', runner=runner, limits=explicit)
    assert wait_result(second['run_id'])['status'] == 'SUCCEEDED'
    assert observed[-1] == explicit


def test_cancel_keeps_completed_observation_and_prevents_next_tool(database):
    def runner(message, **kwargs):
        context = current_runtime()
        executor = ToolExecutor({'get_order_status': tools.get_order_status}, TraceRecorder(context.run_id))
        executor.execute('get_order_status', {'order_id': 1002})
        ControlStore().enqueue('s', context.run_id, 'CANCEL', command_id='stop', expected_revision=0)
        executor.execute('get_order_status', {'order_id': 1003})
        raise AssertionError('不应执行第二步')
    result = run_conversation('查询两个订单', 's', runner=runner)
    assert result['status'] == 'CANCELLED'
    assert [t['args']['order_id'] for t in result['trace']] == [1002]
    assert project_trace(result['events'])['integrity']['complete'] is True


def test_commands_are_scoped_idempotent_and_revision_checked(database):
    claim = ControlStore().claim('s', 'q', '查询订单', {})
    store, run_id = ControlStore(), claim['run_id']
    a = store.enqueue('s', run_id, 'STEER', message='预算300', command_id='one', expected_revision=0)
    assert store.enqueue('s', run_id, 'STEER', message='预算300', command_id='one', expected_revision=0) == a
    with pytest.raises(AgentError) as stale:
        store.enqueue('s', run_id, 'STEER', message='预算500', command_id='stale', expected_revision=0)
    assert stale.value.code == 'STALE_CONTROL_REVISION'
    with pytest.raises(AgentError):
        store.enqueue('other', run_id, 'CANCEL', command_id='other', expected_revision=1)
    store.enqueue('s', run_id, 'FOLLOW_UP', message='查询订单1003', command_id='follow', expected_revision=1)
    store.enqueue('s', run_id, 'CANCEL', command_id='cancel', expected_revision=2)
    assert store.commands('s', run_id)[1]['status'] == 'CANCELLED'


def test_async_follow_up_is_durable_and_sequential(database):
    entered, release = threading.Event(), threading.Event()
    def runner(message, **kwargs):
        if message == '查询订单 1002 的状态':
            entered.set()
            assert release.wait(3)
        return run_orchestrator(message, **kwargs)
    controller = SessionController()
    handle = controller.start('s', '查询订单 1002 的状态', runner=runner)
    assert entered.wait(3)
    ControlStore().enqueue('s', handle['run_id'], 'FOLLOW_UP', message='查询订单 1003 的状态', command_id='later', expected_revision=0)
    release.set()
    assert wait_result(handle['run_id'])['status'] == 'SUCCEEDED'
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        command = ControlStore().commands('s', handle['run_id'])[0]
        if command.get('outcome'):
            assert wait_result(command['outcome']['run_id'])['trace'][0]['args']['order_id'] == 1003
            return
        time.sleep(.01)
    raise AssertionError('后续任务未启动')


def test_cancel_survives_process_interruption(database):
    class Crash(BaseException): pass
    def runner(message, **kwargs):
        context = current_runtime()
        ControlStore().enqueue('s', context.run_id, 'CANCEL', command_id='durable-stop', expected_revision=0)
        raise Crash()
    with pytest.raises(Crash):
        run_conversation('查询订单1002', 's', runner=runner)
    active = ControlStore().conversation('s')['active_run_id']
    result = resume_run('s', active)
    assert result['status'] == 'CANCELLED'
    assert not result['trace']


def test_terminal_event_and_result_commit_atomically(database):
    class Crash(BaseException): pass
    class FailOnceStore(ControlStore):
        def finish(self, *args, **kwargs):
            raise Crash()
    with pytest.raises(Crash):
        run_conversation('查询订单1002状态', 'atomic', store=FailOnceStore())
    active = ControlStore().conversation('atomic')['active_run_id']
    assert not any(e['type'] == 'terminal' for e in ControlStore().run(active)['events'])
    result = resume_run('atomic', active)
    assert sum(e['type'] == 'terminal' for e in result['events']) == 1
    assert project_trace(result['events'])['integrity']['single_terminal'] is True


def test_steer_after_budget_exhaustion_has_terminal_result(database):
    from src.runtime import BudgetLimits
    def runner(message, **kwargs):
        context = current_runtime()
        time.sleep(.02)
        ControlStore().enqueue('expired', context.run_id, 'STEER', message='改预算', command_id='expired', expected_revision=0)
        return {'reply': '尚未完成', 'trace': [], 'agent': 'runtime'}
    result = run_conversation('原始需求', 'expired', limits=BudgetLimits(max_seconds=.01), runner=runner)
    assert result['status'] == 'FAILED'
    assert result['terminal_reason'] == 'TIME_BUDGET_EXHAUSTED'
    assert result['events'][-1]['type'] == 'terminal'
    assert ControlStore().conversation('expired')['active_run_id'] is None


def test_read_retry_checks_new_cancel_before_reinvoking(database):
    from src.runtime import ToolSpec
    calls = []
    def read(value: str):
        calls.append(value)
        active = ControlStore().conversation('retry')['active_run_id']
        ControlStore().enqueue('retry', active, 'CANCEL', command_id='retry-stop', expected_revision=0)
        raise TimeoutError('测试暂时超时')
    def runner(message, **kwargs):
        context = current_runtime()
        ToolExecutor({'read': ToolSpec(read)}, TraceRecorder(context.run_id)).execute('read', {'value': message})
    result = run_conversation('旧目标', 'retry', runner=runner)
    assert result['status'] == 'CANCELLED'
    assert calls == ['旧目标']


def test_drain_continues_after_cached_followup_from_restart(database):
    from src.durable import execute_claim
    from src.runtime import BudgetLimits
    store = ControlStore()
    parent = store.claim('restart-queue', 'parent', '查询订单1002状态', BudgetLimits().__dict__)
    store.enqueue('restart-queue', parent['run_id'], 'FOLLOW_UP', message='查询订单1003状态', command_id='first', expected_revision=0)
    store.enqueue('restart-queue', parent['run_id'], 'FOLLOW_UP', message='查询订单1001状态', command_id='second', expected_revision=1)
    execute_claim(parent, store=store)
    first = store.claim('restart-queue', 'first', '查询订单1003状态', BudgetLimits().__dict__)
    execute_claim(first, store=store)  # 模拟完成后、command确认前崩溃。
    handle = SessionController().drain('restart-queue')
    assert handle and handle['run_id'] != first['run_id']
    assert wait_result(handle['run_id'])['status'] == 'SUCCEEDED'
    commands = store.commands('restart-queue', parent['run_id'])
    assert all(command['status'] == 'APPLIED' for command in commands)


def test_recovery_does_not_reuse_an_old_observation_scope(database):
    from src.runtime import ToolSpec
    class Crash(BaseException): pass
    class CrashOnSecondRead(ControlStore):
        calls = 0
        def save_checkpoint(self, run_id, token, snapshot, event=None):
            super().save_checkpoint(run_id, token, snapshot, event)
            if event and event['type'] == 'tool_call':
                self.calls += 1
                if self.calls == 2:
                    raise Crash()
    reads = []
    def read(value: str):
        reads.append(value)
        return {'value': len(reads)}
    def runner(message, **kwargs):
        context = current_runtime()
        for scope in ['old', 'new']:
            ToolExecutor({'read': ToolSpec(read)}, TraceRecorder(context.run_id), observation_scope=scope).execute('read', {'value': 'stock'})
    with pytest.raises(Crash):
        run_conversation('核验库存', 'scope-recovery', runner=runner, store=CrashOnSecondRead())
    active = ControlStore().conversation('scope-recovery')['active_run_id']
    ControlStore().enqueue('scope-recovery', active, 'CANCEL', command_id='scope-stop', expected_revision=0)
    result = resume_run('scope-recovery', active, runner=runner)
    assert result['status'] == 'CANCELLED'
    assert reads == ['stock']
    assert len([e for e in result['events'] if e['type'] == 'tool_result']) == 1
    assert any(e.get('code') == 'INTERRUPTED_TOOL_CALL' for e in result['events'])
    assert project_trace(result['events'])['integrity']['complete'] is True


@pytest.mark.parametrize('fields', [{}, {'command_id': 'id'}, {'expected_revision': 0},
                                    {'command_id': ' ', 'expected_revision': 0},
                                    {'command_id': 'id', 'expected_revision': True}])
def test_control_contract_requires_id_and_integer_revision(database, fields):
    store = ControlStore()
    run_id = store.claim('contract', 'request', '目标', {})['run_id']
    with pytest.raises(AgentError, match='必须提供'):
        store.enqueue('contract', run_id, 'CANCEL', **fields)
    assert store.commands('contract', run_id) == []


def test_command_id_cannot_be_rebound_to_another_revision(database):
    store = ControlStore()
    run_id = store.claim('revision', 'request', '目标', {})['run_id']
    store.enqueue('revision', run_id, 'STEER', message='新目标', command_id='id', expected_revision=0)
    with pytest.raises(AgentError) as error:
        store.enqueue('revision', run_id, 'STEER', message='新目标', command_id='id', expected_revision=1)
    assert error.value.code == 'COMMAND_ID_CONFLICT'


@pytest.mark.parametrize('kinds,same_id', [(('STEER', 'STEER'), False), (('STEER', 'CANCEL'), False),
                                        (('FOLLOW_UP', 'FOLLOW_UP'), False), (('STEER', 'STEER'), True)])
def test_concurrent_controls_have_one_atomic_winner_or_one_idempotent_command(database, kinds, same_id):
    from concurrent.futures import ThreadPoolExecutor
    store = ControlStore()
    run_id = store.claim('concurrent', 'request', '目标', {})['run_id']
    barrier = threading.Barrier(2)
    def submit(index):
        barrier.wait(timeout=3)
        try:
            return store.enqueue('concurrent', run_id, kinds[index], message='新目标',
                                 command_id='same' if same_id else f'command-{index}', expected_revision=0)
        except AgentError as error:
            return error.code
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(submit, index) for index in range(2)]
        results = [future.result(timeout=5) for future in futures]
    assert len(store.commands('concurrent', run_id)) == 1
    assert store.revision(run_id) == 1
    if same_id:
        assert results[0] == results[1]
    else:
        assert sum(isinstance(r, dict) for r in results) == 1
        assert any(r in ('STALE_CONTROL_REVISION', 'RUN_CANCELLING') for r in results if isinstance(r, str))
