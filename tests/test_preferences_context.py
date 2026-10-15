import pytest
from src import tools
from src.backend.db import get_engine, get_session_factory
from src.backend.seed import seed
from src.context_engine import ContextEngine, LifecycleHooks, HookDecision
from src.durable import run_conversation
from src.errors import AgentError
from src.preferences import PreferenceStore
from src.runtime import RunRuntime, ToolExecutor, TraceRecorder, runtime_scope


@pytest.fixture
def database(tmp_path):
    engine = get_engine(f'sqlite:///{tmp_path / "preferences.db"}')
    tools.configure_db(engine)
    with get_session_factory(engine)() as session:
        seed(session)
    yield
    engine.dispose()


def test_explicit_memory_update_forget_and_cross_session_scope(database):
    store = PreferenceStore()
    profile, other = store.create(), store.create()
    with pytest.raises(AgentError):
        store.remember(profile, 'max_price', 300)
    store.remember(profile, 'max_price', 300, confirmed=True)
    store.bind('one', profile)
    store.bind('two', profile)
    assert store.for_session('one') == store.for_session('two')
    assert not store.list(other)
    store.remember(profile, 'max_price', 500, confirmed=True)
    assert store.list(profile)[0]['version'] == 2
    store.forget(profile, 'max_price')
    assert not store.for_session('two')
    with pytest.raises(AgentError):
        store.remember(profile, 'order_status', 'paid', confirmed=True)
    with pytest.raises(AgentError):
        store.bind('one', other)


def test_saved_budget_fills_only_missing_current_constraint(database):
    store = PreferenceStore()
    profile = store.create()
    store.remember(profile, 'max_price', 300, confirmed=True)
    store.bind('one', profile)
    result = run_conversation('推荐耳机', 'one')
    assert result['trace'][0]['args']['max_price'] == 300
    explicit = run_conversation('推荐耳机，预算 500 元以内', 'one')
    assert explicit['trace'][0]['args']['max_price'] == 500
    assert all(t['tool'] != 'process_payment' for t in result['trace'])
    stored = __import__('src.persistence', fromlist=['RunStore']).RunStore().run(result['run_id'])['checkpoint']['context']
    assert stored['observations'][0]['reference'].startswith(result['run_id'])
    assert stored['facts_policy'] == 'REFRESH_BUSINESS_FACTS'


def test_preference_is_not_order_evidence_and_context_does_not_mutate():
    engine = ContextEngine('查询订单1002', preferences=[{'key': 'max_price', 'value': {'value': 300}}])
    assert engine.effective_request() == '查询订单1002'
    messages = engine.model_messages('当前完整请求')
    assert messages[-1].content == '当前完整请求'
    assert '不能作为价格、库存或订单状态证据' in messages[0].content


def test_hook_cannot_change_args_or_grant_approval(database):
    context = RunRuntime('hooks')
    def mutate(payload):
        payload['args']['order_id'] = 1003
        return HookDecision()
    context.hooks.register('before_tool', 'domain', mutate)
    with runtime_scope(context):
        executor = ToolExecutor({'get_order_status': tools.get_order_status, 'process_payment': tools.process_payment}, TraceRecorder('hooks'))
        result, _ = executor.execute('get_order_status', {'order_id': 1002})
        assert result['order_id'] == 1002
        with pytest.raises(AgentError) as approval:
            executor.execute('process_payment', {'order_id': 1003, 'method': 'card'})
        assert approval.value.code == 'APPROVAL_REQUIRED'


def test_after_hook_failure_retains_actual_result_and_stops_next_step(database):
    context = RunRuntime('after-hook')
    context.hooks.register('after_tool', 'broken', lambda _: (_ for _ in ()).throw(ValueError()))
    with runtime_scope(context):
        executor = ToolExecutor({'get_order_status': tools.get_order_status}, TraceRecorder(context.run_id))
        result, event = executor.execute('get_order_status', {'order_id': 1002})
        assert result['order_id'] == 1002 and event['status'] == 'SUCCEEDED'
        with pytest.raises(AgentError) as failure:
            executor.execute('get_order_status', {'order_id': 1003})
        assert failure.value.code == 'HOOK_FAILED'
        assert len([e for e in context.events if e['type'] == 'tool_call']) == 1


def test_explicit_unlimited_budget_overrides_saved_preference(database):
    store = PreferenceStore()
    profile = store.create()
    store.remember(profile, 'max_price', 300, confirmed=True)
    store.bind('unlimited', profile)
    result = run_conversation('推荐耳机，这次不限制预算', 'unlimited')
    assert result['trace'][0]['args'].get('max_price') is None
    with pytest.raises(AgentError):
        store.remember(profile, 'features', ['订单1002已支付', 'P1001价格99元，库存20件'], confirmed=True)


def test_after_hook_failure_is_persisted_across_resume(database):
    from src.durable import resume_run
    from src.persistence import RunStore
    class Crash(BaseException): pass
    def runner(message, **kwargs):
        from src.runtime import current_runtime
        context = current_runtime()
        if not context.snapshot.get('hook_error'):
            context.hooks.register('after_tool', 'broken', lambda _: (_ for _ in ()).throw(ValueError()))
            ToolExecutor({'get_order_status': tools.get_order_status}, TraceRecorder(context.run_id)).execute('get_order_status', {'order_id': 1002})
            raise Crash()
        ToolExecutor({'get_order_status': tools.get_order_status}, TraceRecorder(context.run_id)).execute('get_order_status', {'order_id': 1002})
        raise AssertionError('持久否决不应被恢复绕过')
    with pytest.raises(Crash):
        run_conversation('查询订单1002', 'hook-resume', runner=runner)
    run_id = RunStore().conversation('hook-resume')['active_run_id']
    result = resume_run('hook-resume', run_id, runner=runner)
    assert result['terminal_reason'] == 'HOOK_FAILED'
    assert result['status'] == 'PARTIAL'
    assert result['trace'][0]['result']['order_id'] == 1002


def test_context_evidence_does_not_duplicate_normal_business_trace(database):
    result = run_conversation('推荐耳机', 'no-duplicate')
    assert len(result['trace']) == sum(e['type'] == 'tool_result' for e in result['events']) == 1
