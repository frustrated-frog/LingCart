import pytest
from src import tools
from src.errors import AgentError
from src.execution_boundaries import apply_dialogue_boundaries, require_allowed
from src.runtime import BudgetLimits, RunRuntime, tool_spec


@pytest.mark.parametrize('name', ['initiate_refund', 'submit_after_sales_case'])
def test_refund_boundary_covers_production_composite_tools(name):
    context = RunRuntime('effects', BudgetLimits())
    apply_dialogue_boundaries(context, [dict(restriction='NO_REFUND', source_quote='不要退款')], '不要退款')
    with pytest.raises(AgentError, match='禁止此操作'):
        require_allowed(context, name, tool_spec(name, getattr(tools, name)))


def test_no_refund_allows_local_draft():
    context = RunRuntime('draft', BudgetLimits())
    apply_dialogue_boundaries(context, [dict(restriction='NO_REFUND', source_quote='不要退款')], '不要退款')
    require_allowed(context, 'prepare_after_sales_case', tool_spec('prepare_after_sales_case', tools.prepare_after_sales_case))


def test_declared_composite_effect_blocks_approval_and_replay_before_preflight():
    from src.runtime import BusinessEffect, ToolSpec, ToolExecutor, TraceRecorder, runtime_scope
    calls = []
    def composite(order_id: int):
        calls.append(order_id)
        return {'success': True}
    context = RunRuntime('composite', BudgetLimits())
    context.approval = {'tool': 'renamed_composite', 'args': {'order_id': 1001}}
    with runtime_scope(context):
        executor = ToolExecutor({'renamed_composite': ToolSpec(composite, side_effect=True,
            business_effects=(BusinessEffect.REFUND,), preflight=lambda args, ctx: calls.append('preflight'))}, TraceRecorder(context.run_id))
        executor.execute('renamed_composite', {'order_id': 1001})
        apply_dialogue_boundaries(context, [dict(restriction='NO_REFUND', source_quote='不要退款')], '不要退款')
        with pytest.raises(AgentError, match='禁止此操作'):
            executor.execute('renamed_composite', {'order_id': 1001})
    assert calls == ['preflight', 1001]


def test_read_tool_cannot_declare_write_effect():
    from src.runtime import ToolSpec, BusinessEffect
    with pytest.raises(AgentError):
        ToolSpec(lambda: None, business_effects=(BusinessEffect.REFUND,))
