"""仅评测入口的有界故障注入；不睡眠、不改变生产执行器或真实副作用。"""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch

from src.errors import AgentError


@contextmanager
def recovery_fault(kind, state, *, remaining_faults=None):
    import src.adaptive as adaptive
    import src.direct_executor as direct
    from src.runtime import ToolSpec, tool_spec, current_runtime
    remaining = 4 if kind == 'repeated_tool_failure' else 2 if kind == 'tool_timeout' else 1
    if remaining_faults is not None:
        if type(remaining_faults) is not int or not 0 <= remaining_faults <= remaining:
            raise ValueError('剩余故障次数无效')
        remaining = remaining_faults
    state['remaining_faults'] = remaining
    state['faults'] = []
    def inject(code, tool):
        nonlocal remaining
        remaining -= 1
        state['remaining_faults'] = remaining
        state['fired'] = True
        state['faults'].append(dict(code=code, tool=tool))
        runtime = current_runtime()
        if runtime:
            runtime.record('evaluation_fault_injected', code=code, tool=tool, kind=kind)
        raise AgentError(code, '评测环境注入的临时故障', retryable=code in {'TOOL_TIMEOUT','TOOL_ERROR'})
    with ExitStack() as stack:
        if kind == 'invalid_arguments':
            original = adaptive._validate_tool
            def validate(decision, *args, **kwargs):
                if remaining:
                    inject('INVALID_TOOL_ARGUMENTS', decision.tool)
                return original(decision, *args, **kwargs)
            stack.enter_context(patch.object(adaptive, '_validate_tool', validate))
        else:
            for name in ('search_products', 'discover_products'):
                spec = tool_spec(name, direct._READ_ONLY_TOOLS[name])
                def wrapped(_fn=spec.function, _name=name, **kwargs):
                    if remaining:
                        inject('TOOL_TIMEOUT' if kind == 'tool_timeout' else 'TOOL_ERROR', _name)
                    return _fn(**kwargs)
                stack.enter_context(patch.dict(direct._READ_ONLY_TOOLS, {name:ToolSpec(wrapped, spec.schema)}))
        yield
