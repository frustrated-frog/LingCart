"""把模型的来源选择编译为精确 HUMAN 原文；不推断材料或授权。"""
from copy import deepcopy

from src.after_sales.model_planning import source_catalog
from src.errors import AgentError


def enabled(context=None):
    from src.runtime import current_runtime
    context = context or current_runtime()
    return bool(context and getattr(context.mechanisms.get('tool_protocol'), 'source_references', False))


def catalog(context, request):
    return [s for s in source_catalog(context, request) if s['origin'] == 'CURRENT_TURN']


INSTRUCTION = '''Use source_ref IDs from human_sources instead of source_quote everywhere; host restores exact text.
Never invent quotes/offsets. Each ID may support separate scoped facts. PRESERVE keeps prior sources;
ADD/REPLACE assert CURRENT facts only, never backfill uncommitted history. References grant no authorization.
'''


def schema_with_refs(schema):
    def visit(value):
        if isinstance(value, list):
            return [visit(v) for v in value]
        if not isinstance(value, dict):
            return value
        result = {k: visit(v) for k, v in value.items()}
        props = result.get('properties', {})
        if 'source_quote' in props:
            original = props.pop('source_quote')
            nullable = {'type': 'null'} in original.get('anyOf', [])
            props['source_ref'] = ({'anyOf': [{'type': 'string'}, {'type': 'null'}]}
                                   if nullable else {'type': 'string'})
            if 'required' in result:
                result['required'] = ['source_ref' if k == 'source_quote' else k for k in result['required']]
        return result
    return visit(deepcopy(schema))


def decode(context, payload, request, *, purpose):
    sources = {s['source_ref']: s for s in catalog(context, request)}
    bindings = []
    def visit(value, path):
        if isinstance(value, list):
            return [visit(v, [*path, i]) for i, v in enumerate(value)]
        if not isinstance(value, dict):
            return value
        if 'source_quote' in value:
            raise AgentError('SOURCE_PROTOCOL_MIXED', '当前协议只接受 source_ref，不能混入 source_quote', retryable=False)
        result = {k: visit(v, [*path, k]) for k, v in value.items() if k != 'source_ref'}
        if 'source_ref' in value:
            if value['source_ref'] is None:
                # PRESERVE 的可空来源不引用任何原文；必填来源仍由完整 schema 拒绝。
                result['source_quote'] = None
                return result
            source = sources.get(value['source_ref']) if isinstance(value['source_ref'], str) else None
            if source is None:
                raise AgentError('SOURCE_REFERENCE_UNKNOWN', 'source_ref 不属于本轮 HUMAN 原文目录；有效 ID：' + ','.join(sources), retryable=False)
            result['source_quote'] = source['text']
            bindings.append(dict(path=path, source=source))
        return result
    decoded = visit(payload, [])
    context.record('request_sources_resolved', purpose=purpose, status='RESOLVED', bindings=bindings)
    return decoded
