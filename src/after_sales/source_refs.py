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


INSTRUCTION = '''Output source_ref instead of source_quote everywhere. Choose an exact ID from human_sources;
the host restores the original HUMAN text. Never copy/rewrite a quote, invent offsets or cite prior runs.
One source may support multiple independently scoped facts. Preserved prior requirements keep their stored sources.
Only extract NEW facts/changes asserted in the CURRENT message; do not backfill old uncommitted material from history.
Material status, goal mode and authorization are separate interpretations of the same source, not implied by its ID.
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
            props.pop('source_quote')
            props['source_ref'] = {'type': 'string'}
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
            source = sources.get(value['source_ref']) if isinstance(value['source_ref'], str) else None
            if source is None:
                raise AgentError('SOURCE_REFERENCE_UNKNOWN', 'source_ref 不属于本轮 HUMAN 原文目录', retryable=False)
            result['source_quote'] = source['text']
            bindings.append(dict(path=path, source=source))
        return result
    decoded = visit(payload, [])
    context.record('request_sources_resolved', purpose=purpose, status='RESOLVED', bindings=bindings)
    return decoded
