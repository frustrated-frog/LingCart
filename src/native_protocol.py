"""Native typed functions for the Agent; no natural-language intent routing."""
from copy import deepcopy
import json
import re


def human_quotes(request, history, limit):
    from src.agents.base import as_text
    sources = [request, *(as_text(m.content) for m in history or [] if getattr(m, 'type', None) == 'human')]
    quotes = {request[:limit]}
    for source in sources:
        if len(source) <= limit:
            quotes.add(source)
        for part in re.findall(r'[^。！？；;，,\n]+[。！？；;，,\n]?', source):
            part = part.strip()
            if 0 < len(part) <= limit:
                quotes.add(part)
                quotes.add(part.rstrip('。！？；;，,'))
    return sorted(q for q in quotes if q)


def strict_schema(schema):
    """Adapt Pydantic schemas to DeepSeek's documented strict JSON Schema subset."""
    definitions = schema.get('$defs', {})
    def visit(value):
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        if '$ref' in value:
            return visit(definitions[value['$ref'].rsplit('/', 1)[-1]])
        result = {k: visit(v) for k, v in value.items()
                  if k not in {'$defs', 'title', 'default', 'minLength', 'maxLength', 'minItems', 'maxItems'}}
        if result.get('type') == 'object':
            result['additionalProperties'] = False
            result['required'] = list(result.get('properties', {}))
        return result
    return visit(deepcopy(schema))


def function(name, description, schema):
    return {'type': 'function', 'function': {'name': name, 'description': description,
            'strict': True, 'parameters': strict_schema(schema)}}


def one_call(response, allowed):
    calls = getattr(response, 'tool_calls', [])
    if len(calls) != 1 or calls[0].get('name') not in allowed:
        raise ValueError('Return exactly one declared function call')
    return calls[0]['name'], calls[0].get('args') or {}, calls[0].get('id')


def size(declarations):
    return len(json.dumps(declarations, ensure_ascii=False, separators=(',', ':')))


def decision_action_surface(state, plan, *, remaining_model_calls):
    """Reserve the terminal audit when a single recommendation can now commit.

    This is a mechanical eligibility test, not a ranking policy.  It only
    narrows actions after registry bindings passed their independent audit and
    the planner will receive complete records for at least one valid candidate.
    """
    full = {'mode': 'full', 'allowed_actions': None, 'remaining_model_calls': remaining_model_calls,
            'reserved_model_calls': 0, 'reason': 'GENERAL_DECISION'}
    transaction = state.get('binding_tx') or {}
    objectives = (state.get('goal') or {}).get('objectives', [])
    if (remaining_model_calls != 2 or transaction.get('source') != 'catalog_registry'
            or transaction.get('status') != 'APPLIED' or len(objectives) != 1
            or objectives[0].get('kind') != 'recommend'):
        return full
    try:
        from src.constraint_bindings import constraint_refs, effective_goal, require_bindings_approved
        from src.adaptive_decision import Constraint, predicate_holds
        from src.errors import AgentError
        require_bindings_approved(state)
        goal = effective_goal(state)
        objective = goal['objectives'][0]
        applicable = [Constraint.model_validate(raw) for ref, raw in constraint_refs(goal).items()
                      if ref.startswith('global:') or ref.startswith(f"objective:{objective['id']}:")]
        if any(constraint.field is None for constraint in applicable):
            return full
    except (AgentError, ValueError, KeyError, TypeError):
        return full

    access = state.get('evidence_access', {})
    candidates = {}
    selected_products = {
        product_id
        for row in state.get('objective_ledger', [])
        if row.get('objective_id') == objective['id']
        for product_id in row.get('selected_product_ids', [])
    }
    for step in plan.steps:
        if (step.tool not in {'search_products', 'discover_products', 'compare_products'}
                or step.status not in {'SUCCEEDED', 'REUSED'} or not isinstance(step.result, dict)):
            continue
        readable = set(access.get(step.step_id, []))
        for product in step.result.get('products', []):
            if (product.get('product_id') in selected_products or product.get('product_id') not in readable
                    or type(product.get('stock')) is not int
                    or product['stock'] <= 0 or not all(predicate_holds(product, item) for item in applicable)):
                continue
            candidates.setdefault(product['product_id'], []).append(step.step_id)
    if not candidates:
        return full
    return {'mode': 'recommendation_commit',
            'allowed_actions': ['recommend', 'clarify', 'finish'],
            'remaining_model_calls': remaining_model_calls, 'reserved_model_calls': 1,
            'reason': 'VALID_CANDIDATE_AND_TERMINAL_AUDIT_RESERVATION',
            'objective_id': objectives[0]['id'],
            'candidates': [{'product_id': identity, 'evidence_refs': list(dict.fromkeys(refs))}
                           for identity, refs in sorted(candidates.items())]}


def decision_functions(state, plan, history=None, *, surface=None):
    from src.adaptive_decision import Decision, IntentDeclaration, ConstraintBinding
    from src.direct_executor import _READ_ONLY_TOOLS
    from src.runtime import tool_spec
    from src.constraint_bindings import constraint_refs
    if state['goal'] is None:
        schema = strict_schema(IntentDeclaration.model_json_schema())
        from src.catalog_concepts import get_catalog_concept_registry
        get_catalog_concept_registry().constrain_schema(schema)
        quotes = human_quotes(state['request'], history, 300)
        def ground(value):
            if isinstance(value, dict):
                if 'source_quote' in value.get('properties', {}):
                    value['properties']['source_quote']['enum'] = quotes
                for item in value.values():
                    ground(item)
            elif isinstance(value, list):
                for item in value:
                    ground(item)
        ground(schema)
        return [function('declare_intent', '完整声明用户目标或提出必要澄清；此函数不执行业务动作。', schema)]
    declarations = []
    allowed_actions = None if not surface else surface.get('allowed_actions')
    allowed_actions = None if allowed_actions is None else set(allowed_actions)
    products = [p for s in plan.steps if isinstance(s.result, dict) for p in s.result.get('products', [])]
    product_ids = sorted({p['product_id'] for p in products})
    if surface and surface.get('mode') == 'recommendation_commit':
        product_ids = [row['product_id'] for row in surface['candidates']]
    for name, item in _READ_ONLY_TOOLS.items():
        if allowed_actions is not None and name not in allowed_actions:
            continue
        if name == 'compare_products' and len(product_ids) < 2:
            continue
        schema = deepcopy(tool_spec(name, item).schema.model_json_schema())
        # Adaptive shopping represents exclusions as audited Constraint rows.
        # Keep the lower-level semantic-router argument out of this native
        # protocol so there is one polarity authority in the shopping loop.
        if name == 'search_products':
            schema['properties'].pop('excluded_terms', None)
            if 'required' in schema:
                schema['required'] = [field for field in schema['required'] if field != 'excluded_terms']
        if name == 'compare_products':
            schema['properties']['product_ids']['items']['enum'] = product_ids
        schema['properties']['reason'] = {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'description': '可选的调用说明；无说明用null。'}
        description = f'调用只读业务工具 {name}；参数按声明提供。'
        if name == 'compare_products':
            description += '只对比2到3个已观察且不同的商品编号。'
        if name == 'search_products':
            description += 'query按空白分词做字面包含匹配；可用宽泛query与category发现真实规格。limit为1到60。'
            schema['properties']['limit'].update(minimum=1, maximum=60)
        if name == 'answer_faq':
            description += '完成FAQ目标时question直接使用目标里的原始question；探索性查询可自行提问。'
        declarations.append(function(name, description, schema))
    properties = strict_schema(Decision.model_json_schema())['properties']
    def prop(name, nonnull=False):
        value = deepcopy(properties[name])
        if nonnull and 'anyOf' in value:
            value = next(v for v in value['anyOf'] if v.get('type') != 'null')
        return value
    def control(name, description, fields):
        declarations.append(function(name, description, {'type': 'object', 'properties': fields}))
    if allowed_actions is None or 'clarify' in allowed_actions:
        control('clarify', '仅当用户意图有实质歧义时提问；沿用未被修改的上下文条件。', {'message': prop('message', True)})
    if allowed_actions is None or 'finish' in allowed_actions:
        control('finish', '如实说明无法完成；不会将任务标为成功。', {'message': prop('message', True)})
    if allowed_actions is None or 'revise_plan' in allowed_actions:
        control('revise_plan', '更新建议计划或工作笔记；不生成或执行业务步骤。',
                {'next_steps': prop('next_steps', True), 'working_memory': prop('working_memory')})
    ids = [s.step_id for s in plan.steps]
    if ids and (allowed_actions is None or {'inspect', 'complete'} & allowed_actions):
        reference = prop('evidence_ref', True)
        reference['enum'] = ids
        if allowed_actions is None or 'inspect' in allowed_actions:
            control('inspect', '读取归档的完整原始记录；历史读取不刷新价格库存。',
                    {'evidence_ref': reference, 'offset': prop('offset'), 'limit': prop('limit')})
        evidence = {o['id']: {'type': 'array', 'items': {'type': 'string', 'enum': ids}}
                    for o in state['goal']['objectives']}
        if allowed_actions is None or 'complete' in allowed_actions:
            control('complete', '提交所有目标的完成证据。推荐需引用对应select_product；可附同商品的已读支持证据。',
                    {'evidence': {'type': 'object', 'properties': evidence}})
    if product_ids and (allowed_actions is None or {'recommend', 'refine_constraints'} & allowed_actions):
        product = prop('product_id', True)
        product['enum'] = product_ids
        objective = prop('objective_id', True)
        objective['enum'] = [o['id'] for o in state['goal']['objectives'] if o['kind'] == 'recommend']
        refs = prop('evidence_refs')
        permitted_refs = ids
        if surface and surface.get('mode') == 'recommendation_commit':
            permitted_refs = sorted({ref for row in surface['candidates'] for ref in row['evidence_refs']})
        refs['items']['enum'] = permitted_refs
        if objective['enum'] and (allowed_actions is None or 'recommend' in allowed_actions):
            control('recommend', '为目标选择一个尚未选过的已读商品并解释依据；运行时重新核验。达到目标quantity前继续选择。',
                    {'product_id': product, 'objective_id': objective, 'evidence_refs': refs, 'reason': prop('reason')})
        conditions = constraint_refs(state['goal'])
        if conditions and (allowed_actions is None or 'refine_constraints' in allowed_actions):
            schema = strict_schema(ConstraintBinding.model_json_schema())
            schema['properties']['constraint_ref']['enum'] = list(conditions)
            fields = {key for p in products for key in p if key != 'specs'}
            fields.update('specs.' + key for p in products for key in p.get('specs', {}))
            permitted = {'product_id', 'name', 'category', 'price', 'stock', 'summary'}
            fields = sorted(f for f in fields if f in permitted or f.startswith('specs.'))
            schema['properties']['field']['enum'] = fields
            schema['properties']['evidence_refs']['items']['enum'] = ids
            control('refine_constraints', '依据已读字段修正现有条件的执行编码，保留原意；批准后旧核验需重做。',
                    {'bindings': {'type': 'array', 'items': schema}})
    return declarations


def decode_decision(response, state):
    from src.adaptive_decision import Decision, IntentDeclaration, decision_from_intent
    from src.direct_executor import _READ_ONLY_TOOLS
    if state['goal'] is None and not getattr(response, 'tool_calls', []):
        from src.agents.base import as_text
        intent = IntentDeclaration.model_validate(json.loads(as_text(response.content)))
        decision = decision_from_intent(intent)
        return decision, dict(function='declare_intent', submission_id=None, transport='validated_json_fallback')
    allowed = set(_READ_ONLY_TOOLS) | {'declare_intent', 'clarify', 'finish', 'revise_plan', 'inspect', 'complete', 'recommend', 'refine_constraints'}
    name, args, call_id = one_call(response, allowed)
    data = deepcopy(args)
    if name == 'declare_intent':
        if state['goal'] is not None:
            raise ValueError('The goal is already declared')
        intent = IntentDeclaration.model_validate(data)
        decision = decision_from_intent(intent)
    elif name in _READ_ONLY_TOOLS:
        reason = data.pop('reason', None) or ''
        decision = Decision(action='tool', tool=name, args=data, reason=reason)
    elif name == 'complete':
        evidence = data['evidence']
        decision = Decision(action='complete', claims=[{'objective_id': k, 'evidence_refs': v} for k, v in evidence.items()])
    else:
        decision = Decision.model_validate({'action': name, **data})
    return decision, dict(function=name, submission_id=call_id, transport='native_function')


def decode_decisions(response, state):
    """原生模型可一次声明独立读操作/选品；终态与修改映射仍单独提交。"""
    from types import SimpleNamespace
    from src.direct_executor import _READ_ONLY_TOOLS
    calls = getattr(response, 'tool_calls', [])
    if len(calls) <= 1:
        return [decode_decision(response, state)]
    if state['goal'] is None or any(c.get('name') not in set(_READ_ONLY_TOOLS) | {'recommend'} for c in calls):
        raise ValueError('Batch only independent readonly queries/recommendations; submit other controls individually')
    batch = [decode_decision(SimpleNamespace(tool_calls=[call]), state) for call in calls]
    recommendations = [d for d, _ in batch if d.action == 'recommend']
    objectives = {row['id']: row for row in (state.get('goal') or {}).get('objectives', [])}
    ledger = {row['objective_id']: row for row in state.get('objective_ledger', [])}
    for objective_id in {decision.objective_id for decision in recommendations}:
        decisions = [decision for decision in recommendations if decision.objective_id == objective_id]
        product_ids = [decision.product_id for decision in decisions]
        existing = set((ledger.get(objective_id) or {}).get('selected_product_ids', []))
        quantity = int((objectives.get(objective_id) or {}).get('quantity', 1))
        if len(product_ids) != len(set(product_ids)) or existing.intersection(product_ids):
            raise ValueError('Recommendations for one objective must select distinct products')
        if len(product_ids) > quantity - len(existing):
            raise ValueError('Recommendation batch exceeds the objective quantity')
    return batch


def audit_function(schema, state, history):
    from src.constraint_bindings import constraint_refs
    from src.agents.base import as_text
    result = strict_schema(schema)
    row = result['properties']['requirements']['items']['properties']
    quotes = set(human_quotes(state['request'], history, 600))
    quotes.update(o['source_quote'] for o in state['goal']['objectives'])
    quotes.update(c['source_quote'] for c in constraint_refs(state['goal']).values())
    row['source_quote']['enum'] = sorted(quotes)
    row['objective_ids']['items']['enum'] = [o['id'] for o in state['goal']['objectives']]
    refs = list(constraint_refs(state['goal']))
    if refs:
        row['constraint_refs']['items']['enum'] = refs
    return function('submit_audit', '一次提交本阶段所有要求的审查，不要逐条分函数调用；引用须选人类原文和已声明编号。', result)
