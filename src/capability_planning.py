"""CGAP 工程适配：声明能力、投影真实观察、滚动求解并复用原执行边界。"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

from src.capability_planner import (CapabilityContract, CapabilityPlanNode, CapabilityPlanView,
                                    CapabilityPlanningProjection, PlanningAction, search_plan)
from src.errors import AgentError
from src.harness.mechanisms import EmbeddedMechanism


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def built_in_contracts():
    """工具语义声明，不包含按问句或业务场景编排的行动序列。"""
    return (
        CapabilityContract(tool='get_order_status', scope='order', automatic=True,
            provides={'order.status': ('order_id', 'status')}),
        CapabilityContract(tool='get_order_details', scope='order', automatic=True,
            provides={'order.status': ('order_id', 'status'), 'order.details': ('order_id', 'item', 'amount')}),
        CapabilityContract(tool='track_shipment', scope='order', automatic=True,
            provides={'order.status': ('order_id', 'status'), 'order.shipment': ('order_id', 'tracking_message')}),
        CapabilityContract(tool='check_refund_eligibility', scope='order', automatic=True,
            requires=('order.status',), provides={'order.refund_check': ('order_id', 'eligible', 'reason')}),
        CapabilityContract(tool='get_payment_status', scope='order', automatic=True,
            provides={'order.payment': ('order_id', 'status', 'amount', 'method')}),
        CapabilityContract(tool='answer_faq', scope='question', automatic=True,
            provides={'policy.answer': ('answer', 'status')}),
        CapabilityContract(tool='search_products', scope='catalog', provides={'catalog.candidates': ('products',)}),
        CapabilityContract(tool='discover_products', scope='catalog', provides={'catalog.candidates': ('products',)}),
        CapabilityContract(tool='compare_products', scope='catalog', requires=('catalog.candidates',),
            provides={'catalog.comparison': ('products',)}),
    )


TARGETS = {'get_order_status': 'order.status', 'get_order_details': 'order.details',
           'track_shipment': 'order.shipment', 'check_refund_eligibility': 'order.refund_check',
           'get_payment_status': 'order.payment', 'answer_faq': 'policy.answer',
           'compare_products': 'catalog.comparison', 'recommend': 'catalog.candidates'}


def scope_for(contract, args):
    if contract.scope == 'order':
        return f"order:{args['order_id']}" if type(args.get('order_id')) is int else None
    if contract.scope == 'question':
        return 'question:' + digest(args['question'])[:16] if args.get('question') else None
    return 'catalog'


def fact(scope, name):
    return f'{scope}/{name}'


def project_capability_planning(events, checkpoint):
    """只投影已持久化的协议，不在 UI 侧重建规划或成功率。"""
    from pydantic import ValidationError
    adaptive = checkpoint.get('adaptive') or {}
    records = [e for e in events if e.get('type') == 'capability_plan_updated']
    if adaptive.get('capability_plan'):
        records.append(adaptive['capability_plan'])
    history, invalid = {}, 0
    for item in records:
        try:
            view = CapabilityPlanView.model_validate({k: v for k, v in item.items()
                                                      if k in CapabilityPlanView.model_fields})
        except (ValidationError, TypeError):
            invalid += 1
            continue
        history[view.plan_id] = view
    ordered = sorted(history.values(), key=lambda v: v.revision)
    current_revision = checkpoint.get('goal_revision', adaptive.get('goal_revision'))
    return CapabilityPlanningProjection(status='PARTIAL' if invalid else 'READY' if ordered else 'EMPTY',
        history=ordered, invalid_records=invalid,
        stale=bool(ordered and current_revision is not None and ordered[-1].goal_revision != current_revision)
    ).model_dump(mode='json')


def compile_problem(state, plan, contracts, tools):
    """从已接受目标与已核验观察建立问题；每次调用均重建，不信任上轮预测。"""
    from src.runtime import ToolEffect, tool_spec
    from src.task_plan import verify_observation

    available = {c.tool: c for c in contracts if c.tool in tools
                 and tool_spec(c.tool, tools[c.tool]).effect == ToolEffect.READ}
    catalog_sha = digest([dict(c.model_dump(mode='json'),
                        input_schema=tool_spec(c.tool, tools[c.tool]).schema.model_json_schema())
                        for c in sorted(available.values(), key=lambda c: c.tool)])
    facts, failed = {}, set()
    floor = state.get('capability_step_floor', 0)
    for step in plan.steps[floor:]:
        contract = available.get(step.tool)
        if contract is None:
            continue
        if step.status in {'FAILED', 'REJECTED'}:
            failed.add((step.tool, digest(step.args)))
            continue
        if step.status not in {'SUCCEEDED', 'REUSED'} or not isinstance(step.result, dict):
            continue
        try:
            verify_observation(step)
        except AgentError:
            continue
        scope = scope_for(contract, step.args)
        if not scope:
            continue
        if contract.scope == 'order' and step.result.get('order_id') != step.args['order_id']:
            continue
        if step.tool == 'answer_faq' and step.result.get('status') != 'ANSWERED':
            continue
        for name, fields in contract.provides.items():
            if all(key in step.result and step.result[key] is not None
                   and step.result[key] != [] and step.result[key] != '' for key in fields):
                facts.setdefault(fact(scope, name), []).append(step.step_id)
        if step.tool == 'compare_products' and step.result.get('products'):
            ids = sorted(p['product_id'] for p in step.result['products'])
            facts.setdefault(fact('catalog', 'comparison:' + digest(ids)[:16]), []).append(step.step_id)

    goals, delegated, action_map = [], [], {}
    for objective in (state.get('goal') or {}).get('objectives', []):
        kind, oid = objective['kind'], objective['id']
        if kind not in TARGETS:
            delegated.append(oid)
            continue
        scope = ('order:' + str(objective['order_id']) if objective.get('order_id') is not None else
                 'question:' + digest(objective['question'])[:16] if objective.get('question') else 'catalog')
        target = TARGETS[kind]
        if kind == 'compare_products' and objective.get('product_ids'):
            target = 'comparison:' + digest(sorted(objective['product_ids']))[:16]
        goals.append(fact(scope, target))
        if kind == 'recommend':
            delegated.append(oid)
        for contract in available.values():
            if contract.scope != scope.split(':')[0]:
                continue
            schema = tool_spec(contract.tool, tools[contract.tool]).schema
            args = {key: deepcopy(objective[key]) for key in schema.model_fields
                    if objective.get(key) not in (None, [])}
            automatic = contract.automatic
            try:
                schema.model_validate(args, strict=True)
            except ValueError:
                automatic = False
            if (contract.tool, digest(args)) in failed:
                continue
            adds = [fact(scope, name) for name in contract.provides]
            if kind == 'compare_products' and contract.tool == 'compare_products':
                adds.append(fact(scope, target))
            identity = digest([contract.tool, scope, args])[:20]
            previous = action_map.get(identity)
            action_map[identity] = PlanningAction(action_id=identity, tool=contract.tool, scope=scope, args=args,
                requires=tuple(fact(scope, name) for name in contract.requires), adds=tuple(sorted(set(adds))),
                objective_ids=tuple(sorted({oid, *(previous.objective_ids if previous else ())})),
                automatic=automatic)
    return tuple(action_map.values()), facts, tuple(sorted(set(goals))), delegated, catalog_sha


@dataclass(frozen=True)
class CapabilityPlanningMechanism(EmbeddedMechanism):
    slot: str = 'planning'
    mechanism_id: str = 'capability-grounded-planning'
    mechanism_version: str = '1'
    compatible_engines: tuple[str, ...] = ('adaptive',)
    source_files: tuple[str, ...] = ('src/capability_planner.py', 'src/capability_planning.py',
        'src/adaptive.py', 'src/adaptive_decision.py', 'src/capability_registry.py', 'src/context_engine.py',
        'src/task_plan.py', 'src/tools.py', 'src/runtime.py')

    def problem(self, state, plan):
        from src.capability_registry import get_capability_registry
        from src.direct_executor import _READ_ONLY_TOOLS
        return compile_problem(state, plan, get_capability_registry().planning_contracts(), _READ_ONLY_TOOLS)

    def refresh(self, context, state, plan):
        if not state.get('goal'):
            return None
        actions, facts, goals, delegated, catalog_sha = self.problem(state, plan)
        remaining = max(0, min(context.limits.max_steps - len(plan.steps),
                               context.limits.max_tool_calls - context.counters['tool_calls']))
        result = search_plan(actions, facts, goals, max_steps=remaining)
        missing = list(result.missing)
        # 一个不可达目标不能阻止其它独立目标取得进展。
        if result.status == 'UNREACHABLE':
            reachable = [g for g in goals if g not in result.missing]
            if reachable:
                result = search_plan(actions, facts, reachable, max_steps=remaining)
        nodes, sources = [], {key: list(refs) for key, refs in facts.items()}
        for action in result.actions:
            deps = sorted({ref for key in action.requires for ref in sources.get(key, [])})
            nodes.append(CapabilityPlanNode(action_id=action.action_id, tool=action.tool, scope=action.scope,
                objective_ids=list(action.objective_ids), requires=list(action.requires), provides=list(action.adds),
                depends_on=[ref for ref in deps if ref in {n.action_id for n in nodes}],
                evidence_refs=[ref for ref in deps if ref.startswith('step-')],
                status=('PLANNED' if not set(action.requires) <= facts.keys() else
                        'READY' if action.automatic else 'NEEDS_ARGUMENTS'), automatic=action.automatic))
            for key in action.adds:
                sources[key] = [action.action_id]
        status = ('SEARCH_LIMIT' if result.status == 'SEARCH_LIMIT' else 'PARTIAL' if missing else
                  'UNREACHABLE' if result.status == 'UNREACHABLE' else
                  'READY' if any(n.status == 'READY' for n in nodes) else
                  'NEEDS_MODEL' if nodes or delegated else 'OBSERVATIONS_READY')
        previous = state.get('capability_plan') or {}
        identity = digest(dict(goal=state['goal'], goal_revision=state['goal_revision'],
            catalog=catalog_sha, facts=facts, nodes=[n.model_dump() for n in nodes], missing=missing,
            status=status, steps=[(s.step_id, s.status, s.error) for s in plan.steps]))
        if previous.get('plan_id') == identity:
            return previous
        reason = ('目标已更新，重新建立依赖' if previous and previous['goal_revision'] != state['goal_revision'] else
                  '根据实际观察重算剩余依赖' if previous else '根据目标与工具能力建立初始计划')
        view = CapabilityPlanView(plan_id=identity, catalog_sha256=catalog_sha,
            goal_revision=state['goal_revision'], revision=previous.get('revision', 0) + 1,
            previous_plan_id=previous.get('plan_id'), status=status, reason=reason, nodes=nodes,
            facts=facts, missing=missing, delegated_objectives=delegated, expanded=result.expanded)
        view = view.model_copy(update={'objective_labels': {
            o['id']: o.get('description', o['id']) for o in state['goal']['objectives']}})
        state['capability_plan'] = view.model_dump(mode='json')
        context.record('capability_plan_updated', **state['capability_plan'])
        return state['capability_plan']

    def next_decision(self, context, state, plan):
        from src.adaptive_decision import Decision
        view = self.refresh(context, state, plan)
        surface = state.get('decision_surface') or {}
        if surface.get('allowed_actions') and 'tool' not in surface['allowed_actions']:
            return None
        ready = next((n for n in (view or {}).get('nodes', []) if n['status'] == 'READY'), None)
        if ready is None:
            return None
        actions, _, _, _, _ = self.problem(state, plan)
        action = next(a for a in actions if a.action_id == ready['action_id'])
        return Decision(action='tool', tool=action.tool, args=deepcopy(action.args),
                        reason='能力规划器选择已具备输入的只读步骤')

    def bind_step(self, state, plan, step):
        from src.capability_registry import get_capability_registry
        contracts = {c.tool: c for c in get_capability_registry().planning_contracts()}
        contract = contracts.get(step.tool)
        if contract is None:
            return
        _, facts, _, _, _ = self.problem(state, plan)
        scope = scope_for(contract, step.args)
        if scope is None:
            raise AgentError('CAPABILITY_INPUT_MISSING', '工具缺少可追溯的对象输入', retryable=False)
        required = [fact(scope, key) for key in contract.requires]
        missing = [key for key in required if key not in facts]
        if missing:
            raise AgentError('CAPABILITY_PRECONDITION_MISSING', '先取得前置观察：' + '、'.join(missing), retryable=False)
        step.depends_on = sorted({ref for key in required for ref in facts[key]})
        step.capability_id = f'cgap:{step.tool}'
        for key in ('order_id', 'question'):
            if key not in step.args:
                continue
            ids = [o['id'] for o in state['goal']['objectives'] if o.get(key) == step.args[key]]
            if not ids:
                raise AgentError('CAPABILITY_INPUT_UNGROUNDED', '规划输入不属于已确认目标', retryable=False)
            step.argument_provenance[key] = dict(source='accepted_goal', objective_ids=ids,
                                                 goal_revision=state['goal_revision'])
            step.objective_ids = ids
