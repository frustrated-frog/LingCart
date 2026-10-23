"""Model-directed shopping loop; code owns safety, evidence and execution budgets."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
import time
import uuid

from pydantic import BaseModel, ConfigDict, Field

from src.adaptive_decision import Decision, ShoppingGoal, decide, goal_payload, predicate_holds, validate_goal
from src.adaptive_goals import check_claims, new_ledger, outcome_report, record_outcome, recommendation_text, verification
from src.direct_executor import _READ_ONLY_TOOLS, _finish
from src.errors import AgentError, classify_exception
from src.intake import (DIRECT, LLM_REQUIRED, PAYMENT_PROCESS, PRODUCT_COMPARE, PRODUCT_DISCOVERY,
                        PRODUCT_SEARCH, REFUND_REQUEST, TaskIntake, analyze_request)
from src.runtime import BudgetLimits, RunRuntime, ToolEffect, ToolExecutor, TraceRecorder, current_runtime, runtime_scope, tool_spec
from src.task_plan import TaskPlan, TaskStep, verify_observation
from src.constraint_bindings import (build_binding_transaction, constraint_refs, effective_goal,
    fingerprint, propose_bindings, record_binding_evidence, require_bindings_approved,
    resolve_catalog_concepts)

MAX_REPLANS = 2
MAX_DECISION_REPAIRS = 2
_SELECT = "select_product"
_DONE = {"SUCCEEDED", "REUSED", "NOT_NEEDED", "REJECTED", "WAITING_APPROVAL", "FAILED"}


class StepDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    step_id: str = Field(min_length=1, max_length=80)
    specialist: str = "shopping"
    tool: str
    args: dict = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class PlanPatch(BaseModel):
    """A small, auditable patch, never arbitrary executable model output."""
    model_config = ConfigDict(extra="forbid", strict=True)
    reason: str = Field(min_length=1, max_length=240)
    replace: list[StepDraft] = Field(default_factory=list, max_length=8)
    append: list[StepDraft] = Field(default_factory=list, max_length=8)
    cancel: list[str] = Field(default_factory=list, max_length=8)


def apply_plan_patch(plan, patch, limits, *, registry=None):
    """Validate atomically, retaining every completed step and its observation."""
    patch = patch if isinstance(patch, PlanPatch) else PlanPatch.model_validate(patch)
    changed = [s.step_id for s in patch.replace] + patch.cancel
    if len(changed) != len(set(changed)):
        raise AgentError("INVALID_PLAN_PATCH", "同一步骤不能被重复修改", retryable=False)
    existing = {s.step_id: s for s in plan.steps}
    if any(identity not in existing or existing[identity].status in _DONE for identity in changed):
        raise AgentError("IMMUTABLE_COMPLETED_STEP", "不能修改已完成步骤或未知步骤", retryable=False)
    replacements = {s.step_id: TaskStep(**s.model_dump()) for s in patch.replace}
    candidate = deepcopy(plan)
    candidate.steps = [replacements.get(s.step_id, s) for s in candidate.steps]
    for step in candidate.steps:
        if step.step_id in patch.cancel:
            step.status = "CANCELLED"
    candidate.steps.extend(TaskStep(**s.model_dump()) for s in patch.append)
    cancelled = {s.step_id for s in candidate.steps if s.status == "CANCELLED"}
    if any(s.status != "CANCELLED" and cancelled.intersection(s.depends_on) for s in candidate.steps):
        raise AgentError("INVALID_PLAN_PATCH", "取消步骤时必须同时处理其未完成依赖", retryable=False)
    allowed = registry if registry is not None else _READ_ONLY_TOOLS
    for step in list(replacements.values()) + candidate.steps[len(plan.steps):]:
        if step.tool != _SELECT:
            if step.tool not in allowed:
                raise AgentError("INVALID_PLAN_PATCH", "重规划不能添加未授权工具或写操作", retryable=False)
            from src.runtime import tool_spec
            if tool_spec(step.tool, allowed[step.tool]).effect != ToolEffect.READ:
                raise AgentError("INVALID_PLAN_PATCH", "重规划只能追加只读业务步骤", retryable=False)
        if step.tool == "compare_products" and step.args.get("product_ids"):
            observed = {p["product_id"] for s in plan.steps if s.tool == "search_products" and s.status in _DONE
                        for p in (s.result or {}).get("products", [])}
            ids = step.args["product_ids"]
            if not 2 <= len(ids) <= 3 or len(set(ids)) != len(ids) or not set(ids) <= observed:
                raise AgentError("UNGROUNDED_PLAN_ENTITY", "对比编号必须来自已获得的搜索证据，最多三款", retryable=False)
    candidate.validate(limits.max_steps, limits.max_depth)
    return candidate


def _verify_products(step):
    verify_observation(step)
    ids = []
    for product in step.result["products"]:
        price, stock = product.get("price"), product.get("stock")
        valid = (isinstance(product.get("product_id"), str) and re.fullmatch(r"P\d{4}S?", product["product_id"])
                 and isinstance(product.get("name"), str) and isinstance(product.get("summary"), str)
                 and isinstance(product.get("category"), str) and isinstance(product.get("specs"), dict)
                 and isinstance(price, (int, float)) and not isinstance(price, bool) and math.isfinite(price) and price >= 0
                 and isinstance(stock, int) and not isinstance(stock, bool) and stock >= 0)
        if not valid:
            raise AgentError("INVALID_TOOL_RESULT", "商品证据缺少有效编号、价格、库存或规格", retryable=False)
        ids.append(product["product_id"])
    if len(ids) != len(set(ids)):
        raise AgentError("INVALID_TOOL_RESULT", "商品证据包含重复编号", retryable=False)


def _restore_plan(payload):
    return TaskPlan(payload["goals"], [TaskStep(**step) for step in payload["steps"]], payload.get("verification", {}))


def _save(context, state, plan):
    state["plan"] = plan.payload()
    context.phase = "PLAN"
    context.checkpoint(adaptive=deepcopy(state), plan=plan.payload())


def _apply_liveplan(context, state, plan):
    """Run one recoverable monitor/advisor transaction at a committed boundary."""

    mechanism = context.mechanisms.get('steering')
    if mechanism is None or not hasattr(mechanism, 'evaluate'):
        return
    assessment, changed = mechanism.evaluate(context, state, plan)
    if changed:
        state['model_cursor'] = context.model_cursor
        _save(context, state, plan)
    if assessment is None:
        return
    mechanism.reserve_advice(context, state, assessment)
    state['model_cursor'] = context.model_cursor
    _save(context, state, plan)
    mechanism.complete_advice(context, state, assessment)
    state['model_cursor'] = context.model_cursor
    _save(context, state, plan)


def _review_liveplan_candidate(context, recorder, state, plan, decision):
    mechanism = context.mechanisms.get('steering')
    if mechanism is None or not hasattr(mechanism, 'review_candidate'):
        return False
    guidance = mechanism.review_candidate(context, state, decision)
    if guidance is None:
        return False
    _discard_native_queue(context, state, 'LIVEPLAN_INTERVENTION')
    state['feedback'] = {
        'code': 'LIVEPLAN_ACTION_REJECTED',
        'message': guidance,
        'instruction': '遵循运行时纠偏建议，选择不同的下一步；不得放宽用户目标、审批与安全边界。',
    }
    state['model_cursor'] = context.model_cursor
    state['pending'] = None
    _save(context, state, plan)
    recorder.record('decision_rejected', code='LIVEPLAN_ACTION_REJECTED', action=decision.action)
    return True


def _apply_binding_transaction(context, recorder, state, plan, history, transaction, *, reject_as_error):
    """Independently audit and atomically apply one content-addressed proposal."""
    from src.request_audit import review_request
    from src.session_control import apply_controls

    transaction = deepcopy(transaction)
    current = state.get('binding_tx') or {}
    if (current.get('transaction_id') == transaction['transaction_id']
            and current.get('status') == 'APPLIED'
            and fingerprint(state.get('constraint_bindings', {})) == transaction['after_bindings_sha256']):
        return True
    if current.get('transaction_id') != transaction['transaction_id'] or current.get('status') != 'PROPOSED':
        state['binding_tx'] = transaction
        state['binding_resolution'] = None
        _save(context, state, plan)
        recorder.record('constraint_binding_proposed', source=transaction['source'],
            transaction_id=transaction['transaction_id'], changed_refs=transaction['changed_refs'],
            registry=(transaction.get('registry') or {}).get('fingerprint'),
            evidence_sha256=transaction['evidence_sha256'])

    proposed_revision = state.get('binding_revision', 0) + 1
    proposed = {**state, 'constraint_bindings': deepcopy(transaction['proposed_bindings']),
                'binding_revision': proposed_revision, 'binding_tx': transaction}
    audit = review_request(context, proposed, plan, history, phase='bindings')
    # A user change invalidates the proposal before it can cross the commit boundary.
    apply_controls(context)
    state['model_cursor'] = context.model_cursor
    if (transaction['goal_revision'] != state['goal_revision']
            or transaction['before_bindings_sha256'] != fingerprint(state.get('constraint_bindings', {}))):
        raise AgentError('BINDING_TRANSACTION_CONFLICT', '映射审查期间目标或现有映射已变化，旧提案不能提交', retryable=False)
    if audit['verdict'] != 'APPROVED':
        transaction.update(status='REJECTED', audit_reference=audit['reference'],
                           audit_sha256=fingerprint(audit))
        state['binding_tx'] = transaction
        state['feedback'] = dict(code='BINDING_AUDIT_REJECTED', audit=audit,
            transaction_id=transaction['transaction_id'],
            message='执行映射未通过原意核对；保留原约束，请补充证据、修正映射或澄清。')
        _save(context, state, plan)
        recorder.record('constraint_binding_rejected', source=transaction['source'],
            transaction_id=transaction['transaction_id'], audit_reference=audit['reference'])
        if reject_as_error:
            raise AgentError('BINDING_AUDIT_REJECTED', '字段映射没有忠实保留用户条件', retryable=False)
        return False

    before = deepcopy(state.get('constraint_bindings', {}))
    transaction.update(status='APPLIED', audit_reference=audit['reference'],
                       audit_sha256=fingerprint(audit), applied_binding_revision=proposed_revision)
    state.update(constraint_bindings=deepcopy(transaction['proposed_bindings']),
                 binding_revision=proposed_revision, binding_audit=audit,
                 binding_tx=transaction, binding_resolution=None, completion_audit=None)
    state.setdefault('binding_sources', {}).update(
        {ref: transaction['source'] for ref in transaction['changed_refs']})
    state.pop('audit_format_feedback', None)
    # 已验证推荐只证明旧执行映射；服务查询证据不受字段映射修订影响。
    recommendation_ids = {o['id'] for o in state['goal']['objectives'] if o['kind'] == 'recommend'}
    for row in state['objective_ledger']:
        if row['objective_id'] in recommendation_ids:
            row.update(status='PENDING', evidence_refs=[], failures=['CONSTRAINT_BINDING_CHANGED'])
    state['journal'].append(dict(reason='CONSTRAINT_BINDING_REFINED', source=transaction['source'],
        transaction_id=transaction['transaction_id'], binding_revision=proposed_revision,
        before=before, after=deepcopy(state['constraint_bindings'])))
    state['feedback'] = dict(code='BINDINGS_REFINED', binding_revision=proposed_revision,
        transaction_id=transaction['transaction_id'],
        message='版本化映射已通过独立审查并生效；以前的核验不能替代按新映射重新核验。')
    _save(context, state, plan)
    recorder.record('constraint_bindings_refined', source=transaction['source'],
        transaction_id=transaction['transaction_id'], binding_revision=proposed_revision,
        before=before, after=state['constraint_bindings'], audit_reference=audit['reference'])
    return True


def _maybe_resolve_catalog_bindings(context, recorder, state, plan, history):
    """Resume or create a registry transaction before spending another planner call."""
    current = state.get('binding_tx') or {}
    if current.get('status') == 'PROPOSED':
        return _apply_binding_transaction(context, recorder, state, plan, history, current,
                                          reject_as_error=False)
    resolution = resolve_catalog_concepts(state, plan)
    if resolution is None:
        return None
    if resolution['status'] == 'UNRESOLVED':
        if (state.get('binding_resolution') or {}).get('resolution_id') == resolution['resolution_id']:
            return None
        state['binding_resolution'] = deepcopy(resolution)
        state['feedback'] = dict(code='CATALOG_CONCEPT_UNRESOLVED',
            message='版本化概念尚未获得兼容的真实字段观察；继续读取正确品类，或在已读证据上提交人工映射。',
            unresolved=deepcopy(resolution['unresolved']))
        _save(context, state, plan)
        recorder.record('constraint_binding_unresolved', resolution_id=resolution['resolution_id'],
                        unresolved=resolution['unresolved'])
        return False
    return _apply_binding_transaction(context, recorder, state, plan, history,
                                      resolution['transaction'], reject_as_error=False)


def _products(step):
    return (step.result or {}).get('products', []) if isinstance(step.result, dict) else []


def _observation(result, event):
    return dict(result, observation=dict(timestamp=event['timestamp'],
        reference=f"{event['run_id']}:{event['sequence']}",
        digest=hashlib.sha256(json.dumps(result, sort_keys=True, ensure_ascii=False).encode()).hexdigest()))


def _discard_native_queue(context, state, reason):
    for queued in state.get('native_queue', []):
        context.record('native_batch_cancelled', reason=reason, **queued['submission'])
    state['native_queue'] = []


def _present(context, recorder, intake, state, plan, *, reason, message, error=None, success=False):
    if success:
        from src.request_audit import require_approved
        require_approved(state, plan)
    _discard_native_queue(context, state, reason)
    plan.verification = verification(state, complete=success)
    success = plan.verification['all_goals_satisfied']
    if not success and state.get('objective_ledger'):
        message += '\n\n' + outcome_report(state, plan)
    _save(context, state, plan)
    recorder.record('verification', **plan.verification)
    trace = [dict(run_id=context.run_id, agent='shopping', tool=e['tool'], args=e.get('args', {}),
                  result=e.get('result'), status=e.get('status'), duration_ms=e.get('duration_ms'))
             for e in context.events if e['type'] == 'tool_result']
    if success:
        from src.dialogue_understanding import DialogueCommand, render_direct_commands
        dialogue_commands = [DialogueCommand.model_validate(command)
                             for command in state.get('dialogue_commands', [])]
        preface = render_direct_commands(dialogue_commands)
        if preface:
            message = preface + '\n\n' + message
        if any(command.kind == 'UNSUPPORTED' for command in dialogue_commands):
            message += '\n\n另外一部分请求超出了当前消费服务助手的能力范围。'
            status, reason = 'PARTIAL', 'PARTIAL_UNSUPPORTED_INTENT'
        else:
            status = 'SUCCEEDED'
    elif error or reason == 'MODEL_STOPPED':
        status = 'PARTIAL' if trace else 'FAILED'
    else:
        status = 'WAITING_CLARIFICATION'
    result = _finish(recorder, intake, context.run_id, 'shopping', trace, reply=message,
                     status=status, terminal_reason=reason, error=error)
    result.update(route='ADAPTIVE', plan=dict(plan.payload(), version=state['version'],
        goal_revision=state['goal_revision'], replan_count=state['replans'],
        journal=deepcopy(state['journal']), goal=deepcopy(state['goal']), next_steps=state['next_steps'],
        working_memory=deepcopy(state.get('working_memory')), objective_ledger=deepcopy(state.get('objective_ledger', []))))
    result['plan'].update(constraint_bindings=deepcopy(state.get('constraint_bindings', {})),
                          binding_revision=state.get('binding_revision', 0),
                          binding_transaction=deepcopy(state.get('binding_tx')))
    result['request_audit'] = deepcopy(state.get('request_audit'))
    result['binding_audit'] = deepcopy(state.get('binding_audit'))
    result['completion_audit'] = deepcopy(state.get('completion_audit'))
    result['liveplan'] = deepcopy(state.get('liveplan'))
    return result


def _present_dialogue(context, recorder, intake, state, plan, decision):
    """Complete a tool-free dialogue command without fake business evidence."""
    _discard_native_queue(context, state, decision.action.upper())
    state['pending'] = None
    state['model_cursor'] = context.model_cursor
    _save(context, state, plan)
    status = 'FAILED' if decision.action == 'unsupported' else 'SUCCEEDED'
    reason = 'UNSUPPORTED_INTENT' if decision.action == 'unsupported' else 'COMPLETED'
    error = ({'code': 'UNSUPPORTED_INTENT', 'message': decision.reason or decision.message, 'retryable': False}
             if decision.action == 'unsupported' else None)
    recorder.record('verification', conversation_complete=status == 'SUCCEEDED',
                    business_evidence_required=False)
    result = _finish(recorder, intake, context.run_id, 'intake', [], reply=decision.message,
                     status=status, terminal_reason=reason, error=error)
    result.update(route='CONVERSATIONAL' if status == 'SUCCEEDED' else 'UNSUPPORTED',
                  understanding={'commands': [command.model_dump() for command in decision.commands],
                                 'objectives': [], 'reason': decision.reason})
    return result


def _validate_tool(decision, state, plan, history):
    if decision.tool not in _READ_ONLY_TOOLS:
        raise AgentError('UNKNOWN_TOOL', '该工具未执行。可调用只读工具：' + ', '.join(sorted(_READ_ONLY_TOOLS)) + '；选择商品请使用 action=recommend，选择并非业务工具调用。', retryable=False)
    spec = tool_spec(decision.tool, _READ_ONLY_TOOLS[decision.tool])
    if spec.effect != ToolEffect.READ:
        raise AgentError('TOOL_POLICY_REJECTED', '自主选购只能执行只读工具', retryable=False)
    try:
        if set(decision.args) - set(spec.schema.model_fields):
            raise ValueError('Unknown tool argument')
        args = spec.schema.model_validate(decision.args, strict=True).model_dump(exclude_unset=True)
    except ValueError as error:
        raise AgentError('INVALID_TOOL_ARGUMENTS', '模型提交的工具参数无效', retryable=False) from error
    if decision.tool == 'compare_products':
        observed = {p['product_id'] for step in plan.steps for p in _products(step)}
        ids = args['product_ids']
        if not 2 <= len(ids) <= 3 or len(set(ids)) != len(ids) or not set(ids) <= observed:
            raise AgentError('UNGROUNDED_PLAN_ENTITY', '对比只能引用已观察的两到三款商品', retryable=False)
    if 'order_id' in args:
        sources = [state['request'], *(str(m.content) for m in history or [] if getattr(m, 'type', None) == 'human')]
        if not any(re.search(r'(?<!\d)' + str(args['order_id']) + r'(?!\d)', source) for source in sources):
            raise AgentError('UNGROUNDED_PLAN_ENTITY', '订单编号必须来自用户', retryable=False)
    provenance = {}
    if decision.tool in {'search_products', 'discover_products'}:
        from src.capability_registry import get_capability_registry

        kind = PRODUCT_SEARCH if decision.tool == 'search_products' else PRODUCT_DISCOVERY
        sources = tuple([state['request'], *(str(message.content) for message in history or []
                                             if getattr(message, 'type', None) == 'human')])
        registry = get_capability_registry()
        accepted = set(registry.registration(kind).spec.accepted_fields)
        normalizable = {name: value for name, value in args.items() if name in accepted}
        normalized, provenance = registry.normalize_arguments({
            'id': f'adaptive-{decision.tool}',
            'kind': kind,
            'source_quote': state['request'],
            **normalizable,
        }, sources)
        args = {**args, **{name: normalized[name] for name in normalizable}}
    return args, provenance


def _normalize_goal_catalog_aliases(goal, request, history):
    """Resolve category aliases through trusted catalog metadata, never model guesswork."""
    from src.capability_registry import get_capability_registry

    payload = goal_payload(goal)
    sources = tuple([request, *(str(message.content) for message in history or []
                               if getattr(message, 'type', None) == 'human')])
    registry = get_capability_registry()
    provenance = {}
    rows = [(f'global:{index}', constraint)
            for index, constraint in enumerate(payload.get('hard_constraints', []))]
    rows.extend((f"objective:{objective['id']}:{index}", constraint)
                for objective in payload.get('objectives', [])
                for index, constraint in enumerate(objective.get('hard_constraints', [])))
    for reference, constraint in rows:
        value = constraint.get('value')
        if constraint.get('field') != 'category' or constraint.get('catalog_concept') is not None:
            continue
        values = value if isinstance(value, list) else [value]
        normalized_values = []
        entries = []
        for item in values:
            if not isinstance(item, str) or not item:
                normalized_values.append(item)
                continue
            normalized, resolved = registry.normalize_arguments({
                'id': f'adaptive-goal-{reference}',
                'kind': PRODUCT_SEARCH,
                'source_quote': constraint['source_quote'],
                'category': item,
            }, sources)
            normalized_values.append(normalized['category'])
            if 'category' in resolved:
                entries.append(resolved['category'])
        normalized_value = normalized_values if isinstance(value, list) else normalized_values[0]
        constraint['value'] = normalized_value
        if entries:
            provenance[reference] = {
                'field': 'category',
                'raw_value': deepcopy(value),
                'normalized_value': deepcopy(normalized_value),
                'resolutions': entries,
            }
    return payload, provenance


def _accept(context, recorder, state, plan, decision, history):
    if state['goal'] is None:
        if decision.goal is None:
            if decision.action in {'clarify', 'respond', 'handoff', 'unsupported'}:
                if decision.commands:
                    from src.dialogue_understanding import validate_command_grounding
                    validate_command_grounding(decision.commands, state['request'], history)
                state['pending'] = dict(decision=decision.model_dump(), step_id=None)
                state['model_cursor'] = context.model_cursor
                _save(context, state, plan)
                recorder.record('adaptive_decision', source='model', decision=decision.model_dump(),
                                goal_revision=state['goal_revision'], step_id=None)
                return
            raise AgentError('MODEL_FORMAT_ERROR', '首次决策必须包含目标及约束', retryable=False)
        if decision.commands:
            from src.dialogue_understanding import validate_command_grounding
            validate_command_grounding(decision.commands, state['request'], history)
            state['dialogue_commands'] = [command.model_dump() for command in decision.commands]
        validate_goal(decision.goal, state['request'], history)
        accepted_goal, goal_provenance = _normalize_goal_catalog_aliases(
            decision.goal, state['request'], history)
        if goal_provenance:
            recorder.record('goal_arguments_normalized', provenance=deepcopy(goal_provenance))
        if decision.action != 'clarify':
            from src.request_audit import review_request
            proposed = {**state, 'goal': accepted_goal}
            audit = review_request(context, proposed, plan, history, phase='goal')
            state['audit_attempts'] = [*state.get('audit_attempts', []), audit]
            if audit['verdict'] != 'APPROVED':
                state['feedback'] = dict(code='REQUEST_AUDIT_REJECTED', audit=audit,
                    message='对照用户原文修复全部目标和约束；不确定则澄清。')
                raise AgentError('REQUEST_AUDIT_REJECTED', '独立审查未确认需求覆盖与语义，不能执行当前目标', retryable=False)
            state['request_audit'] = audit
            state.pop('audit_format_feedback', None)
        state['goal'] = accepted_goal
        state['objective_ledger'] = new_ledger(state['goal'])
        plan.goals = [o['id'] for o in state['goal']['objectives']]
    elif decision.goal is not None:
        normalized_goal, _ = _normalize_goal_catalog_aliases(decision.goal, state['request'], history)
        if normalized_goal != state['goal']:
            raise AgentError('GOAL_CHANGE_REQUIRES_USER', '没有新用户指令时不能修改目标或硬约束', retryable=False)
    surface = state.get('decision_surface') or {}
    if surface.get('mode') != 'full' and surface.get('allowed_actions'):
        if decision.action not in set(surface['allowed_actions']):
            raise AgentError('ACTION_SURFACE_VIOLATION', '当前预算已为最终独立审查预留，只能推荐、澄清或如实结束', retryable=False)
        if decision.action == 'recommend':
            candidates = {row['product_id']: set(row['evidence_refs']) for row in surface.get('candidates', [])}
            if (decision.product_id not in candidates or decision.objective_id != surface.get('objective_id')
                    or not set(decision.evidence_refs) <= candidates[decision.product_id]):
                raise AgentError('ACTION_SURFACE_VIOLATION', '推荐必须来自当前已读、满足硬约束的候选及其证据', retryable=False)
    if decision.action == 'refine_constraints':
        bindings = propose_bindings(state, plan, decision.bindings)
        transaction = build_binding_transaction(state, bindings, decision.bindings, source='model')
        _apply_binding_transaction(context, recorder, state, plan, history, transaction,
                                   reject_as_error=True)
    if decision.working_memory is not None:
        completed = {s.step_id for s in plan.steps if s.status in _DONE and s.result is not None}
        if not set(decision.working_memory.evidence_refs) <= completed:
            raise AgentError('UNGROUNDED_WORKING_MEMORY', '工作笔记只能引用已完成的证据', retryable=False)
        state['working_memory'] = decision.working_memory.model_dump()
    if decision.next_steps is not None:
        if state['next_steps'] is not None and decision.next_steps != state['next_steps']:
            if state['replans'] >= MAX_REPLANS:
                raise AgentError('REPLAN_BUDGET_EXHAUSTED', '计划修改次数已耗尽', retryable=False)
            state['replans'] += 1
            state['version'] += 1
            entry = dict(version=state['version'], goal_revision=state['goal_revision'],
                         reason=decision.reason, before=state['next_steps'], after=decision.next_steps)
            state['journal'].append(entry)
            recorder.record('plan_replanned', source='model', **entry)
        state['next_steps'] = decision.next_steps
    step = None
    if decision.action == 'tool':
        args, argument_provenance = _validate_tool(decision, state, plan, history)
        step = TaskStep(f'step-{len(plan.steps) + 1}', 'shopping', decision.tool, args,
                        argument_provenance=argument_provenance)
        if argument_provenance:
            recorder.record('tool_arguments_normalized', tool=decision.tool,
                raw_args=deepcopy(decision.args), normalized_args=deepcopy(args),
                argument_provenance=deepcopy(argument_provenance))
    elif decision.action == 'inspect':
        step = TaskStep(f'step-{len(plan.steps) + 1}', 'shopping', 'read_evidence',
                        dict(reference=decision.evidence_ref, offset=decision.offset, limit=decision.limit))
    elif decision.action == 'recommend':
        if not any(o['id'] == decision.objective_id and o['kind'] == 'recommend' for o in state['goal']['objectives']):
            raise AgentError('INVALID_RECOMMENDATION_OBJECTIVE', '推荐必须对应当前目标中的选购项', retryable=False)
        indexed = {s.step_id: s for s in plan.steps}
        for ref in decision.evidence_refs:
            if ref not in indexed or not any(p['product_id'] == decision.product_id for p in _products(indexed[ref])):
                raise AgentError('UNGROUNDED_RECOMMENDATION', '推荐必须引用包含该商品的真实观察', retryable=False)
            if decision.product_id not in state.get('evidence_access', {}).get(ref, []):
                raise AgentError('UNREAD_RECOMMENDATION_EVIDENCE', '请先读取商品完整证据，不能仅凭索引推荐', retryable=False)
        step = _selection_step(state, plan, decision)
    if step:
        candidate = deepcopy(plan)
        candidate.steps.append(step)
        candidate.validate(context.limits.max_steps, context.limits.max_depth)
        plan.steps.append(step)
    state['pending'] = dict(decision=decision.model_dump(), step_id=step.step_id if step else None)
    state['model_cursor'] = context.model_cursor
    state['feedback'] = None
    _save(context, state, plan)
    recorder.record('adaptive_decision', source='model', decision=decision.model_dump(),
                    goal_revision=state['goal_revision'], step_id=step.step_id if step else None,
                    native_submission=state.get('active_native_submission'))


def _selection_step(state, plan, decision):
    indexed = {s.step_id: s for s in plan.steps}
    def roots(ref):
        step = indexed.get(ref)
        if step is not None and step.tool == _SELECT:
            return [root for dependency in step.depends_on for root in roots(dependency)]
        return [ref]
    dependencies = list(dict.fromkeys(root for ref in decision.evidence_refs for root in roots(ref)))
    return TaskStep(f'step-{len(plan.steps) + 1}', 'shopping', _SELECT,
        dict(product_id=decision.product_id, objective_id=decision.objective_id,
             goal_revision=state['goal_revision']), dependencies)


def _repair(context, recorder, state, plan, error):
    _discard_native_queue(context, state, error.code)
    state['repair_count'] = state.get('repair_count', 0) + 1
    state['model_cursor'] = context.model_cursor
    if error.code not in {'REQUEST_AUDIT_REJECTED', 'BINDING_AUDIT_REJECTED'}:
        state['feedback'] = dict(code=error.code, message=str(error),
                                instruction='修复协议或参数后提交下一次决策；不要放宽用户目标或审批条件。')
        if error.code == 'AUDIT_FORMAT_ERROR':
            state['audit_format_feedback'] = dict(phase=getattr(error, 'audit_phase', None), message=str(error))
            state['feedback']['instruction'] = '审查响应格式无效不代表原提案被语义否定。可重提同一目标或映射，让审查模型修正协议；不要为此改变用户原意。'
    state['pending'] = None
    _save(context, state, plan)
    recorder.record('decision_rejected', code=error.code, message=str(error), repair_count=state['repair_count'])
    if state['repair_count'] > MAX_DECISION_REPAIRS:
        raise AgentError('DECISION_REPAIR_BUDGET_EXHAUSTED', '模型决策修复次数已耗尽：' + str(error), retryable=False)


def _review_completion(context, recorder, state, plan, history):
    from src.request_audit import require_approved, review_request
    require_approved(state)
    # 恢复后的最终核验已经重新读取事实，不能复用基于旧观察的审查响应。
    cursor = context.model_cursor + 1
    while (cached := context.models.get(str(cursor))) and json.loads(cached['input'])[0] == 'request_completion_audit':
        context.models.pop(str(cursor))
        context.record('audit_invalidated', reason='FRESH_COMPLETION_EVIDENCE')
        cursor += 1
    try:
        audit = review_request(context, state, plan, history, phase='completion')
    except AgentError as error:
        if error.code != 'AUDIT_FORMAT_ERROR':
            raise
        _repair(context, recorder, state, plan, error)
        return False
    state['completion_audit'] = audit
    state['model_cursor'] = context.model_cursor
    _save(context, state, plan)
    if audit['verdict'] != 'APPROVED':
        error = AgentError('COMPLETION_SEMANTIC_REJECTED', '最终证据未完整回应用户原始需求', retryable=False)
        _repair(context, recorder, state, plan, error)
        state['feedback']['audit'] = audit
        _save(context, state, plan)
        return False
    return True


def _verify_recommendation(context, recorder, state, plan, decision, step):
    require_bindings_approved(state)
    # A new scope on EVERY attempt, including process resume. Historical ledger
    # replay must never satisfy the final freshness gate.
    executor = ToolExecutor(_READ_ONLY_TOOLS, recorder, max_calls=context.limits.max_tool_calls,
                            observation_scope=f'adaptive:verify:{uuid.uuid4().hex}')
    try:
        result, event = executor.execute('search_products', dict(query=decision.product_id,
            category=None, max_price=None, available_only=False, limit=60))
    except AgentError as error:
        if error.code not in {'TOOL_TIMEOUT', 'TOOL_ERROR'}:
            raise
        step.status = 'FAILED'
        step.error = dict(code=error.code, message=str(error))
        state['feedback'] = dict(code='RECOMMENDATION_VERIFICATION_FAILED',
                                failures=[error.code], evidence_ref=step.step_id)
        recorder.record('verification_failed', step_id=step.step_id, **state['feedback'])
        return None
    verification = TaskStep('fresh', 'shopping', 'search_products', {}, result=result)
    _verify_products(verification)
    fresh = next((p for p in result['products'] if p['product_id'] == decision.product_id), None)
    failures = []
    constraint_failures = []
    if fresh is None:
        failures.append('PRODUCT_MISSING')
    else:
        if fresh['stock'] <= 0:
            failures.append('OUT_OF_STOCK')
        goal = ShoppingGoal.model_validate(effective_goal(state))
        objective = next(o for o in goal.objectives if o.id == decision.objective_id)
        for ref, data in constraint_refs(goal.model_dump()).items():
            if not (ref.startswith('global:') or ref.startswith(f'objective:{objective.id}:')):
                continue
            from src.adaptive_decision import Constraint
            c = Constraint.model_validate(data)
            if not predicate_holds(fresh, c):
                failures.append('HARD_CONSTRAINT: ' + c.description)
                constraint_failures.append(dict(constraint_ref=ref, field=c.field, op=c.op, value=c.value,
                                                description=c.description))
        for ref in decision.evidence_refs:
            old = next(p for s in plan.steps if s.step_id == ref for p in _products(s) if p['product_id'] == decision.product_id)
            if any(old.get(key) != fresh.get(key) for key in ('price', 'stock', 'category', 'specs', 'name', 'summary')):
                failures.append('STALE_EVIDENCE: ' + ref)
    step.result = _observation(dict(products=[fresh] if fresh else [], failures=failures,
        evidence_refs=decision.evidence_refs, rationale=decision.reason,
        binding_revision=state.get('binding_revision', 0),
        bindings_sha256=fingerprint(state.get('constraint_bindings', {}))), event)
    if failures:
        step.status = 'REJECTED'
        state['feedback'] = dict(code='RECOMMENDATION_VERIFICATION_FAILED', failures=failures, evidence_ref=step.step_id)
        if constraint_failures:
            state['feedback']['constraint_failures'] = constraint_failures
        recorder.record('verification_failed', step_id=step.step_id, **state['feedback'])
        return None
    step.status = 'SUCCEEDED'
    step.result['selected_product'] = deepcopy(fresh)
    return recommendation_text(step)


def _complete(context, recorder, state, plan, decision):
    # A crash can leave a refresh attempt pending after its tool event was saved.
    # Keep its history, but never leave an abandoned attempt presented as live work.
    for step in plan.steps:
        if step.tool == _SELECT and step.status == 'PENDING':
            step.status = 'CANCELLED'
            step.error = dict(code='SUPERSEDED_VERIFICATION', message='恢复后重新读取，旧核验尝试不作为完成证据')
    failures = check_claims(state, plan, decision.claims)
    if failures:
        state['feedback'] = dict(code='COMPLETION_VERIFICATION_FAILED', failures=failures)
        recorder.record('verification_failed', **state['feedback'])
        return None
    # Each recommendation is refreshed at the terminal boundary, even if it was
    # verified earlier while another outcome was still being investigated.
    for objective in state['goal']['objectives']:
        if objective['kind'] != 'recommend':
            continue
        row = next(r for r in state['objective_ledger'] if r['objective_id'] == objective['id'])
        source = next(s for s in plan.steps if s.step_id == row['evidence_refs'][0])
        row.update(status='PENDING', failures=['FINAL_REFRESH_REQUIRED'])
        _save(context, state, plan)
        proposal = Decision(action='recommend', objective_id=objective['id'],
            product_id=source.result['selected_product']['product_id'], evidence_refs=[source.step_id],
            reason=source.result['rationale'])
        step = _selection_step(state, plan, proposal)
        candidate = deepcopy(plan)
        candidate.steps.append(step)
        candidate.validate(context.limits.max_steps, context.limits.max_depth)
        plan.steps.append(step)
        _save(context, state, plan)
        message = _verify_recommendation(context, recorder, state, plan, proposal, step)
        record_outcome(state, objective['id'], step)
        _save(context, state, plan)
        recorder.record('step_completed', step_id=step.step_id, specialist='shopping', status=step.status)
        if message is None:
            return None
    return outcome_report(state, plan, complete=True)


def _run(context, user_message, history, pending_action):
    from src.orchestrator import _execute_understanding, run_orchestrator
    intake = analyze_request(user_message)
    saved = context.snapshot.get('adaptive')
    shopping_hints = {PRODUCT_SEARCH, PRODUCT_DISCOVERY, PRODUCT_COMPARE}
    # Fresh open-set turns first use the same capability protocol as Auto.
    # High-recall shopping hints stay in the deeper Adaptive shopping loop;
    # hints are never themselves permission to execute a capability.
    if (not saved and not pending_action and intake.execution_mode == LLM_REQUIRED
            and not (set(intake.route_hints) & shopping_hints)):
        from src.dialogue_understanding import understand_turn
        from src.llm import get_llm
        understanding = understand_turn(
            user_message,
            history,
            pending_action,
            model=get_llm(purpose='dialogue_understanding'),
        )
        result = _execute_understanding(
            user_message, history, pending_action, intake, context.run_id, understanding=understanding,
        )
        result.setdefault('plan', {
            'goal': None,
            'steps': [],
            'verification': {'goal_outcomes': [], 'all_goals_satisfied': False,
                             'side_effects_executed': False},
        })
        result['execution_mode'] = 'auto'
        return result
    if pending_action or (not saved and (PAYMENT_PROCESS in intake.intents or REFUND_REQUEST in intake.intents
            or (intake.execution_mode == DIRECT and intake.intents and PRODUCT_SEARCH not in intake.intents
                and 'PRODUCT_COMPARE' not in intake.intents))):
        result = run_orchestrator(user_message, history, pending_action)
        result['execution_mode'] = 'auto'
        return result
    recorder = TraceRecorder(context.run_id)
    revision = context.snapshot.get('goal_revision', 0)
    if saved:
        if saved.get('schema_version') not in {4, 5, 6}:
            raise AgentError('ADAPTIVE_CHECKPOINT_INCOMPATIBLE', '旧检查点缺少独立需求审查记录，不能安全恢复，请新建任务；已完成操作记录保留', retryable=False)
        state = deepcopy(saved)
        state.update(schema_version=6)
        state.setdefault('constraint_bindings', {})
        state.setdefault('binding_revision', 0)
        state.setdefault('binding_audit', None)
        state.setdefault('binding_evidence_access', {})
        state.setdefault('binding_sources', {})
        state.setdefault('binding_tx', None)
        state.setdefault('binding_resolution', None)
        state.setdefault('decision_surface', None)
        state.setdefault('native_queue', [])
        state.setdefault('dialogue_commands', [])
        plan = _restore_plan(state['plan'])
        if revision == state['goal_revision'] and user_message != state['request']:
            raise AgentError('CHECKPOINT_CONFLICT', '恢复请求与已保存目标不一致', retryable=False)
        if revision != state['goal_revision']:
            _discard_native_queue(context, state, 'USER_CONSTRAINT_CHANGED')
            for step in plan.steps:
                if step.status == 'PENDING':
                    step.status = 'CANCELLED'
            state['journal'].append(dict(reason='USER_CONSTRAINT_CHANGED', goal_revision=revision,
                                         previous_goal=state['goal']))
            state.update(goal_revision=revision, goal=None, request=user_message, pending=None,
                         model_cursor=0, feedback=None, working_memory=None, objective_ledger=[],
                         next_steps=None, request_audit=None, completion_audit=None,
                         constraint_bindings={}, binding_revision=0, binding_audit=None,
                         binding_evidence_access={}, binding_sources={}, binding_tx=None,
                         binding_resolution=None, decision_surface=None,
                         audit_format_feedback=None,
                         dialogue_commands=[],
                         native_queue=[], active_native_submission=None,
                         liveplan=None,
                         version=state['version'] + 1)
            plan.goals = []
            plan.verification = {}
            _save(context, state, plan)
        context.model_cursor = state['model_cursor']
    else:
        plan = TaskPlan([], [])
        state = dict(schema_version=6, version=1, replans=0, goal_revision=revision, request=user_message,
                     goal=None, next_steps=None, working_memory=None, journal=[], pending=None,
                     objective_ledger=[], feedback=None, repair_count=0, request_audit=None,
                     completion_audit=None, constraint_bindings={}, binding_revision=0, binding_audit=None,
                     binding_evidence_access={}, binding_sources={}, binding_tx=None,
                     binding_resolution=None, decision_surface=None,
                     dialogue_commands=[],
                     native_queue=[], active_native_submission=None,
                     liveplan=None,
                     model_cursor=context.model_cursor)
        _save(context, state, plan)
        recorder.record('plan_created', source='adaptive_model', plan=plan.payload())
    intake = TaskIntake(user_message, (PRODUCT_SEARCH,), {}, execution_mode='adaptive', reason='模型逐步决策')
    try:
        while True:
            from src.session_control import apply_controls
            apply_controls(context)
            context.require()
            if state['pending'] is None:
                try:
                    _apply_liveplan(context, state, plan)
                    if state.get('native_queue'):
                        queued = state['native_queue'].pop(0)
                        decision = Decision.model_validate(queued['decision'])
                        state['active_native_submission'] = queued['submission']
                    else:
                        _maybe_resolve_catalog_bindings(context, recorder, state, plan, history)
                        decision = decide(context, state, plan, history)
                    # 插话也使审查中的旧目标失效，执行任何动作前再次检查。
                    apply_controls(context)
                    if _review_liveplan_candidate(context, recorder, state, plan, decision):
                        continue
                    _accept(context, recorder, state, plan, decision, history)
                    apply_controls(context)
                except AgentError as error:
                    if error.code in {'MODEL_FORMAT_ERROR', 'UNKNOWN_TOOL', 'INVALID_TOOL_ARGUMENTS', 'REQUEST_AUDIT_REJECTED', 'AUDIT_FORMAT_ERROR',
                                      'INVALID_CONSTRAINT_BINDING', 'UNREAD_BINDING_EVIDENCE', 'UNOBSERVED_BINDING_FIELD', 'BINDING_AUDIT_REJECTED'}:
                        _repair(context, recorder, state, plan, error)
                        continue
                    if error.code not in {'UNREAD_RECOMMENDATION_EVIDENCE', 'INVALID_RECOMMENDATION_OBJECTIVE', 'BINDING_ALREADY_CURRENT'}:
                        raise
                    # An index is not sufficient evidence. Let the model choose
                    # which original records to inspect, within its remaining budget.
                    _discard_native_queue(context, state, error.code)
                    state['feedback'] = dict(code=error.code, message=str(error))
                    state['model_cursor'] = context.model_cursor
                    _save(context, state, plan)
                    recorder.record('decision_rejected', code=error.code, action=decision.action)
                    continue
            pending = state['pending']
            decision = Decision.model_validate(pending['decision'])
            step = next((s for s in plan.steps if s.step_id == pending['step_id']), None)
            if decision.action in {'respond', 'unsupported'}:
                return _present_dialogue(context, recorder, intake, state, plan, decision)
            if decision.action == 'handoff':
                from src.orchestrator import _explicit_escalation
                return _explicit_escalation(intake, context.run_id, state['request'])
            if decision.action in {'clarify', 'finish'}:
                return _present(context, recorder, intake, state, plan,
                    reason='MODEL_CLARIFICATION' if decision.action == 'clarify' else 'MODEL_STOPPED', message=decision.message)
            if decision.action == 'recommend':
                if step.status == 'SUCCEEDED':
                    # A previous verification succeeded but its terminal result
                    # wasn't committed. Preserve that historical observation and
                    # create a fresh attempt instead of overwriting completed work.
                    candidate = deepcopy(plan)
                    step = _selection_step(state, plan, decision)
                    candidate.steps.append(step)
                    candidate.validate(context.limits.max_steps, context.limits.max_depth)
                    plan.steps.append(step)
                    state['pending']['step_id'] = step.step_id
                    _save(context, state, plan)
                row = next(r for r in state['objective_ledger'] if r['objective_id'] == decision.objective_id)
                row.update(status='PENDING', failures=['FINAL_REFRESH_REQUIRED'])
                _save(context, state, plan)
                message = _verify_recommendation(context, recorder, state, plan, decision, step)
                record_outcome(state, decision.objective_id, step)
                if message is not None and len(state['objective_ledger']) == 1:
                    # Keep the pending proposal through terminal persistence. If
                    # the process dies before commit, resume must refresh again.
                    apply_controls(context)
                    context.require()
                    if context.hook_error:
                        raise context.hook_error
                    if _review_completion(context, recorder, state, plan, history):
                        apply_controls(context)
                        return _present(context, recorder, intake, state, plan, reason='COMPLETED', message=message, success=True)
                if message is not None and len(state['objective_ledger']) > 1:
                    state['feedback'] = dict(code='OBJECTIVES_REQUIRE_COMPLETION',
                        message='此选购项已核验，请继续处理剩余目标，再逐项提交 complete 证据。')
            elif decision.action == 'complete':
                message = _complete(context, recorder, state, plan, decision)
                if message is not None:
                    apply_controls(context)
                    context.require()
                    if context.hook_error:
                        raise context.hook_error
                    # Preserve the pending completion across the durable commit;
                    # replay verifies current business facts again after a crash.
                    if _review_completion(context, recorder, state, plan, history):
                        apply_controls(context)
                        return _present(context, recorder, intake, state, plan, reason='COMPLETED', message=message, success=True)
            elif decision.action == 'inspect':
                from src.context_engine import ContextEngine
                try:
                    step.result = ContextEngine.read_evidence(plan, **step.args)
                    step.status = 'SUCCEEDED'
                except AgentError as error:
                    step.status = 'FAILED'
                    step.error = dict(code=error.code, message=str(error))
                    state['feedback'] = dict(step_id=step.step_id, **step.error)
                recorder.record('evidence_read', step_id=step.step_id, **step.args, status=step.status)
            elif decision.action == 'refine_constraints':
                state['feedback'] = dict(code='BINDINGS_REFINED', binding_revision=state['binding_revision'],
                    message='新映射已通过独立审查并生效。以前的核验失败属于旧映射，尚未按新映射重新核验商品。')
            elif decision.action == 'tool':
                executor = ToolExecutor(_READ_ONLY_TOOLS, recorder, max_calls=context.limits.max_tool_calls,
                    observation_scope=f"adaptive:{state['goal_revision']}:{step.step_id}")
                try:
                    result, event = executor.execute(step.tool, step.args)
                    step.result = result
                    if event['status'] == 'BUSINESS_ERROR':
                        step.status = 'FAILED'
                    elif step.tool == 'answer_faq' and result.get('status') != 'ANSWERED':
                        step.status = 'REJECTED'
                        state['feedback'] = dict(code='INSUFFICIENT_POLICY_EVIDENCE',
                                                message='政策库未提供有效答案，该目标未完成。')
                    else:
                        if step.tool in {'search_products', 'compare_products'}:
                            _verify_products(step)
                            if not result['products']:
                                state['feedback'] = dict(code='EMPTY_CATALOG_RESULT', message='本次查询无匹配，不代表目录中不存在商品。可以更换检索词、使用品类过滤或澄清；空观察不能支持映射或推荐。')
                        else:
                            verify_observation(step)
                        step.status = event['status']
                    if isinstance(result, dict):
                        step.result = _observation(result, event)
                    record_binding_evidence(state, step)
                except AgentError as error:
                    # Recoverable tool observations go back to the model. Safety
                    # and budget failures remain hard runtime boundaries.
                    if error.code not in {'TOOL_TIMEOUT', 'TOOL_ERROR'}:
                        raise
                    step.status = 'FAILED'
                    step.error = dict(code=error.code, message=str(error))
                    state['feedback'] = dict(step_id=step.step_id, **step.error)
            state['pending'] = None
            if state.get('feedback') and state['feedback'].get('code') != 'OBJECTIVES_REQUIRE_COMPLETION':
                _discard_native_queue(context, state, state['feedback'].get('code', 'OBSERVATION_CHANGED'))
            _save(context, state, plan)
            if step:
                recorder.record('step_completed', step_id=step.step_id, specialist='shopping', status=step.status)
    except Exception as exc:
        error = classify_exception(exc, source='model')
        message = ('未配置可用模型，自主模式未执行。请配置模型后重试。' if error.code == 'MODEL_UNAVAILABLE'
                   and not context.counters['model_calls'] else '自主运行未完成：' + str(error))
        return _present(context, recorder, intake, state, plan, reason=error.code, message=message,
                        error=dict(code=error.code, message=str(error), retryable=error.retryable))


def run_adaptive(user_message, history=None, pending_action=None, *, limits=None):
    context = current_runtime()
    if context is None:
        context = RunRuntime(uuid.uuid4().hex, limits or BudgetLimits.adaptive())
        with runtime_scope(context):
            return run_adaptive(user_message, history, pending_action)
    started = time.perf_counter()
    context.record('intake', request=user_message, execution_mode='adaptive')
    try:
        result = _run(context, user_message, history, pending_action)
    except Exception as exc:
        error = classify_exception(exc, source='model')
        result = _finish(TraceRecorder(context.run_id), analyze_request(user_message), context.run_id, 'runtime', [],
            reply=str(error), status='FAILED', terminal_reason=error.code, pending_action=pending_action,
            error=dict(code=error.code, message=str(error), retryable=error.retryable))
        result['route'] = 'ADAPTIVE'
    result['events'] = context.events
    from src.agents.base import sum_usage
    calls = [e for e in context.events if e['type'] == 'model_call']
    if calls:
        result['usage'] = sum_usage(*(e.get('usage') for e in calls))
    elif any(e['type'] == 'model_started' for e in context.events):
        result['usage'] = dict(input_tokens=None, output_tokens=None, total_tokens=None)
    result['budget'] = dict(limits=context.limits.__dict__, consumed=dict(context.counters))
    result.setdefault('execution_mode', 'adaptive')
    result['autonomous_executed'] = any(e['type'] == 'model_call' and e.get('purpose') == 'adaptive_decision' for e in context.events)
    result['latency_ms'] = round((time.perf_counter() - started) * 1000, 2)
    return result
