"""Evidence-driven shopping plans with durable, bounded changes to unfinished work.

Business tools and model calls keep their existing runtime/approval boundaries.
The adaptive layer chooses only from observations; it never generates catalog IDs.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re
import time
import uuid

from pydantic import BaseModel, ConfigDict, Field

from src.direct_executor import _READ_ONLY_TOOLS, _finish
from src.errors import AgentError, classify_exception
from src.intake import (CLARIFICATION_REQUIRED, DIRECT, LLM_REQUIRED, PAYMENT_PROCESS,
                        PRODUCT_SEARCH, REFUND_REQUEST, TaskIntake, analyze_request)
from src.runtime import (BudgetLimits, RunRuntime, ToolEffect, ToolExecutor,
                         TraceRecorder, current_runtime, runtime_scope)
from src.task_plan import TaskPlan, TaskStep, _facts, build_plan, verify_observation


MAX_REPLANS = 2
_SELECT = "select_product"
_DONE = {"SUCCEEDED", "REUSED", "NOT_NEEDED", "REJECTED", "WAITING_APPROVAL"}
_FEATURES = ("主动降噪", "通话降噪", "降噪", "静音", "三模", "双模", "有线", "无线", "蓝牙",
             "头戴式", "开放式", "入耳式", "4k", "60hz", "30hz", "便携", "通勤", "办公",
             "防水", "红轴", "茶轴", "磁轴", "双屏", "hdmi", "运动", "视频会议", "麦克风", "空间音频")
_BUDGET = re.compile(
    r"(?:预算|价格|价钱|花费|费用)\s*(?:改为|改成|调整为|提高到|提高至|降到|降至|为|是|控制在|低于|最多|不高于|不超过|≤|<=)?\s*[¥￥]?\s*(-?\d+(?:\.\d+)?)"
    r"|(?:不超过|不高于|最多)\s*[¥￥]?\s*(-?\d+(?:\.\d+)?)\s*(?:元|块|人民币)"
    r"|(-?\d+(?:\.\d+)?)\s*(?:元|块)?\s*(?:以内|以下|不超过)"
    r"|(?:under|budget|max(?:imum)?)\s*(?:cny|rmb|[¥￥])?\s*(-?\d+(?:\.\d+)?)", re.I)


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


def _goal(request, intake, *, previous=None, additions=()):
    """Known hard constraints are checked against values, never model prose."""
    category = intake.entities.get("category")
    if not category:
        category = next((s.args.get("category") for s in build_plan(intake).steps if s.tool == "search_products"), None)
    goal = dict(category=category, max_price=intake.entities.get("max_price"), features=[], excluded=[],
                minimums={}, product_ids=intake.entities.get("product_ids", []))
    if previous:
        goal = deepcopy(previous)
        goal.pop("conflict", None)
    texts = list(additions) if previous else [request]
    for text in texts:
        lower = text.casefold()
        if previous and any(intent not in {"PRODUCT_SEARCH", "PRODUCT_COMPARE"} for intent in analyze_request(text).intents):
            goal["conflict"] = "本次补充包含新的服务任务，请另行提交订单、支付或退款请求。已完成的购物查询仍保留。"
        budgets = [float(next(value for value in match.groups() if value is not None)) for match in _BUDGET.finditer(lower)]
        if budgets:
            goal["max_price"] = min(budgets)
        if re.search(r"(?:不限制|不限|无上限|不设上限)\s*(?:预算|价格)|(?:预算|价格)\s*(?:不限|无上限|不设上限)|no\s+(?:budget|price)\s+limit", lower):
            goal['max_price'] = None
        categories = [c for c in ("耳机", "键盘", "扩展坞") if c in lower]
        if len(categories) > 1:
            goal["conflict"] = "一次购物目标包含多个商品类别，请分别说明要选哪类商品。"
        elif categories and previous:
            goal["category"] = categories[0]
        for feature in _FEATURES:
            if feature not in lower:
                continue
            denied = bool(re.search(r"(?:不要|不需要|无需|排除|不想要)\s*" + re.escape(feature), lower))
            destination, other = ("excluded", "features") if denied else ("features", "excluded")
            if denied or previous:
                goal[other] = [value for value in goal[other] if value != feature]
            if previous and feature in {"头戴式", "开放式", "入耳式"} and re.search(r"改成|改为|换成", lower):
                goal["features"] = [value for value in goal["features"] if value not in {"头戴式", "开放式", "入耳式"}]
            if feature not in goal[destination]:
                goal[destination].append(feature)
        # Preserve unfamiliar explicit requirements conservatively as exact terms.
        for match in re.finditer(r"(?:支持|带有|要有|必须有|with\s+)([^，,。;；\n]{1,30})", lower):
            term = re.split(r"(?:的)?(?:耳机|键盘|扩展坞)|预算|价格", match.group(1))[0].strip()
            remainder = term
            for known in sorted(_FEATURES, key=len, reverse=True):
                remainder = remainder.replace(known, "")
            if remainder.strip(" 的和及、 ") and term and term not in goal["features"]:
                goal["features"].append(term)
        for pattern, key in ((r"(?:续航[^\d]{0,6}|至少\s*)(\d+(?:\.\d+)?)\s*(?:个)?小时", "battery_hours"),
                             (r"(?:至少|不少于)\s*(\d+)\s*(?:个)?键", "keys"),
                             (r"(?:至少|不少于)\s*(\d+)\s*(?:个)?接口", "ports"),
                             (r"(?:至少|不少于)\s*(\d+)\s*w", "power_watts")):
            match = re.search(pattern, lower)
            if match:
                goal["minimums"][key] = float(match.group(1))
    if not previous and intake.entities.get("query"):
        # A model's query is a constraint only if grounded in the customer's text.
        for term in intake.entities["query"].casefold().split():
            if term not in request.casefold():
                raise AgentError("UNGROUNDED_PLAN_CONSTRAINT", "规划提出了用户没有提供的商品条件", retryable=False)
            if term not in goal["features"] and term not in goal["excluded"]:
                goal["features"].append(term)
    if goal["max_price"] is not None and (not math.isfinite(goal["max_price"]) or goal["max_price"] < 0):
        goal["conflict"] = "预算必须是非负数，请明确可接受的最高价格。"
    if len(set(goal["features"]) & {"头戴式", "开放式", "入耳式"}) > 1:
        goal["conflict"] = "同一款耳机的佩戴方式条件冲突，请选择头戴式、开放式或入耳式。"
    if set(goal["features"]) & set(goal["excluded"]):
        goal["conflict"] = "商品特征同时被要求和排除，请明确保留哪项条件。"
    if len(goal["product_ids"]) > 3:
        goal["conflict"] = "一次最多对比三款商品，请明确保留哪两到三款。"
    return goal


def _search_args(goal):
    return dict(query=" ".join(goal["features"]), category=goal["category"], max_price=goal["max_price"],
                available_only=True, limit=60)


def _matches(product, goal):
    if product["stock"] <= 0 or (goal["category"] and product["category"] != goal["category"]):
        return False
    if goal["max_price"] is not None and product["price"] > goal["max_price"]:
        return False
    specs = product["specs"]
    # Spec keys such as "降噪" are not evidence that a product has that feature.
    values = " ".join(str(v) for v in specs.values()).casefold()
    text = f"{product['name']} {product['summary']} {values}".casefold()
    def has(feature):
        if "降噪" in feature:
            value = str(specs.get("降噪", "")).casefold()
            return feature in value and value not in {"无", "不支持", "none"}
        return feature in text
    if not all(has(f) for f in goal["features"]) or any(has(f) for f in goal["excluded"]):
        return False
    for key, minimum in goal["minimums"].items():
        value = specs.get({"battery_hours": "综合续航", "keys": "键数", "ports": "接口数量", "power_watts": "PD 供电"}[key])
        number = re.search(r"\d+(?:\.\d+)?", str(value))
        if not number or float(number.group()) < minimum:
            return False
    return True


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


def _scope(state):
    return f"adaptive:{state['goal_revision']}:{state['replans']}"


def _replace_remaining(context, recorder, state, plan, goal, reason):
    if state["replans"] >= MAX_REPLANS:
        raise AgentError("REPLAN_BUDGET_EXHAUSTED", "已达到两次重规划上限，请明确条件后创建新任务", retryable=False)
    pending = [s for s in plan.steps if s.specialist == "shopping" and s.status not in _DONE and s.status != "CANCELLED"]
    identities = [s.step_id for s in pending]
    number = len(plan.steps)
    while len(identities) < 3:
        number += 1
        identities.append(f"step-{number}")
    search, compare, select = identities[:3]
    drafts = [StepDraft(step_id=search, tool="search_products", args=_search_args(goal)),
              StepDraft(step_id=compare, tool="compare_products", depends_on=[search]),
              StepDraft(step_id=select, tool=_SELECT, depends_on=[compare])]
    existing_ids = {s.step_id for s in pending}
    patch = PlanPatch(reason=reason, replace=[s for s in drafts if s.step_id in existing_ids],
                      append=[s for s in drafts if s.step_id not in existing_ids], cancel=identities[3:])
    candidate = apply_plan_patch(plan, patch, context.limits)
    state["version"] += 1
    state["replans"] += 1
    state["goal"] = goal
    state["journal"].append(dict(version=state["version"], goal_revision=state["goal_revision"], **patch.model_dump()))
    for draft in drafts:
        state["scopes"][draft.step_id] = _scope(state)
    _save(context, state, candidate)
    recorder.record("plan_replanned", source="adaptive", version=state["version"], goal_revision=state["goal_revision"],
                    replan_count=state["replans"], patch=patch.model_dump())
    return candidate


def _present(context, recorder, intake, state, plan, *, reason="COMPLETED", message=None, error=None, pending_action=None):
    active = [s for s in plan.steps if s.status != "CANCELLED"]
    selected = [s for s in active if s.tool == _SELECT and s.status in _DONE]
    satisfied = reason == "COMPLETED" and bool(selected) and all(s.status in _DONE for s in active)
    plan.verification = dict(all_goals_satisfied=satisfied, satisfied_steps=[s.step_id for s in active if s.status in _DONE],
                             incomplete_steps=[s.step_id for s in active if s.status not in _DONE], waiting_steps=[],
                             side_effects_executed=False, goal_revision=state["goal_revision"])
    _save(context, state, plan)
    recorder.record("verification", **plan.verification)
    if error:
        status = "PARTIAL" if any(s.status in _DONE for s in active) else "FAILED"
    else:
        status = "SUCCEEDED" if satisfied else "WAITING_CLARIFICATION"
    if message is None and selected:
        product = selected[-1].result["selected_product"]
        latest_compare = next(s for s in active if s.step_id == selected[-1].depends_on[0])
        products = latest_compare.result["products"]
        comparisons = "\n".join(f"- {p['name']}（{p['product_id']}）：¥{p['price']:.2f}，{p['summary']}" for p in products)
        message = (f"按你的条件，在本地演示目录中选择 {product['name']}（{product['product_id']}），"
                   f"价格 ¥{product['price']:.2f}，当前有货。\n{comparisons}\n选择依据：满足预算和已明确的商品条件，优先价格较低的商品。")
        facts = [_facts(s) for s in active if s.specialist != "shopping" and s.status in _DONE]
        if facts:
            message = "\n\n".join([*facts, message])
    trace = [dict(run_id=context.run_id, agent="shopping" if e["tool"] in {"search_products", "compare_products"} else "orchestrator",
                  tool=e["tool"], args=e.get("args", {}), result=e.get("result"), status=e.get("status"), duration_ms=e.get("duration_ms"))
             for e in context.events if e["type"] == "tool_result"]
    result = _finish(recorder, intake, context.run_id, "shopping", trace, reply=message or "请补充商品条件后继续。",
                     status=status, terminal_reason=reason, pending_action=pending_action, error=error)
    result["route"] = "ADAPTIVE"
    result["plan"] = dict(plan.payload(), version=state["version"], goal_revision=state["goal_revision"],
                          replan_count=state["replans"], journal=deepcopy(state["journal"]), goal=deepcopy(state["goal"]))
    return result


def _run(context, user_message, history, pending_action):
    from src.orchestrator import resolve_followup, run_orchestrator
    request = resolve_followup(user_message, history)
    intake = analyze_request(request)
    state = deepcopy(context.snapshot.get("adaptive"))
    recorder = TraceRecorder(context.run_id)
    revision = context.snapshot.get("goal_revision", 0)
    if pending_action or (not state and (PAYMENT_PROCESS in intake.intents or REFUND_REQUEST in intake.intents)):
        return run_orchestrator(user_message, history, pending_action)
    if state:
        plan = _restore_plan(state["plan"])
        intake = TaskIntake(**{**state["intake"], "intents": tuple(state["intake"]["intents"]),
                               "missing_slots": tuple(state["intake"].get("missing_slots", []))})
        if revision == state["goal_revision"] and request != state["request"]:
            raise AgentError("CHECKPOINT_CONFLICT", "恢复请求与已保存目标不一致", retryable=False)
    else:
        planned = None
        if intake.execution_mode == LLM_REQUIRED:
            from src.planner import create_model_plan
            intake, planned = create_model_plan(request, history)
            if planned:
                shopping = [s for s in planned.steps if s.tool == "search_products"]
                if len(shopping) == 1:
                    budget = shopping[0].args.get("max_price")
                    numbers = {float(value) for value in re.findall(r"\d+(?:\.\d+)?", request)}
                    if budget is not None and budget not in numbers:
                        raise AgentError("UNGROUNDED_PLAN_CONSTRAINT", "模型预算没有来自用户提供的金额", retryable=False)
                    ids = list(dict.fromkeys(p for s in planned.steps if s.tool == 'compare_products' for p in s.args.get('product_ids', [])))
                    intake = TaskIntake(request, tuple(planned.goals), {**intake.entities, **shopping[0].args, 'product_ids': ids}, execution_mode=DIRECT,
                                        reason="模型识别目标，商品选择由真实工具证据验证")
        if PRODUCT_SEARCH not in intake.intents or intake.execution_mode == CLARIFICATION_REQUIRED:
            if planned:
                from src.task_plan import execute_plan
                return execute_plan(intake, context.run_id, plan=planned, source="model")
            return run_orchestrator(user_message, history, pending_action)
        if any(intent in intake.intents for intent in (PAYMENT_PROCESS, REFUND_REQUEST)):
            if planned:
                from src.task_plan import execute_plan
                return execute_plan(intake, context.run_id, plan=planned, source="model")
            return run_orchestrator(user_message, history, pending_action)
        goal = _goal(request, intake)
        additions = context.snapshot.get("steering", [])
        if additions:
            goal = _goal(request, intake, previous=goal, additions=additions)
        plan = planned or build_plan(intake)
        searches = [s for s in plan.steps if s.tool == "search_products"]
        if len(searches) != 1:
            raise AgentError("AMBIGUOUS_SHOPPING_GOALS", "请一次提供一个商品选择目标", retryable=False)
        # An explicit comparison is resolved only after search evidence exists.
        plan.steps = [s for s in plan.steps if s.tool != "compare_products"]
        search = searches[0]
        search.args = _search_args(goal)
        next_id = 1 + max([int(s.step_id.split('-')[-1]) for s in plan.steps if re.fullmatch(r'step-\d+', s.step_id)] or [0])
        compare_id = f"step-{next_id}"
        plan.steps += [TaskStep(compare_id, "shopping", "compare_products", {}, [search.step_id]),
                       TaskStep(f"step-{next_id + 1}", "shopping", _SELECT, {}, [compare_id])]
        plan.validate(context.limits.max_steps, context.limits.max_depth)
        state = dict(version=1, replans=0, goal_revision=revision, request=request, intake=intake.to_dict(),
                     goal=goal, journal=[], scopes={s.step_id: f"adaptive:{revision}:0" for s in plan.steps},
                     steering_count=len(context.snapshot.get("steering", [])))
        _save(context, state, plan)
        recorder.record("plan_created", source="adaptive", version=1, goal_revision=revision, plan=plan.payload(), goal=goal)
    try:
        if revision != state["goal_revision"]:
            additions = context.snapshot.get("steering", [])[state.get("steering_count", 0):]
            if not additions:
                raise AgentError("CHECKPOINT_CONFLICT", "目标版本变化但没有对应补充要求", retryable=False)
            goal = _goal(request, intake, previous=state["goal"], additions=additions)
            state["goal_revision"] = revision
            state["request"] = request
            state["steering_count"] = len(context.snapshot.get("steering", []))
            plan = _replace_remaining(context, recorder, state, plan, goal, "USER_CONSTRAINT_CHANGED")
        if state["goal"].get("conflict"):
            return _present(context, recorder, intake, state, plan, reason="CONSTRAINT_CONFLICT", message=state["goal"]["conflict"])
        while True:
            context.require()
            pending = next((s for s in plan.steps if s.status not in _DONE and s.status != "CANCELLED"), None)
            if pending is None:
                return _present(context, recorder, intake, state, plan, pending_action=pending_action)
            step = pending
            completed = {s.step_id: s for s in plan.steps}
            if any(completed[d].status not in _DONE for d in step.depends_on):
                raise AgentError("DEPENDENCY_FAILED", "前置步骤未完成，已停止后续执行", retryable=False)
            recorder.record("step_started", step_id=step.step_id, specialist=step.specialist, dependencies=step.depends_on)
            if step.tool == "compare_products":
                source = completed[step.depends_on[0]]
                products = [p for p in source.result["products"] if _matches(p, state["goal"])]
                requested = state["goal"].get("product_ids", [])
                if requested:
                    products = [p for p in products if p["product_id"] in requested]
                if not products or (requested and not set(requested) <= {p["product_id"] for p in products}):
                    return _present(context, recorder, intake, state, plan, reason="NO_MATCHING_PRODUCTS",
                                    message="本地演示目录中没有同时满足当前预算、库存和商品条件的商品。请明确调整哪项条件后继续；预算和条件已保留。")
                products.sort(key=lambda p: (p["price"], p["product_id"]))
                products = products[:3]
                if len(products) == 1:
                    step.status = "NOT_NEEDED"
                    step.result = dict(products=products, reason="SINGLE_MATCH", evidence_step=source.step_id)
                    _save(context, state, plan)
                    recorder.record("step_completed", step_id=step.step_id, specialist="shopping", status=step.status)
                    continue
                step.args = dict(product_ids=[p["product_id"] for p in products])
                _save(context, state, plan)
            if step.tool == _SELECT:
                source = completed[step.depends_on[0]]
                eligible = [p for p in source.result["products"] if _matches(p, state["goal"])]
                if not eligible:
                    recorder.record("verification_failed", step_id=step.step_id, reason="PRODUCT_CONSTRAINT_CHANGED",
                                    evidence_step=source.step_id)
                    plan = _replace_remaining(context, recorder, state, plan, state["goal"], "PRODUCT_CONSTRAINT_CHANGED")
                    continue
                product = min(eligible, key=lambda p: (p["price"], p["product_id"]))
                step.result = dict(selected_product=deepcopy(product), evidence_step=source.step_id)
                step.status = "SUCCEEDED"
            else:
                registry = {name: value for name, value in _READ_ONLY_TOOLS.items() if name != _SELECT}
                executor = ToolExecutor(registry, recorder, max_calls=context.limits.max_tool_calls,
                                        observation_scope=state["scopes"][step.step_id])
                if executor.registry[step.tool].effect != ToolEffect.READ:
                    raise AgentError("INVALID_PLAN_PATCH", "自主管理的步骤只能调用只读工具", retryable=False)
                step.result, event = executor.execute(step.tool, step.args)
                if event["status"] == "BUSINESS_ERROR":
                    raise AgentError("BUSINESS_DATA_NOT_FOUND", step.result["error"], retryable=False)
                if step.tool in {"search_products", "compare_products"}:
                    _verify_products(step)
                else:
                    verify_observation(step)
                step.status = event["status"]
            _save(context, state, plan)
            recorder.record("step_completed", step_id=step.step_id, specialist=step.specialist, status=step.status)
    except AgentError as error:
        return _present(context, recorder, intake, state, plan, reason=error.code,
                        message="运行已停止，已保留完成步骤和原始条件。" + str(error),
                        error=dict(code=error.code, message=str(error), retryable=error.retryable), pending_action=pending_action)


def run_adaptive(user_message, history=None, pending_action=None, *, limits=None):
    """Production entry point; uses the same runtime and result contract as auto."""
    context = current_runtime()
    if context is None:
        context = RunRuntime(uuid.uuid4().hex, limits or BudgetLimits())
        with runtime_scope(context):
            return run_adaptive(user_message, history, pending_action)
    started = time.perf_counter()
    context.record("intake", request=user_message, execution_mode="adaptive")
    try:
        result = _run(context, user_message, history, pending_action)
    except Exception as exc:
        error = classify_exception(exc, source="model")
        recorder = TraceRecorder(context.run_id)
        result = _finish(recorder, analyze_request(user_message), context.run_id, "runtime", [], reply=str(error),
                         status="FAILED", terminal_reason=error.code, pending_action=pending_action,
                         error=dict(code=error.code, message=str(error), retryable=error.retryable))
        result["route"] = "ADAPTIVE"
    result["events"] = context.events
    from src.agents.base import sum_usage
    calls = [e for e in context.events if e["type"] == "model_call"]
    if calls:
        result["usage"] = sum_usage(*(e.get("usage") for e in calls))
    elif any(e["type"] == "model_started" for e in context.events):
        result["usage"] = dict(input_tokens=None, output_tokens=None, total_tokens=None)
    result["budget"] = dict(limits=context.limits.__dict__, consumed=dict(context.counters))
    result["execution_mode"] = "adaptive"
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result
