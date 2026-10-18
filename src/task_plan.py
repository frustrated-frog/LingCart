"""Bounded deterministic goals, specialist steps, dependencies and evidence checks."""
from dataclasses import asdict, dataclass, field
import math

from src.capability_registry import MODEL_ANSWER_OPERATION, get_capability_registry
from src.direct_executor import _READ_ONLY_TOOLS, _STATUS_ZH, _TRACKING_ZH, _finish, _trace_entry
from src.errors import AgentError
from src.intake import (FAQ, ORDER_DETAILS, ORDER_STATUS, PAYMENT_PROCESS, PAYMENT_STATUS, PRODUCT_COMPARE,
                        PRODUCT_DISCOVERY, PRODUCT_SEARCH, REFUND_ELIGIBILITY, REFUND_REQUEST, SHIPMENT_TRACKING)
from src.runtime import ToolExecutor, TraceRecorder, current_runtime


@dataclass
class TaskStep:
    step_id: str
    specialist: str
    tool: str
    args: dict
    depends_on: list[str] = field(default_factory=list)
    capability_id: str | None = None
    objective_id: str | None = None
    status: str = "PENDING"
    result: object = None
    error: dict | None = None


@dataclass
class TaskPlan:
    goals: list[str]
    steps: list[TaskStep]
    verification: dict = field(default_factory=dict)

    def payload(self):
        return asdict(self)

    def validate(self, max_steps, max_depth):
        if len(self.steps) > max_steps:
            raise AgentError("PLAN_STEP_BUDGET_EXHAUSTED", "请求包含过多步骤，请拆成较小的任务", retryable=False)
        depths = {}
        for step in self.steps:
            if step.step_id in depths or any(dependency not in depths for dependency in step.depends_on):
                raise AgentError("INVALID_TASK_PLAN", "计划包含重复步骤、循环或未知依赖", retryable=False)
            depths[step.step_id] = 1 + max([depths[d] for d in step.depends_on] or [0])
            if depths[step.step_id] > max_depth:
                raise AgentError("PLAN_DEPTH_BUDGET_EXHAUSTED", "任务依赖层数超出限制", retryable=False)


def build_plan(intake):
    steps = []
    known = {}
    def add(actor, tool, args, dependencies=None):
        identity = ToolExecutor.fingerprint(tool, args)
        if identity in known:
            return known[identity]
        step = TaskStep(f"step-{len(steps) + 1}", actor, tool, args, dependencies or [])
        steps.append(step)
        known[identity] = step.step_id
        return step.step_id
    intents = intake.intents
    order_ids = intake.entities.get("order_ids") or ([intake.entities["order_id"]] if "order_id" in intake.entities else [])
    for order_id in order_ids:
        args = {"order_id": order_id}
        if ORDER_DETAILS in intents:
            add("order", "get_order_details", args)
        if ORDER_STATUS in intents:
            add("order", "get_order_status", args)
        if SHIPMENT_TRACKING in intents:
            add("order", "track_shipment", args)
        if REFUND_ELIGIBILITY in intents or REFUND_REQUEST in intents:
            state = add("order", "get_order_status", args)
            eligibility = add("refund", "check_refund_eligibility", args, [state])
            if REFUND_REQUEST in intents:
                add("refund", "prepare_refund", args, [eligibility])
        if PAYMENT_STATUS in intents or PAYMENT_PROCESS in intents:
            payment = add("payment", "get_payment_status", args)
            if PAYMENT_PROCESS in intents:
                add("payment", "prepare_payment", args, [payment])
    if FAQ in intents:
        add("faq", "answer_faq", {"question": intake.request})
    if PRODUCT_SEARCH in intents:
        args = {key: intake.entities[key] for key in ("query", "category", "max_price") if intake.entities.get(key) is not None}
        add("shopping", "search_products", args)
    if PRODUCT_DISCOVERY in intents:
        args = {key: intake.entities[key] for key in ("query", "category", "max_price", "soft_preferences")
                if intake.entities.get(key) not in (None, "", [])}
        args["context_seed"] = intake.request
        add("shopping", "discover_products", args)
    if PRODUCT_COMPARE in intents:
        add("shopping", "compare_products", {"product_ids": intake.entities.get("product_ids", [])})
    return TaskPlan(list(intents), steps)


def _facts(step):
    result, name, order_id = step.result, step.tool, step.args.get("order_id")
    if step.status in {"FAILED", "SKIPPED"}:
        return f"{step.specialist} 步骤未完成：{(step.error or {}).get('message', '前置查询未完成')}"
    if name == "answer_faq":
        return result['answer']
    if name == "read_clock":
        weekdays = {1: "星期一", 2: "星期二", 3: "星期三", 4: "星期四", 5: "星期五", 6: "星期六", 7: "星期日"}
        return (f"当前日期是 {result['date']}，{weekdays[result['weekday']]}，"
                f"时间 {result['time']}（{result['timezone']}）。")
    if name == "calculate":
        return f"{result['expression']} = {result['result']}"
    if name == "get_order_details":
        return f"订单 {order_id} 的商品是 {result['item']}，金额 ¥{result['amount']:.2f}，当前{_STATUS_ZH.get(result['status'], result['status'])}。"
    if name == "get_order_status":
        return f"订单 {order_id} 当前{_STATUS_ZH.get(result['status'], result['status'])}。"
    if name == "track_shipment":
        return f"订单 {order_id} 的物流进度：{_TRACKING_ZH.get(result['tracking_message'], result['tracking_message'])}。"
    if name == "check_refund_eligibility":
        return f"订单 {order_id} 目前符合退款条件；尚未发起退款。" if result['eligible'] else f"订单 {order_id} 目前不符合退款条件：{result['reason']}"
    if name == "get_payment_status":
        return f"订单 {order_id} 的支付状态是“{_STATUS_ZH.get(result['status'], result['status'])}”，金额 ¥{result['amount']:.2f}。"
    if name.startswith("prepare_"):
        return result["message"]
    if name == MODEL_ANSWER_OPERATION:
        return result["answer"]
    if name in {"search_products", "discover_products", "compare_products"}:
        products = result.get("products", [])
        if not products:
            return "当前本地演示目录中没有可验证的候选商品。可以调整预算或条件后继续。"
        if name == "discover_products":
            prefix = "我先根据当前真实库存给你一组方向不同的探索推荐："
        else:
            prefix = f"在本地演示目录中找到 {result['total']} 款符合条件的商品：" if name == "search_products" else "所选商品的实际参数："
        reply = prefix + "\n" + "\n".join(
            f"- {p['name']}（{p['product_id']}）：¥{p['price']:.2f}，{p['summary']}" for p in products[:4]
        )
        if name == "discover_products":
            reply += "\n\n这些是探索候选，不代表我已经知道你的个人偏好。"
            if result.get("unbound_catalog_language"):
                reply += "\n我没有从这句话中识别到可验证的商品属性，因此先按品类、价位和规格差异提供候选。"
            if result.get("follow_up_question"):
                reply += "\n" + result["follow_up_question"]
        return reply
    return str(result)


def verify_observation(step):
    """Evidence must refer to this goal and contain the fields used in replies."""
    result, name = step.result, step.tool
    required = {
        "read_clock": {"date", "time", "weekday", "timezone", "observed_at", "data_source"},
        "calculate": {"expression", "result", "data_source"},
        "get_order_status": {"order_id", "status"},
        "get_order_details": {"order_id", "status", "item", "amount"},
        "track_shipment": {"order_id", "status", "tracking_message"},
        "check_refund_eligibility": {"order_id", "eligible", "reason"},
        "get_payment_status": {"order_id", "status", "amount", "method"},
        "search_products": {"products", "total"},
        "discover_products": {"products", "total", "candidate_set", "interaction_policy", "data_source"},
        "compare_products": {"products"},
        MODEL_ANSWER_OPERATION: {"answer", "question", "evidence_requirement", "provenance"},
    }
    valid = isinstance(result, dict) and required.get(name, set()) <= result.keys()
    if name == 'answer_faq':
        if not isinstance(result, dict) or result.get('status') != 'ANSWERED':
            raise AgentError('INSUFFICIENT_POLICY_EVIDENCE', '政策没有回答当前问题，不能标记完成', retryable=False)
        valid = (result.get('question') == step.args.get('question') and bool(result.get('answer'))
                 and isinstance(result.get('policies'), list) and bool(result['policies'])
                 and all(isinstance(p, dict) and all(p.get(k) for k in ('policy_id', 'version', 'text', 'source')) for p in result['policies']))
    if valid and step.args.get("order_id") is not None:
        valid = result.get("order_id") == step.args["order_id"]
    if valid and name in {"get_order_status", "get_order_details", "track_shipment"}:
        valid = result["status"] in {"placed", "shipped", "delivered", "cancelled"}
    if valid and name == "get_payment_status":
        valid = result["status"] in {"pending", "paid", "refunded"} and isinstance(result["method"], str)
    if valid and name in {"get_payment_status", "get_order_details"}:
        valid = isinstance(result["amount"], (int, float)) and math.isfinite(result["amount"]) and result["amount"] >= 0
    if valid and name == "check_refund_eligibility":
        valid = isinstance(result["eligible"], bool) and isinstance(result["reason"], str)
    if valid and name == "track_shipment":
        valid = isinstance(result["tracking_message"], str)
    if valid and name == "read_clock":
        valid = (result.get("data_source") == "RUNTIME_CLOCK" and result.get("weekday") in range(1, 8)
                 and isinstance(result.get("observed_at"), str))
    if valid and name == "calculate":
        valid = (result.get("data_source") == "DETERMINISTIC_CALCULATOR"
                 and result.get("expression") == step.args.get("expression")
                 and type(result.get("result")) in {int, float} and math.isfinite(result["result"]))
    if valid and name in {"search_products", "discover_products", "compare_products"}:
        valid = isinstance(result["products"], list) and all(isinstance(p, dict) and {"product_id", "name", "price", "summary"} <= p.keys() for p in result["products"])
    if valid and name == "discover_products":
        candidate_set = result["candidate_set"]
        valid = (isinstance(candidate_set, dict)
                 and {"candidate_set_id", "catalog_snapshot", "candidate_ids", "selected_ids", "score_components"} <= candidate_set.keys()
                 and candidate_set["selected_ids"] == [product["product_id"] for product in result["products"]]
                 and set(candidate_set["selected_ids"]) <= set(candidate_set["candidate_ids"])
                 and result.get("data_source") == "DEMO_CATALOG")
    if valid and name == "compare_products":
        valid = {p["product_id"] for p in result["products"]} == set(step.args["product_ids"])
    if valid and name == MODEL_ANSWER_OPERATION:
        valid = (result.get("question") == step.args.get("question") and bool(result.get("answer"))
                 and result.get("provenance") in {"MODEL_GENERAL_KNOWLEDGE", "NO_REALTIME_SOURCE"})
    if not valid:
        raise AgentError("INVALID_TOOL_RESULT", "工具结果未包含本步骤所需的业务证据", retryable=False)


def execute_plan(intake, run_id, *, plan=None, source="deterministic", incomplete_reason=None):
    context = current_runtime()
    recorder = TraceRecorder(run_id)
    plan = plan or build_plan(intake)
    plan.validate(context.limits.max_steps if context else 8, context.limits.max_depth if context else 4)
    proposals = [step for step in plan.steps if step.tool.startswith("prepare_")]
    if len(proposals) > 1:
        # One explicit confirmation must never authorize several independent writes.
        return _finish(recorder, intake, run_id, "intake", [], reply="请分别提交每个订单的支付或退款申请，逐项确认。",
                       status="WAITING_CLARIFICATION", terminal_reason="MULTIPLE_ACTIONS_REQUIRE_SEPARATE_APPROVAL")
    actors = set(step.specialist for step in plan.steps)
    agent = next(iter(actors)) if len(actors) == 1 else ("refund" if actors <= {"order", "refund"} else "orchestrator")
    recorder.record("route_decision", source=source, route="DIRECT", intents=list(intake.intents), confidence=intake.confidence)
    # Restore the same plan shape. Observations are rehydrated through the ledger.
    if context and context.snapshot.get("plan_definition"):
        definition = dict(goals=plan.goals, steps=[dict(step_id=s.step_id, specialist=s.specialist,
            tool=s.tool, args=s.args, depends_on=s.depends_on, capability_id=s.capability_id,
            objective_id=s.objective_id) for s in plan.steps])
        if definition != context.snapshot["plan_definition"]:
            raise AgentError("CHECKPOINT_CONFLICT", "恢复时的任务计划发生变化", retryable=False)
    else:
        definition = dict(goals=plan.goals, steps=[dict(step_id=s.step_id, specialist=s.specialist,
            tool=s.tool, args=s.args, depends_on=s.depends_on, capability_id=s.capability_id,
            objective_id=s.objective_id) for s in plan.steps])
        if context:
            context.checkpoint(plan_definition=definition)
        if len(plan.steps) > 1:
            recorder.record("plan_created", plan=definition)
    registered_tools = get_capability_registry().execution_tools()
    # The direct registry remains the local override seam used by harnesses and
    # tests; capability-only operations are added without replacing it.
    executor = ToolExecutor({**registered_tools, **_READ_ONLY_TOOLS}, recorder,
                            max_calls=context.limits.max_tool_calls if context else 12)
    trace, completed, pending, errors = [], {}, None, []
    stop_error = None
    for step in plan.steps:
        dependency_failed = any(completed[d].status not in {"SUCCEEDED", "REUSED", "WAITING_APPROVAL", "REJECTED"} for d in step.depends_on)
        if dependency_failed or stop_error:
            step.status = "SKIPPED"
            step.error = stop_error or {"code": "DEPENDENCY_FAILED", "message": "前置查询失败，未继续执行依赖步骤", "retryable": False}
            recorder.record("step_skipped", step_id=step.step_id, specialist=step.specialist, error=step.error)
        else:
            recorder.record("step_started", step_id=step.step_id, specialist=step.specialist, dependencies=step.depends_on)
            try:
                if step.tool.startswith("prepare_"):
                    observation = completed[step.depends_on[0]].result
                    eligible = observation.get("eligible") if step.tool == "prepare_refund" else observation.get("status") == "pending"
                    if eligible:
                        pending = dict(type="INITIATE_REFUND" if step.tool == "prepare_refund" else "PROCESS_PAYMENT",
                                       order_id=step.args["order_id"], requires_approval=True)
                        if step.tool == "prepare_refund":
                            pending["reason"] = intake.request
                        else:
                            pending.update(amount=observation["amount"], method=observation.get("method"))
                        step.status = "WAITING_APPROVAL"
                        step.result = {"message": f"订单 {step.args['order_id']} 的操作尚未执行，请明确确认后继续。"}
                        recorder.record("approval_required", step_id=step.step_id, action=pending)
                    else:
                        step.status = "REJECTED"
                        step.result = {"message": f"订单 {step.args['order_id']} 当前不满足操作条件，无需重复提交。"}
                elif step.tool == MODEL_ANSWER_OPERATION:
                    from src.general_answer import generate_general_answer
                    step.result = generate_general_answer(**step.args)
                    step.status = "SUCCEEDED"
                    recorder.record("knowledge_result", step_id=step.step_id,
                                    evidence_requirement=step.args.get("evidence_requirement"),
                                    provenance=step.result.get("provenance"))
                    verify_observation(step)
                else:
                    step.result, event = executor.execute(step.tool, step.args)
                    trace.append(_trace_entry(run_id, step.specialist, step.tool, step.args, step.result, event))
                    step.status = event["status"]
                    if step.status == "BUSINESS_ERROR":
                        code = ("INVALID_PRODUCT_SELECTION" if step.tool == "compare_products"
                                else "INVALID_UTILITY_REQUEST" if step.tool in {"read_clock", "calculate"}
                                else "BUSINESS_DATA_NOT_FOUND")
                        raise AgentError(code, step.result["error"], retryable=False)
                    verify_observation(step)
            except AgentError as error:
                step.status = "FAILED"
                step.error = {"code": error.code, "message": str(error), "retryable": error.retryable}
                errors.append(step.error)
                if "BUDGET" in error.code or error.code in {"RUN_LEASE_LOST", "UNCERTAIN_TOOL_OUTCOME"}:
                    stop_error = step.error
            recorder.record("step_completed", step_id=step.step_id, specialist=step.specialist, status=step.status, error=step.error)
        completed[step.step_id] = step
        if context:
            context.phase = "PLAN"
            context.checkpoint(plan=plan.payload())
    satisfied = [step.step_id for step in plan.steps if step.status in {"SUCCEEDED", "REUSED", "WAITING_APPROVAL", "REJECTED"}]
    missing = [step.step_id for step in plan.steps if step.step_id not in satisfied]
    plan.verification = dict(satisfied_steps=satisfied, incomplete_steps=missing, waiting_steps=[s.step_id for s in plan.steps if s.status == "WAITING_APPROVAL"],
                             all_goals_satisfied=bool(plan.steps) and not missing and pending is None,
                             side_effects_executed=False)
    recorder.record("verification", **plan.verification)
    status = "WAITING_APPROVAL" if pending else ("PARTIAL" if satisfied else "FAILED") if missing else "SUCCEEDED"
    if not plan.steps:
        status = "FAILED"
        errors.append({"code": "EMPTY_TASK_PLAN", "message": "没有可验证的业务步骤", "retryable": False})
    reason = errors[-1]["code"] if errors else "APPROVAL_REQUIRED" if pending else "BUSINESS_RULE_REJECTED" if any(s.status == "REJECTED" for s in plan.steps) else "COMPLETED"
    if not errors and not pending and any(s.tool == "prepare_payment" and s.status == "REJECTED" for s in plan.steps):
        reason = "NO_ACTION_NEEDED"
    # An empty search is a verified outcome, not a claim that products exist.
    if not errors and not pending and len(plan.steps) == 1 and plan.steps[0].tool in {"search_products", "discover_products"} and not plan.steps[0].result.get("products"):
        reason = "NO_CANDIDATE_EVIDENCE" if plan.steps[0].tool == "discover_products" else "NO_MATCHING_PRODUCTS"
    if incomplete_reason and status == "SUCCEEDED":
        status, reason = "PARTIAL", incomplete_reason
    response = _finish(recorder, intake, run_id, agent, trace, reply="\n\n".join(_facts(s) for s in plan.steps) or "没有可执行的业务步骤。",
                       status=status, terminal_reason=reason, pending_action=pending, error=errors[-1] if errors else None)
    if any(step.tool == MODEL_ANSWER_OPERATION for step in plan.steps):
        response["route"] = "KNOWLEDGE" if len(plan.steps) == 1 else "MIXED"
    response["plan"] = plan.payload()
    return response
