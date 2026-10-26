"""Bounded deterministic goals, specialist steps, dependencies and evidence checks."""
from dataclasses import asdict, dataclass, field
import math

from src.capability_registry import MODEL_ANSWER_OPERATION, get_capability_registry
from src.catalog_query import (
    CATALOG_QUERY_OPERATION,
)
from src.constraint_planning import CONSTRAINT_PLAN_OPERATION
from src.direct_executor import _READ_ONLY_TOOLS, _STATUS_ZH, _TRACKING_ZH, _finish, _trace_entry
from src.errors import AgentError
from src.intake import (FAQ, ORDER_DETAILS, ORDER_STATUS, PAYMENT_PROCESS, PAYMENT_STATUS, PRODUCT_COMPARE,
                        PRODUCT_DISCOVERY, PRODUCT_SEARCH, REFUND_ELIGIBILITY, REFUND_REQUEST, SHIPMENT_TRACKING)
from src.runtime import ToolExecutor, TraceRecorder, current_runtime
from src.after_sales.execution import ACTION_PLANNER_VERSION, decide_next_case_action, project_case_execution_state
from src.after_sales.obligation_graph import (
    ACTIVE_CLARIFICATION_PLANNER_VERSION,
    OBLIGATION_ACTION_PLANNER_VERSION,
    abstain_obligation_action,
    build_obligation_graph,
    decide_next_obligation_action,
)
from src.after_sales.clarification import (
    build_clarification_portfolio,
    solve_active_clarification,
)
from src.after_sales.pending_clarification import build_pending_clarification


_PROPOSAL_OPERATIONS = {"prepare_refund", "prepare_payment", "prepare_case_submission"}
_POLICY_AWARE_CASE_OPERATIONS = {
    "prepare_after_sales_case",
    "continue_after_sales_case",
    "record_after_sales_evidence_readiness",
}
_MAX_AUTONOMOUS_CASE_ACTIONS = 4


def _case_execution_mode(context) -> str:
    mechanism = (context.mechanisms or {}).get("planning") if context else None
    return getattr(mechanism, "case_execution_mode", "policy_aware")


def _case_action_selection_mode(context) -> str:
    mechanism = (context.mechanisms or {}).get("planning") if context else None
    return getattr(mechanism, "action_selection_mode", "priority_v2")


@dataclass
class TaskStep:
    step_id: str
    specialist: str
    tool: str
    args: dict
    depends_on: list[str] = field(default_factory=list)
    capability_id: str | None = None
    objective_id: str | None = None
    objective_ids: list[str] = field(default_factory=list)
    argument_provenance: dict[str, dict] = field(default_factory=dict)
    status: str = "PENDING"
    outcome: str = "PENDING"
    result: object = None
    error: dict | None = None


@dataclass
class TaskPlan:
    goals: list[str]
    steps: list[TaskStep]
    verification: dict = field(default_factory=dict)

    def payload(self):
        payload = asdict(self)
        for step in payload["steps"]:
            if not step.get("objective_ids"):
                step.pop("objective_ids", None)
            if not step.get("argument_provenance"):
                step.pop("argument_provenance", None)
            if step.get("outcome") == "PENDING":
                step.pop("outcome", None)
        return payload

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
        args = {key: intake.entities[key] for key in ("query", "category", "max_price", "excluded_terms")
                if intake.entities.get(key) not in (None, [])}
        add("shopping", "search_products", args)
    if PRODUCT_DISCOVERY in intents:
        args = {key: intake.entities[key] for key in ("query", "category", "max_price", "soft_preferences", "excluded_terms")
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
    contract = get_capability_registry().operation_contract(name)
    if contract is not None:
        return contract.render_result(result, step.args)
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
    contract = get_capability_registry().operation_contract(name)
    if contract is not None:
        contract.verify_result(result, step.args)
        return
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


def _advance_after_sales_case(
    result,
    *,
    executor,
    recorder,
    trace,
    run_id,
    max_actions,
    action_selection_mode="priority_v2",
):
    """Consume persisted CasePlan actions without creating a second runtime or bypassing ToolExecutor."""
    current = result
    executed = []
    initial_state_id = None
    final_state_id = None
    final_status = "ABSTAINED"
    final_reason = "CASE_STATE_NOT_EXECUTABLE"
    graph_ids = []
    clarification_portfolio_ids = []
    clarification_decision_ids = []
    final_question = None
    final_question_id = None
    pending_clarification = None
    planner_version = ACTION_PLANNER_VERSION
    for index in range(max_actions + 1):
        state = project_case_execution_state(current)
        initial_state_id = initial_state_id or state.state_id
        final_state_id = state.state_id
        recorder.record(
            "case_state_projected",
            state=state.model_dump(mode="json"),
            autonomous_action_index=index,
        )
        if action_selection_mode in {"obligation_graph", "active_clarification"}:
            graph = build_obligation_graph(state) if state.status == "READY" else None
            clarification_portfolio = None
            clarification_decision = None
            if action_selection_mode == "active_clarification" and graph is not None:
                clarification_portfolio = build_clarification_portfolio(state, graph, current)
                clarification_decision = solve_active_clarification(graph, clarification_portfolio)
                if graph.status != "READY":
                    directive = decide_next_obligation_action(
                        state,
                        graph,
                        planner_version=ACTIVE_CLARIFICATION_PLANNER_VERSION,
                    )
                elif clarification_decision.status == "INVALID":
                    directive = abstain_obligation_action(
                        state,
                        graph,
                        clarification_decision.reason_code,
                        planner_version=ACTIVE_CLARIFICATION_PLANNER_VERSION,
                    )
                else:
                    directive = decide_next_obligation_action(
                        state,
                        graph,
                        selection=clarification_decision.candidate_selection,
                        planner_version=ACTIVE_CLARIFICATION_PLANNER_VERSION,
                    )
                planner_version = ACTIVE_CLARIFICATION_PLANNER_VERSION
            else:
                planner_version = (
                    ACTIVE_CLARIFICATION_PLANNER_VERSION
                    if action_selection_mode == "active_clarification"
                    else OBLIGATION_ACTION_PLANNER_VERSION
                )
                directive = decide_next_obligation_action(
                    state,
                    graph,
                    planner_version=planner_version,
                )
            if graph is not None:
                graph_ids.append(graph.graph_id)
                recorder.record(
                    "case_obligation_graph_solved",
                    graph=graph.model_dump(mode="json"),
                    selection=(
                        directive.selection.model_dump(mode="json")
                        if directive.selection is not None else None
                    ),
                    directive_id=directive.directive_id,
                    autonomous_action_index=index,
                )
            if clarification_portfolio is not None and clarification_decision is not None:
                clarification_portfolio_ids.append(clarification_portfolio.portfolio_id)
                clarification_decision_ids.append(clarification_decision.decision_id)
                final_question_id = clarification_decision.selected_question_id
                recorder.record(
                    "case_clarification_portfolio_solved",
                    portfolio=clarification_portfolio.model_dump(mode="json"),
                    decision=clarification_decision.model_dump(mode="json"),
                    directive_id=directive.directive_id,
                    autonomous_action_index=index,
                )
        elif action_selection_mode == "priority_v2":
            directive = decide_next_case_action(state)
        else:
            raise AgentError(
                "HARNESS_ACTION_SELECTION_UNKNOWN",
                "售后 Harness 配置了未知的候选行动求解模式",
                retryable=False,
            )
        recorder.record(
            "case_obligation_verified",
            directive=directive.model_dump(mode="json"),
            autonomous_action_index=index,
        )
        final_status, final_reason = directive.status, directive.reason_code
        final_question = directive.question
        if (
            action_selection_mode == "active_clarification"
            and clarification_portfolio is not None
            and clarification_decision is not None
            and clarification_decision.status == "SELECTED"
            and clarification_decision.selected_question_id
            and clarification_decision.candidate_selection is not None
            and directive.status == "WAIT_USER_INPUT"
            and directive.reason_code == "CUSTOMER_EVIDENCE_REQUIRED"
        ):
            selected_question = next(
                question for question in clarification_portfolio.questions
                if question.question_id == clarification_decision.selected_question_id
            )
            pending_clarification = build_pending_clarification(
                case_id=state.case_id,
                requirement_id=selected_question.requirement_id,
                question_id=selected_question.question_id,
                question=selected_question.question,
                case_revision=state.case_revision,
                policy_version_id=state.policy_version_id,
                policy_checksum=state.policy_checksum,
                policy_decision_id=state.policy_decision_id,
                state_id=state.state_id,
                graph_id=clarification_portfolio.graph_id,
                portfolio_id=clarification_portfolio.portfolio_id,
                clarification_decision_id=clarification_decision.decision_id,
                selection_id=clarification_decision.candidate_selection.selection_id,
            ).model_dump(mode="json")
            recorder.record(
                "case_clarification_opened",
                **pending_clarification,
                authorization_granted=False,
            )
        if directive.status != "EXECUTE":
            recorder.record(
                "case_execution_paused",
                case_id=state.case_id,
                case_revision=state.case_revision,
                state_id=state.state_id,
                status=directive.status,
                reason_code=directive.reason_code,
                question=directive.question,
                graph_id=getattr(directive, "graph_id", None),
                candidate_id=getattr(directive, "candidate_id", None),
            )
            break
        if index >= max_actions:
            final_status, final_reason = "ABSTAINED", "AUTONOMOUS_STEP_BUDGET_EXHAUSTED"
            recorder.record(
                "case_execution_paused",
                case_id=state.case_id,
                case_revision=state.case_revision,
                state_id=state.state_id,
                status=final_status,
                reason_code=final_reason,
                graph_id=getattr(directive, "graph_id", None),
                candidate_id=getattr(directive, "candidate_id", None),
            )
            break
        recorder.record(
            "case_autonomous_action_selected",
            case_id=state.case_id,
            case_revision=state.case_revision,
            state_id=state.state_id,
            directive_id=directive.directive_id,
            operation=directive.operation,
            arguments=directive.arguments,
            reason_code=directive.reason_code,
            graph_id=getattr(directive, "graph_id", None),
            candidate_id=getattr(directive, "candidate_id", None),
            selection_reason=(
                directive.selection.reason_code
                if getattr(directive, "selection", None) is not None else None
            ),
        )
        next_result, event = executor.execute(directive.operation, directive.arguments)
        trace.append(_trace_entry(
            run_id,
            "after_sales",
            directive.operation,
            directive.arguments,
            next_result,
            event,
        ))
        if isinstance(next_result, dict) and isinstance(next_result.get("clarification_binding"), dict):
            recorder.record("case_clarification_answer_bound", **next_result["clarification_binding"])
        contract = get_capability_registry().operation_contract(directive.operation)
        if contract is None:
            raise AgentError("UNKNOWN_CAPABILITY_OPERATION", "案件自主执行动作没有注册输出契约", retryable=False)
        contract.verify_result(next_result, directive.arguments)
        next_revision = next_result.get("revision") if isinstance(next_result, dict) else None
        if type(next_revision) is not int or next_revision <= state.case_revision:
            raise AgentError("CASE_AUTONOMOUS_NO_PROGRESS", "案件自主执行没有产生可验证的版本进展", retryable=False)
        executed.append({
            "directive_id": directive.directive_id,
            "operation": directive.operation,
            "arguments": directive.arguments,
            "from_revision": state.case_revision,
            "to_revision": next_revision,
            "tool_status": event["status"],
            "graph_id": getattr(directive, "graph_id", None),
            "candidate_id": getattr(directive, "candidate_id", None),
        })
        recorder.record(
            "case_autonomous_action_completed",
            case_id=state.case_id,
            directive_id=directive.directive_id,
            operation=directive.operation,
            from_revision=state.case_revision,
            to_revision=next_revision,
            tool_status=event["status"],
            graph_id=getattr(directive, "graph_id", None),
            candidate_id=getattr(directive, "candidate_id", None),
        )
        current = next_result
    return {
        **current,
        "autonomous_execution": {
            "schema_version": 2,
            "planner_version": planner_version,
            "action_selection_mode": action_selection_mode,
            "initial_state_id": initial_state_id,
            "final_state_id": final_state_id,
            "obligation_graph_ids": graph_ids,
            "clarification_portfolio_ids": clarification_portfolio_ids,
            "clarification_decision_ids": clarification_decision_ids,
            "selected_question_id": final_question_id,
            "question": final_question,
            "pending_clarification": pending_clarification,
            "executed_actions": executed,
            "status": final_status,
            "reason_code": final_reason,
        },
    }


def execute_plan(intake, run_id, *, plan=None, source="deterministic", incomplete_reason=None):
    context = current_runtime()
    recorder = TraceRecorder(run_id)
    plan = plan or build_plan(intake)
    plan.validate(context.limits.max_steps if context else 8, context.limits.max_depth if context else 4)
    proposals = [step for step in plan.steps if step.tool in _PROPOSAL_OPERATIONS]
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
            objective_id=s.objective_id, objective_ids=s.objective_ids,
            argument_provenance=s.argument_provenance) for s in plan.steps])
        if definition != context.snapshot["plan_definition"]:
            raise AgentError("CHECKPOINT_CONFLICT", "恢复时的任务计划发生变化", retryable=False)
    else:
        definition = dict(goals=plan.goals, steps=[dict(step_id=s.step_id, specialist=s.specialist,
            tool=s.tool, args=s.args, depends_on=s.depends_on, capability_id=s.capability_id,
            objective_id=s.objective_id, objective_ids=s.objective_ids,
            argument_provenance=s.argument_provenance) for s in plan.steps])
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
                if step.tool in _PROPOSAL_OPERATIONS:
                    observation = completed[step.depends_on[0]].result
                    if step.tool == "prepare_case_submission":
                        decision = observation.get("policy_decision") or {}
                        outcome = decision.get("outcome")
                        if observation.get("case_status") == "SUBMITTED":
                            step.status = "SUCCEEDED"
                            step.outcome = "SATISFIED"
                            step.result = {
                                "message": f"售后案件 {observation['case_id']} 已经提交，无需重复操作。"
                            }
                        elif outcome == "ELIGIBLE":
                            args = {
                                "case_id": observation["case_id"],
                                "expected_revision": observation["revision"],
                                "decision_id": decision["decision_id"],
                            }
                            pending = {
                                "type": "SUBMIT_AFTER_SALES_CASE",
                                "tool": "submit_after_sales_case",
                                "args": args,
                                "action_fingerprint": ToolExecutor.fingerprint("submit_after_sales_case", args),
                                "case_id": observation["case_id"],
                                "order_id": observation["order_id"],
                                "policy_version_id": decision["policy_version_id"],
                                "policy_decision_id": decision["decision_id"],
                                "requires_approval": True,
                            }
                            step.status = "WAITING_APPROVAL"
                            step.outcome = "WAITING_APPROVAL"
                            step.result = {
                                "message": (
                                    f"售后案件 {observation['case_id']} 已通过政策核验；"
                                    "退款尚未提交，请明确确认后继续。"
                                )
                            }
                            recorder.record("approval_required", step_id=step.step_id, action=pending)
                        elif outcome in {"NEEDS_EVIDENCE", "MANUAL_REVIEW"}:
                            step.status = "SUCCEEDED"
                            step.outcome = "UNAVAILABLE_EVIDENCE"
                            label = "需要补充证据" if outcome == "NEEDS_EVIDENCE" else "需要人工审核"
                            step.result = {
                                "message": (
                                    f"售后案件 {observation['case_id']} 已保存，{label}。"
                                    f"原因：{'、'.join(decision.get('reason_codes') or [])}"
                                )
                            }
                        else:
                            step.status = "REJECTED"
                            step.outcome = "SATISFIED"
                            step.result = {
                                "message": (
                                    f"售后案件 {observation['case_id']} 已核验，当前不符合政策。"
                                    f"原因：{'、'.join(decision.get('reason_codes') or [])}"
                                )
                            }
                        completed[step.step_id] = step
                        recorder.record(
                            "step_completed",
                            step_id=step.step_id,
                            specialist=step.specialist,
                            status=step.status,
                            error=step.error,
                        )
                        if context:
                            context.phase = "PLAN"
                            context.checkpoint(plan=plan.payload())
                        continue
                    eligible = observation.get("eligible") if step.tool == "prepare_refund" else observation.get("status") == "pending"
                    if eligible:
                        pending = dict(type="INITIATE_REFUND" if step.tool == "prepare_refund" else "PROCESS_PAYMENT",
                                       order_id=step.args["order_id"], requires_approval=True)
                        if step.tool == "prepare_refund":
                            pending["reason"] = intake.request
                        else:
                            pending.update(amount=observation["amount"], method=observation.get("method"))
                        step.status = "WAITING_APPROVAL"
                        step.outcome = "WAITING_APPROVAL"
                        step.result = {"message": f"订单 {step.args['order_id']} 的操作尚未执行，请明确确认后继续。"}
                        recorder.record("approval_required", step_id=step.step_id, action=pending)
                    else:
                        step.status = "REJECTED"
                        step.outcome = "SATISFIED"
                        step.result = {"message": f"订单 {step.args['order_id']} 当前不满足操作条件，无需重复提交。"}
                elif step.tool == MODEL_ANSWER_OPERATION:
                    from src.general_answer import generate_general_answer
                    step.result = generate_general_answer(**step.args)
                    step.status = "SUCCEEDED"
                    recorder.record("knowledge_result", step_id=step.step_id,
                                    evidence_requirement=step.args.get("evidence_requirement"),
                                    provenance=step.result.get("provenance"))
                    verify_observation(step)
                    step.outcome = ("UNAVAILABLE_EVIDENCE"
                                    if step.result.get("provenance") == "NO_REALTIME_SOURCE" else "SATISFIED")
                else:
                    step.result, event = executor.execute(step.tool, step.args)
                    trace.append(_trace_entry(run_id, step.specialist, step.tool, step.args, step.result, event))
                    if isinstance(step.result, dict) and isinstance(step.result.get("clarification_binding"), dict):
                        recorder.record("case_clarification_answer_bound", **step.result["clarification_binding"])
                    step.status = event["status"]
                    if step.status == "BUSINESS_ERROR":
                        code = ("INVALID_PRODUCT_SELECTION" if step.tool == "compare_products"
                                else "INVALID_UTILITY_REQUEST" if step.tool in {"read_clock", "calculate"}
                                else "BUSINESS_DATA_NOT_FOUND")
                        raise AgentError(code, step.result["error"], retryable=False)
                    if (step.tool in _POLICY_AWARE_CASE_OPERATIONS
                            and _case_execution_mode(context) == "policy_aware"):
                        autonomous_budget = max(
                            0,
                            (context.limits.max_steps if context else 8) - len(plan.steps),
                        )
                        step.result = _advance_after_sales_case(
                            step.result,
                            executor=executor,
                            recorder=recorder,
                            trace=trace,
                            run_id=run_id,
                            max_actions=min(_MAX_AUTONOMOUS_CASE_ACTIONS, autonomous_budget),
                            action_selection_mode=_case_action_selection_mode(context),
                        )
                    verify_observation(step)
                    if step.tool == CATALOG_QUERY_OPERATION:
                        verification = step.result["artifact"]["verification"]
                        recorder.record(
                            "catalog_goal_verified",
                            step_id=step.step_id,
                            result_set_id=step.result["artifact"]["result_set_id"],
                            **verification["goal"],
                        )
                        recorder.record(
                            "catalog_delivery_verified",
                            step_id=step.step_id,
                            result_set_id=step.result["artifact"]["result_set_id"],
                            **verification["delivery"],
                        )
                    if step.tool == CONSTRAINT_PLAN_OPERATION:
                        verification = step.result["artifact"]["verification"]
                        recorder.record(
                            "global_goal_verified",
                            step_id=step.step_id,
                            solution_set_id=step.result["artifact"]["solution_set_id"],
                            **verification,
                        )
                    contract = get_capability_registry().operation_contract(step.tool)
                    step.outcome = contract.outcome(step.result, step.args) if contract else "SATISFIED"
                    if isinstance(step.result, dict) and step.result.get("data_source") == "AFTER_SALES_CASE_LEDGER":
                        plans = (step.result.get("case") or {}).get("plans") or []
                        if plans:
                            recorder.record("case_plan_revised", step_id=step.step_id, plan=plans[-1])
            except AgentError as error:
                step.status = "FAILED"
                step.outcome = "FAILED"
                step.error = {"code": error.code, "message": str(error), "retryable": error.retryable}
                errors.append(step.error)
                artifact = step.result.get("artifact") if isinstance(step.result, dict) else None
                if (
                    step.tool in {CATALOG_QUERY_OPERATION, CONSTRAINT_PLAN_OPERATION}
                    and isinstance(artifact, dict)
                    and (artifact.get("verification") or {}).get("status") == "REJECTED"
                ):
                    recorder.record(
                        "catalog_verification_rejected" if step.tool == CATALOG_QUERY_OPERATION else "global_goal_rejected",
                        step_id=step.step_id,
                        result_set_id=artifact.get("result_set_id") or artifact.get("solution_set_id"),
                        error=step.error,
                    )
                if "BUDGET" in error.code or error.code in {"RUN_LEASE_LOST", "UNCERTAIN_TOOL_OUTCOME"}:
                    stop_error = step.error
            recorder.record("step_completed", step_id=step.step_id, specialist=step.specialist, status=step.status, error=step.error)
        completed[step.step_id] = step
        if step.status == "SKIPPED":
            step.outcome = "FAILED"
        if context:
            context.phase = "PLAN"
            context.checkpoint(plan=plan.payload())
    satisfied = [step.step_id for step in plan.steps if step.outcome == "SATISFIED"]
    missing = [step.step_id for step in plan.steps if step.outcome not in {"SATISFIED", "WAITING_APPROVAL"}]
    goal_rows: dict[str, list[TaskStep]] = {}
    for index, step in enumerate(plan.steps):
        identities = step.objective_ids or ([step.objective_id] if step.objective_id else [f"goal-{index + 1}"])
        for identity in identities:
            goal_rows.setdefault(identity, []).append(step)
    goal_outcomes = []
    for objective_id, steps in goal_rows.items():
        outcomes = {step.outcome for step in steps}
        outcome = ("FAILED" if "FAILED" in outcomes else
                   "UNAVAILABLE_EVIDENCE" if "UNAVAILABLE_EVIDENCE" in outcomes else
                   "WAITING_APPROVAL" if "WAITING_APPROVAL" in outcomes else "SATISFIED")
        goal_outcomes.append({"objective_id": objective_id, "outcome": outcome,
                              "step_ids": [step.step_id for step in steps]})
    plan.verification = dict(satisfied_steps=satisfied, incomplete_steps=missing,
                             waiting_steps=[s.step_id for s in plan.steps if s.outcome == "WAITING_APPROVAL"],
                             goal_outcomes=goal_outcomes,
                             all_goals_satisfied=bool(plan.steps) and all(row["outcome"] == "SATISFIED" for row in goal_outcomes),
                             side_effects_executed=False)
    recorder.record("verification", **plan.verification)
    unavailable = any(step.outcome == "UNAVAILABLE_EVIDENCE" for step in plan.steps)
    status = ("WAITING_APPROVAL" if pending else "PARTIAL" if unavailable
              else ("PARTIAL" if satisfied else "FAILED") if missing else "SUCCEEDED")
    if not plan.steps:
        status = "FAILED"
        errors.append({"code": "EMPTY_TASK_PLAN", "message": "没有可验证的业务步骤", "retryable": False})
    optimality_not_proven = any(
        step.tool == CONSTRAINT_PLAN_OPERATION
        and (step.result.get("artifact", {}).get("optimality") or {}).get("status") == "BEST_KNOWN"
        for step in plan.steps
        if isinstance(step.result, dict)
    )
    reason = (errors[-1]["code"] if errors else "APPROVAL_REQUIRED" if pending else
              "OPTIMALITY_NOT_PROVEN" if optimality_not_proven else
              "EVIDENCE_UNAVAILABLE" if unavailable else "BUSINESS_RULE_REJECTED"
              if any(s.status == "REJECTED" for s in plan.steps) else "COMPLETED")
    if not errors and not pending and any(s.tool == "prepare_payment" and s.status == "REJECTED" for s in plan.steps):
        reason = "NO_ACTION_NEEDED"
    # An empty search is a verified outcome, not a claim that products exist.
    if not errors and not pending and len(plan.steps) == 1 and plan.steps[0].tool in {"search_products", "discover_products"} and not plan.steps[0].result.get("products"):
        reason = "NO_CANDIDATE_EVIDENCE" if plan.steps[0].tool == "discover_products" else "NO_MATCHING_PRODUCTS"
    if (
        not errors and not pending and len(plan.steps) == 1
        and plan.steps[0].tool == CONSTRAINT_PLAN_OPERATION
        and plan.steps[0].result.get("artifact", {}).get("feasibility") == "INFEASIBLE"
    ):
        reason = "NO_FEASIBLE_SOLUTION"
    if incomplete_reason and status == "SUCCEEDED":
        status, reason = "PARTIAL", incomplete_reason
    response = _finish(recorder, intake, run_id, agent, trace, reply="\n\n".join(_facts(s) for s in plan.steps) or "没有可执行的业务步骤。",
                       status=status, terminal_reason=reason, pending_action=pending, error=errors[-1] if errors else None)
    response["artifacts"] = [
        step.result["artifact"] for step in plan.steps
        if isinstance(step.result, dict)
        and isinstance(step.result.get("artifact"), dict)
    ]
    if any(step.tool == MODEL_ANSWER_OPERATION for step in plan.steps):
        response["route"] = "KNOWLEDGE" if len(plan.steps) == 1 else "MIXED"
    response["plan"] = plan.payload()
    pending_clarifications = [
        autonomous.get("pending_clarification")
        for step in plan.steps
        if isinstance(step.result, dict)
        and isinstance((autonomous := step.result.get("autonomous_execution")), dict)
        and isinstance(autonomous.get("pending_clarification"), dict)
    ]
    if pending_clarifications:
        response["pending_clarification"] = pending_clarifications[-1]
    return response
