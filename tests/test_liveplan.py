import json

from langchain_core.messages import AIMessage

from src.context_engine import ContextEngine
from src.harness import get_runner
from src.liveplan import LivePlanSteeringMechanism
from src.liveplan_monitor import decision_signature
from src.runtime import BudgetLimits, RunRuntime, runtime_scope
from src.task_plan import TaskPlan, TaskStep


def state():
    return {
        "goal_revision": 0,
        "request": "推荐预算500元的耳机",
        "goal": {"objectives": [{"id": "shopping", "kind": "recommend"}]},
        "objective_ledger": [{"objective_id": "shopping", "status": "PENDING", "evidence_refs": []}],
        "binding_revision": 0,
        "next_steps": None,
        "feedback": None,
        "working_memory": None,
        "model_cursor": 0,
    }


def mechanism(**updates):
    values = {
        "advisor_mode": "predefined",
        "max_advisor_calls": 1,
        "cooldown_steps": 2,
        "action_repeat_threshold": 2,
        "phase_dwell_threshold": 4,
        "evidence_stagnation_threshold": 3,
        "verification_repeat_threshold": 2,
        "max_blocking_interventions": 2,
    }
    values.update(updates)
    return LivePlanSteeringMechanism().configure(values)


def search_decision(query="耳机"):
    return {
        "action": "tool",
        "reason": "读取候选",
        "tool": "search_products",
        "args": {"query": query, "category": "耳机", "limit": 2},
    }


def add_search(context, plan, *, query="耳机", reference="run:1"):
    index = len(plan.steps) + 1
    step_id = f"step-{index}"
    decision = search_decision(query)
    context.record("adaptive_decision", source="model", decision=decision,
                   goal_revision=0, step_id=step_id)
    plan.steps.append(TaskStep(
        step_id,
        "shopping",
        "search_products",
        decision["args"],
        status="SUCCEEDED",
        result={
            "products": [{"product_id": "P0001", "name": "耳机", "price": 199, "stock": 2}],
            "observation": {"reference": reference, "timestamp": reference},
        },
    ))
    context.record("step_completed", step_id=step_id, specialist="shopping", status="SUCCEEDED")


def repeated_assessment(*, advisor_mode="predefined"):
    context = RunRuntime("liveplan-run", BudgetLimits(max_model_calls=8, max_tool_calls=8))
    current = state()
    plan = TaskPlan(["shopping"], [])
    monitor = mechanism(advisor_mode=advisor_mode)
    add_search(context, plan, reference="run:1")
    assessment, changed = monitor.evaluate(context, current, plan)
    assert assessment is None and changed
    add_search(context, plan, reference="run:2")
    assessment, changed = monitor.evaluate(context, current, plan)
    assert changed and assessment is not None
    return context, current, plan, monitor, assessment


def test_monitor_detects_value_stagnation_without_counting_new_references_as_evidence():
    context, current, _, _, assessment = repeated_assessment()

    assert assessment.signals[0].code == "ACTION_OSCILLATION"
    assert assessment.signals[0].severity == "blocking"
    evaluations = [event for event in context.events if event["type"] == "trajectory_monitor_evaluated"]
    assert evaluations[0]["progress"]["distinct_evidence"] == 1
    assert evaluations[1]["progress"]["distinct_evidence"] == 1
    assert context.events[-1]["type"] == "trajectory_drift_detected"
    assert current["liveplan"]["pending_drift"]["status"] == "DETECTED"


def test_predefined_advice_blocks_only_the_exact_repeated_candidate():
    context, current, _, monitor, assessment = repeated_assessment()
    monitor.reserve_advice(context, current, assessment)
    advice = monitor.complete_advice(context, current, assessment)

    assert advice.reason_code == "ACTION_OSCILLATION"
    assert current["liveplan"]["active_intervention"]["mode"] == "CONSTRAIN_NEXT_ACTION"
    assert monitor.review_candidate(context, current, search_decision()) == advice.guidance
    assert monitor.review_candidate(context, current, search_decision("头戴式耳机")) is None
    outcomes = [event["outcome"] for event in context.events if event["type"] == "intervention_applied"]
    assert outcomes == ["ACTIVATED", "ACTION_REJECTED", "NEXT_ACTION_OBSERVED"]


def test_model_advisor_is_bounded_and_uses_the_shared_runtime_budget(monkeypatch):
    context, current, _, monitor, assessment = repeated_assessment(advisor_mode="model")

    class Advisor:
        def json_mode(self):
            return self

        def invoke(self, _messages):
            return AIMessage(content=json.dumps({
                "recommended_action": "CHANGE_APPROACH",
                "target_objective_ids": ["shopping"],
                "reason_code": "ACTION_OSCILLATION",
                "guidance": "改用品类和关键属性组合检索，不再重复同一查询。",
                "confidence": 0.8,
            }), usage_metadata={"input_tokens": 30, "output_tokens": 20, "total_tokens": 50})

    monkeypatch.setattr("src.llm.get_llm", lambda **_kwargs: Advisor())
    reservation = monitor.reserve_advice(context, current, assessment)
    assert reservation["source"] == "model"
    advice = monitor.complete_advice(context, current, assessment)

    assert advice.confidence == 0.8
    assert context.counters["model_calls"] == 1
    assert current["liveplan"]["advisor_calls"] == 1
    generated = next(event for event in context.events if event["type"] == "corrective_advice_generated")
    assert generated["source"] == "model"
    assert not any("products" in json.dumps(event) for event in context.events
                   if event["type"] == "corrective_advice_generated")


def test_unknown_previous_advisor_outcome_falls_back_without_another_paid_call(monkeypatch):
    context, current, _, monitor, assessment = repeated_assessment(advisor_mode="model")
    reservation = monitor.reserve_advice(context, current, assessment)
    context.record("model_started", purpose="liveplan_advisor",
                   agent=f"liveplan_advisor:{reservation['attempt_id']}")
    monkeypatch.setattr("src.llm.get_llm", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not retry")))

    advice = monitor.complete_advice(context, current, assessment)

    assert advice.confidence == 1.0
    failure = next(event for event in context.events if event["type"] == "corrective_advice_failed")
    assert failure["code"] == "PREVIOUS_ADVISOR_OUTCOME_UNKNOWN"


def test_advice_and_intervention_events_rehydrate_without_duplicate_generation():
    context, current, _, monitor, assessment = repeated_assessment()
    monitor.reserve_advice(context, current, assessment)
    advice = monitor.complete_advice(context, current, assessment)
    generated_count = sum(event["type"] == "corrective_advice_generated" for event in context.events)
    activated_count = sum(event["type"] == "intervention_applied" and event["outcome"] == "ACTIVATED"
                          for event in context.events)
    current["liveplan"]["pending_drift"]["status"] = "ADVICE_RESERVED"
    current["liveplan"]["active_intervention"] = None

    recovered = monitor.complete_advice(context, current, assessment)

    assert recovered == advice
    assert current["liveplan"]["active_intervention"]["drift_id"] == assessment.evaluation_id
    assert sum(event["type"] == "corrective_advice_generated" for event in context.events) == generated_count
    assert sum(event["type"] == "intervention_applied" and event["outcome"] == "ACTIVATED"
               for event in context.events) == activated_count


def test_recorded_candidate_rejection_is_not_counted_twice_after_checkpoint_gap():
    context, current, _, monitor, assessment = repeated_assessment()
    monitor.reserve_advice(context, current, assessment)
    advice = monitor.complete_advice(context, current, assessment)
    decision = search_decision()
    assert monitor.review_candidate(context, current, decision) == advice.guidance
    rejected_count = sum(event["type"] == "intervention_applied" and event["outcome"] == "ACTION_REJECTED"
                         for event in context.events)
    # Simulate an adaptive checkpoint saved before the event-backed block count.
    current["liveplan"]["active_intervention"]["block_count"] = 0

    assert monitor.review_candidate(context, current, decision) == advice.guidance
    assert sum(event["type"] == "intervention_applied" and event["outcome"] == "ACTION_REJECTED"
               for event in context.events) == rejected_count


def test_budget_risk_never_spends_the_reserved_decision_and_audit_calls():
    context = RunRuntime("budget-risk", BudgetLimits(max_model_calls=2, max_tool_calls=8))
    current = state()
    plan = TaskPlan(["shopping"], [])
    monitor = mechanism(advisor_mode="model")
    add_search(context, plan)

    assessment, _ = monitor.evaluate(context, current, plan)
    assert assessment is not None and assessment.signals[0].code == "BUDGET_RISK"
    reservation = monitor.reserve_advice(context, current, assessment)

    assert reservation["source"] == "predefined"
    assert current["liveplan"]["advisor_calls"] == 0


def test_runtime_guidance_is_injected_as_data_without_mutating_business_state():
    context, current, plan, monitor, assessment = repeated_assessment()
    monitor.reserve_advice(context, current, assessment)
    monitor.complete_advice(context, current, assessment)
    before_goal = json.loads(json.dumps(current["goal"]))
    before_liveplan = json.loads(json.dumps(current["liveplan"]))
    with runtime_scope(context):
        monitor.install(context)
        messages, _ = ContextEngine(current["request"]).decision_messages(
            "system", current, plan, [], max_chars=12000,
        )
    payload = json.loads(messages[-1].content)

    assert payload["runtime_guidance"]["reason_code"] == "ACTION_OSCILLATION"
    assert payload["runtime_guidance"]["policy"] == "ADVISORY_ONLY_EXCEPT_EXACT_REPEAT_PRECOMMIT_BLOCK"
    assert current["goal"] == before_goal
    assert current["liveplan"] == before_liveplan


def test_liveplan_profile_is_a_research_only_single_slot_candidate():
    baseline = get_runner("adaptive-v1").profile
    candidate = get_runner("adaptive-liveplan-v1").profile

    assert candidate.research.status == "candidate"
    assert candidate.research.paper_id == "arXiv:2608.06701v1"
    assert candidate.engine == baseline.engine
    assert [item.slot for item in candidate.mechanisms if item not in baseline.mechanisms] == ["steering"]
    steering = next(item for item in get_runner("adaptive-liveplan-v1").provenance()["mechanisms"]
                    if item["slot"] == "steering")
    assert steering["configuration"]["max_advisor_calls"] == 1
    assert len(steering["implementation_sha256"]) == 64


def test_decision_signature_excludes_explanation_but_preserves_business_arguments():
    left = search_decision()
    right = {**left, "reason": "换一种说法"}
    changed = search_decision("开放式耳机")

    assert decision_signature(left) == decision_signature(right)
    assert decision_signature(left) != decision_signature(changed)
