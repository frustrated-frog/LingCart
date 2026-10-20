"""Evidence-backed, goal-level oracle for semantic-understanding evaluations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

from eval.semantic_contracts import GoalOracle, SemanticCase, ValueAssertion


_ADAPTIVE_KIND_MAP = {
    "recommend": "PRODUCT_DISCOVERY",
    "compare_products": "PRODUCT_COMPARE",
    "get_order_status": "ORDER_STATUS",
    "get_order_details": "ORDER_DETAILS",
    "track_shipment": "SHIPMENT_TRACKING",
    "check_refund_eligibility": "REFUND_ELIGIBILITY",
    "get_payment_status": "PAYMENT_STATUS",
    "answer_faq": "FAQ",
}


def _path(value: object, path: str) -> tuple[bool, object]:
    current = value
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
            continue
        return False, None
    return True, current


def _assertion(assertion: ValueAssertion, actual: object) -> tuple[bool, object]:
    present, value = _path(actual, assertion.path)
    if assertion.operator == "present":
        return present and value is not None, value
    if not present:
        return False, None
    if assertion.operator == "eq":
        return value == assertion.value, value
    if assertion.operator == "contains_all":
        if isinstance(value, str):
            return all(str(item) in value for item in assertion.value), value
        if isinstance(value, (list, tuple, set)):
            return all(item in value for item in assertion.value), value
        return False, value
    if assertion.operator == "set_eq":
        if not isinstance(value, (list, tuple, set)):
            return False, value
        return set(value) == set(assertion.value), value
    return False, value  # pragma: no cover - Pydantic rejects unknown operators.


def _objective_assertions(goal: GoalOracle, objective: dict) -> tuple[bool, list[dict]]:
    checks = []
    for expected in goal.objective:
        passed, actual = _assertion(expected, objective)
        checks.append({
            "dimension": "objective",
            "code": "OBJECTIVE_FIELD_MATCH" if passed else "ENTITY_MISMATCH",
            "passed": passed,
            "path": expected.path,
            "expected": expected.value,
            "actual": actual,
        })
    return all(check["passed"] for check in checks), checks


def _actual_objectives(result: dict) -> tuple[list[dict], str]:
    understanding = result.get("understanding") or {}
    objectives = understanding.get("objectives") or []
    if objectives:
        return [deepcopy(item) for item in objectives if isinstance(item, dict)], "dialogue_understanding"
    adaptive_goal = (result.get("plan") or {}).get("goal") or {}
    if adaptive_goal.get("objectives"):
        normalized = []
        for item in adaptive_goal["objectives"]:
            if not isinstance(item, dict):
                continue
            normalized.append({**deepcopy(item), "kind": _ADAPTIVE_KIND_MAP.get(item.get("kind"), item.get("kind"))})
        return normalized, "adaptive_goal"
    intake = result.get("intake") or {}
    entities = intake.get("entities") or {}
    return [
        {"id": f"goal-{index}", "kind": kind, **deepcopy(entities)}
        for index, kind in enumerate(intake.get("intents") or [], start=1)
    ], "deterministic_intake"


def _actual_commands(result: dict) -> list[str]:
    commands = (result.get("understanding") or {}).get("commands") or []
    if not commands:
        commands = (result.get("plan") or {}).get("dialogue_commands") or []
    return [item.get("kind") for item in commands if isinstance(item, dict) and item.get("kind")]


def _tool_records(events: list[dict]) -> tuple[list[dict], list[str]]:
    calls = {
        event.get("execution_id"): event
        for event in events
        if event.get("type") == "tool_call" and event.get("execution_id")
    }
    records, failures = [], []
    for event in events:
        if event.get("type") != "tool_result":
            continue
        execution_id = event.get("execution_id")
        call = calls.get(execution_id)
        if call is None:
            failures.append("AUTHORITATIVE_EVIDENCE_INVALID")
            continue
        if any(call.get(key) != event.get(key) for key in ("run_id", "tool", "args", "action_fingerprint")):
            failures.append("AUTHORITATIVE_EVIDENCE_INVALID")
            continue
        records.append({
            "operation": event.get("tool"),
            "args": deepcopy(event.get("args") or {}),
            "result": deepcopy(event.get("result")),
            "status": event.get("status"),
            "execution_id": execution_id,
            "sequence": event.get("sequence"),
            "policy": deepcopy(event.get("policy") or call.get("policy") or {}),
        })
    return records, failures


def _perfect_matchings(expected: tuple[GoalOracle, ...], actual: list[dict]) -> list[dict[int, int]]:
    edges: dict[int, list[int]] = {}
    for expected_index, goal in enumerate(expected):
        edges[expected_index] = []
        for actual_index, objective in enumerate(actual):
            if objective.get("kind") != goal.objective_kind:
                continue
            matches, _ = _objective_assertions(goal, objective)
            if matches:
                edges[expected_index].append(actual_index)
    solutions: list[dict[int, int]] = []

    def walk(index: int, used: set[int], assignment: dict[int, int]) -> None:
        if len(solutions) > 1:
            return
        if index == len(expected):
            solutions.append(dict(assignment))
            return
        for actual_index in edges[index]:
            if actual_index in used:
                continue
            assignment[index] = actual_index
            walk(index + 1, {*used, actual_index}, assignment)
            assignment.pop(index, None)

    walk(0, set(), {})
    return solutions


def _step_capabilities(result: dict, objective: dict, operations: set[str]) -> set[str]:
    objective_id = objective.get("id")
    capabilities = set()
    for step in (result.get("plan") or {}).get("steps") or []:
        ids = set(step.get("objective_ids") or [])
        if step.get("objective_id"):
            ids.add(step["objective_id"])
        related = objective_id in ids if ids and objective_id else step.get("tool") in operations
        if related and step.get("capability_id"):
            capabilities.add(step["capability_id"])
    return capabilities


def _goal_outcome(result: dict, objective: dict) -> str | None:
    outcomes = ((result.get("plan") or {}).get("verification") or {}).get("goal_outcomes") or []
    objective_id = objective.get("id")
    exact = next((item for item in outcomes if item.get("objective_id") == objective_id), None)
    if exact:
        return exact.get("outcome")
    if len(outcomes) == 1:
        return outcomes[0].get("outcome")
    ledger = (result.get("plan") or {}).get("objective_ledger") or []
    exact = next((item for item in ledger if item.get("objective_id") == objective_id), None)
    return (exact or {}).get("status")


def _check_operation(expectation, records: list[dict]) -> tuple[list[dict], list[str]]:
    candidates = [record for record in records if record["operation"] == expectation.operation]
    matching = []
    rejected = []
    for record in candidates:
        args_ok = all(_assertion(item, record["args"])[0] for item in expectation.args)
        result_ok = all(_assertion(item, record["result"])[0] for item in expectation.result)
        status_ok = record["status"] in expectation.accepted_statuses
        if args_ok and result_ok and status_ok:
            matching.append(record)
        else:
            rejected.append(record)
    failures = []
    if not candidates:
        failures.append("OPERATION_MISSING")
    elif not expectation.min_calls <= len(matching) <= expectation.max_calls:
        failures.append("OPERATION_CARDINALITY_MISMATCH")
        if any(not all(_assertion(item, row["args"])[0] for item in expectation.args) for row in rejected):
            failures.append("NORMALIZED_PARAMETER_MISMATCH")
        if any(not all(_assertion(item, row["result"])[0] for item in expectation.result) for row in rejected):
            failures.append("AUTHORITATIVE_EVIDENCE_INVALID")
    return matching, list(dict.fromkeys(failures))


@dataclass(frozen=True)
class SemanticScore:
    payload: dict

    @property
    def passed(self) -> bool:
        return bool(self.payload["passed"])


def score_semantic_result(case: SemanticCase, result: dict, *, mode: str) -> SemanticScore:
    """Recompute one case from raw runtime events; presentation trace is never authoritative."""

    actual, objective_source = _actual_objectives(result)
    commands = _actual_commands(result)
    events = list(result.get("events") or [])
    records, evidence_failures = _tool_records(events)
    failures: list[dict] = [
        {"code": code, "message": "工具结果缺少可配对的权威调用证据"}
        for code in evidence_failures
    ]
    solutions = _perfect_matchings(case.oracle.goals, actual)
    assignment = solutions[0] if len(solutions) == 1 else {}
    goal_results = []
    if len(solutions) > 1:
        failures.append({"code": "GOAL_MATCH_AMBIGUOUS", "message": "同类目标存在多个等价匹配"})

    used_actual = set(assignment.values())
    for expected_index, goal in enumerate(case.oracle.goals):
        if expected_index not in assignment:
            same_kind = [item for item in actual if item.get("kind") == goal.objective_kind]
            code = "ENTITY_MISMATCH" if same_kind else "GOAL_MISSING"
            check = {"dimension": "goal", "code": code, "passed": False}
            goal_results.append({
                "oracle_goal_id": goal.oracle_goal_id,
                "actual_objective_id": None,
                "passed": False,
                "checks": [check],
            })
            failures.append({"code": code, "goal_id": goal.oracle_goal_id, "message": "未找到唯一匹配的实际目标"})
            continue
        objective = actual[assignment[expected_index]]
        _, checks = _objective_assertions(goal, objective)
        operation_names = {item.operation for item in goal.required_operations}
        if goal.capability_id:
            capabilities = _step_capabilities(result, objective, operation_names)
            passed = goal.capability_id in capabilities
            checks.append({
                "dimension": "capability",
                "code": "CAPABILITY_MATCH" if passed else "CAPABILITY_MISMATCH",
                "passed": passed,
                "expected": goal.capability_id,
                "actual": sorted(capabilities),
            })
        evidence_refs = []
        for operation in goal.required_operations:
            matches, operation_failures = _check_operation(operation, records)
            evidence_refs.extend(row["execution_id"] for row in matches)
            checks.append({
                "dimension": "operation",
                "code": "OPERATION_MATCH" if not operation_failures else operation_failures[0],
                "passed": not operation_failures,
                "operation": operation.operation,
                "evidence_refs": [row["execution_id"] for row in matches],
            })
            for code in operation_failures:
                failures.append({"code": code, "goal_id": goal.oracle_goal_id,
                                 "operation": operation.operation, "message": "目标执行证据不满足契约"})
        actual_outcome = _goal_outcome(result, objective)
        if goal.outcome is not None:
            passed = actual_outcome == goal.outcome
            checks.append({
                "dimension": "outcome",
                "code": "GOAL_OUTCOME_MATCH" if passed else "GOAL_OUTCOME_MISMATCH",
                "passed": passed,
                "expected": goal.outcome,
                "actual": actual_outcome,
            })
        goal_passed = all(item["passed"] for item in checks)
        if not goal_passed:
            for item in checks:
                if not item["passed"] and item["code"] not in {failure["code"] for failure in failures}:
                    failures.append({"code": item["code"], "goal_id": goal.oracle_goal_id,
                                     "message": "目标维度校验失败"})
        goal_results.append({
            "oracle_goal_id": goal.oracle_goal_id,
            "actual_objective_id": objective.get("id"),
            "passed": goal_passed,
            "checks": checks,
            "evidence_refs": sorted(set(evidence_refs)),
            "projection": {
                "objective_kind": objective.get("kind"),
                "objective": {item.path: _path(objective, item.path)[1] for item in goal.objective},
                "outcome": actual_outcome,
            },
        })

    if case.oracle.goal_set == "exact":
        for index, objective in enumerate(actual):
            if index not in used_actual:
                failures.append({"code": "UNEXPECTED_GOAL", "actual_objective_id": objective.get("id"),
                                 "message": f"出现未声明目标 {objective.get('kind')}"})

    command_contract = case.oracle.commands
    command_set = set(commands)
    for command in command_contract.required_all:
        if command not in command_set:
            failures.append({"code": "COMMAND_MISSING", "command": command, "message": "缺少必需对话命令"})
    if command_contract.required_any and not command_set.intersection(command_contract.required_any):
        failures.append({"code": "COMMAND_MISSING", "commands": list(command_contract.required_any),
                         "message": "未命中任一必需对话命令"})
    for command in command_contract.forbidden:
        if command in command_set:
            failures.append({"code": "FORBIDDEN_COMMAND", "command": command, "message": "出现禁用对话命令"})
    declared_commands = set(command_contract.required_all) | set(command_contract.required_any)
    if not command_contract.allow_extra:
        for command in command_set - declared_commands:
            failures.append({"code": "UNEXPECTED_COMMAND", "command": command, "message": "出现未声明对话命令"})

    for operation in case.oracle.forbidden_operations:
        if any(record["operation"] == operation for record in records) or any(
            event.get("type") == "tool_call" and event.get("tool") == operation for event in events
        ):
            failures.append({"code": "FORBIDDEN_OPERATION_EXECUTED", "operation": operation,
                             "message": "物理事件中出现禁用操作"})

    if case.execution_policy.side_effects == "forbidden":
        unsafe = [
            event.get("tool") for event in events if event.get("type") == "tool_call"
            and (event.get("policy") or {}).get("effect") not in {None, "READ"}
        ]
        if unsafe:
            failures.append({"code": "FORBIDDEN_SIDE_EFFECT", "operations": unsafe,
                             "message": "只读语义评测执行了副作用操作"})

    terminal = case.oracle.terminal
    if result.get("status") not in terminal.statuses:
        failures.append({"code": "TERMINAL_STATUS_MISMATCH", "expected": list(terminal.statuses),
                         "actual": result.get("status"), "message": "顶层状态不符合契约"})
    if terminal.reason is not None and result.get("terminal_reason") != terminal.reason:
        failures.append({"code": "TERMINAL_REASON_MISMATCH", "expected": terminal.reason,
                         "actual": result.get("terminal_reason"), "message": "终止原因不符合契约"})
    terminal_events = [event for event in events if event.get("type") == "terminal"]
    if len(terminal_events) != 1 or terminal_events[-1].get("status") != result.get("status") or (
        terminal_events[-1].get("reason") != result.get("terminal_reason")
    ):
        failures.append({"code": "TERMINAL_EVENT_MISMATCH", "message": "顶层终态与唯一 terminal 事件不一致"})
    verification = (result.get("plan") or {}).get("verification") or {}
    if terminal.all_goals_satisfied is not None and verification.get("all_goals_satisfied") != terminal.all_goals_satisfied:
        failures.append({"code": "GOAL_OUTCOME_MISMATCH", "expected": terminal.all_goals_satisfied,
                         "actual": verification.get("all_goals_satisfied"), "message": "整体目标核验状态不一致"})

    semantic_calls = [
        event for event in events if event.get("type") == "model_call"
        and event.get("status") == "SUCCEEDED" and event.get("purpose") == "dialogue_understanding"
    ]
    if case.execution_policy.semantic_path == "required" and not semantic_calls:
        failures.append({"code": "SEMANTIC_MODEL_EVIDENCE_MISSING", "message": "案例要求语义模型但无成功调用证据"})
    if case.execution_policy.semantic_path == "forbidden" and semantic_calls:
        failures.append({"code": "UNSAFE_SEMANTIC_PATH", "message": "案例禁止模型路径但发生语义调用"})

    if case.execution_policy.trace_integrity == "required":
        try:
            from src.trace_projector import project_trace
            integrity = project_trace(events).get("integrity") or {}
        except Exception:
            integrity = {"complete": False, "issues": ["PROJECTOR_ERROR"]}
        if integrity.get("complete") is not True:
            failures.append({"code": "TRACE_INTEGRITY_FAILURE", "issues": integrity.get("issues", []),
                             "message": "运行事件未满足序列、工具配对或唯一终态不变量"})
    else:
        integrity = None

    goal_recall = (
        sum(item["actual_objective_id"] is not None for item in goal_results) / len(goal_results)
        if goal_results else 1.0
    )
    failure_codes = list(dict.fromkeys(item["code"] for item in failures))
    projection = {
        "goals": [item.get("projection") for item in goal_results if item.get("projection")],
        "commands": sorted(command_set),
        "status": result.get("status"),
        "terminal_reason": result.get("terminal_reason"),
    }
    return SemanticScore({
        "passed": not failures,
        "failure_codes": failure_codes,
        "failures": failures,
        "goal_recall": goal_recall,
        "false_fast_path": case.execution_policy.semantic_path == "required" and not semantic_calls,
        "objective_source": objective_source,
        "goal_results": goal_results,
        "commands": commands,
        "semantic_projection": projection,
        "trace_integrity": integrity,
        "mode": mode,
    })


def apply_mode_consistency(samples: list[dict]) -> list[bool]:
    """Compare goal/evidence projections, not merely top-level terminal strings."""

    grouped: dict[tuple[str, int], dict[str, dict]] = {}
    for sample in samples:
        grouped.setdefault((sample["case_id"], sample["repeat"]), {})[sample["mode"]] = sample
    outcomes = []
    for pair in grouped.values():
        if set(pair) != {"auto", "adaptive"}:
            continue
        same = pair["auto"].get("semantic_projection") == pair["adaptive"].get("semantic_projection")
        outcomes.append(same)
        if not same:
            for sample in pair.values():
                sample["passed"] = False
                sample.setdefault("failure_codes", []).append("AUTO_ADAPTIVE_SEMANTIC_MISMATCH")
                sample.setdefault("failures", []).append({
                    "code": "AUTO_ADAPTIVE_SEMANTIC_MISMATCH",
                    "message": "Auto 与 Adaptive 的目标、实体、结果或终态投影不一致",
                })
    return outcomes
