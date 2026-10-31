"""检查独立审查、请求 IR 与可执行目标的双向一致性，不解析自然语言。"""
from __future__ import annotations

from src.adaptive_decision import ShoppingGoal
from src.constraint_bindings import constraint_refs, fingerprint
from src.errors import AgentError
from src.request_ir import UserRequestIR, _human_sources, locate_source_span


def goal_fingerprint(goal: dict) -> str:
    return fingerprint(ShoppingGoal.model_validate(goal).model_dump())


def _reject(message: str):
    raise AgentError("REQUEST_IR_COMPILE_ERROR", "请求编译一致性检查失败：" + message, retryable=False)


def _same_source(left, right) -> bool:
    return (left.source, left.history_index, left.source_sha256) == (
        right.source, right.history_index, right.source_sha256)


def validate_audit_alignment(request: str, history: list | None, goal: dict, audit: dict):
    """Every executable reference needs a grounded review in the same source range."""
    if audit.get("verdict") != "APPROVED":
        _reject("目标尚未通过独立审查")
    objectives = {row["id"]: row for row in goal["objectives"]}
    constraints = constraint_refs(goal)
    reviewed = []
    for row in audit.get("requirements", []):
        span = locate_source_span(row["source_quote"], request, history)
        scope = row.get("scope", "GOAL_REQUIREMENT")
        if row.get("status") != "COVERED":
            _reject("审查仍有未覆盖要求")
        if not set(row.get("objective_ids", [])) <= objectives.keys() or not set(row.get("constraint_refs", [])) <= constraints.keys():
            _reject("审查引用未知目标或条件")
        if scope != "GOAL_REQUIREMENT" and (row.get("objective_ids") or row.get("constraint_refs")):
            _reject("非执行要求不能成为执行引用")
        reviewed.append((row, scope, span))
    for refs, key in ((objectives, "objective_ids"), (constraints, "constraint_refs")):
        for identity, item in refs.items():
            span = locate_source_span(item["source_quote"], request, history)
            aligned = any(
                scope == "GOAL_REQUIREMENT" and identity in row.get(key, [])
                and _same_source(span, source)
                and max(span.start, source.start) < min(span.end, source.end)
                for row, scope, source in reviewed
            )
            if not aligned:
                _reject(f"{identity} 的审查原文与执行来源不对应")
            if any(scope in {"PROHIBITION", "EXECUTION_BOUNDARY", "USER_FACT", "SOFT_PREFERENCE", "UNKNOWN"}
                   and span == source for _, scope, source in reviewed):
                _reject(f"{identity} 把非执行要求编译成目标或条件")


def validate_adaptive_compilation(payload: dict, goal: dict, request: str, history: list | None):
    """Validate at acceptance and again before executing a restored checkpoint.

    The model owns semantic classification. This compiler checks source integrity,
    complete reference coverage, scope, cardinality and immutable target identity.
    """
    try:
        ir = UserRequestIR.model_validate({key: value for key, value in payload.items() if key != "sha256"})
    except ValueError as error:
        _reject(str(error))
    if ir.sha256 != payload.get("sha256") or ir.adaptive_goal_sha256 != goal_fingerprint(goal):
        _reject("IR 或目标指纹已改变")
    sources = {(kind, index): text for kind, index, text in _human_sources(request, history)}
    import hashlib
    for row in ir.requirements:
        source = sources.get((row.source.source, row.source.history_index))
        if (source is None or hashlib.sha256(source.encode()).hexdigest() != row.source.source_sha256
                or source[row.source.start:row.source.end] != row.source.text):
            _reject(f"{row.requirement_id} 的 HUMAN 来源已改变")
    objectives = {row["id"]: row for row in goal["objectives"]}
    constraints = constraint_refs(goal)
    outcomes = [row for row in ir.requirements if row.kind == "BUSINESS_OUTCOME"]
    predicates = [row for row in ir.requirements if row.kind == "PRODUCT_PREDICATE"]
    if len(outcomes) != len(objectives) or len(predicates) != len(constraints):
        _reject("执行目标或条件数量与 IR 不一致")
    seen_objectives, seen_constraints = set(), set()
    recommendation_ids = {key for key, row in objectives.items() if row["kind"] == "recommend"}
    for row in outcomes:
        if len(row.objective_ids) != 1 or row.objective_ids[0] not in objectives:
            _reject("结果要求引用了未知目标")
        identity = row.objective_ids[0]
        target = objectives[identity]
        if identity in seen_objectives or row.quantity != target.get("quantity", 1):
            _reject(f"{identity} 的数量或目标身份不一致")
        if row.source != locate_source_span(target["source_quote"], request, history):
            _reject(f"{identity} 的来源与目标不一致")
        seen_objectives.add(identity)
    for row in predicates:
        if len(row.constraint_refs) != 1 or row.constraint_refs[0] not in constraints:
            _reject("条件要求引用了未知条件")
        ref = row.constraint_refs[0]
        scope = recommendation_ids if ref.startswith("global:") else {ref.split(":")[1]}
        if ref in seen_constraints or set(row.objective_ids) != scope:
            _reject(f"{ref} 的作用范围不一致")
        if row.source != locate_source_span(constraints[ref]["source_quote"], request, history):
            _reject(f"{ref} 的来源与条件不一致")
        seen_constraints.add(ref)
    if seen_objectives != objectives.keys() or seen_constraints != constraints.keys():
        _reject("存在未编译或额外的执行引用")
    if any(row.kind not in {"BUSINESS_OUTCOME", "PRODUCT_PREDICATE"}
           and (row.objective_ids or row.constraint_refs) for row in ir.requirements):
        _reject("Adaptive 上下文要求不能携带执行引用")
