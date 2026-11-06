from dataclasses import replace

import pytest

from src import tools
from src.capability_planner import search_plan
from src.capability_planning import CapabilityPlanningMechanism
from src.capability_registry import (
    CapabilityContract, CapabilityRegistration, CapabilityRegistry, CapabilitySpec,
    CompiledOperation, get_capability_registry, use_capability_registry,
)
from src.errors import AgentError
from src.intake import REFUND_ELIGIBILITY, REFUND_REQUEST, TaskIntake
from src.runtime import ToolSpec
from src.task_plan import TaskPlan, TaskStep, build_plan


def read(order_id: int):
    raise AssertionError("编译依赖不得调用工具")


READ = ToolSpec(read)


def registration(kind, contracts=(), handlers=None):
    return CapabilityRegistration(
        CapabilitySpec(kind, kind, "测试能力", "order", "BUSINESS_DATA"),
        lambda _objective, _request: (),
        tools=handlers if handlers is not None else {c.tool: READ for c in contracts},
        planning_contracts=tuple(contracts),
    )


def contract(tool, output, *, requires=(), default_for=()):
    return CapabilityContract(tool=tool, scope="order", automatic=True,
                              requires=requires, provides={output: ("order_id",)}, default_for=default_for)


def compile_read(registry, tool="target", args=None):
    return registry.compile_read_dependencies(
        CompiledOperation("result", "order", tool, {"order_id": 1001} if args is None else args)
    )


def test_registry_does_not_infer_builtin_contracts_from_tool_names():
    registry = CapabilityRegistry((registration("legacy", handlers={"get_order_status": tools.get_order_status}),))
    assert registry.planning_contracts() == ()
    with pytest.raises(AgentError) as error:
        compile_read(registry, "get_order_status")
    assert error.value.code == "CAPABILITY_CONTRACT_MISSING"


def test_contract_snapshots_and_registration_input_cannot_mutate_registry():
    declaration = contract("target", "state")
    registry = CapabilityRegistry((registration("read", [declaration]),))
    version = registry.version
    declaration.provides["injected"] = ("field",)
    registry.planning_contracts()[0].provides.clear()
    assert registry.planning_contracts()[0].provides == {"state": ("order_id",)}
    assert registry.version == version


@pytest.mark.parametrize("duplicate_in_one_registration", [False, True])
def test_conflicting_contracts_are_rejected_atomically(duplicate_in_one_registration):
    original = contract("target", "state")
    registry = CapabilityRegistry(() if duplicate_in_one_registration else (registration("first", [original]),))
    before = (registry.version, registry.execution_tools(), registry.prompt_catalog(), registry.planning_contracts())
    changed = contract("target", "other")
    additions = [contract("new_tool", "new"), changed]
    if duplicate_in_one_registration:
        additions.insert(0, original)
    with pytest.raises(ValueError, match="conflicting planning contracts"):
        registry.register(registration("second", additions))
    assert (registry.version, registry.execution_tools(), registry.prompt_catalog(), registry.planning_contracts()) == before


def test_equal_shared_contracts_are_allowed_but_default_provider_conflicts_are_atomic():
    first = contract("first", "state", default_for=("state",))
    registry = CapabilityRegistry((registration("one", [first]), registration("two", [first])))
    assert len(registry.planning_contracts()) == 1
    version = registry.version
    with pytest.raises(ValueError, match="conflicting default providers"):
        registry.register(registration("three", [contract("second", "state", default_for=("state",))]))
    assert registry.version == version
    assert "second" not in registry.execution_tools()


@pytest.mark.parametrize("requires,provider", [("order.details", "get_order_details"), ("order.payment", "get_payment_status")])
def test_one_dependency_declaration_changes_fast_semantic_and_cgap_plans(requires, provider):
    baseline = get_capability_registry()
    registrations = []
    for row in baseline.prompt_catalog():
        entry = baseline.registration(row["objective_kind"])
        declarations = tuple(c.model_copy(update={"requires": (requires,)})
                             if c.tool == "check_refund_eligibility" else c for c in entry.planning_contracts)
        registrations.append(replace(entry, planning_contracts=declarations))
    registry = CapabilityRegistry(tuple(registrations))
    state = {"goal": {"objectives": [{"id": "refund", "kind": "check_refund_eligibility", "order_id": 1001}]}}
    mechanism = CapabilityPlanningMechanism()
    baseline_sha = mechanism.problem(state, TaskPlan([], []))[-1]
    # 显式实例编译必须读取该实例，而不是全局默认注册表。
    operations = registry.compile({"kind": REFUND_ELIGIBILITY, "order_id": 1001}, "查询退款资格")
    assert get_capability_registry() is baseline
    assert [op.operation for op in operations] == [provider, "check_refund_eligibility"]
    assert operations[-1].depends_on == (operations[0].ref,)
    with use_capability_registry(registry):
        plan = build_plan(TaskIntake("查询退款资格", (REFUND_ELIGIBILITY,), {"order_id": 1001}))
        assert [s.tool for s in plan.steps] == [provider, "check_refund_eligibility"]
        assert plan.steps[-1].depends_on == [plan.steps[0].step_id]
        actions, facts, goals, _, catalog_sha = mechanism.problem(state, TaskPlan([], []))
        assert [a.tool for a in search_plan(actions, facts, goals).actions] == [provider, "check_refund_eligibility"]
        assert catalog_sha != baseline_sha
    assert registry.version != baseline.version


def test_changing_default_provider_does_not_remove_cgap_alternatives_or_observation_reuse():
    baseline = get_capability_registry()
    registrations = []
    for row in baseline.prompt_catalog():
        entry = baseline.registration(row["objective_kind"])
        declarations = tuple(c.model_copy(update={"default_for": ("order.status",) if c.tool == "get_order_details" else ()})
                             if c.tool in {"get_order_details", "get_order_status"} else c
                             for c in entry.planning_contracts)
        registrations.append(replace(entry, planning_contracts=declarations))
    registry = CapabilityRegistry(tuple(registrations))
    with use_capability_registry(registry):
        plan = build_plan(TaskIntake("申请退款", (REFUND_REQUEST,), {"order_id": 1001}))
        assert [s.tool for s in plan.steps] == ["get_order_details", "check_refund_eligibility", "prepare_refund"]
        assert plan.steps[-1].depends_on == [plan.steps[1].step_id]
        state = {"goal": {"objectives": [{"id": "refund", "kind": "check_refund_eligibility", "order_id": 1001}]}}
        observed = TaskStep("observed", "order", "get_order_status", {"order_id": 1001}, status="REUSED",
                            result={"order_id": 1001, "status": "delivered"})
        actions, facts, goals, _, _ = CapabilityPlanningMechanism().problem(state, TaskPlan([], [observed]))
        assert {"get_order_status", "get_order_details", "track_shipment"} <= {a.tool for a in actions}
        assert [a.tool for a in search_plan(actions, facts, goals).actions] == ["check_refund_eligibility"]


@pytest.mark.parametrize("contracts,code", [
    ([contract("target", "done", requires=("missing",))], "CAPABILITY_DEPENDENCY_MISSING"),
    ([contract("target", "done", requires=("state",)), contract("a", "state"), contract("b", "state")],
     "CAPABILITY_DEPENDENCY_AMBIGUOUS"),
    ([contract("target", "done", requires=("state",)), contract("a", "state", requires=("done",))],
     "CAPABILITY_DEPENDENCY_CYCLE"),
])
def test_invalid_dependency_graphs_fail_before_any_tool_call(contracts, code):
    with pytest.raises(AgentError) as error:
        compile_read(CapabilityRegistry((registration("test", contracts),)))
    assert error.value.code == code


def test_dependency_expansion_deduplicates_shared_producers_and_keeps_object_binding():
    shared = CapabilityContract(tool="shared", scope="order", automatic=True,
                                provides={"first": ("order_id",), "second": ("order_id",)})
    registry = CapabilityRegistry((registration("test", [
        shared, contract("target", "done", requires=("first", "second")),
    ]),))
    operations = compile_read(registry)
    assert [op.operation for op in operations] == ["shared", "target"]
    assert operations[-1].depends_on == (operations[0].ref,)
    assert all(op.arguments == {"order_id": 1001} for op in operations)


def test_dependency_expansion_requires_valid_inputs_and_cannot_introduce_writes():
    declaration = contract("target", "done")
    registry = CapabilityRegistry((registration("test", [declaration]),))
    with pytest.raises(AgentError) as error:
        compile_read(registry, args={})
    assert error.value.code == "CAPABILITY_INPUT_MISSING"
    write = contract("write", "state")
    registry = CapabilityRegistry((registration("test", [
        declaration.model_copy(update={"requires": ("state",)}), write,
    ], handlers={"target": READ, "write": ToolSpec(read, side_effect=True)}),))
    with pytest.raises(AgentError) as error:
        compile_read(registry)
    assert error.value.code == "CAPABILITY_DEPENDENCY_NOT_READ_ONLY"


def test_dependency_expansion_is_bounded():
    declarations = [contract(f"tool-{i}", f"fact-{i}", requires=(f"fact-{i+1}",) if i < 64 else ())
                    for i in range(65)]
    registry = CapabilityRegistry((registration("test", declarations),))
    with pytest.raises(AgentError) as error:
        compile_read(registry, "tool-0")
    assert error.value.code == "CAPABILITY_DEPENDENCY_LIMIT"
