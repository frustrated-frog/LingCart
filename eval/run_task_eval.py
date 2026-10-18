"""Offline task evaluation: real tools, isolated SQLite, scripted models only.

Run ``python eval/run_task_eval.py --dry-run`` to validate without importing the
application, or ``python eval/run_task_eval.py --output /tmp/task-eval.json``.
The live dialogue-understanding benchmark in run_eval.py is intentionally separate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import socket
import statistics
import sys
import threading
import time
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(__file__).with_name("task_cases.json")
DEFAULT_BASELINE = Path(__file__).with_name("baseline_v4.json")
TOOLS = {
    "get_order_status", "get_order_details", "track_shipment",
    "check_refund_eligibility", "initiate_refund", "get_payment_status",
    "process_payment", "answer_faq", "escalate_to_human", "search_products",
    "compare_products",
}
MUTATIONS = {"process_payment", "initiate_refund"}
STATUSES = {
    "SUCCEEDED", "FAILED", "WAITING_APPROVAL", "WAITING_CLARIFICATION",
    "ESCALATED", "WAITING_HUMAN", "CANCELLED", "REJECTED", "PARTIAL",
}
EXPECT_KEYS = {
    "statuses", "terminal_reason", "tools", "forbidden_tools", "pending_action",
    "pending_present", "reply_contains_all", "reply_contains_any", "db",
    "allowed_mutations", "max_model_invocations", "max_tool_calls",
    "error_code", "error_present", "max_physical_calls_per_tool",
}
_MISSING = object()


def load_dataset(path: str | Path = DEFAULT_DATASET) -> dict:
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_dataset(dataset)
    return dataset


def dataset_fingerprint(dataset: dict) -> str:
    encoded = json.dumps(dataset, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_dataset(dataset: dict) -> None:
    """Reject broken expectations before the application or any model loads."""
    if not isinstance(dataset, dict) or dataset.get("schema_version") != 1:
        raise ValueError("dataset.schema_version must be 1")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("dataset.cases must be a nonempty list")
    ids = set()
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError(f"missing or duplicate case id: {case_id!r}")
        ids.add(case_id)
        for key in ("category", "language", "description"):
            if not isinstance(case.get(key), str) or not case[key]:
                raise ValueError(f"{case_id}: {key} must be nonempty")
        turns = case.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"{case_id}: turns must be nonempty")
        for index, turn in enumerate(turns):
            prefix = f"{case_id} turn {index + 1}"
            if not isinstance(turn.get("message"), str) or not turn["message"].strip():
                raise ValueError(f"{prefix}: message must be nonempty")
            expect = turn.get("expect")
            if not isinstance(expect, dict) or not expect.get("statuses"):
                raise ValueError(f"{prefix}: expected statuses are required")
            if set(expect) - EXPECT_KEYS:
                raise ValueError(f"{prefix}: unknown expectation keys {set(expect) - EXPECT_KEYS}")
            if not set(expect["statuses"]) <= STATUSES:
                raise ValueError(f"{prefix}: invalid expected status")
            for tool in expect.get("tools", []):
                if tool.get("name") not in TOOLS:
                    raise ValueError(f"{prefix}: unknown required tool {tool.get('name')}")
                if not isinstance(tool.get("args", {}), dict):
                    raise ValueError(f"{prefix}: tool args must be an object")
            if not set(expect.get("forbidden_tools", [])) <= TOOLS:
                raise ValueError(f"{prefix}: unknown forbidden tool")
            if "pending_from_turn" in turn and not 0 <= turn["pending_from_turn"] < index:
                raise ValueError(f"{prefix}: pending_from_turn must reference an earlier turn")
            for message in turn.get("history", []):
                if message.get("role") not in {"user", "assistant"} or not isinstance(message.get("content"), str):
                    raise ValueError(f"{prefix}: invalid history message")
        if case.get("model", {}).get("mode", "disabled") not in {"disabled", "scripted"}:
            raise ValueError(f"{case_id}: model mode must be disabled or scripted")
        script = case.get("model", {}).get("responses", [])
        if not isinstance(script, list):
            raise ValueError(f"{case_id}: model responses must be a list")
        for response in script:
            if not isinstance(response, dict) or not set(response) <= {"content", "tool_calls", "raise"}:
                raise ValueError(f"{case_id}: invalid scripted model response")


def _json(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _subset(expected, actual) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _subset(value, actual[key]) for key, value in expected.items()
        )
    return expected == actual


def _get_path(value, path: str):
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part, _MISSING)
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return _MISSING
    return value


def _diff(before, after, prefix="") -> list[str]:
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(set(before) | set(after)):
            path = f"{prefix}.{key}" if prefix else str(key)
            result.extend(_diff(before.get(key, _MISSING), after.get(key, _MISSING), path))
        return result
    return [] if before == after else [prefix]


@contextmanager
def _offline_guard():
    attempts = []

    def blocked(*args, **kwargs):
        # Do not include network arguments: they could contain credentials.
        attempts.append("network access attempted")
        raise RuntimeError("Network access is disabled by the offline task evaluator")

    with ExitStack() as stack:
        for target in ("socket.create_connection", "socket.getaddrinfo", "socket.socket.connect", "socket.socket.connect_ex"):
            stack.enter_context(patch(target, blocked))
        yield attempts


def _runtime() -> dict:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # tools.py initializes/auto-seeds its DB at import. Point even this initial
    # import at memory; never touch support_agent.db or the user's env DB URL.
    with patch.dict(os.environ, {"SUPPORT_AGENT_DB_URL": "sqlite:///:memory:"}):
        return {
            "tools": importlib.import_module("src.tools"),
            "db": importlib.import_module("src.backend.db"),
            "seed": importlib.import_module("src.backend.seed").seed,
            "orchestrator": importlib.import_module("src.orchestrator"),
            "messages": importlib.import_module("langchain_core.messages"),
        }


def _snapshot(runtime: dict, engine) -> dict:
    db = runtime["db"]
    state = {}
    with db.get_session_factory(engine)() as session:
        for label, cls in (
            ("orders", db.Order), ("payments", db.Payment), ("refunds", db.Refund),
            ("products", db.Product), ("shipments", db.ShipmentEvent),
            ("tickets", db.SupportTicket),
        ):
            key_name = "order_id" if label in {"orders", "payments", "refunds"} else (
                "product_id" if label == "products" else "ticket_id" if label == 'tickets' else "event_id"
            )
            state[label] = {
                str(getattr(row, key_name)): {
                    column.key: _json(getattr(row, column.key)) for column in cls.__table__.columns
                }
                for row in session.query(cls).all()
            }
    return state


@contextmanager
def _isolated_db(runtime: dict, fixture: dict):
    tools, db = runtime["tools"], runtime["db"]
    original_engine, original_factory = tools._engine, tools._SessionLocal
    engine = db.get_engine("sqlite:///:memory:")
    try:
        tools.configure_db(engine)
        with db.get_session_factory(engine)() as session:
            runtime["seed"](session)
            for order_id in fixture.get("remove_payments", []):
                session.query(db.Payment).filter_by(order_id=order_id).delete()
            for updates_key, cls, id_key in (
                ("payment_updates", db.Payment, "order_id"),
                ("refund_updates", db.Refund, "order_id"),
                ("product_updates", db.Product, "product_id"),
            ):
                for identifier, updates in fixture.get(updates_key, {}).items():
                    identifier = identifier if id_key == "product_id" else int(identifier)
                    row = session.query(cls).filter_by(**{id_key: identifier}).one()
                    for key, value in updates.items():
                        setattr(row, key, value)
            session.commit()
        yield engine
    finally:
        # Direct restoration avoids init_db() issuing DDL on a caller's DB.
        tools._engine, tools._SessionLocal = original_engine, original_factory
        engine.dispose()


class OfflineModel:
    """Count attempted calls, including exceptions; never instantiate a provider."""

    def __init__(self, runtime: dict, config: dict):
        self.runtime = runtime
        self.responses = list(config.get("responses", []))
        self.mode = config.get("mode", "disabled")
        self.acquisitions = 0
        self.invocations = 0
        self.responses_returned = 0

    def get_llm(self, *args, **kwargs):
        self.acquisitions += 1
        if self.mode == "disabled":
            raise RuntimeError("No LLM configured. Offline evaluation disables model providers.")
        return self

    def bind_tools(self, *args, **kwargs):
        return self

    def invoke(self, *args, **kwargs):
        index = self.invocations
        self.invocations += 1
        if index >= len(self.responses):
            raise RuntimeError("Offline FakeLLM script exhausted")
        response = self.responses[index]
        if response.get("raise"):
            kind = response["raise"]
            if kind == "timeout":
                raise TimeoutError("Offline injected model timeout")
            if kind == "unavailable":
                raise RuntimeError("No LLM configured. Offline injected model outage.")
            if kind == "rate_limit":
                raise RuntimeError("429 Offline injected model rate limit")
            raise ValueError("Offline injected malformed model response")
        self.responses_returned += 1
        return self.runtime["messages"].AIMessage(
            content=response.get("content", ""),
            tool_calls=response.get("tool_calls", []),
            usage_metadata={"input_tokens": 8, "output_tokens": 2, "total_tokens": 10},
        )


@contextmanager
def _patched_model(model: OfflineModel):
    provider_getter = sys.modules["src.llm"].get_llm
    original_modules = set(sys.modules)
    with ExitStack() as stack:
        # Source modules import get_llm by value. Patch every loaded src alias,
        # including aliases added by later Agent Core modules.
        for name, module in list(sys.modules.items()):
            if (name == "src" or name.startswith("src.")) and hasattr(module, "get_llm"):
                stack.enter_context(patch.object(module, "get_llm", model.get_llm))
        try:
            yield
        finally:
            # A module imported lazily during a task may retain the patched
            # src.llm alias. Restore it as well before returning to the caller.
            for name in set(sys.modules) - original_modules:
                module = sys.modules[name]
                if name.startswith("src.") and getattr(module, "get_llm", None) == model.get_llm:
                    module.get_llm = provider_getter


@contextmanager
def _observe_tools(runtime: dict):
    """Observe original backend function execution even through cached registries.

    A trace entry alone is not evidence: it must match a physical tool return.
    Profiling avoids replacing real tools with mocked business behavior.
    """
    filename = str(Path(runtime["tools"].__file__).resolve())
    observed, active = [], {}
    old_profile, old_thread_profile = sys.getprofile(), threading.getprofile()

    def profiler(frame, event, arg):
        if frame.f_code.co_filename != filename or frame.f_code.co_name not in TOOLS:
            return
        identity = (threading.get_ident(), id(frame))
        if event == "call":
            count = frame.f_code.co_argcount + frame.f_code.co_kwonlyargcount
            record = {
                "tool": frame.f_code.co_name,
                "args": _json({name: frame.f_locals[name] for name in frame.f_code.co_varnames[:count]}),
                "result": None,
            }
            observed.append(record)
            active[identity] = record
        elif event == "return" and identity in active:
            active.pop(identity)["result"] = _json(arg)

    sys.setprofile(profiler)
    threading.setprofile(profiler)
    try:
        yield observed
    finally:
        sys.setprofile(old_profile)
        threading.setprofile(old_thread_profile)


def _history(runtime: dict, history: list) -> list:
    classes = {"user": runtime["messages"].HumanMessage, "assistant": runtime["messages"].AIMessage}
    return [classes[item["role"]](content=item["content"]) for item in history]


def judge_turn(expect: dict, result: dict, before: dict, after: dict,
               observed: list, model_invocations: int, network_attempts: int = 0) -> list[str]:
    """Return concrete unmet criteria; there is no LLM-as-judge or text scoring."""
    failures = []

    def require(condition, message):
        if not condition:
            failures.append(message)

    require(result.get("status") in expect["statuses"],
            f"status: expected {expect['statuses']}, got {result.get('status')!r}")
    if "terminal_reason" in expect:
        require(result.get("terminal_reason") == expect["terminal_reason"],
                f"terminal_reason: expected {expect['terminal_reason']}, got {result.get('terminal_reason')!r}")
    if "error_code" in expect:
        require((result.get("error") or {}).get("code") == expect["error_code"],
                f"expected error code {expect['error_code']}")
    if "error_present" in expect:
        require(bool(result.get("error")) == expect["error_present"], "error presence does not match")
    require(bool(result.get("reply")), "reply is empty")
    reply = str(result.get("reply", "")).casefold()
    for phrase in expect.get("reply_contains_all", []):
        require(phrase.casefold() in reply, f"reply missing {phrase!r}")
    any_phrases = expect.get("reply_contains_any", [])
    if any_phrases:
        require(any(phrase.casefold() in reply for phrase in any_phrases),
                f"reply missing all of {any_phrases!r}")

    trace = result.get("trace") or []
    for required in expect.get("tools", []):
        matching = [entry for entry in trace if entry.get("tool") == required["name"]
                    and _subset(required.get("args", {}), entry.get("args") or {})]
        require(bool(matching), f"missing tool evidence {required['name']}({required.get('args', {})})")
        if not matching:
            continue
        if "result" in required:
            require(any(_subset(required["result"], entry.get("result")) for entry in matching),
                    f"{required['name']}: result does not contain {required['result']!r}")
        if "products" in required:
            constraints = required["products"]
            products = matching[0].get("result", {}).get("products", [])
            require(len(products) >= constraints.get("min_count", 0), "too few matching products")
            if "ids" in constraints:
                require({p.get("product_id") for p in products} == set(constraints["ids"]),
                        "compared/searched product IDs do not match")
            for product in products:
                catalog = before["products"].get(product.get("product_id"))
                require(catalog is not None, "product not present in seeded catalog")
                if catalog:
                    require(_subset(catalog, product), "product values differ from actual catalog row")
                if "category" in constraints:
                    require(product.get("category") == constraints["category"], "wrong product category")
                if "max_price" in constraints:
                    require(product.get("price", float("inf")) <= constraints["max_price"], "product exceeds budget")
                require(product.get("data_source") == "DEMO_CATALOG", "product is not labeled as demo catalog")

    for entry in trace:
        # Failed unknown-tool/validation attempts have no backend return.
        if entry.get("status") == "FAILED":
            continue
        real_returns = [record for record in observed if record["tool"] == entry.get("tool")
                        and _subset(entry.get("args") or {}, record["args"])
                        and record["result"] == _json(entry.get("result"))]
        require(bool(real_returns), f"trace lacks physical backend evidence: {entry.get('tool')}")
    forbidden = set(expect.get("forbidden_tools", MUTATIONS | {"escalate_to_human"}))
    actual_names = {record["tool"] for record in observed} | {entry.get("tool") for entry in trace}
    require(not (actual_names & forbidden), f"forbidden tools executed: {sorted(actual_names & forbidden)}")

    pending = result.get("pending_action")
    if "pending_present" in expect:
        require(bool(pending) == expect["pending_present"], "pending action presence does not match")
    if "pending_action" in expect:
        require(_subset(expect["pending_action"], pending), "pending action differs from expected proposal")
    changed = _diff(before, after)
    for entry in trace:
        if entry.get('tool') == 'escalate_to_human' and entry.get('status') != 'FAILED':
            output = entry.get('result') or {}
            ticket = after.get('tickets', {}).get(output.get('ticket_id'))
            require(bool(ticket) and ticket.get('context') == entry['args'].get('context')
                    and ticket.get('reason') == entry['args'].get('reason') and ticket.get('status') == 'QUEUED',
                    'human request did not create the claimed durable ticket')
            require(output.get('mode') == 'DEMO' and output.get('delivered') is False and output.get('escalated') is False,
                    'demo ticket must not claim real human delivery')
    permitted = expect.get("allowed_mutations", [])
    unexpected = [path for path in changed if not any(path == allowed or path.startswith(allowed + ".") for allowed in permitted)]
    require(not unexpected, f"unexpected business DB mutations: {unexpected}")
    for path, expected_value in expect.get("db", {}).items():
        actual = _get_path(after, path)
        require(actual is not _MISSING and actual == expected_value,
                f"DB {path}: expected {expected_value!r}, got {actual if actual is not _MISSING else '<missing>'!r}")

    if "max_model_invocations" in expect:
        require(model_invocations <= expect["max_model_invocations"], "model invocation budget exceeded")
    if "max_tool_calls" in expect:
        require(len(observed) <= expect["max_tool_calls"], "physical backend call budget exceeded")
    for name, maximum in expect.get("max_physical_calls_per_tool", {}).items():
        require(sum(record["tool"] == name for record in observed) <= maximum, f"{name} executed too many times")
    require(network_attempts == 0, "offline evaluator observed a network attempt")
    events, run_id = result.get("events") or [], result.get("run_id")
    require(bool(run_id), "run_id is missing")
    require(bool(events) and events[-1].get("type") == "terminal", "terminal event is missing")
    require([event.get("sequence") for event in events] == list(range(1, len(events) + 1)), "events are not sequential")
    require(all(event.get("run_id") == run_id for event in events), "events are not linked to the same run")
    return failures


def evaluate_case(case: dict, runtime: dict | None = None) -> dict:
    with _offline_guard() as network_attempts:
        runtime = runtime or _runtime()
        with _isolated_db(runtime, case.get("fixture", {})) as engine:
            model = OfflineModel(runtime, case.get("model", {}))
            history, pending, results, turns = [], None, [], []
            with _patched_model(model):
                for turn in case["turns"]:
                    if "history" in turn:
                        history = list(turn["history"])
                    if "pending_action" in turn:
                        pending = turn["pending_action"]
                    if "pending_from_turn" in turn:
                        pending = results[turn["pending_from_turn"]].get("pending_action")
                    before = _snapshot(runtime, engine)
                    invocations_before, acquisitions_before = model.invocations, model.acquisitions
                    network_before = len(network_attempts)
                    started_at = time.perf_counter()
                    exception = None
                    with _observe_tools(runtime) as observed:
                        try:
                            result = runtime["orchestrator"].run_orchestrator(
                                turn["message"], history=_history(runtime, history), pending_action=pending,
                            )
                        except Exception as exc:  # One broken task must not hide the other results.
                            exception = {"type": type(exc).__name__, "message": str(exc)}
                            result = {"status": "EXCEPTION", "trace": [], "events": [], "reply": ""}
                    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
                    after = _snapshot(runtime, engine)
                    invocations = model.invocations - invocations_before
                    failures = judge_turn(turn["expect"], result, before, after, observed, invocations,
                                          len(network_attempts) - network_before)
                    if exception:
                        failures.insert(0, f"uncaught {exception['type']}: {exception['message']}")
                    turns.append({
                        "message": turn["message"], "passed": not failures, "failures": failures,
                        "actual_status": result.get("status"), "terminal_reason": result.get("terminal_reason"),
                        "expected_statuses": turn["expect"]["statuses"],
                        "trace_tools": [entry.get("tool") for entry in result.get("trace") or []],
                        "physical_tools": [record["tool"] for record in observed],
                        "trace": _json(result.get("trace") or []), "events": _json(result.get("events") or []),
                        "observed_tools": observed, "changed_db_paths": _diff(before, after),
                        "pending_action": _json(result.get("pending_action")), "reply": result.get("reply"),
                        "error": _json(result.get("error")), "exception": exception,
                        "metrics": {"latency_ms": elapsed_ms, "model_invocations": invocations,
                                    "model_acquisitions": model.acquisitions - acquisitions_before,
                                    "tool_calls": len(result.get("trace") or []), "physical_tool_calls": len(observed),
                                    "usage": _json(result.get("usage") or {}),
                                    "usage_source": "synthetic_fake_llm" if model.mode != "disabled" else "runtime_no_provider"},
                    })
                    results.append(result)
                    history += [{"role": "user", "content": turn["message"]},
                                {"role": "assistant", "content": str(result.get("reply", ""))}]
                    pending = result.get("pending_action")
            return {"id": case["id"], "category": case["category"], "language": case["language"],
                    "description": case["description"], "model_mode": model.mode,
                    "passed": all(turn["passed"] for turn in turns), "turns": turns}


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 3)


def build_report(dataset: dict, cases: list[dict]) -> dict:
    turns = [turn for case in cases for turn in case["turns"]]
    metrics = [turn["metrics"] for turn in turns]
    latencies = [item["latency_ms"] for item in metrics]
    groups = {}
    for key in ("category", "language", "model_mode"):
        groups[key] = {}
        for case in cases:
            bucket = groups[key].setdefault(case[key], {"passed": 0, "total": 0})
            bucket["passed"] += int(case["passed"])
            bucket["total"] += 1
        for bucket in groups[key].values():
            bucket["success_rate"] = bucket["passed"] / bucket["total"]
    passed = sum(case["passed"] for case in cases)
    return {
        "schema_version": 1, "dataset_name": dataset.get("name"),
        "dataset_sha256": dataset_fingerprint(dataset),
        "generated_at": datetime.now(UTC).isoformat(), "offline": True,
        "summary": {"passed": passed, "total": len(cases), "success_rate": passed / len(cases) if cases else 0.0,
                    "turns_passed": sum(turn["passed"] for turn in turns), "turns_total": len(turns),
                    "by_category": groups["category"], "by_language": groups["language"], "by_model_mode": groups["model_mode"]},
        "metrics": {
            "model_invocations": sum(item["model_invocations"] for item in metrics),
            "model_acquisitions": sum(item["model_acquisitions"] for item in metrics),
            "tool_calls": sum(item["tool_calls"] for item in metrics),
            "physical_tool_calls": sum(item["physical_tool_calls"] for item in metrics),
            "latency_ms": {"sum": round(sum(latencies), 3), "mean": round(statistics.mean(latencies), 3) if latencies else 0,
                           "p50": _percentile(latencies, .5), "p95": _percentile(latencies, .95)},
            "tokens": {key: sum((item["usage"].get(key) or 0) for item in metrics)
                       for key in ("input_tokens", "output_tokens", "total_tokens")},
            "cost": {"estimated_amount": None, "currency": None, "billable_provider_requests": 0,
                     "reason": "No live providers or verified pricing. FakeLLM tokens are synthetic; no dollar estimate is inferred."},
        },
        "limitations": [
            "This is a fixed offline regression/task suite, not a measurement of live model quality.",
            "Each task uses a fresh demo seed and a single sequential run_orchestrator conversation.",
            "Physical tool counts include nested backend calls; trace tool counts describe runtime-visible calls.",
            "Latency includes tool-observer overhead and excludes initial imports, DB seeding and snapshots.",
            "No before/after benchmark is claimed; run an identical dataset against each revision for a comparison.",
        ],
        "cases": cases,
    }


def run_evaluation(dataset: dict, case_ids: list[str] | None = None) -> dict:
    validate_dataset(dataset)
    chosen = [case for case in dataset["cases"] if not case_ids or case["id"] in case_ids]
    if case_ids and set(case_ids) - {case["id"] for case in chosen}:
        raise ValueError(f"Unknown case ids: {set(case_ids) - {case['id'] for case in chosen}}")
    logger = logging.getLogger("support_agent")
    previous_level = logger.level
    try:
        logger.setLevel(logging.CRITICAL)
        with _offline_guard():
            runtime = _runtime()
            # Import configure_logging may assign INFO, so mute after import too.
            logger.setLevel(logging.CRITICAL)
            results = [evaluate_case(case, runtime) for case in chosen]
        return build_report(dataset, results)
    finally:
        logger.setLevel(previous_level)


def evaluate_gate(report: dict, baseline: dict) -> list[str]:
    """Compare deterministic quality/cost counters against a frozen baseline."""

    failures = []
    if baseline.get("schema_version") != 1:
        return ["baseline schema_version must be 1"]
    if report.get("dataset_sha256") != baseline.get("dataset_sha256"):
        failures.append("dataset fingerprint differs from the selected baseline")
    summary = report.get("summary", {})
    expected = baseline.get("expected", {})
    for report_key, baseline_key in (
        ("passed", "cases_passed"),
        ("total", "cases_total"),
        ("turns_passed", "turns_passed"),
        ("turns_total", "turns_total"),
    ):
        if summary.get(report_key) != expected.get(baseline_key):
            failures.append(
                f"{report_key}: expected {expected.get(baseline_key)!r}, got {summary.get(report_key)!r}"
            )
    metrics = report.get("metrics", {})
    for key, maximum in baseline.get("maximums", {}).items():
        actual = metrics.get(key)
        if not isinstance(actual, (int, float)) or actual > maximum:
            failures.append(f"{key}: maximum {maximum!r}, got {actual!r}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dry-run", action="store_true", help="validate JSON only; no application imports")
    parser.add_argument("--output", type=Path, help="write a detailed JSON report")
    parser.add_argument("--case", action="append", help="run one case ID; repeat to select several")
    parser.add_argument("--gate", action="store_true", help="compare the full report with eval/baseline_v4.json")
    parser.add_argument("--baseline", type=Path, help="use this baseline and enable the evaluation gate")
    args = parser.parse_args(argv)
    dataset = load_dataset(args.dataset)
    if args.dry_run:
        report = {"dataset_name": dataset.get("name"), "schema_version": 1, "validated": True,
                  "case_count": len(dataset["cases"]), "turn_count": sum(len(c["turns"]) for c in dataset["cases"]),
                  "categories": sorted({c["category"] for c in dataset["cases"]}), "application_imported": False}
        print(f"Dataset OK: {report['case_count']} tasks / {report['turn_count']} turns. No application imports or network calls.")
        exit_code = 0
    else:
        report = run_evaluation(dataset, args.case)
        for case in report["cases"]:
            print(f"{'PASS' if case['passed'] else 'FAIL'} {case['id']}: {case['description']}")
            for index, turn in enumerate(case["turns"], 1):
                for failure in turn["failures"]:
                    print(f"  turn {index}: {failure}")
        summary, metrics = report["summary"], report["metrics"]
        print(f"Task success: {summary['passed']}/{summary['total']} ({summary['success_rate']:.1%}); "
              f"turns: {summary['turns_passed']}/{summary['turns_total']}")
        print(f"Model invokes: {metrics['model_invocations']} (scripted only); "
              f"tools: {metrics['tool_calls']} trace / {metrics['physical_tool_calls']} physical; "
              f"latency p50/p95: {metrics['latency_ms']['p50']:.2f}/{metrics['latency_ms']['p95']:.2f} ms; "
              "monetary cost: unavailable (no verified pricing)")
        exit_code = 0 if summary["passed"] == summary["total"] else 1
        if args.gate or args.baseline:
            if args.case:
                parser.error("evaluation gate requires the full dataset; remove --case")
            baseline_path = args.baseline or DEFAULT_BASELINE
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            failures = evaluate_gate(report, baseline)
            report["gate"] = {
                "passed": not failures,
                "baseline": baseline.get("name", baseline_path.name),
                "failures": failures,
            }
            print(f"Evaluation gate: {'PASS' if not failures else 'FAIL'} ({baseline_path})")
            for failure in failures:
                print(f"  {failure}")
            if failures:
                exit_code = 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"JSON report: {args.output.resolve()}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
