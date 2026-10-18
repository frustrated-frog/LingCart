"""Built-in engine registry and profile-backed runner selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import uuid

from src.errors import AgentError
from src.harness.contracts import AgentEngine, RunRequest
from src.harness.mechanisms import HarnessMechanism, built_in_mechanisms, source_fingerprint
from src.harness.profile import HarnessProfile
from src.runtime import BudgetLimits, RunRuntime, current_runtime, runtime_scope


ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = ROOT / "configs" / "harness"


def harness_execution_identity(provenance: dict) -> dict:
    """Canonical executable identity; source revision remains audit metadata."""

    # Persisted selection events add trace fields (type, sequence, timestamp),
    # so identity must be an allow-list rather than "everything except".
    keys = ("profile_id", "profile_version", "profile_sha256", "engine", "mechanisms", "research")
    return {key: provenance.get(key) for key in keys}


def harness_execution_matches(left: dict, right: dict) -> bool:
    """Compare executable components while accepting legacy ID-only records."""

    if left.get("profile_id") != right.get("profile_id"):
        return False
    if left.get("legacy_profile_reference") or right.get("legacy_profile_reference"):
        return True
    if isinstance(left.get("engine"), dict) and isinstance(right.get("engine"), dict):
        return harness_execution_identity(left) == harness_execution_identity(right)
    left_sha, right_sha = left.get("execution_sha256"), right.get("execution_sha256")
    return not (left_sha and right_sha and left_sha != right_sha)


class FunctionEngine:
    """Adapter retaining the two production-tested engine entry points."""

    def __init__(self, engine_id: str, engine_version: str, execution_mode: str):
        self.engine_id = engine_id
        self.engine_version = engine_version
        self.execution_mode = execution_mode

    @property
    def source_files(self) -> tuple[str, ...]:
        target = "src/adaptive.py" if self.engine_id == "adaptive" else "src/orchestrator.py"
        return ("src/harness/registry.py", target)

    def provenance(self) -> dict:
        return {
            "id": self.engine_id,
            "version": self.engine_version,
            "execution_mode": self.execution_mode,
            "implementation_sha256": source_fingerprint(self.source_files),
            "source_files": list(self.source_files),
        }

    def execute(self, request: RunRequest) -> dict:
        if self.engine_id == "adaptive":
            from src.adaptive import run_adaptive

            runner = run_adaptive
        elif self.engine_id == "orchestrator":
            from src.orchestrator import run_orchestrator

            runner = run_orchestrator
        else:  # pragma: no cover - registry validation makes this unreachable
            raise AgentError("HARNESS_ENGINE_UNKNOWN", "Harness 引擎不存在", retryable=False)
        return runner(
            request.message,
            request.history,
            request.pending_action,
            limits=request.limits,
        )


class HarnessRunner:
    """Callable compatible with durable.run_conversation and evaluation runners."""

    def __init__(self, profile: HarnessProfile, engine: AgentEngine, mechanisms: tuple[HarnessMechanism, ...] = ()):
        self.profile = profile
        self.engine = engine
        self.execution_mode = engine.execution_mode
        self.mechanisms = {mechanism.slot: mechanism for mechanism in mechanisms}
        self._provenance = self._build_provenance()

    def _build_provenance(self) -> dict:
        engine = self.engine.provenance() if hasattr(self.engine, "provenance") else {
            "id": self.engine.engine_id, "version": self.engine.engine_version,
            "execution_mode": self.engine.execution_mode,
        }
        mechanisms = []
        for mechanism in self.mechanisms.values():
            if hasattr(mechanism, "provenance"):
                mechanisms.append(mechanism.provenance())
            else:
                mechanisms.append({"slot": mechanism.slot, "id": mechanism.mechanism_id,
                                   "version": mechanism.mechanism_version})
        mechanisms.sort(key=lambda item: item["slot"])
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
        payload = {
            **self.profile.provenance(),
            "engine": engine,
            "mechanisms": mechanisms,
            "source_revision": revision,
        }
        encoded = json.dumps(harness_execution_identity(payload), ensure_ascii=False,
                             sort_keys=True, separators=(",", ":"))
        payload["execution_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
        return payload

    def provenance(self) -> dict:
        return json.loads(json.dumps(self._provenance, ensure_ascii=False))

    def bind(self, context) -> None:
        selected = [event for event in context.events if event.get("type") == "harness_profile_selected"]
        if selected:
            saved = selected[-1]
            if saved.get("profile_id") != self.profile.profile_id or saved.get("profile_sha256") != self.profile.fingerprint:
                raise AgentError("HARNESS_PROFILE_CONFLICT", "同一次运行不能切换 Harness Profile", retryable=False)
            if not harness_execution_matches(saved, self._provenance):
                raise AgentError("HARNESS_IMPLEMENTATION_MISMATCH", "当前机制实现与任务开始时的版本不一致", retryable=False)
        else:
            context.snapshot["harness"] = self.provenance()
            context.record("harness_profile_selected", **self.provenance())
        context.snapshot["harness"] = self.provenance()
        context.harness_profile = self.provenance()
        for mechanism in self.mechanisms.values():
            mechanism.install(context)

    def build_context(self, goal, *, preferences=None, observations=None, revision=0):
        mechanism = self.mechanisms.get("context")
        if mechanism is not None and hasattr(mechanism, "build_context"):
            return mechanism.build_context(goal, preferences=preferences, observations=observations, revision=revision)
        from src.context_engine import ContextEngine

        return ContextEngine(goal, preferences=preferences, observations=observations, revision=revision)

    def __call__(self, message, history=None, pending_action=None, *, limits=None):
        context = current_runtime()
        if context is None:
            default_limits = BudgetLimits.adaptive() if self.execution_mode == "adaptive" else BudgetLimits()
            context = RunRuntime(uuid.uuid4().hex, limits or default_limits)
            with runtime_scope(context):
                return self(message, history, pending_action)
        self.bind(context)
        result = dict(self.engine.execute(RunRequest(message, history, pending_action, limits)))
        result["harness"] = self.provenance()
        result["events"] = list(context.events)
        return result


class HarnessRegistry:
    """Fail-closed registry: a profile is runnable only with an exact engine version."""

    def __init__(self):
        self._engines: dict[tuple[str, str], AgentEngine] = {}
        self._mechanisms: dict[tuple[str, str, str], HarnessMechanism] = {}
        self._profiles: dict[str, HarnessProfile] = {}

    def register_engine(self, engine: AgentEngine):
        key = (engine.engine_id, engine.engine_version)
        if key in self._engines:
            raise ValueError(f"引擎已注册：{engine.engine_id}@{engine.engine_version}")
        self._engines[key] = engine

    def register_mechanism(self, mechanism: HarnessMechanism):
        key = (mechanism.slot, mechanism.mechanism_id, mechanism.mechanism_version)
        if key in self._mechanisms:
            raise ValueError(f"机制已注册：{mechanism.slot}/{mechanism.mechanism_id}@{mechanism.mechanism_version}")
        self._mechanisms[key] = mechanism

    def register_profile(self, profile: HarnessProfile):
        if profile.profile_id in self._profiles:
            raise ValueError(f"Profile 已注册：{profile.profile_id}")
        key = (profile.engine.id, profile.engine.version)
        if key not in self._engines:
            raise ValueError(f"Profile 引用了未注册引擎：{profile.engine.id}@{profile.engine.version}")
        for reference in profile.mechanisms:
            mechanism = self._mechanisms.get((reference.slot, reference.id, reference.version))
            if mechanism is None:
                raise ValueError(f"Profile 引用了未注册机制：{reference.slot}/{reference.id}@{reference.version}")
            if profile.engine.id not in mechanism.compatible_engines:
                raise ValueError(f"机制 {reference.id} 不兼容引擎 {profile.engine.id}")
            mechanism.configure(dict(reference.configuration))
        self._profiles[profile.profile_id] = profile

    def load_profiles(self, directory: Path = PROFILE_DIR):
        for path in sorted(directory.glob("*.json")):
            # JSON arrays are intentionally accepted as immutable tuples while
            # strict Python callers must provide the exact tuple types.
            self.register_profile(HarnessProfile.model_validate_json(path.read_text()))

    def profiles(self) -> tuple[HarnessProfile, ...]:
        return tuple(self._profiles[key] for key in sorted(self._profiles))

    def resolve(self, profile_id: str) -> HarnessRunner:
        profile = self._profiles.get(profile_id)
        if profile is None:
            available = "、".join(sorted(self._profiles)) or "无"
            raise AgentError("HARNESS_PROFILE_UNKNOWN", f"未知 Harness Profile：{profile_id}；可用项：{available}", retryable=False)
        mechanisms = tuple(
            self._mechanisms[(item.slot, item.id, item.version)].configure(dict(item.configuration))
            for item in profile.mechanisms
        )
        return HarnessRunner(profile, self._engines[(profile.engine.id, profile.engine.version)], mechanisms)


def build_registry(profile_dir: Path = PROFILE_DIR) -> HarnessRegistry:
    registry = HarnessRegistry()
    registry.register_engine(FunctionEngine("orchestrator", "1", "auto"))
    registry.register_engine(FunctionEngine("adaptive", "1", "adaptive"))
    for mechanism in built_in_mechanisms():
        registry.register_mechanism(mechanism)
    registry.load_profiles(profile_dir)
    return registry


_REGISTRY: HarnessRegistry | None = None


def get_registry() -> HarnessRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = build_registry()
    return _REGISTRY


def get_runner(profile_id: str) -> HarnessRunner:
    return get_registry().resolve(profile_id)


def get_product_runner(profile_id: str) -> HarnessRunner:
    """Resolve only profiles approved for ordinary product traffic."""

    runner = get_runner(profile_id)
    if runner.profile.research.status not in {"baseline", "promoted"}:
        raise AgentError(
            "HARNESS_PROFILE_NOT_APPROVED",
            f"Harness Profile {profile_id} 仅限研究评测，尚未批准用于产品请求",
            retryable=False,
        )
    return runner


def unwrap_harness_runner(runner) -> HarnessRunner | None:
    candidate = getattr(runner, "harness_runner", runner)
    return candidate if isinstance(candidate, HarnessRunner) else None


def runner_execution_contract(runner, *, execution_mode: str | None = None) -> dict:
    selected = unwrap_harness_runner(runner)
    mode = getattr(runner, "execution_mode", None) or execution_mode or "auto"
    if mode not in {"auto", "adaptive"}:
        raise AgentError("INVALID_EXECUTION_MODE", "不支持该执行方式", retryable=False)
    return {"execution_mode": mode, "harness": selected.provenance() if selected else None}


def persisted_harness(claim: dict) -> dict | None:
    snapshot = claim.get("snapshot") or claim.get("checkpoint") or {}
    saved = snapshot.get("harness") or (snapshot.get("execution_contract") or {}).get("harness")
    if saved:
        return saved
    event = next((event for event in reversed(claim.get("events", []))
                  if event.get("type") == "harness_profile_selected"), None)
    if event:
        return event
    # Compatibility for runs created before atomic execution contracts.
    profile_id = snapshot.get("harness_profile_id")
    return {"profile_id": profile_id, "legacy_profile_reference": True} if profile_id else None


def resolve_claim_runner(claim: dict, runner=None):
    """Restore the original Profile and reject execution with changed code."""

    saved = persisted_harness(claim)
    selected = unwrap_harness_runner(runner)
    if saved and selected is None:
        if runner is not None:
            raise AgentError("HARNESS_PROFILE_REQUIRED", "该任务必须使用开始时的 Harness Profile 恢复", retryable=False)
        runner = selected = get_runner(saved["profile_id"])
    if saved and selected is not None:
        current = selected.provenance()
        if (current["profile_id"] != saved.get("profile_id") or
                (saved.get("profile_sha256") and current["profile_sha256"] != saved.get("profile_sha256"))):
            raise AgentError("HARNESS_PROFILE_CONFLICT", "恢复 Runner 与任务原 Profile 不一致", retryable=False)
        if not harness_execution_matches(saved, current):
            raise AgentError("HARNESS_IMPLEMENTATION_MISMATCH", "当前源码无法恢复任务开始时的机制实现", retryable=False)
    return runner

