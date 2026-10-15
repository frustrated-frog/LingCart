"""Built-in engine registry and profile-backed runner selection."""
from __future__ import annotations

from pathlib import Path
import uuid

from src.errors import AgentError
from src.harness.contracts import AgentEngine, RunRequest
from src.harness.profile import HarnessProfile
from src.runtime import BudgetLimits, RunRuntime, current_runtime, runtime_scope


ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = ROOT / "configs" / "harness"


class FunctionEngine:
    """Adapter retaining the two production-tested engine entry points."""

    def __init__(self, engine_id: str, engine_version: str, execution_mode: str):
        self.engine_id = engine_id
        self.engine_version = engine_version
        self.execution_mode = execution_mode

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

    def __init__(self, profile: HarnessProfile, engine: AgentEngine):
        self.profile = profile
        self.engine = engine
        self.execution_mode = engine.execution_mode

    def __call__(self, message, history=None, pending_action=None, *, limits=None):
        context = current_runtime()
        if context is None:
            default_limits = BudgetLimits.adaptive() if self.execution_mode == "adaptive" else BudgetLimits()
            context = RunRuntime(uuid.uuid4().hex, limits or default_limits)
            with runtime_scope(context):
                return self(message, history, pending_action)

        selected = [event for event in context.events if event.get("type") == "harness_profile_selected"]
        if selected and selected[-1].get("profile_sha256") != self.profile.fingerprint:
            raise AgentError("HARNESS_PROFILE_CONFLICT", "同一次运行不能切换 Harness Profile", retryable=False)
        if not selected:
            context.record("harness_profile_selected", **self.profile.provenance())
        result = dict(self.engine.execute(RunRequest(message, history, pending_action, limits)))
        result["harness"] = self.profile.provenance()
        result["events"] = list(context.events)
        return result


class HarnessRegistry:
    """Fail-closed registry: a profile is runnable only with an exact engine version."""

    def __init__(self):
        self._engines: dict[tuple[str, str], AgentEngine] = {}
        self._profiles: dict[str, HarnessProfile] = {}

    def register_engine(self, engine: AgentEngine):
        key = (engine.engine_id, engine.engine_version)
        if key in self._engines:
            raise ValueError(f"引擎已注册：{engine.engine_id}@{engine.engine_version}")
        self._engines[key] = engine

    def register_profile(self, profile: HarnessProfile):
        if profile.profile_id in self._profiles:
            raise ValueError(f"Profile 已注册：{profile.profile_id}")
        key = (profile.engine.id, profile.engine.version)
        if key not in self._engines:
            raise ValueError(f"Profile 引用了未注册引擎：{profile.engine.id}@{profile.engine.version}")
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
        return HarnessRunner(profile, self._engines[(profile.engine.id, profile.engine.version)])


def build_registry(profile_dir: Path = PROFILE_DIR) -> HarnessRegistry:
    registry = HarnessRegistry()
    registry.register_engine(FunctionEngine("orchestrator", "1", "auto"))
    registry.register_engine(FunctionEngine("adaptive", "1", "adaptive"))
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

