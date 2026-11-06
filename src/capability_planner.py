"""有界能力状态空间搜索；预测事实只存在于搜索节点中。"""
from collections import deque
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlanningModel(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)


class CapabilityContract(PlanningModel):
    tool: str
    scope: Literal['order', 'catalog', 'question']
    requires: tuple[str, ...] = ()
    # 每个语义输出要求真实结果具备哪些非空字段；False 和 0 都是有效值。
    provides: dict[str, tuple[str, ...]]
    automatic: bool = False


class PlanningAction(PlanningModel):
    action_id: str
    tool: str
    scope: str
    args: dict = Field(default_factory=dict)
    requires: tuple[str, ...] = ()
    adds: tuple[str, ...]
    deletes: tuple[str, ...] = ()
    objective_ids: tuple[str, ...] = ()
    automatic: bool = False

    @model_validator(mode='after')
    def consistent_effects(self):
        if set(self.adds) & set(self.deletes):
            raise ValueError('规划动作不能同时增加和删除同一事实')
        return self


class SearchResult(PlanningModel):
    status: Literal['SOLVED', 'UNREACHABLE', 'SEARCH_LIMIT']
    actions: tuple[PlanningAction, ...] = ()
    expanded: int = 0
    missing: tuple[str, ...] = ()


def search_plan(actions, initial, goals, *, max_nodes=2048, max_steps=16):
    """BFS 返回最短动作序列；有界失败不伪装为无解证明。"""
    if not 1 <= max_nodes <= 10000 or not 0 <= max_steps <= 64:
        raise ValueError('能力规划搜索预算无效')
    if len(actions) > 128 or len({a.action_id for a in actions}) != len(actions):
        raise ValueError('能力动作过多或身份重复')
    initial, goals = frozenset(initial), frozenset(goals)
    if goals <= initial:
        return SearchResult(status='SOLVED')
    # 忽略删除效果的闭包只用于证明不可达，绝不用于宣称计划可执行。
    reachable = set(initial)
    while True:
        previous = set(reachable)
        for action in actions:
            if set(action.requires) <= reachable:
                reachable.update(action.adds)
        if previous == reachable:
            break
    if not goals <= reachable:
        return SearchResult(status='UNREACHABLE', missing=tuple(sorted(goals - reachable)))
    queue = deque([(initial, ())])
    seen = {initial}
    expanded, depth_limited = 0, False
    # 优先直接推进目标的动作只是确定性的展开顺序，不改变最短路径语义。
    ordered = sorted(actions, key=lambda a: (-len(goals.intersection(a.adds)), a.action_id))
    while queue:
        facts, path = queue.popleft()
        if expanded >= max_nodes:
            return SearchResult(status='SEARCH_LIMIT', expanded=expanded)
        expanded += 1
        if len(path) >= max_steps:
            depth_limited = True
            continue
        for action in ordered:
            if not set(action.requires) <= facts:
                continue
            after = (facts - set(action.deletes)) | set(action.adds)
            if after == facts:
                continue
            next_path = (*path, action)
            if goals <= after:
                return SearchResult(status='SOLVED', actions=next_path, expanded=expanded)
            frozen = frozenset(after)
            if frozen not in seen:
                # 队列和 visited 也必须有界，不能只限制已出队节点。
                if len(seen) >= max_nodes:
                    return SearchResult(status='SEARCH_LIMIT', expanded=expanded)
                seen.add(frozen)
                queue.append((frozen, next_path))
    return SearchResult(status='SEARCH_LIMIT' if depth_limited else 'UNREACHABLE', expanded=expanded,
                        missing=tuple(sorted(goals - initial)))


class CapabilityPlanNode(PlanningModel):
    action_id: str
    tool: str
    scope: str
    objective_ids: list[str]
    requires: list[str]
    provides: list[str]
    depends_on: list[str]
    evidence_refs: list[str]
    status: Literal['READY', 'PLANNED', 'NEEDS_ARGUMENTS']
    automatic: bool


class CapabilityPlanView(PlanningModel):
    schema_version: Literal[1] = 1
    plan_id: str
    catalog_sha256: str
    goal_revision: int
    revision: int
    previous_plan_id: str | None
    status: Literal['READY', 'NEEDS_MODEL', 'OBSERVATIONS_READY', 'PARTIAL', 'UNREACHABLE', 'SEARCH_LIMIT']
    reason: str
    nodes: list[CapabilityPlanNode]
    facts: dict[str, list[str]]
    missing: list[str]
    delegated_objectives: list[str]
    objective_labels: dict[str, str] = Field(default_factory=dict)
    expanded: int
    # 这是观察获取计划；任务是否完成仍由目标账本和独立审查决定。
    completion_verified: Literal[False] = False


class CapabilityPlanningProjection(PlanningModel):
    status: Literal['READY', 'EMPTY', 'PARTIAL']
    history: list[CapabilityPlanView]
    stale: bool = False
    invalid_records: int = 0
