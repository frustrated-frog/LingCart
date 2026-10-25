"""不可变程序知识图、局部检索与事务式结构修改。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.pg_conditions import EdgeCondition

PAPER_ID = "arXiv:2609.09153v1"


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


def allowed_actions() -> set[str]:
    from src.adaptive_decision import Decision
    from src.direct_executor import _READ_ONLY_TOOLS
    return set(get_args(Decision.model_fields["action"].annotation)) - {"tool"} | set(_READ_ONLY_TOOLS)


def strategy_text(value: str) -> str:
    # 与经验产物采用相同边界；不声称这些规则能证明自由文本安全。
    import re
    if re.search(r"(?<![A-Za-z0-9])P\d{4}S?(?![A-Za-z0-9])", value, re.I):
        raise ValueError("程序知识不能固化商品编号")
    unsafe = re.compile(r"(?:放宽|放松|忽略|绕过|取消).{0,12}(?:预算|价格|硬约束|排除项|审批|安全)")
    for match in unsafe.finditer(value):
        prefix = value[max(0, match.start() - 6):match.start()]
        # Refiner 必须能够表达“禁止绕过审批”这类安全约束；只豁免紧邻动作的明确否定。
        if re.search(r"(?:不得|不能|不可|不要|禁止|严禁|切勿|避免)\s*$", prefix):
            continue
        raise ValueError("程序知识不能放宽硬约束")
    if re.search(r"(?:订单|order)[\s:#：-]*\d+|\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b|(?:api[_ -]?key|bearer)\s*[:= ]\s*\S+", value, re.I):
        raise ValueError("程序知识不能包含订单事实、个人联系信息或凭证")
    return value.strip()


class GraphNode(FrozenModel):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    kind: Literal["ACTION", "REASONING", "STATE"]
    description: str = Field(min_length=1, max_length=400)

    _strategy = field_validator("description")(strategy_text)

    @model_validator(mode="after")
    def known_action(self):
        if self.kind == "ACTION" and self.id not in allowed_actions():
            raise ValueError("行动节点必须属于已注册工具或决策协议")
        if self.id in {"Start", "End"} and self.kind != "STATE":
            raise ValueError("Start / End 必须是状态节点")
        return self


class GraphEdge(FrozenModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    source: str = Field(min_length=1, max_length=64)
    target: str = Field(min_length=1, max_length=64)
    relation: Literal["LEADS_TO", "TRIGGERS", "DEPENDS_ON", "CONVERGES_TO", "PROVIDES_INPUT_FOR"]
    condition: str = Field(default="", max_length=400)
    guidance: str = Field(min_length=1, max_length=600)
    pitfalls: str = Field(default="", max_length=400)
    applicability: EdgeCondition | None = None

    _strategy = field_validator("condition", "guidance", "pitfalls")(strategy_text)


class ProceduralGraph(FrozenModel):
    schema_version: Literal[1, 2] = 1
    graph_id: str = Field(pattern=r"^[a-z][a-z0-9-]{0,79}$")
    version: int = Field(ge=0)
    parent_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    training_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    nodes: tuple[GraphNode, ...] = Field(min_length=2, max_length=64)
    edges: tuple[GraphEdge, ...] = Field(min_length=1, max_length=192)

    @model_validator(mode="after")
    def topology(self):
        nodes = {n.id for n in self.nodes}
        if len(nodes) != len(self.nodes) or not {"Start", "End"} <= nodes:
            raise ValueError("图节点必须唯一且包含 Start / End")
        if len({e.id for e in self.edges}) != len(self.edges) or len({(e.source, e.relation, e.target) for e in self.edges}) != len(self.edges):
            raise ValueError("边编号和关系三元组不能重复")
        if any(e.source not in nodes or e.target not in nodes or e.source == "End" or e.target == "Start" for e in self.edges):
            raise ValueError("边端点不存在或违反 Start / End 方向")
        reachable = {"End"}
        while True:
            expanded = reachable | {e.source for e in self.edges if e.target in reachable}
            if expanded == reachable:
                break
            reachable = expanded
        if reachable != nodes:
            raise ValueError("所有节点必须存在到 End 的路径")
        if (self.version == 0) != (self.parent_sha256 is None and self.training_sha256 is None):
            raise ValueError("演化图必须有父版本和训练来源指纹")
        if self.version > 0 and (self.parent_sha256 is None or self.training_sha256 is None):
            raise ValueError("演化图缺少来源")
        typed = [edge for edge in self.edges if edge.applicability is not None]
        if self.schema_version == 1 and typed:
            raise ValueError("v1 程序图不能混入类型化边条件")
        if self.schema_version == 2 and len(typed) != len(self.edges):
            raise ValueError("v2 程序图的每条边都必须声明类型化适用条件")
        return self

    def _identity_payload(self):
        payload = self.model_dump(mode="json")
        # 新字段缺失时保持历史 v1 图的内容身份不变。
        for edge in payload["edges"]:
            if edge.get("applicability") is None:
                edge.pop("applicability", None)
        return payload

    @property
    def fingerprint(self):
        return fingerprint(self._identity_payload())

    @property
    def structure_sha256(self):
        edges = []
        for edge in sorted(self.edges, key=lambda item: item.id):
            payload = edge.model_dump(mode="json")
            if payload.get("applicability") is None:
                payload.pop("applicability", None)
            edges.append(payload)
        return fingerprint({"nodes": [n.model_dump() for n in sorted(self.nodes, key=lambda n: n.id)],
                            "edges": edges})

    @classmethod
    def from_path(cls, path: Path):
        return cls.model_validate_json(path.read_text())

    def retrieve(self, action: str = "Start", *, hops: int = 2) -> dict:
        if not 1 <= hops <= 3:
            raise ValueError("局部图跳数必须为 1 至 3")
        fallback = action not in {n.id for n in self.nodes}
        reached, frontier, edges = {action}, {action}, []
        for _ in range(hops):
            selected = [e for e in self.edges if e.source in frontier and e not in edges]
            edges.extend(selected)
            targets = {e.target for e in selected}
            frontier, reached = targets - reached, reached | targets
        if fallback:
            reached, edges = {n.id for n in self.nodes}, list(self.edges)
        return {"anchor": action, "fallback": fallback, "hops": hops,
                "nodes": [n.model_dump() for n in self.nodes if n.id in reached],
                "edges": [e.model_dump() for e in edges]}


class GraphPatch(FrozenModel):
    reason: str = Field(min_length=1, max_length=1200)
    source_cases: tuple[str, ...] = Field(min_length=1, max_length=64)
    delete_edges: tuple[str, ...] = Field(default=(), max_length=24)
    delete_nodes: tuple[str, ...] = Field(default=(), max_length=8)
    add_nodes: tuple[GraphNode, ...] = Field(default=(), max_length=8)
    add_edges: tuple[GraphEdge, ...] = Field(default=(), max_length=24)


def apply_patch(graph: ProceduralGraph, patch: GraphPatch, *, training_sha256: str) -> ProceduralGraph:
    nodes, edges = {n.id: n for n in graph.nodes}, {e.id: e for e in graph.edges}
    for identity in patch.delete_edges:
        if identity not in edges:
            raise ValueError("不能删除不存在或重复的边")
        del edges[identity]
    for identity in patch.delete_nodes:
        if identity not in nodes or identity in {"Start", "End"}:
            raise ValueError("不能删除不存在或保留的节点")
        del nodes[identity]
        edges = {key: edge for key, edge in edges.items() if identity not in {edge.source, edge.target}}
    for node in patch.add_nodes:
        if node.id in nodes:
            raise ValueError("添加节点前必须显式删除旧节点")
        nodes[node.id] = node
    for edge in patch.add_edges:
        if edge.id in edges:
            raise ValueError("更新边必须显式删除后添加")
        edges[edge.id] = edge
    result = ProceduralGraph(schema_version=graph.schema_version, graph_id=graph.graph_id, version=graph.version + 1,
                             parent_sha256=graph.fingerprint, training_sha256=training_sha256,
                             nodes=tuple(sorted(nodes.values(), key=lambda n: n.id)),
                             edges=tuple(sorted(edges.values(), key=lambda e: e.id)))
    if result.structure_sha256 == graph.structure_sha256:
        raise ValueError("提案没有改变程序图")
    return result
