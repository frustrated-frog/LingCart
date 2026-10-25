"""受限修改假设及其验证反馈；拒绝记忆不向 Refiner 泄露验证案例。"""
from typing import Literal
from pydantic import Field
from src.procedural_graph import FrozenModel, GraphPatch, fingerprint


class EvolutionHypothesis(FrozenModel):
    problem: str = Field(min_length=1, max_length=600)
    failure_mode: Literal['NO_INFORMATION_GAIN', 'UNRESOLVED_GOAL', 'ACTION_FAILURE', 'GUIDANCE_NOT_ADOPTED', 'COVERAGE_GAP']
    expected_change: Literal['NEW_INFORMATION', 'OBJECTIVE_PROGRESS', 'RECOVERY', 'BETTER_ADOPTION']
    target_edges: tuple[str, ...] = Field(min_length=1, max_length=12,
        description='必须完整列出 Patch 实际增删的边 ID，包括新增边和删除节点所移除的边；不是问题所在的旧边 ID。')
    source_samples: tuple[str, ...] = Field(min_length=1, max_length=32,
        description='训练样本引用，格式 case_id:repetition；与 patch.source_cases 的纯 case_id 区分。')
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=32)


class EvolutionProposal(FrozenModel):
    hypothesis: EvolutionHypothesis
    patch: GraphPatch


class HypothesisError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def hypothesis_identity(graph, proposal, credit):
    hypothesis, patch = proposal.hypothesis, proposal.patch
    if (not set(hypothesis.source_samples) <= set(credit['sample_refs'])
            or len(set(hypothesis.source_samples)) != len(hypothesis.source_samples)
            or not {ref.split(':')[0] for ref in hypothesis.source_samples} <= set(patch.source_cases)):
        raise HypothesisError('HYPOTHESIS_SOURCE_MISMATCH', '假设只能引用 Patch 对应的训练样本')
    evidence = {e['evidence_id']: e for edge in credit['edges'] for e in edge['evidence']}
    if any(ref not in evidence or evidence[ref]['sample_ref'] not in hypothesis.source_samples for ref in hypothesis.evidence_ids):
        raise HypothesisError('HYPOTHESIS_EVIDENCE_MISMATCH', '假设证据不属于当前训练报告')
    edges = {edge.id: edge for edge in (*graph.edges, *patch.add_edges)}
    if not set(hypothesis.target_edges) <= edges.keys() or len(set(hypothesis.target_edges)) != len(hypothesis.target_edges):
        raise HypothesisError('HYPOTHESIS_TARGET_MISMATCH', '假设目标关系不存在或重复')
    changed = set(patch.delete_edges) | {e.id for e in patch.add_edges}
    changed.update(e.id for e in graph.edges if e.source in patch.delete_nodes or e.target in patch.delete_nodes)
    if set(hypothesis.target_edges) != changed:
        raise HypothesisError('HYPOTHESIS_TARGET_MISMATCH', '假设必须覆盖全部实际修改的关系')
    # 同父图下忽略理由与指导文案，防止用措辞变化重复被否定的行动方向。
    directions = sorted((dict(source=edges[key].source, target=edges[key].target,
        relation=edges[key].relation, applicability=edges[key].applicability.model_dump(mode='json')
        if edges[key].applicability else None, operation='delete' if key in patch.delete_edges
        and key not in {e.id for e in patch.add_edges} else 'upsert') for key in hypothesis.target_edges),
        key=lambda item: fingerprint(item))
    return fingerprint(dict(parent_sha256=graph.fingerprint, failure_mode=hypothesis.failure_mode,
                            expected_change=hypothesis.expected_change, directions=directions))


def validation_changes(before, after):
    """保存样本级验证差异供审查；该字段不进入 Refiner 输入。"""
    if not before or not after:
        return []
    indexed = {(s['case_id'], s['repetition']): s for s in before['samples']}
    return [dict(case_id=s['case_id'], repetition=s['repetition'], before_passed=indexed[key]['passed'],
                 after_passed=s['passed'], model_attempt_delta=s['model_attempts']-indexed[key]['model_attempts'])
            for s in after['samples'] if (key := (s['case_id'], s['repetition'])) in indexed]


DECISIVE_REJECTIONS = {'VALIDATION_REGRESSION', 'SAFETY_OR_PROTOCOL_VIOLATION', 'PG_NOT_EXPOSED', 'COST_REGRESSION'}


def memory_entry(record):
    changes = record.get('validation_changes') or []
    return dict(schema_version=1, hypothesis_id=record.get('hypothesis_id'),
        hypothesis=record.get('hypothesis'), parent_sha256=record['parent_sha256'], iteration=record['iteration'],
        status=record['status'], reason=record['reason'],
        blocks_retry=record['status'] == 'REJECTED' and record['reason'] in DECISIVE_REJECTIONS,
        validation_summary=dict(regressed_samples=sum(c['before_passed'] and not c['after_passed'] for c in changes),
                                improved_samples=sum(not c['before_passed'] and c['after_passed'] for c in changes),
                                compared_samples=len(changes)),
        incumbent_metrics=record.get('incumbent_metrics'), candidate_metrics=record.get('candidate_metrics'))
