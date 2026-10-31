"""有界信息覆盖搜索；假设分支只用于选择，不生成事实或执行凭证。"""
from itertools import permutations
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.after_sales.clarification import ActiveClarificationDecision, ClarificationPortfolio, solve_active_clarification
from src.after_sales.obligation_graph import ObligationGraph, build_candidate_selection
from src.after_sales.service import _checksum

SOLVER = 'case-lookahead-solver-v1'
PLANNER = 'policy-aware-lookahead-v1'
HORIZON = 2
MAX_CANDIDATES = 12
RANKING = ('safe_authoritative_read_first', 'remaining_obligations_asc', 'user_turns_asc',
           'tool_calls_asc', 'response_burden_asc', 'first_action_burden_asc', 'stable_identity')


class InformationPath(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    candidate_ids: tuple[str, ...]
    requirement_ids: tuple[str, ...]
    remaining_if_sufficient: tuple[str, ...]
    remaining_if_no_information: tuple[str, ...]
    user_turns: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    response_burden: int = Field(ge=0)
    first_action_burden: int = Field(ge=0)


class LookaheadReport(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    version: Literal['case-lookahead-v1'] = 'case-lookahead-v1'
    report_id: str
    state_id: str
    graph_id: str
    portfolio_id: str
    status: Literal['SELECTED', 'FALLBACK', 'INVALID', 'NOT_REQUIRED']
    reason_code: str
    horizon: Literal[2] = HORIZON
    expanded_paths: int = Field(default=0, ge=0, le=144)
    selected_candidate_id: str | None = None
    paths: tuple[InformationPath, ...] = ()
    assumption: Literal['SUFFICIENT_INFORMATION_NOT_OBSERVED'] = 'SUFFICIENT_INFORMATION_NOT_OBSERVED'


def solve_lookahead(
    graph: ObligationGraph, portfolio: ClarificationPortfolio,
) -> tuple[ActiveClarificationDecision, LookaheadReport]:
    """返回兼容现有执行器的单步决策及独立的前瞻证据。"""
    base = solve_active_clarification(graph, portfolio)

    def report(status, reason, decision=base, paths=(), expanded=0):
        body = dict(state_id=graph.state_id, graph_id=graph.graph_id, portfolio_id=portfolio.portfolio_id,
                    status=status, reason_code=reason, expanded_paths=expanded, paths=paths,
                    selected_candidate_id=(decision.candidate_selection.selected_candidate_id
                                           if decision.candidate_selection else None))
        identity = {**body, 'version': 'case-lookahead-v1', 'horizon': HORIZON,
                    'paths': [p.model_dump(mode='json') for p in paths]}
        return decision, LookaheadReport(report_id=_checksum(identity), **body)

    if (graph.state_id != portfolio.state_id or graph.graph_id != portfolio.graph_id
            or base.status == 'INVALID'):
        # 输入身份冲突不能回退到仍可执行的旧决策。
        invalid = ActiveClarificationDecision(
            decision_id=_checksum({'graph': graph.graph_id, 'portfolio': portfolio.portfolio_id, 'invalid': True}),
            graph_id=graph.graph_id, portfolio_id=portfolio.portfolio_id,
            status='INVALID', reason_code='LOOKAHEAD_INPUT_INVALID')
        return report('INVALID', 'LOOKAHEAD_INPUT_INVALID', invalid)
    if portfolio.status == 'INVALID':
        return report('FALLBACK', 'LOOKAHEAD_PORTFOLIO_UNAVAILABLE')
    candidates = [c for c in graph.candidates if c.admissible and c.mode in {'AUTONOMOUS', 'ASK_USER'}]
    if not candidates:
        return report('NOT_REQUIRED', base.reason_code)
    if len(candidates) > MAX_CANDIDATES:
        return report('FALLBACK', 'LOOKAHEAD_CANDIDATE_LIMIT')
    questions = {q.obligation_candidate_id: q for q in portfolio.questions}
    if any(c.mode == 'ASK_USER' and c.candidate_id not in questions for c in candidates):
        return report('FALLBACK', 'LOOKAHEAD_QUESTION_MISSING')
    unresolved = {n.requirement_id for n in graph.nodes if n.kind == 'REQUIREMENT' and n.status != 'SATISFIED'}
    by_id = {c.candidate_id: c for c in candidates}
    burden = {c.candidate_id: questions[c.candidate_id].response_burden if c.mode == 'ASK_USER' else 0
              for c in candidates}
    paths = []
    for depth in range(1, min(HORIZON, len(candidates)) + 1):
        for sequence in permutations(sorted(by_id), depth):
            actions = [by_id[k] for k in sequence]
            requirements = tuple(c.requirement_id for c in actions)
            if len(set(requirements)) != len(requirements) or not set(requirements) <= unresolved:
                continue
            paths.append(InformationPath(candidate_ids=sequence, requirement_ids=requirements,
                remaining_if_sufficient=tuple(sorted(unresolved - set(requirements))),
                remaining_if_no_information=tuple(sorted(unresolved)),
                user_turns=sum(c.user_interaction_cost for c in actions),
                tool_calls=sum(c.tool_call_cost for c in actions),
                response_burden=sum(burden[k] for k in sequence),
                first_action_burden=burden[sequence[0]]))
    if not paths:
        return report('FALLBACK', 'LOOKAHEAD_NO_PATH')

    def rank(path):
        first = by_id[path.candidate_ids[0]]
        return (first.mode != 'AUTONOMOUS', len(path.remaining_if_sufficient), path.user_turns,
                path.tool_calls, path.response_burden, path.first_action_burden,
                tuple(by_id[k].requirement_id for k in path.candidate_ids), path.candidate_ids)

    # 每个首步只保留最佳延续；所有候选均被搜索，显示的路径不冒充执行计划。
    best = {}
    for path in sorted(paths, key=rank):
        best.setdefault(path.candidate_ids[0], path)
    ordered = sorted(best.values(), key=rank)
    candidate_order = [by_id[p.candidate_ids[0]] for p in ordered]
    candidate_order.extend(c for c in graph.candidates if c.admissible and c.candidate_id not in best)
    selection = build_candidate_selection(graph, candidate_order, reason_code='BOUNDED_INFORMATION_COVERAGE',
                                         solver_version=SOLVER, ranking_policy=RANKING)
    selected = candidate_order[0]
    question = questions.get(selected.candidate_id)
    decision = ActiveClarificationDecision(solver_version=SOLVER,
        decision_id=_checksum({'solver': SOLVER, 'selection': selection.selection_id, 'portfolio': portfolio.portfolio_id}),
        graph_id=graph.graph_id, portfolio_id=portfolio.portfolio_id,
        status='SELECTED' if question else 'BYPASSED', reason_code='BOUNDED_INFORMATION_COVERAGE',
        selected_question_id=question.question_id if question else None,
        selected_obligation_candidate_id=selected.candidate_id,
        candidate_selection=selection, ranking_policy=RANKING)
    return report('SELECTED', 'BOUNDED_INFORMATION_COVERAGE', decision, tuple(ordered), len(paths))
