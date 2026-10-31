"""两步搜索的选择差异、假设边界、稳定性与安全回退。"""
from copy import deepcopy

import json
import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import select
from src import tools, orchestrator, llm
from src.backend.db import Refund, ToolLedger
from src.durable import run_conversation, resume_run
from src.persistence import RunStore
from src.harness import get_runner
from tests.test_after_sales_runtime import (database, MESSAGE, require_two_customer_evidence,
    ActiveClarificationSemanticModel, CrashAfterCaseWrite, CrashAfterAutonomousCaseToolResult)


from src.after_sales.clarification import build_clarification_portfolio, solve_active_clarification
from src.after_sales.execution import project_case_execution_state
from src.after_sales.obligation_graph import build_obligation_graph, decide_next_obligation_action
from src.after_sales.lookahead import solve_lookahead, PLANNER
from tests.test_active_clarification import _two_question_result
from tests.test_after_sales_execution import _case_result


def inputs(result):
    state = project_case_execution_state(result)
    graph = build_obligation_graph(state)
    return state, graph, build_clarification_portfolio(state, graph, result)


def test_lookahead_prefers_less_burdensome_first_step_with_equal_two_step_coverage():
    result = _two_question_result()
    original = deepcopy(result)
    state, graph, portfolio = inputs(result)
    baseline = solve_active_clarification(graph, portfolio)
    decision, report = solve_lookahead(graph, portfolio)
    assert baseline.selected_question_id != decision.selected_question_id
    chosen = next(q for q in portfolio.questions if q.question_id == decision.selected_question_id)
    assert chosen.requirement_id == 'policy:v1:SERIAL_LABEL'
    assert report.expanded_paths == 4
    assert len(report.paths[0].candidate_ids) == 2
    assert report.paths[0].remaining_if_sufficient == ()
    assert len(report.paths[0].remaining_if_no_information) == 2
    assert result == original  # 前瞻没有把假设写成已满足义务。
    directive = decide_next_obligation_action(state, graph, selection=decision.candidate_selection, planner_version=PLANNER)
    assert directive.status == 'WAIT_USER_INPUT'
    assert directive.requirement_id == chosen.requirement_id
    assert directive.operation is None
    assert decision.candidate_selection.solver_version == 'case-lookahead-solver-v1'


def test_candidate_order_does_not_change_search_or_identity():
    result = _two_question_result()
    _, graph, portfolio = inputs(result)
    expected = solve_lookahead(graph, portfolio)
    graph = graph.model_copy(update={'candidates': tuple(reversed(graph.candidates))})
    assert solve_lookahead(graph, portfolio) == expected


def test_authoritative_query_executes_only_first_step_and_does_not_predict_success():
    result = _case_result({'kind':'COLLECT_BUSINESS_EVIDENCE', 'goal_id':'evidence:ORDER_FACT',
                          'requirement_id':'policy:v1:ORDER_FACT', 'operation':'continue_after_sales_case'})
    state, graph, portfolio = inputs(result)
    decision, report = solve_lookahead(graph, portfolio)
    directive = decide_next_obligation_action(state, graph, selection=decision.candidate_selection, planner_version=PLANNER)
    assert directive.status == 'EXECUTE'
    assert directive.arguments['expected_revision'] == state.case_revision
    assert report.paths[0].user_turns == 0
    assert report.paths[0].remaining_if_no_information == ('policy:v1:ORDER_FACT',)


def test_stale_portfolio_never_selects_an_action():
    _, graph, portfolio = inputs(_two_question_result())
    decision, report = solve_lookahead(graph, portfolio.model_copy(update={'state_id':'0'*64}))
    assert report.status == 'INVALID'
    assert decision.candidate_selection is None


def test_exceeding_search_bound_falls_back_without_partial_optimality_claim():
    _, graph, portfolio = inputs(_two_question_result())
    # 超过上限不搜索、不中断原安全策略。
    import src.after_sales.lookahead as module
    from unittest.mock import patch
    with patch.object(module, 'MAX_CANDIDATES', 1):
        decision, report = solve_lookahead(graph, portfolio)
    assert report.status == 'FALLBACK' and not report.paths
    assert report.expanded_paths == 0
    assert decision == solve_active_clarification(graph, portfolio)


def test_unavailable_obligation_is_not_counted_as_resolved():
    _, graph, portfolio = inputs(_two_question_result())
    node = graph.nodes[0].model_copy(update={'node_id':'requirement:deferred', 'requirement_id':'deferred'})
    extended = graph.model_copy(update={'nodes': (*graph.nodes, node)})
    _, report = solve_lookahead(extended, portfolio)
    assert report.paths[0].remaining_if_sufficient == ('deferred',)



@pytest.mark.parametrize('crash', [False, True])
def test_real_ledger_answer_replans_and_resume_does_not_duplicate_write(database, monkeypatch, crash):
    answer = '序列号标签清晰完整'
    class Understanding:
        def json_mode(self): return self
        def invoke(self, prompt):
            return AIMessage(content=json.dumps(dict(commands=[], objectives=[dict(
                id='answer', kind='AFTER_SALES_CONTINUE', source_quote=answer, evidence_text=answer)])))
    class Semantic(ActiveClarificationSemanticModel):
        def invoke(self, prompt):
            payload = json.loads(prompt.split('\nEvidence input:\n')[-1])
            output = json.loads(super().invoke(prompt).content)
            if any(e['evidence_id'] == 'incoming' and e['evidence_type'] == 'SERIAL_LABEL' for e in payload['evidence']):
                for row in output['requirements']:
                    if row['requirement_id'].endswith(':SERIAL_LABEL'):
                        row.update(status='SUFFICIENT', citations=[dict(evidence_id='incoming', quote=answer)],
                                   missing_information=[], question='')
            return AIMessage(content=json.dumps(output))
    with tools._session() as session: require_two_customer_evidence(session)
    monkeypatch.setattr(llm, 'get_llm', lambda **_: Semantic())
    runner = get_runner('after-sales-lookahead-v1')
    opened = run_conversation(MESSAGE, 'lookahead-ledger', runner=runner)
    assert opened['pending_clarification']['requirement_id'].endswith(':SERIAL_LABEL')
    first_report = next(e['report'] for e in opened['events'] if e['type']=='case_lookahead_solved')
    assert len(first_report['paths']) == 2
    monkeypatch.setattr(orchestrator, 'get_llm', lambda **_: Understanding())
    if crash:
        with pytest.raises(CrashAfterCaseWrite):
            run_conversation(answer, 'lookahead-ledger', runner=runner, store=CrashAfterAutonomousCaseToolResult())
        active = RunStore().conversation('lookahead-ledger')['active_run_id']
        result = resume_run('lookahead-ledger', active)
        assert next(t for t in result['trace'] if t['tool']=='continue_after_sales_case')['status']=='REUSED'
    else:
        result = run_conversation(answer, 'lookahead-ledger', runner=runner)
    assert result['pending_clarification']['requirement_id'].endswith(':ORIGINAL_PACKAGING')
    last = [e['report'] for e in result['events'] if e['type']=='case_lookahead_solved'][-1]
    assert last['state_id'] != first_report['state_id']
    assert len(last['paths']) == 1
    with tools._session() as session:
        ledger = session.scalars(select(ToolLedger).where(ToolLedger.run_id==result['run_id'],
                                                        ToolLedger.tool_name=='continue_after_sales_case')).all()
        assert len(ledger)==1
        assert session.scalar(select(Refund).where(Refund.order_id==1001)).status=='none'
