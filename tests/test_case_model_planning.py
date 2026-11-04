"""模型选择必须改变真实执行，同时保留权威状态、恢复与确认语义。"""
import json

import pytest
from sqlalchemy import select

from src import llm, tools
from src.backend.db import Refund, ToolLedger, PendingAction
from src.durable import run_conversation, resume_run
from src.harness import get_runner
from src.persistence import RunStore
from src.after_sales.model_planning import decide_model_action
from src.runtime import RunRuntime
from tests.test_case_lookahead import inputs
from tests.test_active_clarification import _two_question_result
from tests.test_incremental_case_state import database, models, Adapter, MESSAGE, SERIAL, PACKAGING
from tests.test_evidence_patch import patch_models
from tests.test_after_sales_runtime import require_authoritative_order_evidence, CrashAfterCaseWrite, CrashAfterAutonomousCaseToolResult

PROFILE = 'after-sales-model-planning-v1'


class Planner:
    def __init__(self):
        self.inputs = []
        self.prefer = 'ORIGINAL_PACKAGING'
        self.invalid = None
        self.clarify = False

    def decide(self, prompt):
        payload = json.loads(prompt.split('\nPlanning input:\n')[-1].split('\nRepair feedback:\n')[0])
        self.inputs.append(payload)
        choices = payload['candidates']
        choice = next((c for c in choices if (c['requirement_id'] or '').endswith(self.prefer)), choices[0])
        result = dict(action='SELECT', candidate_id=choice['candidate_id'], source_quote=payload['current_request'],
                      reason='先核对用户当前关注的材料，再依据实际结果调整。',
                      expected_observation='获得这项材料的真实状态。', replan_when='收到材料或查询结果后重新选择。')
        if self.clarify:
            result.update(action='CLARIFY_GOAL', candidate_id=None, question='请明确当前希望推进哪一种处理结果？')
        if self.invalid == 'foreign': result['candidate_id'] = 'not-a-current-candidate'
        if self.invalid == 'ungrounded': result['source_quote'] = '不存在的用户原文'
        if self.invalid == 'tool': result['arguments'] = {'operation':'submit_after_sales_case'}
        return result


@pytest.fixture
def planner(patch_models, monkeypatch):
    planner = Planner()
    monkeypatch.setattr(llm, 'get_llm', lambda *, purpose, **_: Adapter(planner.decide if purpose=='planner' else patch_models.semantic))
    return planner


def run(text=MESSAGE, **kwargs):
    return run_conversation(text, 'model-plan', runner=get_runner(PROFILE), **kwargs)


def test_model_choice_changes_actual_question_against_deterministic_baseline(planner):
    baseline = run_conversation(MESSAGE, 'baseline', runner=get_runner('after-sales-evidence-patch-v1'))
    planner.prefer = 'SERIAL_LABEL'
    actual = run()
    assert baseline['pending_clarification']['requirement_id'].endswith('ORIGINAL_PACKAGING')
    assert actual['pending_clarification']['requirement_id'].endswith('SERIAL_LABEL'), actual
    assert actual['pending_clarification']['question'] in actual['reply']
    decision = next(e for e in actual['events'] if e['type']=='case_model_decision')
    assert decision['decision']['candidate_id']
    assert decision['status']=='ACCEPTED' and decision['observed'] is False
    assert actual['pending_action'] is None


def test_feedback_replans_remaining_material_and_keeps_confirmation(planner):
    first = run()
    second = run(PACKAGING)
    third = run(SERIAL)
    assert second['pending_clarification']['requirement_id'].endswith('SERIAL_LABEL'),second
    assert third['status']=='WAITING_APPROVAL',third
    assert first['pending_clarification']['state_id'] != second['pending_clarification']['state_id']
    assert any(i['state']['policy_outcome']=='ELIGIBLE' for i in planner.inputs)
    with tools._session() as session:
        assert session.scalar(select(Refund).where(Refund.order_id==1001)).status=='none'


def test_goal_clarification_blocks_approval_even_after_evidence_is_sufficient(planner):
    run()
    run(PACKAGING)
    planner.clarify = True
    result = run(SERIAL)
    assert result['status']=='WAITING_CLARIFICATION',result
    assert result['terminal_reason']=='MODEL_GOAL_CLARIFICATION'
    assert result['pending_action'] is None
    assert not result.get('pending_clarification')
    assert '请明确当前希望' in result['reply']


def test_goal_change_does_not_keep_an_old_submission_confirmation(planner):
    run()
    run(PACKAGING)
    ready = run(SERIAL)
    planner.clarify = True
    result = run('先明确处理方向，暂时不确认提交')
    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert result['pending_action'] is None
    with tools._session() as session:
        assert session.get(PendingAction, ready['pending_action']['action_id']).status == 'CLOSED'


def test_repair_can_choose_a_valid_action_without_increasing_retry_limit(planner):
    original = planner.decide
    def repair(prompt):
        result = original(prompt)
        if len(planner.inputs) == 1:
            result['candidate_id'] = 'foreign'
        return result
    planner.decide = repair
    result = run()
    assert result['pending_clarification'] and len(planner.inputs) == 2
    assert sum(e['type']=='case_model_decision_rejected' for e in result['events']) == 1


@pytest.mark.parametrize('invalid',['foreign','ungrounded','tool'])
def test_invalid_action_has_one_repair_and_cannot_execute(planner,invalid):
    planner.invalid=invalid
    result=run()
    assert result['error']['code']=='INVALID_CASE_MODEL_DECISION',result
    assert len(planner.inputs)==2
    assert [t['tool'] for t in result['trace']]==['prepare_after_sales_case']
    assert not result['pending_action']


@pytest.mark.parametrize('crash_at',['decision','tool'])
def test_model_decision_and_tool_ledger_resume_without_duplicate_calls(planner,crash_at):
    with tools._session() as session: require_authoritative_order_evidence(session)
    class CrashAfterDecision(RunStore):
        def save_checkpoint(self, run_id, token, snapshot, event=None):
            super().save_checkpoint(run_id, token, snapshot, event)
            if event and event['type']=='case_model_decision':raise CrashAfterCaseWrite()
    store=CrashAfterDecision() if crash_at=='decision' else CrashAfterAutonomousCaseToolResult()
    with pytest.raises(CrashAfterCaseWrite):run(store=store)
    active=RunStore().conversation('model-plan')['active_run_id']
    before=len(planner.inputs)
    result=resume_run('model-plan',active)
    assert result['status']=='WAITING_APPROVAL',result
    assert any(e['type']=='model_reused' and e['purpose']=='case_action_decision' for e in result['events'])
    assert len(planner.inputs)==before+1  # 新政策状态只调用一次；旧决策从持久缓存恢复。
    assert planner.inputs[-1]['observations'][0]['to_revision']>planner.inputs[-1]['observations'][0]['from_revision']
    with tools._session() as session:
        rows=session.scalars(select(ToolLedger).where(ToolLedger.run_id==active,ToolLedger.tool_name=='continue_after_sales_case')).all()
        assert len(rows)==1
        assert session.scalar(select(Refund).where(Refund.order_id==1001)).status=='none'


def test_stale_state_never_calls_model():
    current=_two_question_result()
    state,graph,portfolio=inputs(current)
    context=RunRuntime('test')
    directive,_=decide_model_action(context,state,graph,portfolio.model_copy(update={'state_id':'0'*64}),current,[])
    assert directive.status=='ABSTAINED'
    assert not context.counters['model_calls']


def test_planning_budget_exhaustion_stops_before_autonomous_write(planner):
    from src.runtime import BudgetLimits
    result = run(limits=BudgetLimits(max_model_calls=0))
    assert result['error']['code'] == 'MODEL_ROUND_BUDGET_EXHAUSTED'
    assert not planner.inputs
    assert not result['pending_action']


def test_profile_changes_only_planning_and_stays_research_only():
    from pathlib import Path
    from eval.experiment_contract import ExperimentSpec
    from src.harness import get_product_runner
    from src.errors import AgentError
    spec = ExperimentSpec.from_path(Path('research/experiments/after-sales-model-planning-v1/experiment-spec.json'))
    baseline, candidate = [get_runner(p).provenance() for p in (spec.baseline_profile, spec.candidate_profile)]
    assert baseline['engine'] == candidate['engine']
    assert [m for m in baseline['mechanisms'] if m['slot'] != 'planning'] == [m for m in candidate['mechanisms'] if m['slot'] != 'planning']
    with pytest.raises(AgentError, match='仅限研究'):
        get_product_runner(PROFILE)
