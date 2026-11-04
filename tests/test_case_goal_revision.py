"""目标修订通过真实持久化、工具账本和确认状态验证；脚本模型不代表效果收益。"""
import json

import pytest
from sqlalchemy import select

from src import tools
from src.backend.db import AfterSalesEvidence, AfterSalesPendingClarification, PendingAction, Refund, ToolLedger
from src.durable import run_conversation, resume_run
from src.harness import get_runner
from src.persistence import RunStore
from tests.test_incremental_case_state import database, models, Adapter, MESSAGE, SERIAL, PACKAGING
from tests.test_evidence_patch import patch_models
from tests.test_case_model_planning_v2 import planner, BATCH
from tests.test_after_sales_runtime import CrashAfterCaseWrite, CrashAfterAutonomousCaseToolResult

PROFILE = 'after-sales-goal-revision-v1'
DRAFT = BATCH + '，先保存为草稿'
PAUSE = '暂停这个案件，保留已有材料'
RESUME = '继续处理这个案件，先重新核验'
AMBIGUOUS = SERIAL + '，但后面是维修还是退款我还没想好'


@pytest.fixture
def goal_models(planner, patch_models, monkeypatch):
    patch_models.goal_modes = {DRAFT: 'DRAFT', PAUSE: 'PAUSED', RESUME: 'CONTINUE', AMBIGUOUS: 'CLARIFY'}
    patch_models.scopes[DRAFT] = patch_models.scopes[BATCH]
    patch_models.scopes[AMBIGUOUS] = [('SERIAL_LABEL', 'PROVIDED')]
    patch_models.boundary = None
    extract = patch_models.extract
    def goals(messages):
        result = extract(messages)
        payload = json.loads(messages[-1].content)
        request = payload['request']
        if request == patch_models.boundary:
            result['changes'].append(dict(operation='ADD', requirement=dict(requirement_id='global-no-refund',
                kind='EXECUTION_BOUNDARY', description='全局禁止退款', source_quote=request, restriction='NO_REFUND')))
        if request in patch_models.goal_modes:
            changes = result['changes']
            existing = next((r for r in changes if r.get('previous_id') == 'case-mode'), None)
            if existing:
                changes.remove(existing)
            changes.append(dict(operation='REPLACE' if existing else 'ADD',
                **(dict(previous_id='case-mode', source_quote=request) if existing else {}),
                requirement=dict(requirement_id='case-mode', kind='EXECUTION_BOUNDARY',
                    description='案件目标模式', source_quote=request,
                    case_goal=dict(case_id=payload['case_context'][0]['case_id'], mode=patch_models.goal_modes[request]))))
        return result
    monkeypatch.setattr('src.request_semantics_runtime.get_llm', lambda **_: Adapter(goals))
    return patch_models


def run(text=MESSAGE, **kwargs):
    return run_conversation(text, 'goal-revision', runner=get_runner(PROFILE), **kwargs)


def mode(result):
    return next(iter(result['case_goals'].values()))


@pytest.mark.parametrize('crash', [False, True])
def test_draft_keeps_both_materials_without_planning_or_confirmation(goal_models, planner, crash):
    first = run()
    before = len(planner.inputs)
    if crash:
        with pytest.raises(CrashAfterCaseWrite):
            run(DRAFT, store=CrashAfterAutonomousCaseToolResult())
        active = RunStore().conversation('goal-revision')['active_run_id']
        result = resume_run('goal-revision', active)
    else:
        result = run(DRAFT)
    assert result['status'] == 'SUCCEEDED', result
    assert mode(result)['mode'] == 'DRAFT' and mode(result)['revision'] == 1
    assert len(planner.inputs) == before
    assert not result.get('pending_action') and not result.get('pending_clarification')
    assert '草稿' in result['reply']
    assert result['reply'].count('已保存为草稿') == 1
    assert result['plan']['steps'][-1]['result']['suppressed_by_goal'] == 'DRAFT'
    assert any(e['type'] == 'case_goal_revised' for e in result['events'])
    with tools._session() as session:
        assert len(session.scalars(select(AfterSalesEvidence).where(AfterSalesEvidence.source_ref == f"agent_runs:{result['run_id']}:request")).all()) == 2
        assert len(session.scalars(select(ToolLedger).where(ToolLedger.run_id == result['run_id'], ToolLedger.tool_name == 'continue_after_sales_case')).all()) == 2
        assert session.get(AfterSalesPendingClarification, first['pending_clarification']['clarification_id']).status == 'SUPERSEDED'
        assert session.scalar(select(Refund).where(Refund.order_id == 1001)).status == 'none'


def test_pause_closes_old_confirmation_and_resume_reassesses(goal_models, planner):
    run()
    ready = run(BATCH)
    assert ready['status'] == 'WAITING_APPROVAL', ready
    before = len(planner.inputs)
    paused = run(PAUSE)
    assert paused['status'] == 'SUCCEEDED' and not paused['trace'], paused
    assert mode(paused)['mode'] == 'PAUSED'
    assert len(planner.inputs) == before
    with tools._session() as session:
        assert session.get(PendingAction, ready['pending_action']['action_id']).status == 'CLOSED'
    resumed = run(RESUME)
    assert resumed['status'] == 'WAITING_APPROVAL', resumed
    assert mode(resumed)['mode'] == 'CONTINUE' and mode(resumed)['revision'] == 2
    assert resumed['pending_action']['action_id'] != ready['pending_action']['action_id']
    assert any(t['tool'] == 'continue_after_sales_case' for t in resumed['trace'])
    with tools._session() as session:
        assert session.scalar(select(Refund).where(Refund.order_id == 1001)).status == 'none'


def test_goal_clarification_keeps_unambiguous_material(goal_models):
    run()
    def clarify(result):
        result['commands'] = [dict(kind='CLARIFY', source_quote=AMBIGUOUS, message='希望申请维修还是退款？')]
        return result
    goal_models.raw_override = clarify
    result = run(AMBIGUOUS)
    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert result['terminal_reason'] == 'MODEL_GOAL_CLARIFICATION'
    assert mode(result)['mode'] == 'CLARIFY'
    assert not result.get('pending_action') and not result.get('pending_clarification')
    with tools._session() as session:
        rows = session.scalars(select(AfterSalesEvidence).where(AfterSalesEvidence.source_ref == f"agent_runs:{result['run_id']}:request")).all()
        assert [row.evidence_type for row in rows] == ['SERIAL_LABEL']


def test_invalid_material_cannot_release_a_paused_goal(goal_models):
    run()
    run(PAUSE)
    goal_models.goal_modes[SERIAL] = 'CONTINUE'
    def foreign(result):
        result['objectives'][0]['material_changes'][0]['requirement_id'] = 'foreign'
        return result
    goal_models.raw_override = foreign
    result = run(SERIAL)
    assert result['status'] == 'WAITING_CLARIFICATION' and not result['trace'], result
    state = RunStore().conversation('goal-revision')['state']
    assert next(iter(state['case_goals'].values()))['mode'] == 'PAUSED'
    assert not any(e['type'] == 'case_goal_revised' for e in result['events'])


def test_goal_checkpoint_recovery_does_not_increment_revision(goal_models):
    run()
    class CrashAfterGoal(RunStore):
        def save_checkpoint(self, run_id, token, snapshot, event=None):
            super().save_checkpoint(run_id, token, snapshot, event)
            if snapshot.get('case_goals') and not event:
                raise CrashAfterCaseWrite()
    with pytest.raises(CrashAfterCaseWrite):
        run(PAUSE, store=CrashAfterGoal())
    active = RunStore().conversation('goal-revision')['active_run_id']
    result = resume_run('goal-revision', active)
    assert mode(result)['revision'] == 1 and result['status'] == 'SUCCEEDED', result


def test_candidate_only_changes_planning_and_cannot_enter_product():
    from src.harness import get_product_runner
    from src.errors import AgentError
    a, b = [get_runner(p).provenance() for p in ['after-sales-model-planning-v2', PROFILE]]
    assert [m for m in a['mechanisms'] if m['slot'] != 'planning'] == [m for m in b['mechanisms'] if m['slot'] != 'planning']
    with pytest.raises(AgentError):
        get_product_runner(PROFILE)


def test_held_case_blocks_old_authorization_before_preflight_across_profiles():
    from src.after_sales.goals import CaseGoal
    from src.errors import AgentError
    from src.request_ir import locate_source_span
    from src.runtime import BusinessEffect, RunRuntime, ToolSpec, ToolExecutor, TraceRecorder, runtime_scope
    context = RunRuntime('held')  # Stable Kernel 同样保护没有候选机制的执行。
    context.snapshot['conversation'] = {'case_goals': {'case-a': CaseGoal(case_id='case-a', order_id=1001,
        mode='PAUSED', revision=1, source=locate_source_span(PAUSE, PAUSE), run_id='prior').model_dump(mode='json')}}
    calls = []
    def composite(order_id: int):
        calls.append(order_id)
        return {'success': True}
    context.approval = {'tool': 'submit', 'args': {'order_id': 1001}}
    with runtime_scope(context):
        executor = ToolExecutor({'submit': ToolSpec(composite, side_effect=True,
            business_effects=(BusinessEffect.REFUND,), preflight=lambda args, ctx: calls.append('preflight'))}, TraceRecorder(context.run_id))
        with pytest.raises(AgentError) as error:
            executor.execute('submit', {'order_id': 1001})
        assert error.value.code == 'CASE_GOAL_HELD'
    assert not calls


def test_resume_preserves_explicit_global_prohibition(goal_models):
    run()
    request = DRAFT + '；整个会话都不要退款'
    goal_models.boundary = request
    goal_models.goal_modes[request] = 'DRAFT'
    goal_models.scopes[request] = goal_models.scopes[DRAFT]
    run(request)
    result = run(RESUME)
    assert mode(result)['mode'] == 'CONTINUE'
    assert 'NO_REFUND' in {row['restriction'] for row in result['execution_boundaries']}
    assert not result.get('pending_action'), result


@pytest.mark.parametrize('reject', [False, True])
def test_api_response_preserves_goal_state_on_success_and_rejection(goal_models, reject):
    from api import ChatResponse
    run()
    result = run(PAUSE)
    if reject:
        goal_models.raw_override = lambda result: {'commands': [], 'objectives': []}
        result = run(SERIAL)
        assert result['status'] != 'SUCCEEDED'
    response = ChatResponse.model_validate(dict(result, session_id='goal-revision'))
    assert next(iter(response.case_goals.values())).mode == 'PAUSED'


def test_preserved_other_case_does_not_force_this_turn_to_modify_it(monkeypatch):
    from types import SimpleNamespace
    from langchain_core.messages import HumanMessage
    from tests.test_incremental_case_state import real_verify as verify_intent
    from src.request_semantics import apply_extraction
    from src.runtime import RunRuntime
    from src.errors import AgentError
    source = '订单1001和订单1002分别建立售后案件'
    prior = apply_extraction({'changes': [dict(operation='ADD', requirement=dict(requirement_id=f'case{order}',
        kind='OUTCOME', description='售后', source_quote=source,
        outcome=dict(kind='after_sales_case', order_id=order, reason_code='NOT_AS_DESCRIBED'))) for order in [1001, 1002]]}, source, [])
    request = '暂停订单1001的案件'
    changes = [dict(operation='PRESERVE', previous_id=f'case{order}') for order in [1001, 1002]]
    contract = apply_extraction({'changes': changes}, request, [HumanMessage(content=source)], previous=prior)
    context = RunRuntime('multiple')
    context.mechanisms['planning'] = SimpleNamespace(goal_revision=True)
    understanding = SimpleNamespace(objectives=[SimpleNamespace(kind='AFTER_SALES_CONTINUE', case_id='case-a')], execution_boundaries=[], commands=[])
    monkeypatch.setattr('src.after_sales.context.current_case_context', lambda: [dict(case_id='case-a', order_id=1001, reason_code='NOT_AS_DESCRIBED')])
    monkeypatch.setattr('src.after_sales.intent.extract_contract', lambda *a, **k: contract)
    verify_intent(context, understanding, request, [])
    # 新增但未被执行目标覆盖的结果仍须拒绝，不能把所有其他订单一概忽略。
    request = '暂停订单1001，订单1003新建售后案件'
    changes.append(dict(operation='ADD', requirement=dict(requirement_id='case1003', kind='OUTCOME', description='售后', source_quote=request,
        outcome=dict(kind='after_sales_case', order_id=1003, reason_code='NOT_AS_DESCRIBED'))))
    contract = apply_extraction({'changes': changes}, request, [HumanMessage(content=source)], previous=prior)
    with pytest.raises(AgentError) as error:
        verify_intent(context, understanding, request, [])
    assert error.value.code == 'REQUEST_SEMANTICS_MISMATCH'


def test_goal_scope_must_match_the_current_objective():
    from types import SimpleNamespace
    from src.after_sales.goals import goal_changes
    from src.after_sales.turn_verification import CaseTurnMismatch
    from src.request_semantics import IntentRequirement
    from src.runtime import RunRuntime
    context = RunRuntime('scope')
    context.mechanisms['planning'] = SimpleNamespace(goal_revision=True)
    understanding = SimpleNamespace(objectives=[SimpleNamespace(kind='AFTER_SALES_CONTINUE', case_id='own')], commands=[])
    row = IntentRequirement(requirement_id='goal', kind='EXECUTION_BOUNDARY', description='暂停', source_quote=PAUSE,
                            case_goal=dict(case_id='foreign', mode='PAUSED'))
    with pytest.raises(CaseTurnMismatch):
        goal_changes(context, understanding, [row], {'revisions': [dict(operation='ADD', requirement_id='goal')]})


def test_frozen_dev_diagnostic_and_experiment_load():
    from pathlib import Path
    from eval.run_live_eval import load_dataset
    from eval.experiment_contract import ExperimentSpec
    data = load_dataset(Path('eval/after_sales_goal_revision_dev_tasks.json'))
    assert len(data['cases']) == 5 and all(c['split'] == 'dev' for c in data['cases'])
    spec = ExperimentSpec.from_path(Path('research/experiments/after-sales-goal-revision-v1/experiment-spec.json'))
    assert spec.baseline_profile == 'after-sales-model-planning-v2' and spec.candidate_profile == PROFILE
