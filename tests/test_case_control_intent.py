"""控制语义独立核验、持久化范围与旧授权失效。"""
import json

import pytest
from sqlalchemy import select

from src import tools
from src.backend.db import PendingAction, Refund
from src.durable import run_conversation
from src.harness import get_runner
from tests.test_incremental_case_state import database, models, Adapter, MESSAGE
from tests.test_evidence_patch import patch_models
from tests.test_case_model_planning_v2 import planner, BATCH
from tests.test_case_goal_revision import goal_models, DRAFT, RESUME
from tests.test_case_source_refs import references

PROFILE = 'after-sales-control-intent-v1'
BAN = '只禁止这个案件提交，其他案件不受影响'
RELEASE = '解除这个案件的提交禁令，重新核验后让我确认'
REVOKE = '撤回这个案件之前的授权'


@pytest.fixture
def controls(references, monkeypatch):
    from src import request_semantics_runtime as semantic
    references.controls = {RESUME:'REQUIRE_CONFIRMATION', BAN:'PROHIBIT', RELEASE:'RELEASE_PROHIBITION', REVOKE:'REVOKE_APPROVAL'}
    def build(request, case_id):
        kind = references.controls.get(request)
        if not kind: return None
        return dict(kind=kind, scope='CASE', case_id=case_id,
                    restriction='NO_SUBMISSION' if kind in {'PROHIBIT','RELEASE_PROHIBITION'} else None)
    def intake(result):
        payload = json.loads(references.prompts[-1].split('\nConversation input:\n')[-1].split('\nPrevious output was invalid.')[0])
        request = result['objectives'][0]['source_quote']
        intent = build(request, payload['session_cases'][0]['case_id'])
        if intent: result['control_intents'] = [dict(**intent, source_quote=request)]
        return result
    references.raw_override = intake
    original = semantic.get_llm
    def extract(messages):
        payload = json.loads(messages[-1].content)
        result = original().fn(messages)
        intent = build(payload['request'], payload['case_context'][0]['case_id']) if payload.get('case_context') else None
        if intent:
            ref = payload['human_sources'][0]['source_ref']
            identity = intent['kind'].lower()
            existing = next((c for c in result['changes'] if c.get('previous_id') == identity), None)
            if existing: result['changes'].remove(existing)
            if intent['kind'] == 'RELEASE_PROHIBITION':
                for change in result['changes']:
                    if change.get('previous_id') == 'prohibit': change.update(operation='RETRACT', source_ref=ref)
            result['changes'].append(dict(operation='REPLACE' if existing else 'ADD',
                **(dict(previous_id=identity, source_ref=ref) if existing else {}),
                requirement=dict(requirement_id=identity,kind='EXECUTION_BOUNDARY',description='控制意图',source_ref=ref,control_intent=intent)))
        return result
    monkeypatch.setattr(semantic, 'get_llm', lambda **_: Adapter(extract))
    return references


def run(text=MESSAGE):
    result = run_conversation(text, 'controls', runner=get_runner(PROFILE))
    assert not result.get('error'), result.get('error')
    return result


def test_resume_without_consent_still_proposes_confirmation(controls):
    run(); run(DRAFT)
    result = run(RESUME)
    assert result['status'] == 'WAITING_APPROVAL', result
    assert not result['execution_boundaries']
    assert any(e['type'] == 'case_control_interpreted' and e['control']['kind'] == 'REQUIRE_CONFIRMATION' for e in result['events'])
    with tools._session() as session:
        assert session.scalar(select(Refund).where(Refund.order_id == 1001)).status == 'none'


def test_explicit_case_ban_survives_waiting_intent_until_explicit_release(controls):
    run(); run(BATCH)
    banned = run(BAN)
    assert banned['status'] == 'SUCCEEDED' and not banned['pending_action'], banned
    assert banned['execution_boundaries'][0]['case_id']
    waiting = run(RESUME)
    assert not waiting['pending_action'] and waiting['execution_boundaries'], waiting
    released = run(RELEASE)
    assert released['status'] == 'WAITING_APPROVAL' and not released['execution_boundaries'], released


def test_revocation_closes_old_confirmation_and_does_not_block_future_assessment(controls):
    run(); old = run(BATCH)
    result = run(REVOKE)
    assert not result['pending_action'] and not result['execution_boundaries'], result
    with tools._session() as session:
        assert session.get(PendingAction, old['pending_action']['action_id']).status == 'CLOSED'
    resumed = run(RESUME)
    assert resumed['status'] == 'WAITING_APPROVAL', resumed
    assert resumed['pending_action']['action_id'] != old['pending_action']['action_id']


def test_case_restriction_does_not_block_other_order_and_revoked_approval_cannot_execute():
    from src.execution_boundaries import require_allowed
    from src.request_ir import locate_source_span
    from src.runtime import RunRuntime, tool_spec
    from src.errors import AgentError
    context = RunRuntime('scope')
    context.snapshot['execution_boundaries'] = [dict(restriction='NO_REFUND', origin='control', case_id='case-a', order_id=1001,
        source=locate_source_span(BAN, BAN).model_dump())]
    spec = tool_spec('initiate_refund', tools.initiate_refund)
    require_allowed(context, 'initiate_refund', spec, {'order_id':1002})
    with pytest.raises(AgentError): require_allowed(context, 'initiate_refund', spec, {'order_id':1001})
    context.approval = {'action_id':'old'}
    context.snapshot['revoked_approvals'] = ['old']
    with pytest.raises(AgentError) as error: require_allowed(context, 'initiate_refund', spec, {'order_id':1002})
    assert error.value.code == 'APPROVAL_REVOKED'


def test_second_candidate_only_changes_planning():
    from src.harness import get_product_runner
    from src.errors import AgentError
    a,b = [get_runner(p).provenance() for p in ['after-sales-source-refs-v1',PROFILE]]
    assert [m for m in a['mechanisms'] if m['slot'] != 'planning'] == [m for m in b['mechanisms'] if m['slot'] != 'planning']
    with pytest.raises(AgentError): get_product_runner(PROFILE)


def test_revocation_checkpoint_recovery_keeps_old_approval_closed(controls):
    from src.durable import resume_run
    from src.persistence import RunStore
    from tests.test_after_sales_runtime import CrashAfterCaseWrite
    run(); old = run(BATCH)
    class CrashAfterControl(RunStore):
        def save_checkpoint(self, run_id, token, snapshot, event=None):
            super().save_checkpoint(run_id, token, snapshot, event)
            if snapshot.get('revoked_approvals') and not event:
                raise CrashAfterCaseWrite()
    with pytest.raises(CrashAfterCaseWrite):
        run_conversation(REVOKE, 'controls', runner=get_runner(PROFILE), store=CrashAfterControl())
    result = resume_run('controls', RunStore().conversation('controls')['active_run_id'])
    assert result['status'] == 'SUCCEEDED' and not result['pending_action'], result
    with tools._session() as session:
        assert session.get(PendingAction, old['pending_action']['action_id']).status == 'CLOSED'
    assert run(RESUME)['status'] == 'WAITING_APPROVAL'


def test_independent_control_disagreement_does_not_release_ban(controls):
    run(); run(BATCH); run(BAN)
    original = controls.raw_override
    def disagree(result):
        value = original(result)
        value['control_intents'] = []
        return value
    controls.raw_override = disagree
    result = run_conversation(RELEASE, 'controls', runner=get_runner(PROFILE))
    assert result['execution_boundaries'] and not result['pending_action']
    assert not any(e['type'] == 'case_control_interpreted' for e in result['events'])


def test_legacy_prohibition_cannot_be_retracted_without_matching_control():
    from types import SimpleNamespace
    from src.after_sales.control_intent import verify_controls
    from src.after_sales.turn_verification import CaseTurnMismatch
    from src.runtime import RunRuntime
    context = RunRuntime('legacy')
    context.mechanisms['planning'] = SimpleNamespace(control_intents=True)
    context.snapshot['after_sales_previous_semantics'] = {'requirements': [{'requirement': {'restriction': 'NO_REFUND'}}]}
    with pytest.raises(CaseTurnMismatch):
        verify_controls(context, SimpleNamespace(control_intents=[], objectives=[], execution_boundaries=[]),
                        [], {'revisions': []}, {}, RESUME)


def test_case_ban_removes_old_confirmation_from_projection_before_tool_write():
    from src.execution_boundaries import action_restricted
    from src.request_ir import locate_source_span
    from src.runtime import RunRuntime
    context = RunRuntime('restrict-pending')
    context.snapshot['execution_boundaries'] = [dict(restriction='NO_SUBMISSION', origin='control', case_id='a', order_id=1001,
        source=locate_source_span(BAN, BAN).model_dump())]
    assert action_restricted(context, {'tool':'submit_after_sales_case', 'case_id':'a'})
    assert not action_restricted(context, {'tool':'submit_after_sales_case', 'case_id':'b'})
