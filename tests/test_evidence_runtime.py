"""跨轮补证真实数据库、Tool Ledger 与授权边界回归。"""
from datetime import UTC, datetime
import json
import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import select

from tests.test_after_sales_runtime import database, MESSAGE
from src import orchestrator, tools
from src.backend.db import AfterSalesCase, AfterSalesCaseRun, AfterSalesEvidence, PendingAction, Refund, ToolLedger
from src.durable import run_conversation, resume_run
from src.persistence import RunStore

DETAIL = "我补充一下：左侧外壳有一条贯穿裂缝，开机时会漏音。"
REVOKE = "撤回我刚才提交的损坏说明，那是我看错了。"


class EvidenceModel:
    def json_mode(self):
        return self

    def invoke(self, prompt):
        payload = json.loads(str(prompt).split('\nConversation input:\n')[-1])
        message = payload['latest_human_message']
        if not payload['session_cases']:
            objective = {'id': 'case', 'kind': 'AFTER_SALES_CASE', 'source_quote': message,
                         'order_id': 1001, 'reason_code': 'DAMAGED'}
        else:
            case = next((c for c in payload['session_cases'] if c.get('requirements')), payload['session_cases'][0])
            objective = {'id': 'evidence', 'kind': 'AFTER_SALES_CONTINUE', 'source_quote': message}
            if message == DETAIL:
                objective.update(requirement_id=case['requirements'][0]['requirement_id'], evidence_text=message)
            if message == REVOKE:
                objective['revoke_evidence_id'] = next(e['evidence_id'] for e in case['evidence']
                                                     if e['evidence_type'] == 'DAMAGE_DESCRIPTION')
        return AIMessage(content=json.dumps({'reason': '根据政策要求继续案件', 'commands': [], 'objectives': [objective]}, ensure_ascii=False))


def start_case(monkeypatch, session_id='evidence-session'):
    monkeypatch.setattr(orchestrator, 'get_llm', lambda **kw: EvidenceModel())
    result = run_conversation(MESSAGE, session_id, request_id='open')
    assert not result.get('pending_action')
    with tools._session() as session:
        case = session.scalar(select(AfterSalesCase).where(AfterSalesCase.origin_session_id == session_id))
        assert case is not None
        return case.case_id


def test_cross_run_evidence_reassesses_same_case_without_refunding(database, monkeypatch):
    case_id = start_case(monkeypatch)
    result = run_conversation(DETAIL, 'evidence-session', request_id='add')
    assert result['status'] == 'WAITING_APPROVAL', result
    assert result['pending_action']['case_id'] == case_id
    with tools._session() as session:
        assert len(session.scalars(select(AfterSalesCase)).all()) == 1
        assert len(session.scalars(select(AfterSalesCaseRun)).all()) == 2
        assert len(session.scalars(select(AfterSalesEvidence)).all()) == 2
        refund = session.scalar(select(Refund).where(Refund.order_id == 1001))
        assert refund.status == 'none'
    replay = run_conversation(DETAIL, 'evidence-session', request_id='add')
    assert replay['run_id'] == result['run_id']
    confirmed = run_conversation('确认', 'evidence-session', request_id='confirm')
    assert confirmed['status'] == 'SUCCEEDED', confirmed


def test_retract_evidence_closes_old_approval_and_cannot_refund(database, monkeypatch):
    case_id = start_case(monkeypatch)
    first = run_conversation(DETAIL, 'evidence-session', request_id='add')
    action_id = first['pending_action']['action_id']
    result = run_conversation(REVOKE, 'evidence-session', request_id='revoke')
    assert result['pending_action'] is None, result
    with tools._session() as session:
        assert session.get(PendingAction, action_id).status == 'CLOSED'
        assert session.get(AfterSalesCase, case_id).status == 'AWAITING_EVIDENCE'
        assert session.scalar(select(Refund).where(Refund.order_id == 1001)).status == 'none'
        evidence = session.scalar(select(AfterSalesEvidence).where(AfterSalesEvidence.evidence_type == 'DAMAGE_DESCRIPTION'))
        assert evidence.status == 'REVOKED'


def test_multiple_cases_need_clarification_without_evidence_write(database, monkeypatch):
    case_id = start_case(monkeypatch)
    from src.after_sales import AfterSalesCaseService, CaseCreateCommand
    with tools._session() as session:
        original = AfterSalesCaseService(session).get_case(case_id)
        AfterSalesCaseService(session).open_case(CaseCreateCommand(
            request_key='another', case_type='REFUND', order_id=1001, requested_resolution='REFUND',
            reason_text='另一个诉求', reason_code='DAMAGED', origin_session_id='evidence-session'))
        session.commit()
    result = run_conversation(DETAIL, 'evidence-session', request_id='ambiguous')
    assert result['status'] == 'WAITING_CLARIFICATION', result
    with tools._session() as session:
        assert len(session.scalars(select(AfterSalesEvidence)).all()) == 1


def test_committed_evidence_tool_recovery_does_not_duplicate(database, monkeypatch):
    case_id = start_case(monkeypatch)
    original = RunStore.save_checkpoint
    interrupted = False

    def crash_after_commit(self, run_id, token, snapshot, event=None):
        nonlocal interrupted
        if not interrupted and event and event.get('type') == 'tool_result' and event.get('tool') == 'continue_after_sales_case':
            interrupted = True
            raise KeyboardInterrupt('simulated evidence commit crash')
        return original(self, run_id, token, snapshot, event)

    monkeypatch.setattr(RunStore, 'save_checkpoint', crash_after_commit)
    with pytest.raises(KeyboardInterrupt):
        run_conversation(DETAIL, 'evidence-session', request_id='recover')
    with tools._session() as session:
        ledger = session.scalar(select(ToolLedger).where(ToolLedger.tool_name == 'continue_after_sales_case'))
        assert ledger.status == 'SUCCEEDED'
        run_id = ledger.run_id
    monkeypatch.setattr(RunStore, 'save_checkpoint', original)
    result = resume_run('evidence-session', run_id)
    assert result['status'] == 'WAITING_APPROVAL', result
    with tools._session() as session:
        assert len(session.scalars(select(AfterSalesEvidence)).all()) == 2
        assert len(session.scalars(select(AfterSalesCaseRun)).all()) == 2

@pytest.mark.parametrize('mode,code', [('foreign', 'CASE_CONTEXT_FORBIDDEN'), ('revision', 'CASE_REVISION_CONFLICT'),
                                      ('fabricated', 'UNGROUNDED_EVIDENCE'), ('requirement', 'EVIDENCE_REQUIREMENT_STALE')])
def test_continuation_tool_rejects_invalid_context_before_writing(database, monkeypatch, mode, code):
    from types import SimpleNamespace
    from src.backend.db import AgentRun, Conversation
    from src.after_sales import AfterSalesCaseService
    from src.errors import AgentError
    import src.runtime
    case_id = start_case(monkeypatch)
    with tools._session() as session:
        case = AfterSalesCaseService(session).get_case(case_id)
        session_id = 'foreign-session' if mode == 'foreign' else 'evidence-session'
        if mode == 'foreign':
            session.add(Conversation(session_id=session_id, state={}))
            session.flush()
        session.add(AgentRun(run_id='boundary-run', session_id=session_id, request_id='boundary', request=DETAIL,
                             status='RUNNING', created_at=datetime.now(UTC).replace(tzinfo=None), limits={}))
        session.commit()
    monkeypatch.setattr(src.runtime, 'current_runtime', lambda: SimpleNamespace(run_id='boundary-run'))
    with pytest.raises(AgentError) as exc:
        tools.continue_after_sales_case(case_id, case.revision - 1 if mode == 'revision' else case.revision,
            requirement_id='invented' if mode == 'requirement' else case.latest_decision['facts']['evidence_requirements'][0]['requirement_id'],
            evidence_text='模型伪造' if mode == 'fabricated' else DETAIL)
    assert exc.value.code == code
    with tools._session() as session:
        assert session.get(AfterSalesCase, case_id).revision == case.revision
        assert len(session.scalars(select(AfterSalesEvidence)).all()) == 1
