import pytest
from sqlalchemy import select
from src import tools
from src.backend.db import SupportTicket, get_engine, get_session_factory
from src.backend.seed import seed
from src.errors import AgentError
from src.task_plan import TaskStep, verify_observation
from src.runtime import RunRuntime, runtime_scope


@pytest.fixture(autouse=True)
def database(tmp_path):
    engine = get_engine(f'sqlite:///{tmp_path / "policy.db"}')
    tools.configure_db(engine)
    with get_session_factory(engine)() as session:
        seed(session)
    yield
    engine.dispose()


def test_unknown_invoice_policy_is_not_answered():
    question = '能否开具增值税专用发票'
    result = tools.answer_faq(question)
    assert result['status'] == 'NOT_FOUND' and not result['policies']
    with pytest.raises(AgentError, match='不能标记完成'):
        verify_observation(TaskStep('faq', 'faq', 'answer_faq', {'question': question}, result=result))


def test_legacy_fallback_text_cannot_prove_a_policy_answer():
    with pytest.raises(AgentError):
        verify_observation(TaskStep('faq', 'faq', 'answer_faq', {'question': '发票'}, result='请联系人工客服'))


def test_ticket_is_durable_idempotent_and_honest_about_delivery():
    with runtime_scope(RunRuntime('ticket-run')):
        first = tools.escalate_to_human('用户要求人工', '订单1001')
        second = tools.escalate_to_human('用户要求人工', '订单1001')
    assert first == second
    assert first['mode'] == 'DEMO' and not first['delivered'] and not first['escalated']
    with tools._session() as session:
        rows = session.scalars(select(SupportTicket)).all()
        assert len(rows) == 1
        assert rows[0].ticket_id == first['ticket_id'] and rows[0].status == 'QUEUED'
