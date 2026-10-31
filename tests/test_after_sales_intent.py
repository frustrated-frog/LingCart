"""独立售后需求入口测试；不从规划输出反向生成需求契约。"""
import json
import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import select
from src import orchestrator, tools
from src.after_sales.intent import verify_intent as real_verify
from src.backend.db import AfterSalesCase, Refund
from src.durable import run_conversation
from tests.test_after_sales_runtime import database, MESSAGE


class Extractor:
    def __init__(self, order=1001): self.order=order; self.inputs=[]
    def json_mode(self): return self
    def invoke(self, messages):
        payload=json.loads(messages[-1].content);self.inputs.append(payload)
        assert 'goal' not in payload and 'objectives' not in payload
        if payload.get('prior_contract'):
            return AIMessage(content=json.dumps({'changes': [dict(operation='PRESERVE',previous_id=r['requirement']['requirement_id']) for r in payload['prior_contract']['requirements']]}))
        return AIMessage(content=json.dumps({'changes':[dict(operation='ADD', requirement=dict(
            requirement_id='case',kind='OUTCOME',description='建立售后案件',source_quote=MESSAGE,
            outcome=dict(kind='after_sales_case',order_id=self.order,reason_code='NOT_AS_DESCRIBED')))]}))


@pytest.mark.parametrize('order', [1001,1002])
def test_independent_case_intent_checks_order_before_any_write(database,monkeypatch,order):
    import src.request_semantics_runtime as semantics
    model=Extractor(order)
    monkeypatch.setattr('src.after_sales.intent.verify_intent',real_verify)
    monkeypatch.setattr(semantics,'get_llm',lambda **_:model)
    result=run_conversation(MESSAGE,'case-intent')
    with tools._session() as session:
        cases=session.scalars(select(AfterSalesCase)).all()
        assert len(cases)==(1 if order==1001 else 0)
    if order==1001:
        assert any(e['type']=='after_sales_intent_verified' for e in result['events'])
    else:
        assert not result['trace']
        assert result['terminal_reason']=='REQUEST_SEMANTICS_MISMATCH'
    assert len(model.inputs)==1


class BatchModel:
    def json_mode(self): return self
    def invoke(self,prompt):
        payload=json.loads(prompt.split('\nConversation input:\n')[-1])
        case=payload['session_cases'][0]
        requirements={r['evidence_type']:r['requirement_id'] for r in case['requirements']}
        return AIMessage(content=json.dumps(dict(commands=[],objectives=[dict(id='CaseA',kind='AFTER_SALES_CONTINUE',
            source_quote=payload['latest_human_message'],evidence_updates=[
                dict(requirement_id=requirements['ORIGINAL_PACKAGING'],disposition='PROVIDED',source_quote='原包装和附件完整'),
                dict(requirement_id=requirements['SERIAL_LABEL'],disposition='DEFERRED',source_quote='序列号照片明天补')])]),ensure_ascii=False))


def test_batch_materials_follow_committed_versions_and_keep_boundaries(database,monkeypatch):
    from tests.test_after_sales_runtime import require_two_customer_evidence
    import src.request_semantics_runtime as semantics
    model=Extractor()
    monkeypatch.setattr('src.after_sales.intent.verify_intent',real_verify)
    monkeypatch.setattr(semantics,'get_llm',lambda **_:model)
    with tools._session() as session: require_two_customer_evidence(session)
    opened=run_conversation(MESSAGE,'batch-materials')
    # 使用真实对话请求接入禁令，无需直接篡改数据库状态。
    class BoundaryModel:
        def json_mode(self):return self
        def invoke(self,prompt):return AIMessage(content=json.dumps(dict(commands=[],objectives=[],execution_boundaries=[dict(restriction='NO_REFUND',operation='DENY',source_quote='不要退款')])))
    monkeypatch.setattr(orchestrator,'get_llm',lambda **_:BoundaryModel())
    run_conversation('不要退款','batch-materials')
    monkeypatch.setattr(orchestrator,'get_llm',lambda **_:BatchModel())
    result=run_conversation('原包装和附件完整，序列号照片明天补','batch-materials',request_id='materials')
    updates=[e for e in result['trace'] if e['tool'] in {'continue_after_sales_case','record_after_sales_evidence_readiness'}]
    assert len(updates)==2,result
    assert updates[1]['result']['revision']>updates[0]['result']['revision']
    bindings=[e for e in result['events'] if e['type']=='case_update_revision_bound']
    assert bindings[0]['case_revision']==updates[0]['result']['revision']
    assert any(b['restriction']=='NO_REFUND' for b in result['execution_boundaries'])
    assert result['pending_action'] is None
    assert result['request_semantics']['requirements'][0]['requirement']['outcome']['kind']=='after_sales_case'
    replay=run_conversation('原包装和附件完整，序列号照片明天补','batch-materials',request_id='materials')
    assert replay['run_id']==result['run_id']
    with tools._session() as session:
        assert session.scalar(select(Refund).where(Refund.order_id==1001)).status=='none'


def test_batch_restart_replays_first_write_and_resumes_second(database,monkeypatch):
    from tests.test_after_sales_runtime import require_two_customer_evidence, CrashAfterCaseWrite, CrashAfterAutonomousCaseToolResult
    from src.persistence import RunStore
    from src.durable import resume_run
    from src.backend.db import ToolLedger
    with tools._session() as session: require_two_customer_evidence(session)
    run_conversation(MESSAGE,'batch-restart')
    monkeypatch.setattr(orchestrator,'get_llm',lambda **_:BatchModel())
    with pytest.raises(CrashAfterCaseWrite):
        run_conversation('原包装和附件完整，序列号照片明天补','batch-restart',store=CrashAfterAutonomousCaseToolResult())
    run_id=RunStore().conversation('batch-restart')['active_run_id']
    result=resume_run('batch-restart',run_id)
    updates=[t for t in result['trace'] if t['tool'] in {'continue_after_sales_case','record_after_sales_evidence_readiness'}]
    assert len(updates)==2,result
    assert updates[0]['status']=='REUSED'
    assert updates[1]['status']=='SUCCEEDED'
    with tools._session() as session:
        ledger=session.scalars(select(ToolLedger).where(ToolLedger.run_id==run_id)).all()
        assert len([r for r in ledger if r.tool_name=='continue_after_sales_case'])==1
        assert len([r for r in ledger if r.tool_name=='record_after_sales_evidence_readiness'])==1


def test_explicit_draft_only_succeeds_without_proposing_refund(database,monkeypatch):
    from tests.test_after_sales_runtime import AfterSalesUnderstandingModel
    import src.request_semantics_runtime as semantics
    class DraftPlanner(AfterSalesUnderstandingModel):
        def invoke(self,prompt):
            result=json.loads(super().invoke(prompt).content)
            result['execution_boundaries']=[dict(restriction='NO_REFUND',operation='DENY',source_quote='不要退款')]
            return AIMessage(content=json.dumps(result))
    class DraftExtractor(Extractor):
        def invoke(self,messages):
            result=json.loads(super().invoke(messages).content)
            result['changes'].append(dict(operation='ADD',requirement=dict(requirement_id='no_refund',
                kind='PROHIBITION',description='不要退款',source_quote='不要退款',restriction='NO_REFUND')))
            return AIMessage(content=json.dumps(result))
    monkeypatch.setattr('src.after_sales.intent.verify_intent',real_verify)
    monkeypatch.setattr(orchestrator,'get_llm',lambda **_:DraftPlanner())
    monkeypatch.setattr(semantics,'get_llm',lambda **_:DraftExtractor())
    result=run_conversation(MESSAGE+'；不要退款','draft-only')
    assert result['status']=='SUCCEEDED',result
    assert result['pending_action'] is None
    assert any(e['type']=='execution_boundary_blocked' for e in result['events'])
    with tools._session() as session:
        assert len(session.scalars(select(AfterSalesCase)).all())==1
        assert session.scalar(select(Refund).where(Refund.order_id==1001)).status=='none'


def test_independent_extra_outcome_cannot_cover_a_missing_planned_goal(database, monkeypatch):
    import src.request_semantics_runtime as semantics

    class MultiIntentExtractor(Extractor):
        def invoke(self, messages):
            result = json.loads(super().invoke(messages).content)
            result['changes'].append(dict(operation='ADD', requirement=dict(
                requirement_id='shipment', kind='OUTCOME', description='查询物流',
                source_quote='查询订单1002的物流',
                outcome=dict(kind='track_shipment', order_id=1002))))
            return AIMessage(content=json.dumps(result))

    model = MultiIntentExtractor()
    monkeypatch.setattr('src.after_sales.intent.verify_intent', real_verify)
    monkeypatch.setattr(semantics, 'get_llm', lambda **_: model)
    result = run_conversation(MESSAGE + '；再查询订单1002的物流', 'missing-goal')
    assert result['terminal_reason'] == 'INCOMPLETE_REQUEST_COVERAGE', result
    assert not result['trace']
    assert len(model.inputs) == 1  # 规划修复复用独立提取结果，不重复计费。
    with tools._session() as session:
        assert not session.scalars(select(AfterSalesCase)).all()


def test_model_schema_leaves_fixed_capability_metadata_to_host():
    from src.dialogue_understanding import _prompt
    prompt = _prompt(MESSAGE, [], None, None, None)
    schema = json.loads(prompt.split('\nSchema:\n')[1].split('\nConversation input:\n')[0])
    fields = schema['$defs']['BusinessObjective']['properties']
    assert 'target_type' not in fields
    assert set(fields['evidence_requirement']['enum']) == {'MODEL_KNOWLEDGE', 'REALTIME_EXTERNAL'}


@pytest.mark.parametrize('metadata', [dict(target_type='order'), dict(evidence_requirement='AFTER_SALES_CASE')])
def test_conflicting_legacy_capability_metadata_still_rejected(metadata):
    from src.capability_registry import get_capability_registry
    from src.errors import AgentError
    with pytest.raises(AgentError, match='target_type|证据类型'):
        get_capability_registry().validate_arguments(dict(
            kind='AFTER_SALES_CASE', order_id=1001, reason_code='NOT_AS_DESCRIBED', **metadata))
