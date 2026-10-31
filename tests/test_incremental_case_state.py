"""独立材料归属、增量结论与真实账本回归；脚本模型不计入效果评测。"""
from copy import deepcopy
import json

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import select

from src import llm, orchestrator, tools
from src.after_sales.intent import verify_intent as real_verify
from src.backend.db import AfterSalesEvidence, Refund, ToolLedger, PendingAction
from src.durable import run_conversation, resume_run
from src.harness import get_runner
from src.persistence import RunStore
from tests.test_after_sales_runtime import (database, MESSAGE, require_two_customer_evidence,
    CrashAfterCaseWrite, CrashAfterAutonomousCaseToolResult)

PROFILE = 'after-sales-incremental-state-v1'
SERIAL = '序列号标签清晰完整'
PACKAGING = '原包装完整，附件齐全'


class Models:
    def __init__(self):
        self.scopes = {SERIAL: [('SERIAL_LABEL', 'PROVIDED')], PACKAGING: [('ORIGINAL_PACKAGING', 'PROVIDED')]}
        self.batch = False
        self.repair_batch = False
        self.invalid_batch_parent = False
        self.explicit_target = None
        self.bad_boundary = False
        self.repair_boundary = True
        self.revoke = None
        self.conflict = False
        self.semantic_calls = 0
        self.prompts = []

    def intake(self, prompt):
        self.prompts.append(prompt)
        payload = json.loads(prompt.split('\nConversation input:\n')[-1].split('\nPrevious output was invalid.')[0])
        request = payload['latest_human_message']
        if not payload['session_cases']:
            objective = dict(id='case', kind='AFTER_SALES_CASE', order_id=1001, reason_code='NOT_AS_DESCRIBED', source_quote=request)
        else:
            case = payload['session_cases'][0]
            requirements = {r['evidence_type']: r['requirement_id'] for r in case['requirements']}
            objective = dict(id='material', kind='AFTER_SALES_CONTINUE', source_quote=request)
            if self.batch or (self.repair_batch and 'Previous output was invalid' in prompt):
                objective['evidence_updates'] = [dict(requirement_id=requirements[kind], disposition=disposition, source_quote=request)
                                                 for kind, disposition in self.scopes.get(request, [])]
                if self.invalid_batch_parent:
                    objective['source_quote']=request[:4]
            else:
                objective['evidence_text'] = request
                if self.explicit_target:
                    objective['requirement_id'] = requirements[self.explicit_target]
            if self.revoke:
                objective['revoke_evidence_id'] = self.revoke
        result = dict(commands=[], objectives=[objective])
        if self.bad_boundary and (not self.repair_boundary or 'Previous output was invalid' not in prompt):
            result['execution_boundaries'] = [dict(restriction='NO_REFUND', operation='DENY', source_quote=request)]
        return result

    def extract(self, messages):
        payload = json.loads(messages[-1].content)
        assert 'goal' not in payload and 'objectives' not in payload
        prior = payload.get('prior_contract')
        changes = [dict(operation='PRESERVE', previous_id=row['requirement']['requirement_id'])
                   for row in (prior or {}).get('requirements', [])]
        if not prior:
            changes.append(dict(operation='ADD', requirement=dict(requirement_id='case', kind='OUTCOME',
                description='建立售后案件', source_quote=MESSAGE,
                outcome=dict(kind='after_sales_case', order_id=1001, reason_code='NOT_AS_DESCRIBED'))))
        request = payload['request']
        for kind, disposition in self.scopes.get(request, []):
            case = payload['case_context'][0]
            req = next(r for r in case['requirements'] if r['evidence_type'] == kind)
            identity = kind.lower()
            existing = next((r for r in changes if r.get('previous_id') == identity), None)
            if existing:
                changes.remove(existing)
            changes.append(dict(operation='REPLACE' if existing else 'ADD',
                **(dict(previous_id=identity, source_quote=request) if existing else {}),
                requirement=dict(requirement_id=identity, kind='USER_FACT', description='本轮材料声明',
                    source_quote=request, material_scope=dict(case_id=case['case_id'], requirement_id=req['requirement_id'],
                                                              disposition=disposition))))
        return dict(changes=changes)

    def semantic(self, prompt):
        self.semantic_calls += 1
        data = json.loads(prompt.split('\nEvidence input:\n')[-1])
        rows = []
        for req in data['requirements']:
            old = data.get('prior_assessments', {}).get(req['requirement_id'], {})
            if old.get('preserve_allowed'):
                rows.append(dict(requirement_id=req['requirement_id'], action='PRESERVE', reason='新增材料不改变这项既有判断'))
                continue
            evidence = [e for e in data['evidence'] if e['evidence_type'] == req['evidence_type']
                        and e['evidence_id'] != data.get('requested_withdrawal')]
            status = ('CONFLICT' if self.conflict and req['evidence_type']=='SERIAL_LABEL' else 'SUFFICIENT') if evidence else 'UNKNOWN'
            assessment = dict(requirement_id=req['requirement_id'], status=status,
                citations=[dict(evidence_id=e['evidence_id'], quote=e['statement']) for e in evidence],
                missing_information=[] if status=='SUFFICIENT' else (['状态', '清晰程度'] if req['evidence_type']=='SERIAL_LABEL' else ['包装状态']),
                explanation='脚本核验，不代表模型效果', question='' if status=='SUFFICIENT' else '请确认'+req['description'])
            rows.append(dict(requirement_id=req['requirement_id'], action='REASSESS', assessment=assessment, reason='该要求需要本轮核验'))
        return dict(requirements=rows, withdrawal_confirmed=bool(data.get('requested_withdrawal')),
                    withdrawal_quote=data['latest_request'] if data.get('requested_withdrawal') else '')


class Adapter:
    def __init__(self, invoke): self.fn=invoke
    def json_mode(self): return self
    def invoke(self, prompt):
        return AIMessage(content=json.dumps(self.fn(prompt), ensure_ascii=False),
                         usage_metadata={'input_tokens':1000,'output_tokens':150,'total_tokens':1150})


@pytest.fixture
def models(database, monkeypatch):
    model = Models()
    with tools._session() as session: require_two_customer_evidence(session)
    monkeypatch.setattr('src.after_sales.intent.verify_intent', real_verify)
    monkeypatch.setattr('src.request_semantics_runtime.get_llm', lambda **_: Adapter(model.extract))
    monkeypatch.setattr(orchestrator, 'get_llm', lambda **_: Adapter(model.intake))
    monkeypatch.setattr(llm, 'get_llm', lambda **_: Adapter(model.semantic))
    return model


def run(text, session='incremental', **kwargs):
    return run_conversation(text, session, runner=get_runner(PROFILE), **kwargs)


def transitions(result):
    return [e['result']['evidence_transition'] for e in result['events']
            if e['type']=='tool_result' and isinstance(e.get('result'),dict) and e['result'].get('evidence_transition')]


@pytest.mark.parametrize('crash', [False, True])
def test_out_of_order_material_is_rebound_then_preserved_until_confirmation(models, crash):
    opened = run(MESSAGE)
    assert opened['pending_clarification']['requirement_id'].endswith(':ORIGINAL_PACKAGING')
    if crash:
        with pytest.raises(CrashAfterCaseWrite): run(SERIAL, store=CrashAfterAutonomousCaseToolResult())
        active = RunStore().conversation('incremental')['active_run_id']
        calls = models.semantic_calls
        second = resume_run('incremental', active)
        assert models.semantic_calls == calls
        assert any(t['tool']=='continue_after_sales_case' and t['status']=='REUSED' for t in second['trace'])
    else:
        second = run(SERIAL)
    assert any(e['type']=='case_material_target_reconciled' and e['to_requirement_id'].endswith(':SERIAL_LABEL')
               for e in second['events'])
    assert second['pending_clarification']['requirement_id'].endswith(':ORIGINAL_PACKAGING')
    final = run(PACKAGING)
    assert final['status']=='WAITING_APPROVAL', final
    report = transitions(final)[-1]
    serial = next(c for c in report['changes'] if c['evidence_type']=='SERIAL_LABEL')
    assert serial['action']=='PRESERVE' and serial['from_status']==serial['to_status']=='SUFFICIENT'
    assert report['commit_status']=='COMMITTED' and report['to_revision'] > report['from_revision']
    with tools._session() as session:
        rows=session.scalars(select(AfterSalesEvidence).where(AfterSalesEvidence.evidence_type=='SERIAL_LABEL')).all()
        assert len(rows)==1 and rows[0].payload['statement']==SERIAL
        assert session.scalar(select(Refund).where(Refund.order_id==1001)).status=='none'
        assert len(session.scalars(select(ToolLedger).where(ToolLedger.run_id==second['run_id'], ToolLedger.tool_name=='continue_after_sales_case')).all())==1


def test_explicit_scope_conflict_clarifies_without_writing_and_can_continue(models):
    run(MESSAGE)
    models.explicit_target='ORIGINAL_PACKAGING'
    result=run(SERIAL)
    assert result['status']=='WAITING_CLARIFICATION', result
    assert not result['trace']
    assert '归属判断不一致' in result['reply']
    assert any('independent_requirement_id' in p for p in models.prompts)
    models.explicit_target='SERIAL_LABEL'
    assert run(SERIAL).get('error') is None


@pytest.mark.parametrize('repairable', [True, False])
def test_boundary_disagreement_uses_specific_feedback_then_clarifies_if_needed(models, repairable):
    run(MESSAGE)
    models.bad_boundary=True
    models.repair_boundary=repairable
    result=run(SERIAL)
    assert any('proposed_change' in p and 'independent_restrictions' in p for p in models.prompts)
    if repairable:
        assert any(t['tool']=='continue_after_sales_case' for t in result['trace'])
        assert result.get('error') is None
    else:
        assert result['status']=='WAITING_CLARIFICATION' and not result['trace']
        assert '修改执行限制' in result['reply']


def test_batch_shared_quote_preserves_evidence_type_binding(models):
    run(MESSAGE)
    text='原包装附件和序列号标签均完整清晰'
    models.scopes[text]=[('ORIGINAL_PACKAGING','PROVIDED'),('SERIAL_LABEL','PROVIDED')]
    models.batch=True
    result=run(text)
    assert result['status']=='WAITING_APPROVAL', result
    assert result.get('pending_clarification') is None
    with tools._session() as session:
        evidence=session.scalars(select(AfterSalesEvidence).where(
            AfterSalesEvidence.source_ref==f"agent_runs:{result['run_id']}:request")).all()
        assert {e.evidence_type for e in evidence}=={'ORIGINAL_PACKAGING','SERIAL_LABEL'}
        assert len({e.evidence_id for e in evidence})==2


def test_conflict_revokes_approval_and_explicit_correction_rechecks_material(models):
    run(MESSAGE); run(SERIAL)
    approved=run(PACKAGING)
    action=approved['pending_action']['action_id']
    text='刚才说错了，序列号标签已经破损'
    models.scopes[text]=[('SERIAL_LABEL','PROVIDED')]
    models.explicit_target='SERIAL_LABEL'
    models.conflict=True
    conflict=run(text)
    assert not conflict.get('pending_action')
    change=next(c for c in transitions(conflict)[-1]['changes'] if c['evidence_type']=='SERIAL_LABEL')
    assert change['action']=='REASSESS' and change['to_status']=='CONFLICT'
    with tools._session() as session:
        assert session.get(PendingAction,action).status=='CLOSED'
        bad=session.scalar(select(AfterSalesEvidence).where(AfterSalesEvidence.source_ref==f"agent_runs:{conflict['run_id']}:request"))
        models.revoke=bad.evidence_id
    corrected='撤回刚才标签破损的陈述，我已重新核对，序列号标签清晰完整'
    models.scopes[corrected]=[('SERIAL_LABEL','PROVIDED')]
    models.conflict=False
    final=run(corrected)
    assert final['status']=='WAITING_APPROVAL', final
    assert final['pending_action']['action_id']!=action
    with tools._session() as session:
        assert session.get(AfterSalesEvidence,models.revoke).status!='ACTIVE'
        assert session.scalar(select(Refund).where(Refund.order_id==1001)).status=='none'


def test_deferred_material_then_provided_does_not_invent_evidence(models):
    run(MESSAGE)
    text='包装稍后再找，暂时无法确认'
    models.scopes[text]=[('ORIGINAL_PACKAGING','DEFERRED')]
    models.batch=True
    result=run(text)
    assert not result.get('pending_action')
    with tools._session() as session:
        assert not session.scalars(select(AfterSalesEvidence).where(AfterSalesEvidence.evidence_type=='ORIGINAL_PACKAGING')).all()
    models.batch=False
    run(SERIAL)
    assert run(PACKAGING)['status']=='WAITING_APPROVAL'


@pytest.mark.parametrize('invalid_parent',[False,True])
def test_batch_repair_is_bounded_and_bad_quotes_end_in_targeted_clarification(models,invalid_parent):
    run(MESSAGE)
    text='原包装附件和序列号标签均完整清晰'
    models.scopes[text]=[('ORIGINAL_PACKAGING','PROVIDED'),('SERIAL_LABEL','PROVIDED')]
    models.repair_batch=True
    models.invalid_batch_parent=invalid_parent
    result=run(text)
    assert any('objective.source_quote covers all update quotes' in p for p in models.prompts)
    if invalid_parent:
        assert result['status']=='WAITING_CLARIFICATION' and not result['trace']
        assert '分别说明每项材料' in result['reply']
        assert len([e for e in result['events'] if e['type']=='dialogue_protocol_retry'])==1
    else:
        assert result['status']=='WAITING_APPROVAL' and result.get('pending_clarification') is None


def test_incremental_model_failure_closes_approval(models, monkeypatch):
    run(MESSAGE); run(SERIAL)
    approved=run(PACKAGING)
    monkeypatch.setattr(llm, 'get_llm', lambda **_: Adapter(lambda _: {}))
    models.explicit_target='ORIGINAL_PACKAGING'
    result=run(PACKAGING)
    assert not result.get('pending_action')
    assert all(c['to_status']=='UNKNOWN' for c in transitions(result)[-1]['changes'])
    with tools._session() as session:
        assert session.get(PendingAction, approved['pending_action']['action_id']).status=='CLOSED'
