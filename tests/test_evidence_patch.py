"""统一材料协议使用真实 Runtime/SQLite；脚本模型不代表效果评测。"""
import json
from copy import deepcopy

import pytest
from sqlalchemy import select

from src import orchestrator, tools
from src.backend.db import AfterSalesEvidence, PendingAction, Refund
from src.durable import run_conversation, resume_run
from src.harness import get_runner
from src.persistence import RunStore
from src.after_sales.evidence_patch import PATCHES
from tests.test_incremental_case_state import (database, models, Adapter, MESSAGE, SERIAL, PACKAGING,
    CrashAfterCaseWrite, CrashAfterAutonomousCaseToolResult)

PROFILE='after-sales-evidence-patch-v1'


@pytest.fixture
def patch_models(models,monkeypatch):
    models.raw_override=None
    def intake(prompt):
        result=models.intake(prompt)
        payload=json.loads(prompt.split('\nConversation input:\n')[-1].split('\nPrevious output was invalid.')[0])
        if payload['session_cases']:
            schema=json.loads(prompt.split('\nSchema:\n')[-1].split('\nConversation input:\n')[0])
            props=schema['$defs']['BusinessObjective']['properties']
            assert 'material_changes' in props
            assert not {'evidence_text','evidence_updates','clarification_id','case_revision','revoke_evidence_id','evidence_readiness_claims'} & set(props)
            request=payload['latest_human_message']
            case=payload['session_cases'][0]
            requirements={r['evidence_type']:r['requirement_id'] for r in case['requirements']}
            patches=[]
            for kind,disposition in models.scopes.get(request,[]):
                operation='DEFER' if disposition=='DEFERRED' else 'RETRACT' if disposition=='RETRACTED' else 'REPLACE' if models.revoke else 'PROVIDE'
                patch=dict(operation=operation,source_quote=request)
                if operation!='RETRACT':patch['requirement_id']=requirements[kind]
                if models.revoke:patch['evidence_id']=models.revoke
                patches.append(patch)
            result=dict(commands=[],objectives=[dict(id='material',kind='AFTER_SALES_CONTINUE',source_quote=request,material_changes=patches)])
            if models.raw_override:result=models.raw_override(result)
        return result
    monkeypatch.setattr(orchestrator,'get_llm',lambda **_:Adapter(intake))
    return models


def run(text, **kwargs):
    return run_conversation(text,'patch',runner=get_runner(PROFILE),**kwargs)


@pytest.mark.parametrize('crash',[False,True])
def test_batch_patch_compiles_to_existing_ledger_and_recovers_once(patch_models,crash):
    run(MESSAGE)
    text='原包装附件和序列号标签均完整清晰'
    patch_models.scopes[text]=[('ORIGINAL_PACKAGING','PROVIDED'),('SERIAL_LABEL','PROVIDED')]
    if crash:
        with pytest.raises(CrashAfterCaseWrite):run(text,store=CrashAfterAutonomousCaseToolResult())
        active=RunStore().conversation('patch')['active_run_id']
        result=resume_run('patch',active)
    else: result=run(text)
    assert result['status']=='WAITING_APPROVAL',result
    assert not result.get('pending_clarification')
    with tools._session() as session:
        rows=session.scalars(select(AfterSalesEvidence).where(AfterSalesEvidence.evidence_type.in_(['ORIGINAL_PACKAGING','SERIAL_LABEL']))).all()
        assert len(rows)==2
        assert session.scalar(select(Refund).where(Refund.order_id==1001)).status=='none'


def test_defer_then_provide_and_explicit_replace_then_retract(patch_models):
    run(MESSAGE)
    text='包装晚点核对'
    patch_models.scopes[text]=[('ORIGINAL_PACKAGING','DEFERRED')]
    assert not run(text).get('pending_action')
    run(SERIAL)
    approved=run(PACKAGING)
    assert approved['status']=='WAITING_APPROVAL'
    with tools._session() as session:
        old=session.scalar(select(AfterSalesEvidence).where(AfterSalesEvidence.evidence_type=='SERIAL_LABEL'))
        patch_models.revoke=old.evidence_id
    replacement='撤回之前标签完整的陈述，重新检查后发现标签破损'
    patch_models.scopes[replacement]=[('SERIAL_LABEL','PROVIDED')]
    patch_models.conflict=True
    result=run(replacement)
    assert not result.get('pending_action')
    with tools._session() as session:
        assert session.get(AfterSalesEvidence,patch_models.revoke).status!='ACTIVE'
        assert session.get(PendingAction,approved['pending_action']['action_id']).status=='CLOSED'
        active=session.scalar(select(AfterSalesEvidence).where(AfterSalesEvidence.evidence_type=='SERIAL_LABEL',AfterSalesEvidence.status=='ACTIVE'))
        patch_models.revoke=active.evidence_id
    withdrawal='撤回刚才标签破损的陈述，暂时不能确认标签情况'
    patch_models.scopes[withdrawal]=[('SERIAL_LABEL','RETRACTED')]
    patch_models.conflict=False
    result=run(withdrawal)
    assert not result.get('pending_action')
    with tools._session() as session:
        assert session.get(AfterSalesEvidence,patch_models.revoke).status!='ACTIVE'


@pytest.mark.parametrize('mode',['legacy','foreign','duplicate','unquoted'])
def test_invalid_patch_cannot_write(patch_models,mode):
    run(MESSAGE)
    def invalid(result):
        objective=result['objectives'][0]
        if mode=='legacy':objective['evidence_text']=SERIAL
        if mode=='foreign':objective['material_changes'][0]['requirement_id']='foreign'
        if mode=='duplicate':objective['material_changes']*=2
        if mode=='unquoted':objective['material_changes'][0]['source_quote']='虚构材料'
        return result
    patch_models.raw_override=invalid
    result=run(SERIAL)
    assert result['status']=='WAITING_CLARIFICATION' and not result['trace'],result


@pytest.mark.parametrize('patch',[
    dict(operation='PROVIDE',source_quote='完整',requirement_id='p',evidence_id='old'),
    dict(operation='RETRACT',source_quote='撤回'),
    dict(operation='REPLACE',source_quote='改为破损',requirement_id='p'),
    dict(operation='DEFER',source_quote='稍后',requirement_id='p',revoke_evidence_id='old'),
])
def test_patch_variants_reject_mixed_or_missing_fields(patch):
    with pytest.raises(ValueError):PATCHES.validate_python([patch])


def test_candidate_only_changes_tool_protocol_and_is_not_a_product_profile():
    from pathlib import Path
    from eval.experiment_contract import ExperimentSpec
    from src.harness import get_product_runner
    from src.errors import AgentError
    experiment=ExperimentSpec.from_path(Path('research/experiments/after-sales-evidence-patch-v1/experiment-spec.json'))
    left=get_runner(experiment.baseline_profile).provenance()
    right=get_runner(experiment.candidate_profile).provenance()
    assert left['engine']==right['engine']
    assert [m for m in left['mechanisms'] if m['slot']!='tool_protocol']==[m for m in right['mechanisms'] if m['slot']!='tool_protocol']
    with pytest.raises(AgentError,match='仅限研究'):get_product_runner(PROFILE)
