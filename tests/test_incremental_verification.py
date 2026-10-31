"""增量核验的协议、跨类型影响与失效依赖。"""
from copy import deepcopy
from types import SimpleNamespace
import json

import pytest
from langchain_core.messages import AIMessage

from src.after_sales.incremental import add_prior_context, assess_incremental, validate_transition
from src.after_sales.service import _checksum
from src.errors import AgentError
from src.runtime import BudgetLimits


def inputs():
    assessment=dict(requirement_id='p:SERIAL', status='SUFFICIENT',
                    citations=[dict(evidence_id='old', quote='标签完整')],
                    explanation='原文已满足要求', missing_information=[], question='')
    return dict(case_id='case', case_revision=2, latest_request='包装完整，但刚才的标签情况说错了', requested_withdrawal=None,
                requirements=[dict(requirement_id='p:SERIAL', evidence_type='SERIAL', description='标签状态')],
                evidence=[dict(evidence_id='old', evidence_type='SERIAL', statement='标签完整', source_text='标签完整'),
                          dict(evidence_id='incoming', evidence_type='PACKAGING', statement='包装完整，但刚才的标签情况说错了',
                               source_text='包装完整，但刚才的标签情况说错了')],
                prior_assessments={'p:SERIAL':dict(assessment_id='prior',assessment=assessment,
                    preserve_allowed=True,reason_code='UNCHANGED_DEPENDENCIES')})


class Runtime:
    limits=BudgetLimits()
    def __init__(self, response): self.response=response; self.calls=0
    def model_call(self, *_, **__):
        self.calls+=1
        if isinstance(self.response, Exception): raise self.response
        return AIMessage(content=json.dumps(self.response, ensure_ascii=False))


def response(action='PRESERVE', assessment=None):
    return dict(requirements=[dict(requirement_id='p:SERIAL', action=action, assessment=assessment,
                                  reason='检查完整原文后决定')])


def test_preserve_copies_host_result_without_rewriting_citations():
    data=inputs()
    result=assess_incremental(data, model=None, runtime=Runtime(response()))
    assert result['requirements']==[data['prior_assessments']['p:SERIAL']['assessment']]
    assert result['requirements'][0] is not data['prior_assessments']['p:SERIAL']['assessment']


@pytest.mark.parametrize('citation', ['incoming', 'old'])
def test_cross_type_correction_needs_incoming_impact_citation(citation):
    data=inputs()
    item=next(e for e in data['evidence'] if e['evidence_id']==citation)
    assessment=dict(requirement_id='p:SERIAL',status='CONFLICT',citations=[dict(evidence_id=citation,quote=item['statement'])],
                    missing_information=['标签真实情况'], explanation='新增原文影响旧材料', question='标签到底是否完整？')
    result=assess_incremental(data,model=None,runtime=Runtime(response('REASSESS',assessment)))
    assert result['requirements'][0]['status']==('CONFLICT' if citation=='incoming' else 'UNKNOWN')


@pytest.mark.parametrize('reason', ['matching', 'withdrawal', 'expired', 'explicit', 'missing_source'])
def test_changed_dependencies_cannot_preserve(reason):
    data=inputs()
    old=deepcopy(data['prior_assessments']['p:SERIAL']['assessment'])
    semantic={**old,'assessment_id':'prior'}
    row=dict(requirement_id='p:SERIAL',semantic_assessment=semantic,rejected_evidence={})
    args={}
    if reason=='matching': data['evidence'][-1]['evidence_type']='SERIAL'
    if reason=='withdrawal': args['revoke_evidence_id']='old'
    if reason=='expired': row['rejected_evidence']={'old':'EXPIRED'}
    if reason=='explicit': data['evidence'].pop()
    if reason=='missing_source': data['evidence'].pop(0)
    add_prior_context(data,SimpleNamespace(facts={'evidence_requirements':[row]}),args)
    assert not data['prior_assessments']['p:SERIAL']['preserve_allowed']
    result=assess_incremental(data,model=None,runtime=Runtime(response()))
    assert result['error_code']=='SEMANTIC_PROTOCOL_INVALID'
    assert result['requirements'][0]['status']=='UNKNOWN'


def test_missing_valid_assessment_after_policy_or_source_change_cannot_preserve():
    data=inputs()
    add_prior_context(data,SimpleNamespace(facts={'evidence_requirements':[dict(requirement_id='p:SERIAL')]}),{})
    assert data['prior_assessments']=={}
    assert assess_incremental(data,model=None,runtime=Runtime(response()))['requirements'][0]['status']=='UNKNOWN'


@pytest.mark.parametrize('field,value', [('description','虚假材料'),('action','INVENTED'),('prior_assessment_id','foreign'),
                                      ('to_status','UNKNOWN'),('reason_code','invented')])
def test_even_rehashed_report_cannot_misrepresent_transition(field,value):
    data=inputs(); result=assess_incremental(data,model=None,runtime=Runtime(response()))
    report=result.pop('transition_report'); report['changes'][0][field]=value
    report['report_id']=_checksum({k:v for k,v in report.items() if k!='report_id'})
    with pytest.raises(ValueError): validate_transition(result,data,report)


def test_model_error_abstains_kernel_error_propagates_and_empty_skips_model():
    result=assess_incremental(inputs(),model=None,runtime=Runtime(AgentError('MODEL_TIMEOUT','timeout',retryable=False)))
    assert result['error_code']=='SEMANTIC_MODEL_UNAVAILABLE'
    with pytest.raises(AgentError):
        assess_incremental(inputs(),model=None,runtime=Runtime(AgentError('RUN_CANCELLED','cancelled',retryable=False)))
    data=inputs(); data['requirements']=[]
    runtime=Runtime(None)
    assert assess_incremental(data,model=None,runtime=runtime)['requirements']==[]
    assert runtime.calls==0
