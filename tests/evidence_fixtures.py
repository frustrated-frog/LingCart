"""显式脚本核验替身；仅测试使用，不证明自然语言泛化。"""
import json
from datetime import datetime
from langchain_core.messages import AIMessage


def response_for(inputs, status="SUFFICIENT", *, withdrawal=True):
    withdrawn = inputs.get('requested_withdrawal') if withdrawal else None
    rows = []
    for requirement in inputs['requirements']:
        evidence = [e for e in inputs['evidence'] if e['evidence_type'] == requirement['evidence_type']
                    and e['evidence_id'] != withdrawn]
        verdict = status if evidence else 'INCOMPLETE'
        rows.append({'requirement_id': requirement['requirement_id'], 'status': verdict,
                     'citations': [{'evidence_id': e['evidence_id'], 'quote': e['statement']} for e in evidence],
                     'missing_information': [] if verdict == 'SUFFICIENT' else ['需要确认实际情况'],
                     'explanation': '脚本核验测试结果',
                     'question': '' if verdict == 'SUFFICIENT' else '请确认商品本身是否损坏？'})
    return {'requirements': rows, 'withdrawal_confirmed': bool(withdrawn),
            'withdrawal_quote': inputs['latest_request'] if withdrawn else ''}


class SemanticModel:
    def __init__(self, status='SUFFICIENT', *, withdrawal=True):
        self.status, self.withdrawal = status, withdrawal
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        inputs = json.loads(prompt.split('\nEvidence input:\n')[-1])
        return AIMessage(content=json.dumps(response_for(inputs, self.status, withdrawal=self.withdrawal), ensure_ascii=False))


def record_assessment_fixture(session, case, *, now=datetime(2026, 6, 20, 12)):
    """为只测试确定性政策/账本的用例显式提供独立核验产物。"""
    from src.after_sales.policy import AfterSalesPolicyEngine
    from src.after_sales.semantic import evidence_basis, persist_assessment
    from src.after_sales.service import _checksum
    from src.backend.db import AgentRun
    evaluation = AfterSalesPolicyEngine(session).evaluate(case.case_id, as_of=now)
    inputs = {'requirements': [{k: r[k] for k in ('requirement_id', 'description', 'evidence_type')}
                                for r in evaluation.facts['evidence_requirements']],
              'evidence': [{'evidence_id': e['evidence_id'], 'evidence_type': e['evidence_type'],
                            'statement': e['payload'].get('statement', ''),
                            'source_text': (session.get(AgentRun, e['source_ref'].removeprefix('agent_runs:').removesuffix(':request')).request
                                            if session.get(AgentRun, e['source_ref'].removeprefix('agent_runs:').removesuffix(':request')) else '')}
                           for e in case.evidence
                           if e['status'] == 'ACTIVE' and e['source_type'] == 'CUSTOMER_STATEMENT'],
              'policy_version_id': evaluation.policy_version_id, 'policy_checksum': evaluation.policy_checksum,
              'basis': evidence_basis(case.evidence)}
    from src.backend.db import AfterSalesSemanticAssessment
    from sqlalchemy import select
    if session.scalar(select(AfterSalesSemanticAssessment).where(
            AfterSalesSemanticAssessment.case_id == case.case_id,
            AfterSalesSemanticAssessment.input_hash == _checksum(inputs))):
        return
    persist_assessment(session, case, 'evidence-run', {
        'input': inputs, 'input_hash': _checksum(inputs), 'result': response_for(inputs),
        'verifier_version': 'test-scripted-verifier'}, now=now)
