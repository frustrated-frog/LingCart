"""由政策约束驱动的案件计划修订，复用目标身份与未失效的证据依赖。"""
from __future__ import annotations

from datetime import datetime
import uuid
from sqlalchemy import select

from src.after_sales.service import _checksum
from src.backend.db import AfterSalesCasePlan

PLANNER_VERSION = "evidence-driven-case-plan-v1"


def revise_case_plan(session, case, decision, now: datetime):
    existing = session.scalar(select(AfterSalesCasePlan).where(
        AfterSalesCasePlan.case_id == case.case_id, AfterSalesCasePlan.case_revision == case.revision))
    if existing:
        return existing.payload
    previous = session.scalar(select(AfterSalesCasePlan).where(
        AfterSalesCasePlan.case_id == case.case_id).order_by(AfterSalesCasePlan.revision.desc()).limit(1))
    before = previous.payload if previous else {}
    old_goals = {g['goal_id']: g for g in before.get('goals', [])}
    goals, actions, retained, changed = [], [], [], []
    submitted = case.status == 'SUBMITTED'
    requirements = decision.facts.get('evidence_requirements', [])
    for requirement in requirements:
        goal_id = f"evidence:{requirement['evidence_type']}"
        basis = dict(requirement)
        if basis.get('semantic_assessment'):
            basis['semantic_assessment'] = {k: v for k, v in basis['semantic_assessment'].items()
                                            if k != 'assessment_id'}
        goal = {'goal_id': goal_id, 'requirement_id': requirement['requirement_id'],
                'description': requirement['description'], 'status': requirement['status'],
                'evidence_refs': requirement['evidence_refs'], 'depends_on': [],
                'basis_hash': _checksum(basis)}
        if old_goals.get(goal_id) == goal:
            goals.append(old_goals[goal_id])
            retained.append(goal_id)
        else:
            goals.append(goal)
            changed.append(goal_id)
        if requirement['status'] != 'SATISFIED' and not submitted and 'ADD_EVIDENCE' in decision.allowed_actions:
            if requirement['validation_condition'] == 'AUTHORITATIVE_ORDER' and 'ORDER_RECORD' in requirement['accepted_sources']:
                actions.append({'kind': 'COLLECT_BUSINESS_EVIDENCE', 'goal_id': goal_id,
                                'operation': 'continue_after_sales_case',
                                'requirement_id': requirement['requirement_id']})
            elif requirement['validation_condition'] == 'CUSTOMER_ATTESTATION' and 'CUSTOMER_STATEMENT' in requirement['accepted_sources']:
                semantic = requirement.get('semantic_assessment') or {}
                actions.append({'kind': 'REQUEST_CUSTOMER_EVIDENCE', 'goal_id': goal_id,
                                'requirement_id': requirement['requirement_id'],
                                'question': semantic.get('question') or requirement['description'],
                                'semantic_status': semantic.get('status', 'NOT_ASSESSED')})
            else:
                actions.append({'kind': 'REQUIRES_HUMAN_CAPABILITY', 'goal_id': goal_id})
    removed = sorted(set(old_goals) - {g['goal_id'] for g in goals} - {'resolution'})
    root = {'goal_id': 'resolution', 'description': case.requested_resolution,
            'status': 'SUBMITTED' if submitted else case.status,
            'depends_on': [g['goal_id'] for g in goals], 'evidence_refs': list(decision.evidence_refs),
            'basis_hash': _checksum({'decision_id': decision.decision_id, 'case_status': case.status})}
    goals.append(root)
    if old_goals.get('resolution') == root:
        retained.append('resolution')
    else:
        changed.append('resolution')
    if not submitted:
        if 'REQUEST_CUSTOMER_CONFIRMATION' in decision.allowed_actions:
            actions.append({'kind': 'REQUEST_CUSTOMER_CONFIRMATION', 'goal_id': 'resolution',
                            'operation': 'submit_after_sales_case', 'requires_approval': True})
        elif case.status == 'MANUAL_REVIEW':
            actions.append({'kind': 'REQUIRES_HUMAN_CAPABILITY', 'goal_id': 'resolution'})
    payload = {
        'schema_version': 1, 'planner_version': PLANNER_VERSION, 'plan_id': uuid.uuid4().hex,
        'case_id': case.case_id, 'revision': (previous.revision + 1) if previous else 1,
        'case_revision': case.revision, 'decision_id': decision.decision_id,
        'policy_version_id': decision.policy_version_id, 'status': case.status,
        'supersedes_plan_id': previous.plan_id if previous else None,
        'goals': goals, 'next_actions': actions,
        'retained_goal_ids': retained, 'changed_goal_ids': changed, 'removed_goal_ids': removed,
        'invalidated_decision_id': before.get('decision_id') if before.get('decision_id') != decision.decision_id else None,
        'created_at': now.isoformat(),
    }
    session.add(AfterSalesCasePlan(plan_id=payload['plan_id'], case_id=case.case_id,
                                  revision=payload['revision'], case_revision=case.revision,
                                  decision_id=decision.decision_id, payload=payload, created_at=now))
    session.flush()
    return payload


def plans_for_case(session, case, decision_valid):
    rows = session.scalars(select(AfterSalesCasePlan).where(AfterSalesCasePlan.case_id == case.case_id)
                           .order_by(AfterSalesCasePlan.revision)).all()
    return tuple({**row.payload,
                  'valid': bool(row == rows[-1] and decision_valid and row.payload['status'] == case.status)}
                 for row in rows)
