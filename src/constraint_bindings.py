"""Evidence-grounded execution predicates, separate from immutable user intent."""
from copy import deepcopy
import hashlib
import json

from src.adaptive_decision import Constraint
from src.errors import AgentError


def constraint_refs(goal):
    return {**{f'global:{i}': c for i, c in enumerate(goal.get('hard_constraints', []))},
            **{f"objective:{o['id']}:{i}": c for o in goal['objectives']
               for i, c in enumerate(o.get('hard_constraints', []))}}


def fingerprint(bindings):
    return hashlib.sha256(json.dumps(bindings, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def effective_goal(state):
    goal = deepcopy(state['goal'])
    refs = constraint_refs(goal)
    for ref, binding in state.get('constraint_bindings', {}).items():
        if ref not in refs:
            raise AgentError('INVALID_CONSTRAINT_BINDING', '字段映射引用了未知用户条件', retryable=False)
        refs[ref].update({key: binding[key] for key in ('field', 'op', 'value')})
    return goal


def propose_bindings(state, plan, proposals):
    """Validate provenance and predicate shape; a model separately reviews meaning."""
    constraints = constraint_refs(state['goal'])
    indexed = {s.step_id: s for s in plan.steps}
    bindings = deepcopy(state.get('constraint_bindings', {}))
    for proposal in proposals:
        ref = proposal.constraint_ref
        if ref not in constraints:
            raise AgentError('INVALID_CONSTRAINT_BINDING', '字段映射必须引用现有硬约束，不能增删用户要求', retryable=False)
        try:
            predicate = Constraint.model_validate({**constraints[ref], **proposal.model_dump(include={'field', 'op', 'value'})})
        except ValueError as error:
            raise AgentError('INVALID_CONSTRAINT_BINDING', '映射谓词的操作符与数值类型不匹配', retryable=False) from error
        rows = []
        for evidence_ref in proposal.evidence_refs:
            step = indexed.get(evidence_ref)
            read_ids = set(state.get('evidence_access', {}).get(evidence_ref, []))
            if (step is None or step.status not in {'SUCCEEDED', 'REUSED', 'REJECTED'}
                    or not isinstance(step.result, dict) or not read_ids):
                raise AgentError('UNREAD_BINDING_EVIDENCE', '字段映射只能引用实际读取过的商品证据', retryable=False)
            rows.extend(p for p in step.result.get('products', []) if p['product_id'] in read_ids)
        observed = []
        for product in rows:
            value = product
            for part in predicate.field.split('.'):
                if not isinstance(value, dict) or part not in value:
                    break
                value = value[part]
            else:
                observed.append(value)
        if not observed or not any(v is not None and not isinstance(v, (dict, list, bool)) for v in observed):
            raise AgentError('UNOBSERVED_BINDING_FIELD', '映射字段必须存在于已读商品记录中；缺失字段不能证明否定条件', retryable=False)
        bindings[ref] = proposal.model_dump(exclude={'constraint_ref'})
    proposed = {**state, 'constraint_bindings': bindings}
    if constraint_refs(effective_goal(proposed)) == constraint_refs(effective_goal(state)):
        raise AgentError('BINDING_ALREADY_CURRENT', '该映射已生效，无需重复审查；请继续选择工具、推荐、完成或澄清。软偏好不需要字段绑定。', retryable=False)
    return bindings


def require_bindings_approved(state):
    if not state.get('constraint_bindings'):
        return
    from src.request_audit import binding
    audit = state.get('binding_audit') or {}
    if (audit.get('verdict') != 'APPROVED'
            or audit.get('binding') != binding(state['request'], state['goal'], state['goal_revision'])
            or audit.get('bindings_sha256') != fingerprint(state['constraint_bindings'])
            or audit.get('binding_revision') != state.get('binding_revision', 0)):
        raise AgentError('BINDINGS_NOT_AUDITED', '执行映射尚未通过与当前用户原意绑定的独立审查', retryable=False)
