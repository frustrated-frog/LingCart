"""Evidence-grounded execution predicates, separate from immutable user intent."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from src.adaptive_decision import Constraint, ConstraintBinding
from src.catalog_concepts import CatalogConceptError, get_catalog_concept_registry
from src.errors import AgentError


_PRODUCT_TOOLS = {'search_products', 'discover_products', 'compare_products'}
_READABLE_STATUSES = {'SUCCEEDED', 'REUSED'}


def _sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def constraint_refs(goal):
    return {**{f'global:{i}': c for i, c in enumerate(goal.get('hard_constraints', []))},
            **{f"objective:{o['id']}:{i}": c for o in goal['objectives']
               for i, c in enumerate(o.get('hard_constraints', []))}}


def fingerprint(bindings):
    return _sha256(bindings)


def effective_goal(state):
    goal = deepcopy(state['goal'])
    refs = constraint_refs(goal)
    for ref, binding in state.get('constraint_bindings', {}).items():
        if ref not in refs:
            raise AgentError('INVALID_CONSTRAINT_BINDING', '字段映射引用了未知用户条件', retryable=False)
        refs[ref].update({key: binding[key] for key in ('field', 'op', 'value')})
    return goal


def _path_value(product, field):
    value = product
    for part in field.split('.'):
        if not isinstance(value, dict) or part not in value:
            return None, False
        value = value[part]
    return value, True


def record_binding_evidence(state, step):
    """Persist kernel-readable rows without granting planner read access."""
    if (step.tool not in _PRODUCT_TOOLS or step.status not in _READABLE_STATUSES
            or not isinstance(step.result, dict)):
        return
    ids = sorted({product.get('product_id') for product in step.result.get('products', [])
                  if isinstance(product, dict) and isinstance(product.get('product_id'), str)})
    if ids:
        state.setdefault('binding_evidence_access', {})[step.step_id] = ids


def propose_bindings(state, plan, proposals, *, evidence_access=None):
    """Validate provenance and predicate shape; a model separately reviews meaning."""
    constraints = constraint_refs(state['goal'])
    indexed = {s.step_id: s for s in plan.steps}
    access = state.get('evidence_access', {}) if evidence_access is None else evidence_access
    bindings = deepcopy(state.get('constraint_bindings', {}))
    for raw_proposal in proposals:
        proposal = (raw_proposal if isinstance(raw_proposal, ConstraintBinding)
                    else ConstraintBinding.model_validate(raw_proposal))
        ref = proposal.constraint_ref
        if ref not in constraints:
            raise AgentError('INVALID_CONSTRAINT_BINDING', '未知约束引用；可用引用：' + ', '.join(constraints), retryable=False)
        try:
            predicate = Constraint.model_validate({**constraints[ref],
                **proposal.model_dump(include={'field', 'op', 'value'})})
        except ValueError as error:
            raise AgentError('INVALID_CONSTRAINT_BINDING', '操作符、数值类型或概念 codec 不匹配；数值范围使用单个数字，集合操作使用字符串数组。', retryable=False) from error
        rows = []
        for evidence_ref in proposal.evidence_refs:
            step = indexed.get(evidence_ref)
            read_ids = set(access.get(evidence_ref, []))
            if (step is None or step.status not in _READABLE_STATUSES
                    or not isinstance(step.result, dict) or not read_ids):
                raise AgentError('UNREAD_BINDING_EVIDENCE', '字段映射只能引用当前边界实际读取并校验过的商品证据', retryable=False)
            rows.extend(product for product in step.result.get('products', [])
                        if isinstance(product, dict) and product.get('product_id') in read_ids)
        observed = []
        for product in rows:
            value, found = _path_value(product, predicate.field)
            if found:
                observed.append(value)
        if not observed or not any(value is not None and not isinstance(value, (dict, list, bool))
                                   for value in observed):
            raise AgentError('UNOBSERVED_BINDING_FIELD', '映射字段必须存在于已校验的商品记录中；缺失字段不能证明否定条件', retryable=False)
        bindings[ref] = proposal.model_dump(exclude={'constraint_ref'})
    proposed = {**state, 'constraint_bindings': bindings}
    if constraint_refs(effective_goal(proposed)) == constraint_refs(effective_goal(state)):
        raise AgentError('BINDING_ALREADY_CURRENT', '该映射已生效，无需重复审查；请继续选择工具、推荐、完成或澄清。软偏好不需要字段绑定。', retryable=False)
    return bindings


def _combined_binding_access(state):
    result = deepcopy(state.get('binding_evidence_access', {}))
    for step_id, ids in state.get('evidence_access', {}).items():
        result[step_id] = sorted(set(result.get(step_id, [])) | set(ids))
    return result


def catalog_evidence(state, plan):
    """只提供映射所引用的已校验字段事实，不把旧核验状态混进映射审查。"""
    indexed = {s.step_id: s for s in plan.steps}
    access = _combined_binding_access(state)
    evidence = []
    for ref, binding in state.get('constraint_bindings', {}).items():
        observations = []
        for step_id in binding['evidence_refs']:
            step = indexed.get(step_id)
            if step is None or not isinstance(step.result, dict):
                continue
            for product in step.result.get('products', []):
                if product.get('product_id') not in access.get(step_id, []):
                    continue
                value, found = _path_value(product, binding['field'])
                if found:
                    observations.append(dict(evidence_ref=step_id,
                        product_id=product['product_id'], value=value))
        evidence.append(dict(constraint_ref=ref, field=binding['field'], observations=observations))
    return evidence


def build_binding_transaction(state, bindings, proposals, *, source, evidence_rows=(), registry=None):
    """Create a content-addressed transaction before an independent audit runs."""
    serialized = [proposal.model_dump() if isinstance(proposal, ConstraintBinding) else deepcopy(proposal)
                  for proposal in proposals]
    registry_identity = registry.manifest() if registry is not None else None
    basis = {
        'source': source,
        'goal_revision': state['goal_revision'],
        'goal_sha256': _sha256(state['goal']),
        'before_bindings_sha256': fingerprint(state.get('constraint_bindings', {})),
        'after_bindings_sha256': fingerprint(bindings),
        'proposals': serialized,
        'evidence_sha256': _sha256(list(evidence_rows)),
        'registry': registry_identity,
    }
    transaction_id = _sha256(basis)
    return {**basis, 'transaction_id': transaction_id, 'status': 'PROPOSED',
            'changed_refs': sorted(proposal['constraint_ref'] for proposal in serialized),
            'proposed_bindings': deepcopy(bindings), 'audit_reference': None}


def resolve_catalog_concepts(state, plan):
    """Build an all-or-nothing registry proposal from verified catalog rows."""
    if not state.get('goal') or not (state.get('request_audit') or {}).get('verdict') == 'APPROVED':
        return None
    constraints = constraint_refs(state['goal'])
    pending = []
    for ref, raw in constraints.items():
        constraint = Constraint.model_validate(raw)
        if constraint.catalog_concept is not None and constraint.field is None and ref not in state.get('constraint_bindings', {}):
            pending.append((ref, constraint.catalog_concept))
    if not pending:
        return None

    registry = get_catalog_concept_registry()
    access = state.get('binding_evidence_access', {})
    if not access:
        return None
    indexed = {step.step_id: step for step in plan.steps}
    resolution_id = _sha256({
        'goal_revision': state['goal_revision'],
        'pending': [(ref, concept.model_dump()) for ref, concept in pending],
        'access': access,
        'observations': {step_id: ((indexed.get(step_id).result or {}).get('observation') or {}).get('digest')
                         for step_id in sorted(access) if indexed.get(step_id) is not None},
    })
    proposals = []
    evidence_rows = []
    unresolved = []
    for ref, concept in pending:
        field = registry.field_for(concept)
        values = []
        evidence_refs = []
        rows = []
        for step_id in sorted(access):
            step = indexed.get(step_id)
            if (step is None or step.status not in _READABLE_STATUSES
                    or step.tool not in _PRODUCT_TOOLS or not isinstance(step.result, dict)):
                continue
            allowed_ids = set(access[step_id])
            for product in step.result.get('products', []):
                if product.get('product_id') not in allowed_ids:
                    continue
                value, found = _path_value(product, field)
                if not found or value is None or isinstance(value, (dict, list, bool)):
                    continue
                values.append(value)
                evidence_refs.append(step_id)
                rows.append({'constraint_ref': ref, 'evidence_ref': step_id,
                             'product_id': product['product_id'], 'field': field, 'value': value,
                             'observation_digest': (step.result.get('observation') or {}).get('digest')})
        try:
            resolved = registry.resolve(concept, tuple(values))
        except CatalogConceptError as error:
            unresolved.append({'constraint_ref': ref, 'concept_id': concept.concept_id,
                               'code': 'CODEC_EVIDENCE_INCOMPATIBLE', 'message': str(error)})
            continue
        refs = list(dict.fromkeys(evidence_refs))[:8]
        if not refs:
            unresolved.append({'constraint_ref': ref, 'concept_id': concept.concept_id,
                               'code': 'CATALOG_FIELD_UNOBSERVED', 'message': '尚无已校验字段观察'})
            continue
        proposals.append(ConstraintBinding(constraint_ref=ref, field=resolved.field,
            op=resolved.op, value=resolved.value, evidence_refs=refs,
            reason=f'由版本化概念 {concept.concept_id} 经 {resolved.codec_id} 无损编码；仍需独立语义审查'))
        evidence_rows.extend(rows)
    if unresolved:
        return {'status': 'UNRESOLVED', 'resolution_id': resolution_id,
                'unresolved': unresolved, 'proposals': []}

    bindings = propose_bindings(state, plan, proposals, evidence_access=access)
    transaction = build_binding_transaction(state, bindings, proposals, source='catalog_registry',
        evidence_rows=evidence_rows, registry=registry)
    previous = state.get('binding_tx') or {}
    if (previous.get('transaction_id') == transaction['transaction_id']
            and previous.get('status') in {'PROPOSED', 'REJECTED', 'APPLIED'}):
        return None
    return {'status': 'PROPOSED', 'resolution_id': resolution_id,
            'bindings': bindings, 'proposals': proposals,
            'transaction': transaction, 'evidence_rows': evidence_rows}


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
    transaction = state.get('binding_tx') or {}
    if transaction and (transaction.get('status') != 'APPLIED'
            or transaction.get('after_bindings_sha256') != fingerprint(state['constraint_bindings'])
            or transaction.get('applied_binding_revision') != state.get('binding_revision', 0)
            or transaction.get('audit_reference') != audit.get('reference')
            or audit.get('binding_transaction_id') != transaction.get('transaction_id')):
        raise AgentError('BINDING_TRANSACTION_NOT_COMMITTED', '映射事务、独立审查与当前执行谓词不一致', retryable=False)
