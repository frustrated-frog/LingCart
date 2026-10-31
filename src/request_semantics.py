"""独立需求契约及显式修订；只解释 HUMAN 原文，不授予工具权限。"""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from src.adaptive_decision import Constraint, StrictModel
from src.constraint_bindings import fingerprint
from src.errors import AgentError
from src.request_ir import SourceSpan, locate_source_span


class OutcomeSpec(StrictModel):
    kind: Literal['recommend', 'get_order_status', 'get_order_details', 'track_shipment',
                  'check_refund_eligibility', 'get_payment_status', 'answer_faq', 'compare_products', 'unresolved']
    quantity: int = Field(1, ge=1, le=100)
    order_id: int | None = Field(None, ge=1)
    question: str | None = Field(None, min_length=1, max_length=500)
    product_ids: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode='after')
    def valid_target(self):
        order_kind = self.kind in {'get_order_status', 'get_order_details', 'track_shipment',
                                  'check_refund_eligibility', 'get_payment_status'}
        if order_kind != (self.order_id is not None):
            raise ValueError('Order outcomes require an order ID, other outcomes cannot carry one')
        if (self.kind == 'answer_faq') != (self.question is not None):
            raise ValueError('Only FAQ outcomes require a question')
        if self.kind != 'recommend' and self.quantity != 1:
            raise ValueError('Only recommendations carry cardinality')
        if self.product_ids and (self.kind != 'compare_products' or len(set(self.product_ids)) != len(self.product_ids)):
            raise ValueError('Only comparisons carry distinct product IDs')
        return self


class IntentRequirement(StrictModel):
    requirement_id: str = Field(pattern=r'^[a-zA-Z][a-zA-Z0-9_-]{0,39}$')
    kind: Literal['OUTCOME', 'CONDITION', 'PROHIBITION', 'EXECUTION_BOUNDARY', 'USER_FACT', 'SOFT_PREFERENCE', 'UNKNOWN']
    description: str = Field(min_length=1, max_length=500)
    source_quote: str = Field(min_length=1, max_length=500)
    outcome: OutcomeSpec | None = None
    predicate: Constraint | None = None
    targets: list[str] = Field(default_factory=list, max_length=8)
    restriction: Literal['READ_ONLY', 'NO_PAYMENT', 'NO_REFUND', 'NO_SUBMISSION'] | None = None

    @model_validator(mode='after')
    def typed_requirement(self):
        if (self.kind == 'OUTCOME') != (self.outcome is not None):
            raise ValueError('Only OUTCOME requires outcome')
        if (self.kind == 'CONDITION') != (self.predicate is not None):
            raise ValueError('Only CONDITION requires a predicate')
        if (self.kind == 'CONDITION') != bool(self.targets) or len(self.targets) != len(set(self.targets)):
            raise ValueError('Conditions require explicit unique outcome targets; context has none')
        if self.predicate is not None and self.predicate.source_quote != self.source_quote:
            raise ValueError('Predicate source must equal requirement source')
        if self.restriction and self.kind not in {'PROHIBITION', 'EXECUTION_BOUNDARY'}:
            raise ValueError('Only an execution boundary may restrict effects')
        return self


class RequirementChange(StrictModel):
    operation: Literal['PRESERVE', 'REPLACE', 'RETRACT', 'ADD']
    previous_id: str | None = None
    source_quote: str | None = Field(None, min_length=1, max_length=500)
    requirement: IntentRequirement | None = None

    @model_validator(mode='after')
    def coherent_change(self):
        if (self.operation == 'ADD') != (self.previous_id is None):
            raise ValueError('ADD has no previous_id; every other operation needs one')
        if (self.operation in {'ADD', 'REPLACE'}) != (self.requirement is not None):
            raise ValueError('Only ADD/REPLACE provide a requirement')
        if self.operation in {'REPLACE', 'RETRACT'} and not self.source_quote:
            raise ValueError('Changing prior intent requires a current HUMAN quote')
        if self.operation == 'REPLACE' and self.requirement.requirement_id != self.previous_id:
            raise ValueError('REPLACE retains stable requirement identity')
        return self


class IntentExtraction(StrictModel):
    changes: list[RequirementChange] = Field(min_length=1, max_length=64)


def reject(message, *, code='REQUEST_SEMANTICS_MISMATCH'):
    raise AgentError(code, '需求语义契约未通过：' + message, retryable=False)


def validate_contract(contract):
    """校验保存内容的身份；原文完整性在当前对话绑定时再次验证。"""
    if not isinstance(contract, dict) or contract.get('version') != 'request-semantics/v1':
        reject('未知语义契约版本')
    body = {key: value for key, value in contract.items() if key != 'sha256'}
    if contract.get('sha256') != fingerprint(body):
        reject('持久化语义契约指纹不一致')
    rows = [IntentRequirement.model_validate(row['requirement']) for row in contract['requirements']]
    for stored, row in zip(contract['requirements'], rows):
        source = SourceSpan.model_validate(stored['source'])
        if source.text != row.source_quote:
            reject('保存的原文范围与需求引用不一致')
    ids = [row.requirement_id for row in rows]
    if len(set(ids)) != len(ids):
        reject('需求身份重复')
    outcomes = {row.requirement_id for row in rows if row.kind == 'OUTCOME'}
    for row in rows:
        if not set(row.targets) <= outcomes:
            reject('条件引用了不存在或已撤销的结果')
    return rows


def apply_extraction(extraction, request, history=None, *, previous=None, change_text=None):
    """宿主应用完整修订集合。PRESERVE 直接复制，不允许模型悄悄重写。"""
    extraction = IntentExtraction.model_validate(extraction)
    old = {}
    if previous:
        old = {row.requirement_id: row for row in validate_contract(previous)}
    seen, active, revisions = set(), {}, []
    for change in extraction.changes:
        identity = change.previous_id
        if identity is not None:
            if identity not in old or identity in seen:
                reject('旧需求必须逐项且仅处理一次')
            seen.add(identity)
        if change.source_quote is not None:
            # STEER 的完整请求含旧文本；只有本次追加的原文可以授权修订。
            locate_source_span(change.source_quote, change_text if change_text is not None else request)
        if change.operation == 'PRESERVE':
            row = old[identity]
        elif change.operation == 'RETRACT':
            row = None
        else:
            row = change.requirement
            if change.operation == 'ADD' and row.requirement_id in old:
                reject('新增需求不能复用旧身份')
            if previous and change.operation == 'ADD':
                locate_source_span(change.source_quote or row.source_quote,
                                   change_text if change_text is not None else request)
        if row is not None:
            if row.requirement_id in active:
                reject('有效需求身份重复')
            source = locate_source_span(row.source_quote, request, history)
            if row.outcome and row.outcome.question:
                locate_source_span(row.outcome.question, request, history)
            active[row.requirement_id] = dict(requirement=row.model_dump(), source=source.model_dump())
        revisions.append(dict(operation=change.operation, previous_id=identity,
                              requirement_id=row.requirement_id if row else None,
                              source_quote=change.source_quote))
    if seen != set(old):
        reject('有旧需求未声明保留、替换或撤销')
    body = dict(version='request-semantics/v1', requirements=list(active.values()),
                previous_sha256=previous['sha256'] if previous else None, revisions=revisions)
    contract = {**body, 'sha256': fingerprint(body)}
    validate_contract(contract)
    return contract


def _predicate_identity(predicate, request, history):
    if predicate.get('catalog_concept'):
        from src.catalog_concepts import CatalogConceptRef, get_catalog_concept_registry
        encoded = get_catalog_concept_registry().encode(CatalogConceptRef.model_validate(predicate['catalog_concept']))
        predicate = {**predicate, 'field': encoded.field, 'op': encoded.op, 'value': encoded.value}
    elif predicate.get('field') == 'category':
        from src.capability_registry import get_capability_registry
        from src.request_ir import _human_sources
        values = predicate['value'] if isinstance(predicate['value'], list) else [predicate['value']]
        normalized = []
        for value in values:
            if not isinstance(value, str):
                normalized.append(value)
                continue
            args, _ = get_capability_registry().normalize_arguments(dict(
                id='semantic-category', kind='PRODUCT_SEARCH', source_quote=predicate['source_quote'], category=value),
                tuple(text for _, _, text in _human_sources(request, history)))
            normalized.append(args['category'])
        predicate = {**predicate, 'value': normalized if isinstance(predicate['value'], list) else normalized[0]}
    return ('predicate', fingerprint({key: predicate.get(key) for key in ('field', 'op', 'value')}))


def compare_goal(contract, goal, request, history=None):
    """独立提取的数量、目标和条件范围与规划结果双向匹配，不读取模型理由。"""
    from src.adaptive_decision import ShoppingGoal
    goal = ShoppingGoal.model_validate(goal).model_dump()
    rows = validate_contract(contract)
    for stored, row in zip(contract['requirements'], rows):
        span = locate_source_span(row.source_quote, request, history)
        # 修订阶段已重新绑定原文；执行/恢复阶段禁止来源偏移漂移。
        if span.model_dump() != stored['source']:
            reject('需求原文来源发生变化')
    if any(row.kind == 'UNKNOWN' for row in rows):
        reject('存在需要用户澄清的条件', code='REQUEST_SEMANTICS_UNCERTAIN')
    outcomes = [row for row in rows if row.kind == 'OUTCOME']
    if len(outcomes) != len(goal['objectives']):
        reject('目标遗漏或额外增加')
    mapping, used = {}, set()
    for row in outcomes:
        expected = row.outcome.model_dump()
        span = locate_source_span(row.source_quote, request, history)
        candidates = []
        for objective in goal['objectives']:
            actual = {key: objective[key] for key in expected}
            source = locate_source_span(objective['source_quote'], request, history)
            if (actual == expected and source.source_sha256 == span.source_sha256
                    and max(source.start, span.start) < min(source.end, span.end)):
                candidates.append(objective['id'])
        if len(candidates) != 1 or candidates[0] in used:
            reject(f'{row.requirement_id} 的数量、目标或来源不一致/不明确')
        mapping[row.requirement_id] = candidates[0]
        used.add(candidates[0])
    expected_conditions = []
    for row in rows:
        if row.kind == 'CONDITION':
            for target in row.targets:
                if next(item for item in outcomes if item.requirement_id == target).outcome.kind != 'recommend':
                    reject('商品条件不能约束非推荐目标')
                expected_conditions.append((mapping[target], _predicate_identity(row.predicate.model_dump(), request, history),
                                            locate_source_span(row.source_quote, request, history)))
    actual_conditions = []
    for objective in goal['objectives']:
        constraints = [*goal['hard_constraints'], *objective['hard_constraints']] if objective['kind'] == 'recommend' else []
        for constraint in constraints:
            actual_conditions.append((objective['id'], _predicate_identity(constraint, request, history),
                                      locate_source_span(constraint['source_quote'], request, history)))
    # 保留来源和作用范围；重复条件不额外改变语义。
    def matches(left, right):
        return (left[:2] == right[:2] and left[2].source_sha256 == right[2].source_sha256
                and max(left[2].start, right[2].start) < min(left[2].end, right[2].end))
    if (any(not any(matches(row, item) for item in actual_conditions) for row in expected_conditions)
            or any(not any(matches(row, item) for item in expected_conditions) for row in actual_conditions)):
        reject('条件遗漏、新增或作用范围/比较值不一致')
    return mapping


def affected_outcomes(previous, current):
    """从结果自身及其适用条件计算变更，不按整轮版本把所有任务一起作废。"""
    def identities(contract):
        rows = validate_contract(contract)
        return {row.requirement_id: fingerprint(dict(
            outcome=row.outcome.model_dump(), source_quote=row.source_quote,
            conditions=sorted((fingerprint(item.model_dump()) for item in rows
                               if item.kind == 'CONDITION' and row.requirement_id in item.targets))))
            for row in rows if row.kind == 'OUTCOME'}
    before, after = identities(previous), identities(current)
    return dict(preserved=sorted(key for key in after if before.get(key) == after[key]),
                changed=sorted(key for key in after if key in before and before[key] != after[key]),
                added=sorted(set(after) - set(before)), retracted=sorted(set(before) - set(after)))
