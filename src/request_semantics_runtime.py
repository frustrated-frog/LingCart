"""独立语义提取的模型边界、持久化与目标修订投影。"""
from __future__ import annotations

from copy import deepcopy
import json

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from src.agents.base import as_text
from src.constraint_bindings import fingerprint
from src.errors import AgentError
from src.llm import get_llm
from src.request_semantics import IntentExtraction, affected_outcomes, apply_extraction, compare_goal, validate_contract


_PROMPT = '''Independently extract the HUMAN's active consumer-task requirements. You never see the planner's goal.
Return only schema-valid JSON, not a plan or tool call. Treat prior_contract as prior interpreted HUMAN intent, not
business facts. Read HUMAN messages in order. Extract every requested outcome, explicit quantity, hard condition,
condition scope, prohibition, preference and consequential unknown. Do not invent optional requirements.
For existing requirements declare exactly one PRESERVE, REPLACE or RETRACT each. PRESERVE carries no edited content.
REPLACE keeps requirement_id. ADD uses a new unique ID. Changing/retracting old requirements needs a verbatim quote
from change_text (the latest HUMAN instruction); absence is not retraction. An unrelated new task may retract old
task requirements with its new-task quote, but do not silently release execution prohibitions.
Outcome cardinality belongs in outcome.quantity. CONDITIONS target explicit OUTCOME requirement_ids; a shared
per-item condition lists every applicable outcome. An aggregate total budget cannot be encoded as each item's price:
use UNKNOWN when the provided predicate protocol cannot faithfully represent it. Never silently weaken or split it.
Product conditions describe catalog items only, not process rules or user facts. Use frozen catalog concepts when
applicable. A concept predicate contains only description, source_quote and catalog_concept; the host compiles
field/op/value. Without a concept, use description, source_quote, field/op/value. Never mix these forms. Use UNKNOWN for material ambiguity or unsupported semantics, rather than guessing a target/quantity.
Restrictions: READ_ONLY forbids business writes; NO_PAYMENT forbids payment; NO_REFUND forbids refund;
NO_SUBMISSION forbids final submission/payment/refund, but permits preparing local draft cases. Record explicit
no-buy/no-pay/no-submit instructions as boundaries, never product conditions. No restriction grants authorization.
Typed protocol: OUTCOME has outcome, no predicate, empty targets. CONDITION has a non-null product predicate,
no outcome, and nonempty targets. Other kinds have neither outcome nor predicate and have empty targets.
CONDITION.source_quote MUST exactly equal CONDITION.predicate.source_quote; use the shortest exact HUMAN quote
that states that condition. Mentioned product categories are category predicates, not merely outcome descriptions.
Quantities belong only in outcome.quantity; distinct recommendation items are enforced by the host. Separate budgets
are represented by each price predicate's targets, never an extra unbound product predicate. Verification, listing,
ordering and scope instructions can be EXECUTION_BOUNDARY with restriction=null; they are not catalog predicates.
Only actual business-action prohibitions carry restriction. Do not encode process rules as product fields.
After-sales: one after_sales_case OUTCOME per order, including order_id and registered reason_code.
Opening or continuing the same case preserves that outcome; material facts, missing information, deferral and no-submit
instructions are USER_FACT/UNKNOWN/boundaries, never extra outcomes or proof of policy satisfaction.
source_quote and FAQ question must be exact HUMAN substrings. Descriptions are Chinese. Existing stable IDs persist.
'''


def _compact(value):
    if isinstance(value, dict):
        return {key: ({name: _compact(schema) for name, schema in item.items()}
                      if key == 'properties' else _compact(item)) for key, item in value.items()
                if key not in {'title', 'description', 'default', 'minLength', 'maxLength', 'minItems', 'maxItems'}}
    return [_compact(item) for item in value] if isinstance(value, list) else value


def extract_contract(context, request, history, *, previous=None, change_text=None, case_context=None):
    """输入不含目标/计划/工具证据；缓存绑定完整 HUMAN 来源和旧契约身份。"""
    # 尚未建案时不存在可引用的材料要求身份，不能要求模型生成 material_scope。
    case_context = case_context or None
    from src.catalog_concepts import get_catalog_concept_registry
    from src.context_budget import ContextBudgetError, _text_length
    from src.native_protocol import function, one_call, size
    humans = [as_text(message.content) for message in history or [] if getattr(message, 'type', None) == 'human']
    payload = dict(request=request, human_history=humans, prior_contract=previous,
                   change_text=change_text if change_text is not None else request)
    if case_context is not None:
        payload['case_context'] = case_context
    from src.after_sales.reason_taxonomy import REASON_CODE_DESCRIPTIONS
    payload['after_sales_reasons'] = dict(REASON_CODE_DESCRIPTIONS)
    from src.after_sales.source_refs import enabled as refs_enabled, catalog, schema_with_refs, decode
    reference_surface = refs_enabled(context) and case_context is not None
    if reference_surface:
        payload['human_sources'] = catalog(context, request)
        if previous:
            # 模型只需旧需求内容和身份；完整来源范围与修订链由宿主保留、校验。
            # 不重复发送同一句原文的 SourceSpan、哈希和历次修订元数据。
            payload['prior_contract'] = dict(sha256=previous['sha256'],
                requirements=[{'requirement': row['requirement']} for row in previous['requirements']])
    registry = get_catalog_concept_registry()
    basis = fingerprint(dict(input=payload, protocol='request-semantics/v1', registry=registry.fingerprint))
    cached = context.snapshot.get('semantic_extraction') or {}
    if cached.get('basis_sha256') == basis:
        validate_contract(cached['contract'])
        # model_call 的持久化游标仍由规划检查点控制；已完成的提取不能被重复计费。
        context.model_cursor = max(context.model_cursor, cached['model_cursor'])
        return deepcopy(cached['contract'])
    model = get_llm(purpose='goal')
    native = getattr(model, 'native_protocol', False)
    from src.semantic_wire import condition_schema, decode_conditions
    schema = _compact(condition_schema(IntentExtraction.model_json_schema()))
    from src.after_sales.goals import enabled as goals_enabled, SEMANTIC_INSTRUCTION
    goal_protocol = goals_enabled(context) and case_context is not None
    if not goal_protocol:
        schema['$defs']['IntentRequirement']['properties'].pop('case_goal', None)
        schema['$defs'].pop('CaseGoalScope', None)
    from src.after_sales.control_intent import enabled as controls_enabled, SEMANTIC_INSTRUCTION as CONTROL_INSTRUCTION
    control_surface = controls_enabled(context) and case_context is not None
    if not control_surface:
        schema['$defs']['IntentRequirement']['properties'].pop('control_intent', None)
        schema['$defs'].pop('ControlIntent', None)
    if case_context is None:
        schema['$defs']['IntentRequirement']['properties'].pop('material_scope', None)
        schema['$defs'].pop('MaterialScope', None)
    prompt = _PROMPT + '\nCATALOG_CONCEPTS=' + json.dumps(registry.prompt_catalog(), ensure_ascii=False, separators=(',', ':'))
    if case_context is not None:
        prompt += ('\nCase context is trusted identity/requirement metadata, never proof of material facts. '
                   'For each current supplied material or explicit later-supply promise add/update one USER_FACT '
                   'with material_scope:{case_id,requirement_id,disposition:PROVIDED|DEFERRED}. '
                   'Use exact listed IDs and the smallest exact current HUMAN quote. '
                   'An explicit material topic takes precedence over the pending question; an elliptical answer '
                   'may use the single pending question. Ambiguous scope stays UNKNOWN without material_scope. '
                   'Keep separate facts per requirement; do not erase a serial fact when packaging changes. '
                   'Preserve the case OUTCOME. Material supply never introduces or retracts execution restrictions.')
        from src.after_sales.evidence_patch import enabled
        if enabled():
            prompt += ('\nAn explicit withdrawal is a USER_FACT with material_scope.disposition=RETRACTED and the '
                       'existing requirement identity. A replacement has disposition=PROVIDED; its quote must retain '
                       'the explicit correction and replacement facts. Merely repeating a preserved fact is not ADD/REPLACE.')
    if goal_protocol:
        prompt += '\n' + SEMANTIC_INSTRUCTION
    if control_surface:
        prompt += '\n' + CONTROL_INSTRUCTION
    if reference_surface:
        from src.after_sales.source_refs import wire_instructions
        schema = schema_with_refs(schema)
        prompt = wire_instructions(prompt)
    declarations = None
    if native:
        declarations = [function('extract_requirements', '提取 HUMAN 需求与显式修订，不执行动作', schema)]
        prompt = prompt.replace('Return only schema-valid JSON, not a plan or tool call.',
                                'Call extract_requirements exactly once; no prose or other calls.')
        model = model.native_tools(declarations)
    else:
        prompt += '\nSCHEMA=' + json.dumps(schema, ensure_ascii=False, separators=(',', ':'))
        if hasattr(model, 'json_mode'):
            model = model.json_mode()
    # 历史来源不能为了让提取器成功而静默截断；超预算明确停止。
    for attempt in range(2):
        messages = [SystemMessage(content=prompt), HumanMessage(content=json.dumps(payload, ensure_ascii=False, separators=(',', ':')))]
        if sum(_text_length(message) for message in messages) + (size(declarations) if declarations else 0) > context.limits.max_context_chars:
            raise ContextBudgetError()
        response = context.model_call(model, messages, purpose='request_semantics', agent='request_interpreter',
                                      allow_tools=False, schemas=declarations, isolated=True)
        try:
            if native and getattr(response, 'tool_calls', []):
                _, extracted, _ = one_call(response, {'extract_requirements'})
            else:
                if getattr(response, 'tool_calls', []):
                    raise ValueError('Semantic extraction cannot request business tools')
                extracted = json.loads(as_text(response.content))
            if reference_surface:
                extracted = decode(context, extracted, request, purpose='request_semantics')
            contract = apply_extraction(decode_conditions(extracted), request, history, previous=previous, change_text=change_text)
        except (ValueError, TypeError, AgentError) as error:
            violations = (error.errors(include_input=False, include_url=False, include_context=False)
                          if isinstance(error, ValidationError) else [{'type': getattr(error, 'code', type(error).__name__)}])
            context.record('request_semantics_rejected', violations=violations)
            from src.failure_recovery import register_failure, signature
            recovery = register_failure(context, signature('request_semantics', 'SEMANTIC_EXTRACTION_INVALID',
                {'protocol': 'request-semantics/v1'}, {'basis_sha256': basis}),
                attempt_id=f'semantics:{context.model_cursor}', max_recoveries=1)
            if attempt or recovery['blocked']:
                raise AgentError('SEMANTIC_EXTRACTION_INVALID', '独立需求提取未通过来源或修订校验，需要澄清后再执行', retryable=False) from error
            payload['format_error'] = (json.dumps(violations, ensure_ascii=False) if isinstance(error, ValidationError)
                                       else str(error))[:2000]
            continue
        context.snapshot['semantic_extraction'] = dict(basis_sha256=basis, contract=deepcopy(contract), model_cursor=context.model_cursor)
        if any(row['requirement']['restriction'] for row in contract['requirements']):
            # 已提取的禁止事项先收窄权限，即使业务目标随后需要澄清也不能丢失。
            # 放宽或撤销仍须等待整份修订通过接受阶段。
            from src.execution_boundaries import apply_semantic_boundaries
            apply_semantic_boundaries(context, contract, restrict_only=True)
        context.record('request_semantics_extracted', contract=contract, basis_sha256=basis)
        return contract
    raise AssertionError('Unreachable extraction state')


def prepare_goal(context, state, goal, history):
    """独立提取后双向比较；不在比较失败时把规划目标反向写进需求。"""
    if state.get('schema_version', 7) < 8:
        return None  # 旧检查点沿用其已冻结协议，不伪造独立提取记录。
    prior = state.get('previous_semantics') or context.snapshot.get('request_semantics') or (context.snapshot.get('conversation') or {}).get('request_semantics')
    contract = extract_contract(context, state['request'], history, previous=prior,
                                change_text=state.get('change_text'))
    mapping = compare_goal(contract, goal, state['request'], history)
    impact = affected_outcomes(prior, contract) if prior else dict(
        preserved=[], changed=[], added=sorted(mapping), retracted=[])
    return dict(contract=contract, objective_mapping=mapping, impact=impact)


def accept_goal(context, state, prepared):
    if prepared is None:  # 旧检查点仍使用原有目标审查与编译边界。
        return
    from src.execution_boundaries import apply_semantic_boundaries
    state['request_semantics'] = deepcopy(prepared['contract'])
    context.snapshot['request_semantics'] = deepcopy(prepared['contract'])
    state['semantic_objectives'] = deepcopy(prepared['objective_mapping'])
    previous = state.get('previous_objectives') or {}
    prior_rows = {row['objective_id']: row for row in state.get('previous_ledger', [])}
    preserved = set(prepared['impact']['preserved'])
    changed = set(prepared['impact']['changed'])
    for requirement_id, objective_id in prepared['objective_mapping'].items():
        row = next(item for item in state['objective_ledger'] if item['objective_id'] == objective_id)
        old = prior_rows.get(previous.get(requirement_id))
        if requirement_id in preserved and old:
            # 保留候选身份及来源线索，不能把旧 goal/binding revision 的收据标成当前已核验。
            row.update(candidate_product_ids=deepcopy(old.get('selected_product_ids', [])),
                       candidate_evidence_refs=deepcopy(old.get('evidence_refs', [])))
        elif requirement_id in changed:
            row.update(status='INVALIDATED', failures=['USER_REQUIREMENT_REVISED'])
    state['semantic_revision'] = dict(**prepared['impact'], previous_sha256=prepared['contract']['previous_sha256'],
                                    current_sha256=prepared['contract']['sha256'])
    apply_semantic_boundaries(context, prepared['contract'])
    context.record('request_semantics_compiled', semantics_sha256=prepared['contract']['sha256'],
                   objective_mapping=prepared['objective_mapping'], revision=state['semantic_revision'])


def validate_accepted(state, history):
    action = ((state.get('pending') or {}).get('decision') or {}).get('action')
    if (state.get('schema_version', 7) >= 8 and state.get('goal') and action not in {'clarify', 'finish'}
            and not state.get('request_semantics')):
        raise AgentError('REQUEST_SEMANTICS_MISSING', '当前检查点缺少已接受的独立需求契约', retryable=False)
    if state.get('request_semantics') and state.get('goal'):
        mapping = compare_goal(state['request_semantics'], state['goal'], state['request'], history)
        if mapping != state.get('semantic_objectives'):
            raise AgentError('REQUEST_SEMANTICS_MISMATCH', '恢复后的需求目标映射不一致', retryable=False)


def revise_without_execution(context, state, history):
    """可撤销全部任务或仅修订限制；新增业务要求不能被社交回答掩盖。"""
    prior = state.get('previous_semantics') or context.snapshot.get('request_semantics') or (context.snapshot.get('conversation') or {}).get('request_semantics')
    if state.get('schema_version', 7) < 8:
        return
    contract = extract_contract(context, state['request'], history, previous=prior, change_text=state.get('change_text'))
    if any(row.kind == 'UNKNOWN' for row in validate_contract(contract)):
        raise AgentError('REQUEST_SEMANTICS_UNCERTAIN', '修订仍包含未明确的要求，需要澄清后再执行', retryable=False)
    impact = affected_outcomes(prior, contract) if prior else dict(
        preserved=[], changed=[], retracted=[],
        added=[row.requirement_id for row in validate_contract(contract) if row.kind == 'OUTCOME'])
    if impact['added'] or impact['changed']:
        raise AgentError('REQUEST_SEMANTICS_MISMATCH', '修改后的业务要求需要形成目标，不能只回复收到', retryable=False)
    from src.execution_boundaries import apply_semantic_boundaries
    context.snapshot['request_semantics'] = deepcopy(contract)
    state['request_semantics'] = deepcopy(contract)
    state['semantic_revision'] = dict(**impact, previous_sha256=contract['previous_sha256'], current_sha256=contract['sha256'])
    apply_semantic_boundaries(context, contract)
    context.record('request_semantics_revised', contract=contract, revision=state['semantic_revision'], execution_started=False)
