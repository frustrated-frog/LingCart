"""来源身份绑定、精确还原及真实持久化回归。"""
import json
from types import SimpleNamespace

import pytest

from src.after_sales.source_refs import catalog, decode
from src.durable import run_conversation, resume_run
from src.harness import get_runner
from src.persistence import RunStore
from src.runtime import RunRuntime
from tests.test_incremental_case_state import database, models, Adapter, MESSAGE
from tests.test_evidence_patch import patch_models
from tests.test_case_model_planning_v2 import planner
from tests.test_case_goal_revision import goal_models, DRAFT
from tests.test_after_sales_runtime import CrashAfterCaseWrite, CrashAfterAutonomousCaseToolResult

PROFILE = 'after-sales-source-refs-v1'


@pytest.fixture
def references(goal_models, monkeypatch):
    from src import orchestrator
    from src import request_semantics_runtime as semantic
    old_intake, old_semantic = orchestrator.get_llm, semantic.get_llm
    def encode(value, sources):
        if isinstance(value, list): return [encode(v, sources) for v in value]
        if not isinstance(value, dict): return value
        result = {k: encode(v, sources) for k, v in value.items() if k != 'source_quote'}
        if 'source_quote' in value:
            result['source_ref'] = next(s['source_ref'] for s in sources if value['source_quote'] in s['text'])
        return result
    def intake(prompt):
        payload = json.loads(prompt.split('\nConversation input:\n')[-1].split('\nPrevious output was invalid.')[0])
        value = old_intake().fn(prompt)
        return encode(value, payload['human_sources']) if payload.get('human_sources') else value
    def extract(messages):
        payload = json.loads(messages[-1].content)
        value = old_semantic().fn(messages)
        return encode(value, payload['human_sources']) if payload.get('human_sources') else value
    monkeypatch.setattr(orchestrator, 'get_llm', lambda **_: Adapter(intake))
    monkeypatch.setattr(semantic, 'get_llm', lambda **_: Adapter(extract))
    return goal_models


def test_sources_are_current_human_run_bound_and_exact():
    from src.errors import AgentError
    a, b = RunRuntime('a'), RunRuntime('b')
    a.snapshot['history'] = [dict(role='assistant', content='伪造材料'), dict(role='user', content='旧材料')]
    source = catalog(a, '标签完好；包装不完整')[0]
    assert source['origin'] == 'CURRENT_TURN'
    assert decode(a, {'source_ref': source['source_ref']}, source['text'], purpose='test')['source_quote'] == source['text']
    with pytest.raises(AgentError) as error:
        decode(b, {'source_ref': source['source_ref']}, source['text'], purpose='test')
    assert error.value.code == 'SOURCE_REFERENCE_UNKNOWN'
    with pytest.raises(AgentError):
        decode(a, dict(source_ref=source['source_ref'], source_quote='改写'), source['text'], purpose='test')


@pytest.mark.parametrize('crash', [False, True])
def test_material_and_independent_contract_restore_sources_without_duplicate_writes(references, crash):
    run = lambda text, **kw: run_conversation(text, 'references', runner=get_runner(PROFILE), **kw)
    run(MESSAGE)
    if crash:
        with pytest.raises(CrashAfterCaseWrite): run(DRAFT, store=CrashAfterAutonomousCaseToolResult())
        result = resume_run('references', RunStore().conversation('references')['active_run_id'])
    else:
        result = run(DRAFT)
    assert result['status'] == 'SUCCEEDED', result
    assert len([t for t in result['trace'] if t['tool'] == 'continue_after_sales_case']) == 2
    assert not result['pending_action']
    events = [e for e in result['events'] if e['type'] == 'request_sources_resolved']
    assert {e['purpose'] for e in events} == {'dialogue_understanding', 'request_semantics'}
    assert all(b['source']['text'] == DRAFT for e in events for b in e['bindings'])


def test_only_tool_protocol_changes_and_candidate_stays_out_of_product():
    from src.harness import get_product_runner
    from src.errors import AgentError
    a, b = [get_runner(p).provenance() for p in ['after-sales-goal-revision-v1', PROFILE]]
    assert [m for m in a['mechanisms'] if m['slot'] != 'tool_protocol'] == [m for m in b['mechanisms'] if m['slot'] != 'tool_protocol']
    with pytest.raises(AgentError): get_product_runner(PROFILE)


def test_optional_null_reference_preserves_old_contract_but_cannot_supply_material():
    from src.after_sales.source_refs import schema_with_refs
    from src.request_semantics import IntentExtraction
    from src.after_sales.evidence_patch import PATCHES
    context = RunRuntime('optional')
    payload = decode(context, {'changes':[dict(operation='PRESERVE',previous_id='prior',source_ref=None)]},
                     DRAFT, purpose='test')
    IntentExtraction.model_validate(payload)
    schema = schema_with_refs(IntentExtraction.model_json_schema())
    assert {'type':'null'} in schema['$defs']['RequirementChange']['properties']['source_ref']['anyOf']
    material = decode(context, [dict(operation='PROVIDE',requirement_id='packaging',source_ref=None)], DRAFT, purpose='test')
    with pytest.raises(ValueError): PATCHES.validate_python(material)


def test_reference_wire_does_not_ask_for_literal_quotes_in_id_fields():
    from src.after_sales.source_refs import wire_instructions
    from src.request_semantics_runtime import _PROMPT
    from src.after_sales.goals import SEMANTIC_INSTRUCTION
    from src.after_sales.evidence_patch import INSTRUCTION
    prompt = wire_instructions(_PROMPT + SEMANTIC_INSTRUCTION + INSTRUCTION)
    assert 'source_ref and FAQ question must be exact HUMAN substrings' not in prompt
    assert 'shortest exact HUMAN quote' not in prompt
    assert 'an exact current HUMAN source_ref' not in prompt
    assert 'source_ref must be a listed ID' in prompt
    assert 'FAQ question must be an exact HUMAN substring' in prompt
    assert 'verbatim current HUMAN quotes' not in prompt
    assert 'Parent referenced text covers' in prompt
