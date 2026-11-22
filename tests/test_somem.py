from copy import deepcopy
import json
from pathlib import Path
import tempfile
import pytest
from langchain_core.messages import AIMessage
from src.somem import Memory, Source, Organization, organize, read, views, search
from src.somem_context import SomemContextEngine, SomemContextMechanism
from src.context_engine import ContextEngine
from src.context_budget import _text_length
from src.procedural_graph import canonical
from src.runtime import RunRuntime, BudgetLimits, runtime_scope
from src.task_plan import TaskPlan, TaskStep


def empty():
    return Memory(memory_id='test-memory', version=0, origin='empty')


def sources():
    return (Source(source_id='first', text='办公室里需要安静的键盘。', event_time='2026-10'),
            Source(source_id='second', text='现在改成带去出差，优先轻便。', event_time='2026-11-20'))


def proposal():
    return Organization.model_validate_json(canonical(dict(paragraphs=[
        dict(paragraph_id='office', topic='办公设备', statements=[dict(text='办公室键盘要安静。', event_time='2026-10',
            citations=[dict(source_id='first', quote='办公室里需要安静的键盘。')])], links=['travel']),
        dict(paragraph_id='travel', topic='出差设备', statements=[dict(text='出差设备优先轻便。', event_time='2026-11-20',
            citations=[dict(source_id='second', quote='现在改成带去出差，优先轻便。')])])], core=['travel'])))


def memory():
    return organize(empty(), sources(), proposal(), origin='fixture')


def state():
    return dict(request='推荐键盘', goal=None, goal_revision=0, next_steps=None, feedback=None,
                working_memory=None, objective_ledger=[])


def test_views_preserve_precision_and_creation_order_and_rebuild_links():
    initial = memory()
    assert views(initial)['graph']['travel']['backlinks'] == ['office']
    assert set(views(initial)['timeline']) == {'2026-10', '2026-11-20'}
    updated = proposal().model_copy(update={'paragraphs': tuple(reversed(proposal().paragraphs))})
    result = organize(initial, (), updated, origin='fixture')
    assert views(result)['recent'] == ['travel', 'office']  # 重排不改变创建时间
    assert result.parent_sha256 == initial.fingerprint
    assert initial.sources == result.sources
    assert read(result, 'somem:graph/travel')['items'][0]['backlinks'] == ['office']
    assert search(result, '出差')[0]['score'] > 0
    assert read(result, 'somem:search/不存在的词')['items'] == []


@pytest.mark.parametrize('bad', ['quote', 'date', 'link', 'core', 'duplicate'])
def test_invalid_edits_are_atomic(bad):
    original = memory()
    data = proposal().model_dump(mode='json')
    if bad == 'quote': data['paragraphs'][0]['statements'][0]['citations'][0]['quote'] = '不存在的原文'
    if bad == 'date': data['paragraphs'][0]['statements'][0]['event_time'] = '2026-10-01'
    if bad == 'link': data['paragraphs'][0]['links'] = ['absent']
    if bad == 'core': data['core'] = ['absent']
    if bad == 'duplicate': data['paragraphs'].append(data['paragraphs'][0])
    before = original.model_dump_json()
    with pytest.raises(ValueError):
        organize(original, (), Organization.model_validate_json(canonical(data)), origin='fixture')
    assert original.model_dump_json() == before


def test_sources_cannot_be_rewritten_and_deleted_paragraph_ids_cannot_be_recycled():
    original = memory()
    with pytest.raises(ValueError, match='不可改写'):
        organize(original, (sources()[0].model_copy(update={'text': '伪造原文'}),), proposal(), origin='fixture')
    deleted = organize(original, (), Organization(), origin='fixture')
    assert len(deleted.sources) == 2 and views(deleted)['recent'] == []
    with pytest.raises(ValueError, match='重新分配'):
        organize(deleted, (), proposal(), origin='fixture')


@pytest.mark.parametrize('value', ['2026-02-30', '2026-13', '20261120', 'yesterday'])
def test_dates_are_calendar_valid(value):
    with pytest.raises(ValueError): Source(source_id='bad', text='日期', event_time=value)


def test_read_is_paged_scoped_and_cannot_be_mistaken_for_business_evidence():
    current = organize(empty(), (Source(source_id='long', text='原文'*1500),), Organization(), origin='fixture')
    first = read(current, 'somem:source/long')
    assert first['memory_only'] and first['refreshed'] is False
    assert first['next_offset'] == 1 and first['items'][0]['char_end'] == 1000
    assert 'products' not in first and 'order_id' not in first
    assert read(current, 'somem:source/long', offset=2)['next_offset'] is None
    for reference, offset, limit in [('somem:source/../../secret',0,1), ('somem:source/long',4,1), ('somem:core',0,4)]:
        with pytest.raises(Exception) as error: read(current, reference, offset=offset, limit=limit)
        assert error.value.code in {'SOMEM_NOT_FOUND', 'SOMEM_INVALID_PAGE'}


def test_context_is_budgeted_and_keeps_auditor_and_current_goal_separate():
    engine = SomemContextEngine(ContextEngine('推荐键盘'), memory(), max_chars=1500)
    messages, manifest = engine.decision_messages('规则', state(), TaskPlan([], []), [], max_chars=4000)
    assert sum(_text_length(m) for m in messages) <= 4000
    assert manifest['somem']['status'] == 'EXPOSED'
    assert json.loads(messages[-1].content)['request'] == '推荐键盘'
    assert 'somem:search/' in messages[0].content
    _, low = engine.decision_messages('规则', state(), TaskPlan([], []), [], max_chars=700)
    assert low['somem']['status'] == 'BUDGET_OMITTED'
    messages, audit = engine.decision_messages('独立核验', {**state(), 'feedback': {'audit_phase':'goal'}}, TaskPlan([], []), [], max_chars=4000)
    assert len(messages) == 2 and audit['somem']['status'] == 'AUDIT_BYPASS'
    assert '办公室' not in canonical([m.content for m in messages])


def test_inspection_records_identity_and_preserves_regular_archive_reader():
    engine = SomemContextEngine(ContextEngine('推荐键盘'), memory())
    context = RunRuntime('somem-test', BudgetLimits())
    plan = TaskPlan([], [TaskStep('step-1','shopping','search_products',{},status='SUCCEEDED',result={'products':[]})])
    with runtime_scope(context):
        assert engine.read_evidence(plan,'somem:source/first')['items'][0]['text'] == sources()[0].text
        assert engine.read_evidence(plan,'step-1')['products'] == []
        with pytest.raises(Exception): engine.read_evidence(plan,'somem:source/missing')
    assert [e['status'] for e in context.events if e['type']=='somem_read'] == ['SUCCEEDED','FAILED']


def test_mechanism_rejects_escaping_paths_changes_and_product_use():
    from src.harness import get_product_runner
    from src.harness.mechanisms import ROOT
    from src.errors import AgentError
    with pytest.raises(AgentError): get_product_runner('adaptive-somem-context-v1')
    for path in ['../secret.json', 'research/playbooks/ace-empty-v1.json', '/tmp/secret.json']:
        with pytest.raises(ValueError): SomemContextMechanism().configure({'memory_path':path,'max_chars':12000})
    with tempfile.TemporaryDirectory(dir=ROOT/'research/memories') as folder:
        path = Path(folder)/'memory.json'
        path.write_text(memory().model_dump_json())
        configured = SomemContextMechanism().configure({'memory_path':str(path.relative_to(ROOT)), 'max_chars':12000})
        assert configured.provenance()['configuration']['memory_sha256'] == memory().fingerprint
        path.write_text(empty().model_dump_json())
        with pytest.raises(ValueError, match='发生变化'): configured.build_context('查询')


def test_cli_accept_reject_and_dry_run_never_call_models(tmp_path, monkeypatch):
    from eval.run_somem_organize import main
    monkeypatch.setattr('src.somem_learning.get_llm', lambda **_: pytest.fail('离线不能调用模型'))
    source = tmp_path/'sources.json'
    source.write_text(canonical({'split':'dev','sources':[s.model_dump(mode='json') for s in sources()]}))
    proposed = tmp_path/'proposal.json'; proposed.write_text(proposal().model_dump_json())
    args = ['--sources',str(source)]
    assert main(args) == 0
    output=tmp_path/'accepted'
    assert main([*args,'--proposal',str(proposed),'--output-dir',str(output)]) == 0
    assert json.loads((output/'report.json').read_text())['execution_source'] == 'test_harness'
    assert Memory.model_validate_json((output/'memory.json').read_text()).organization.core == ('travel',)
    with pytest.raises(FileExistsError): main([*args,'--proposal',str(proposed),'--output-dir',str(output)])
    proposed.write_text('{} invalid')
    rejected=tmp_path/'rejected'
    assert main([*args,'--proposal',str(proposed),'--output-dir',str(rejected)]) == 1
    assert not (rejected/'memory.json').exists() and (rejected/'writer.json').exists()
    source.write_text(canonical({'split':'test','sources':[s.model_dump(mode='json') for s in sources()]}))
    with pytest.raises(ValueError): main(args)


def test_model_organizer_uses_gateway_and_preserves_tool_call_rejection(monkeypatch):
    from src.somem_learning import ModelOrganizer
    seen=[]
    class Model:
        def invoke(self,messages):
            seen.append(messages)
            return AIMessage(content=proposal().model_dump_json())
    monkeypatch.setattr('src.somem_learning.get_llm',lambda **_:Model())
    result=ModelOrganizer().propose(empty(),sources())
    assert result['error'] is None and result['writer_sha256']
    assert 'Schema' in seen[0][0].content
    assert json.loads(seen[0][1].content)['incoming'][0]['text'] == sources()[0].text


def test_durable_agent_reads_memory_and_resume_keeps_profile_identity(tmp_path, monkeypatch):
    from src.backend.db import get_engine, get_session_factory
    from src.backend.seed import seed
    from src import tools
    from src.harness.registry import build_registry
    from src.harness.profile import HarnessProfile
    from src.harness.mechanisms import ROOT
    from src.persistence import RunStore
    from src.durable import run_conversation, resume_run
    from tests.test_adaptive import model, goal, clarification
    database = get_engine(f'sqlite:///{tmp_path / "runtime.db"}')
    tools.configure_db(database)
    with get_session_factory(database)() as session: seed(session)
    captured = {}
    class CrashAfterRead(RunStore):
        def save_checkpoint(self, run_id, token, snapshot, event=None):
            result = super().save_checkpoint(run_id, token, snapshot, event)
            if event and event.get('type') == 'step_completed':
                captured['run_id'] = run_id
                raise SystemExit('simulated process exit')
            return result
    with tempfile.TemporaryDirectory(dir=ROOT/'research/memories') as folder:
        path=Path(folder)/'memory.json'; path.write_text(memory().model_dump_json())
        profile=json.loads((ROOT/'configs/harness/adaptive-somem-context-v1.json').read_text())
        profile['profile_id']='test-somem-agent'
        next(m for m in profile['mechanisms'] if m['slot']=='context')['configuration']['memory_path']=str(path.relative_to(ROOT))
        registry=build_registry(); registry.register_profile(HarnessProfile.model_validate_json(canonical(profile)))
        runner=registry.resolve('test-somem-agent')
        script=model(monkeypatch, dict(action='inspect', goal=goal(), reason='读取历史记录', evidence_ref='somem:source/first'), clarification())
        with pytest.raises(SystemExit):
            result = run_conversation('推荐预算500的耳机','somem-resume',runner=runner,store=CrashAfterRead())
        result=resume_run('somem-resume',captured['run_id'],runner=runner)
        events=[e for e in result['events'] if e['type']=='somem_read']
        assert len(events)==1 and events[0]['memory_sha256']==memory().fingerprint
        assert any(o.get('result',{}).get('memory_only') for o in script.inputs[-1]['observations'])
        assert result['status']=='WAITING_CLARIFICATION'
    database.dispose()


def test_native_inspect_can_search_without_prior_steps_and_completion_stays_scoped():
    from src.native_protocol import decision_functions
    from tests.test_adaptive import goal
    runtime = RunRuntime('native-memory', BudgetLimits())
    runtime.context_engine = SomemContextEngine(ContextEngine('推荐耳机'), memory())
    current = {**state(), 'goal': goal()}
    with runtime_scope(runtime):
        declarations = decision_functions(current, TaskPlan([], []), [])
    inspect = next(t['function'] for t in declarations if t['function']['name']=='inspect')
    reference = inspect['parameters']['properties']['evidence_ref']
    assert 'somem:source/first' in reference['anyOf'][0]['enum']
    assert reference['anyOf'][1]['pattern'].startswith('^somem:search/')
    assert not any(t['function']['name']=='complete' for t in declarations)
    assert not any(t['function']['name']=='inspect' for t in decision_functions(current, TaskPlan([], []), []))


def test_archived_memory_is_not_acceptable_business_completion_evidence():
    from src.adaptive_goals import _claim_error
    step=TaskStep('step-1','shopping','read_evidence',{'reference':'somem:source/first'},status='SUCCEEDED',result=read(memory(),'somem:source/first'))
    objective=dict(id='order',kind='get_order_status',order_id=1001,question=None)
    assert _claim_error({'read_evidence_refs':['step-1']},objective,step,{'step-1':step}) == 'EVIDENCE_NOT_SUCCESSFUL'


def test_somem_contract_changes_only_context_and_has_no_promoted_claim():
    from src.harness.registry import get_runner
    from eval.experiment_contract import ExperimentSpec
    spec=ExperimentSpec.from_path(Path('research/experiments/somem-context-v1/experiment-spec.json'))
    left=get_runner(spec.baseline_profile).provenance(); right=get_runner(spec.candidate_profile).provenance()
    assert left['engine']==right['engine']
    assert [m for m in left['mechanisms'] if m['slot']!='context']==[m for m in right['mechanisms'] if m['slot']!='context']
    assert right['research']['status']=='candidate'
