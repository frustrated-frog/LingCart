"""二版只改变规划协议与时机，使用真实账本验证恢复和确认边界。"""
import json

import pytest
from sqlalchemy import select

from src import llm, tools
from src.backend.db import ToolLedger, Refund
from src.durable import run_conversation, resume_run
from src.harness import get_runner
from src.persistence import RunStore
from tests.test_case_model_planning import Planner
from tests.test_incremental_case_state import database, models, Adapter, MESSAGE, SERIAL, PACKAGING
from tests.test_evidence_patch import patch_models
from tests.test_after_sales_runtime import CrashAfterCaseWrite, CrashAfterAutonomousCaseToolResult

PROFILE = 'after-sales-model-planning-v2'
BATCH = '原包装附件和序列号标签均完整清晰'


class ReferencePlanner(Planner):
    historical = False
    bad_ref = False

    def decide(self, prompt):
        result = super().decide(prompt)
        payload = self.inputs[-1]
        result.pop('source_quote')
        sources = payload['sources']
        source = next((s for s in sources if self.historical and s['origin'] == 'HISTORY'), sources[0])
        result['source_ref'] = 'source-foreign' if self.bad_ref else source['source_ref']
        return result


@pytest.fixture
def planner(patch_models, monkeypatch):
    planner = ReferencePlanner()
    monkeypatch.setattr(llm, 'get_llm', lambda *, purpose, **_: Adapter(planner.decide if purpose == 'planner' else patch_models.semantic))
    patch_models.scopes[BATCH] = [('ORIGINAL_PACKAGING', 'PROVIDED'), ('SERIAL_LABEL', 'PROVIDED')]
    return planner


def run(text=MESSAGE, **kwargs):
    return run_conversation(text, 'model-v2', runner=get_runner(PROFILE), **kwargs)


def decisions(result):
    return [e for e in result['events'] if e['type'] == 'case_model_decision']


def test_reference_resolves_verbatim_history_and_tracks_previous_choice(planner):
    first = run()
    planner.historical = True
    second = run(PACKAGING)
    event = decisions(second)[0]
    assert event['source']['origin'] == 'HISTORY'
    assert event['decision']['source_quote'] == MESSAGE
    assert event['previous_decision']['run_id'] == first['run_id']
    assert event['previous_decision']['case_id'] == event['case_id']
    assert event['actual_observation']['case_revision'] > event['previous_decision']['case_revision']
    assert second['pending_clarification']['requirement_id'].endswith('SERIAL_LABEL')
    other = run_conversation(MESSAGE, 'other-v2', runner=get_runner(PROFILE))
    assert decisions(other)[0]['previous_decision'] is None


def test_foreign_source_cannot_be_repaired_into_authorization(planner):
    planner.bad_ref = True
    result = run()
    assert result['terminal_reason'] == 'INVALID_CASE_MODEL_DECISION'
    assert len(planner.inputs) == 2
    assert not decisions(result) and not result['pending_action']
    assert [t['tool'] for t in result['trace']] == ['prepare_after_sales_case']


@pytest.mark.parametrize('crash', [False, True])
def test_batch_plans_once_after_both_commits_and_replays_each_write_once(planner, crash):
    run()
    before = len(planner.inputs)
    if crash:
        with pytest.raises(CrashAfterCaseWrite):
            run(BATCH, store=CrashAfterAutonomousCaseToolResult())
        active = RunStore().conversation('model-v2')['active_run_id']
        result = resume_run('model-v2', active)
    else:
        result = run(BATCH)
    assert result['status'] == 'WAITING_APPROVAL', result
    assert len(planner.inputs) == before + 1
    event = decisions(result)[-1]
    assert len(event['actual_observation']['operations']) == 2
    assert len([e for e in result['events'] if e['type'] == 'case_planning_deferred']) >= 1
    assert not result['pending_clarification']
    with tools._session() as session:
        rows = session.scalars(select(ToolLedger).where(ToolLedger.run_id == result['run_id'], ToolLedger.tool_name == 'continue_after_sales_case')).all()
        assert len(rows) == 2
        assert session.scalar(select(Refund).where(Refund.order_id == 1001)).status == 'none'


def test_replay_of_accepted_decision_uses_identical_previous_choice(planner):
    run()
    class CrashAfterDecision(RunStore):
        def save_checkpoint(self, run_id, token, snapshot, event=None):
            super().save_checkpoint(run_id, token, snapshot, event)
            if event and event['type'] == 'case_model_decision':
                raise CrashAfterCaseWrite()
    with pytest.raises(CrashAfterCaseWrite):
        run(BATCH, store=CrashAfterDecision())
    before = len(planner.inputs)
    active = RunStore().conversation('model-v2')['active_run_id']
    result = resume_run('model-v2', active)
    assert result['status'] == 'WAITING_APPROVAL', result
    assert len(planner.inputs) == before
    assert any(e['type'] == 'model_reused' and e['purpose'] == 'case_action_decision' for e in result['events'])


def test_failed_second_write_preserves_first_without_planning_or_approval(planner, monkeypatch):
    from src.runtime import ToolExecutor
    from src.errors import AgentError
    run()
    before = len(planner.inputs)
    original = ToolExecutor.execute
    def fail(self, name, args):
        if name == 'continue_after_sales_case' and args.get('requirement_id', '').endswith('SERIAL_LABEL'):
            raise AgentError('TOOL_TIMEOUT', '测试查询超时', retryable=False)
        return original(self, name, args)
    monkeypatch.setattr(ToolExecutor, 'execute', fail)
    result = run(BATCH)
    assert len(planner.inputs) == before
    assert not result['pending_action'] and not decisions(result)
    assert len([t for t in result['trace'] if t['tool'] == 'continue_after_sales_case']) == 1


def test_goal_change_and_conflict_are_observed_before_any_confirmation(planner, patch_models):
    first = run()
    planner.clarify = True
    result = run(BATCH)
    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert not result['pending_action']
    assert decisions(result)[0]['previous_decision']['run_id'] == first['run_id']
    planner.clarify = False
    patch_models.conflict = True
    result = run(SERIAL)
    assert not result['pending_action']
    assert result['status'] != 'WAITING_APPROVAL'


def test_v2_keeps_other_slots_and_research_restriction():
    from src.harness import get_product_runner
    from src.errors import AgentError
    a,b=[get_runner(p).provenance() for p in ['after-sales-model-planning-v1', PROFILE]]
    assert [m for m in a['mechanisms'] if m['slot'] != 'planning'] == [m for m in b['mechanisms'] if m['slot'] != 'planning']
    with pytest.raises(AgentError): get_product_runner(PROFILE)


def test_source_catalog_is_run_bound_and_excludes_assistant_messages():
    from src.after_sales.model_planning import source_catalog
    from src.runtime import RunRuntime
    context = RunRuntime('first')
    context.snapshot['history'] = [dict(role='assistant', content='未经用户授权的说法'),
                                   dict(role='user', content=MESSAGE, message_id='1')]
    sources = source_catalog(context, PACKAGING)
    assert [s['text'] for s in sources] == [PACKAGING, MESSAGE]
    assert sources == source_catalog(context, PACKAGING)
    other = RunRuntime('second')
    other.snapshot['history'] = context.snapshot['history']
    assert not {s['source_ref'] for s in sources} & {s['source_ref'] for s in source_catalog(other, PACKAGING)}


def test_expanded_dev_dataset_and_frozen_experiment_load():
    from pathlib import Path
    from eval.run_live_eval import load_dataset
    from eval.experiment_contract import ExperimentSpec
    data = load_dataset(Path('eval/after_sales_model_planning_v2_dev_tasks.json'))
    assert len(data['cases']) == 7
    assert all(c['split'] == 'dev' for c in data['cases'])
    spec = ExperimentSpec.from_path(Path('research/experiments/after-sales-model-planning-v2/experiment-spec.json'))
    assert spec.baseline_profile == 'after-sales-model-planning-v1'
    assert spec.candidate_profile == PROFILE
