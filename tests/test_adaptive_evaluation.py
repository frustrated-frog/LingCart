import json
import pytest
from langchain_core.messages import AIMessage
from eval import run_adaptive_eval as specialized
from eval import run_live_eval as live


def test_adaptive_suite_is_separate_and_dry_run_never_loads_application(monkeypatch, capsys):
    monkeypatch.setattr(live.offline, '_runtime', lambda: (_ for _ in ()).throw(AssertionError('不应导入应用')))
    assert specialized.main(['--dry-run']) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['suite'] == 'adaptive' and output['execution_mode'] == 'adaptive'
    assert output['planned_samples'] == 60
    with pytest.raises(ValueError):
        specialized.main(['--execution-mode', 'auto'])


def test_first_context_experiment_dry_run_is_bound_to_holdout_protocol(monkeypatch, capsys):
    monkeypatch.setattr(live.offline, '_runtime', lambda: (_ for _ in ()).throw(AssertionError('不能导入应用')))
    spec = live.ROOT / 'research/experiments/objective-context-v1/experiment-spec.json'

    assert specialized.main(['--dry-run', '--split', 'test', '--repeats', '3',
                             '--harness-profile', 'adaptive-objective-context-v1',
                             '--experiment-spec', str(spec)]) == 0
    output = capsys.readouterr().out
    assert 'objective-context-v1' in output
    assert '未调用模型' in output


def test_adaptive_dataset_contains_no_model_scripts_and_covers_regressions():
    dataset = live.load_dataset(specialized.DATASET)
    assert len(dataset['cases']) == 20
    assert all('model' not in c for c in dataset['cases'])
    assert {'coverage','negation_scope','price_change','stock_change','long_history','steering','unknown_policy'} <= {c['category'] for c in dataset['cases']}


def test_adaptive_suite_refuses_to_silently_use_auto():
    with pytest.raises(ValueError, match='自主专项'):
        live.evaluate(live.load_dataset(specialized.DATASET), execution_mode='auto')


def test_specialized_entry_cannot_relabel_the_support_suite():
    with pytest.raises(ValueError, match='通用任务集'):
        specialized.main(['--dataset=' + str(live.DATASET), '--dry-run'])


def test_unknown_configuration_does_not_create_adaptive_baseline():
    report = live.evaluate(live.load_dataset(specialized.DATASET), execution_mode='adaptive', readiness={'configured':False})
    assert report['status'] == 'NOT_RUN' and not report['samples']
    assert report['manifest']['suite'] == 'adaptive'
    assert live.evaluate_gate(report)


def test_adaptive_judge_rejects_auto_results_and_missing_goals():
    from eval.adaptive_checks import judge
    case = {'adaptive_expect':{'objectives':[{'kind':'get_order_status','order_id':1001}], 'require_goal_coverage':True}}
    failures = judge(case, {'execution_mode':'auto','status':'WAITING_CLARIFICATION'}, {'products':{}})
    assert any('自主决策' in e for e in failures)
    assert any('没有覆盖' in e for e in failures)


def test_evaluation_steering_injection_is_once_per_durable_run(monkeypatch):
    runtime = live.offline._runtime()
    import src.adaptive_decision as decisions
    import src.request_audit as auditing
    from src.adaptive import run_adaptive
    class Model:
        calls = 0
        def invoke(self, messages):
            self.calls += 1
            payload = json.loads(messages[-1].content)
            if payload['goal'] is None:
                quote = '200元' if '200元' in payload['request'] else '500元'
                return AIMessage(content=json.dumps({'action':'tool','reason':'搜索候选',
                    'goal':{'summary':'选耳机','objectives':[{'id':'shopping','description':'推荐耳机','source_quote':'耳机','kind':'recommend'}],
                            'hard_constraints':[{'description':'预算','source_quote':quote,'field':'price','op':'lte','value':200 if '200元' in quote else 500}]},
                    'tool':'search_products','args':{'category':'耳机','limit':2}}),
                    usage_metadata={'input_tokens':100,'output_tokens':80,'total_tokens':180})
            product = payload['observations'][-1]['result']['products'][0]
            return AIMessage(content=json.dumps({'action':'recommend','reason':'满足预算','objective_id':'shopping',
                'product_id':product['product_id'],'evidence_refs':[payload['observations'][-1]['step_id']]}),
                usage_metadata={'input_tokens':100,'output_tokens':80,'total_tokens':180})
    class Auditor:
        def invoke(self, messages):
            payload=json.loads(messages[-1].content)
            goal=payload['goal']
            return AIMessage(content=json.dumps({'verdict':'APPROVED','issues':[], 'requirements':[
                {'source_quote':goal['hard_constraints'][0]['source_quote'],'interpretation':'预算及耳机',
                 'objective_ids':['shopping'],'constraint_refs':['global:0'],'status':'COVERED'}]}),
                usage_metadata={'input_tokens':100,'output_tokens':80,'total_tokens':180})
    model = Model()
    monkeypatch.setattr(decisions,'get_llm',lambda **_:model)
    monkeypatch.setattr(auditing,'get_llm',lambda **_:Auditor())
    dataset = live.load_dataset(specialized.DATASET)
    dataset['cases'] = [dataset['cases'][12]]
    report = live.evaluate(dataset, repeats=1, execution_mode='adaptive', runner=run_adaptive, readiness={'configured':False})
    assert report['execution_source']=='test_harness' and not report['live_verified']
    sample=report['samples'][0]
    assert sample['passed'], sample['turns'][0]['failures']
    result=sample['turns'][0]['result']
    assert result['plan']['goal_revision']==1
    assert sum(e['type']=='control_applied' for e in result['events'])==1
    assert report['adaptive_verified_samples']==1
    assert live.evaluate_gate(report)
