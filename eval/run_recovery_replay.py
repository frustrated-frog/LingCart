"""从同一失败检查点比较原始、配对控制与有界搜索。默认校验，--live 才调用模型。"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from eval.run_live_eval import environment_manifest, load_dataset
from src.adaptive_decision import StrictModel
from src.procedural_graph import fingerprint
from src.pg_learning import write_artifact
from src.recovery_usage import recovery_usage

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ('adaptive-v1', 'adaptive-recovery-control-v1', 'adaptive-recovery-search-v1')
CAPTURE_PROFILE = 'adaptive-recovery-control-v1'
ReplayProfile = Literal['adaptive-v1', 'adaptive-recovery-control-v1', 'adaptive-recovery-search-v1',
                        'adaptive-recovery-eager-v2', 'adaptive-recovery-search-v2']


class ReplaySpec(StrictModel):
    schema_version: Literal[1] = 1
    dataset: str = Field(pattern=r'^eval/[a-z][a-z0-9_]*_tasks\.json$')
    split: Literal['dev', 'test']
    case_ids: list[str] = Field(min_length=1, max_length=6)
    repeats: int = Field(2, ge=1, le=5)
    max_model_calls: int = Field(16, ge=6, le=16)
    max_seconds: int = Field(90, ge=30, le=180)
    # 顺序为参考组、配对控制、候选；旧 spec 默认保留 v1 的三组身份。
    profiles: list[ReplayProfile] = Field(default_factory=lambda: list(PROFILES), min_length=3, max_length=3)

    @model_validator(mode='after')
    def unique(self):
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError('任务编号不能重复')
        if len(set(self.profiles)) != len(self.profiles):
            raise ValueError('回放策略不能重复')
        return self

    def cases(self, dataset):
        cases = {case['id']: case for case in dataset['cases']}
        if dataset.get('suite') != 'adaptive' or any(i not in cases or cases[i]['split'] != self.split for i in self.case_ids):
            raise ValueError('数据集分组与回放合同不一致')
        selected = [cases[i] for i in self.case_ids]
        if any(len(c['turns']) != 1 or c.get('scenario', {}).get('kind') == 'steer' for c in selected):
            raise ValueError('当前回放仅接受单轮任务，运行中用户修改需另建控制命令快照协议')
        return selected


def summary(samples, *, expected_points, repeats, captures_complete, profiles=PROFILES):
    metrics = {}
    valid = captures_complete and len(samples) == expected_points * repeats * len(profiles)
    expected_pairs = None
    for profile in profiles:
        rows = [s for s in samples if s['profile_id'] == profile]
        pairs = {(s['origin_sha256'], s['repetition']) for s in rows}
        if expected_pairs is None:
            expected_pairs = pairs
        valid &= pairs == expected_pairs
        costs = [(recovery_usage(s)['usage'] or {}).get('total_tokens') for s in rows]
        complete = (len(rows) == expected_points * repeats == len(pairs)
                    and all(type(t) is int and t > 0 for t in costs))
        valid &= complete and all(s['integrity'] and not s['changed_db_paths'] for s in rows)
        metrics[profile] = dict(samples=len(rows), passed=sum(s['passed'] for s in rows),
            success_rate=sum(s['passed'] for s in rows) / len(rows) if rows else None,
            suffix_tokens=sum(costs) if costs and all(type(t) is int and t > 0 for t in costs) else None,
            suffix_steps=sum(s['suffix_steps'] for s in rows),
            model_calls=sum(s['suffix_budget']['model_calls'] for s in rows),
            tool_calls=sum(s['suffix_budget']['tool_calls'] for s in rows),
            searches=sum(e.get('type')=='recovery_search_started' for s in rows for e in s.get('result',{}).get('events',[])),
            selected_searches=sum(e.get('type')=='recovery_search_consumed' for s in rows for e in s.get('result',{}).get('events',[])),
            invalid_searches=sum(e.get('type')=='recovery_search_completed' and e.get('status')=='INVALID'
                                 for s in rows for e in s.get('result',{}).get('events',[])),
            deferred_gates=sum(e.get('type')=='recovery_search_gate' and e.get('status')=='DEFERRED'
                              for s in rows for e in s.get('result',{}).get('events',[])))
    controls = {(s['origin_sha256'],s['repetition']):s for s in samples if s['profile_id'] == profiles[1]}
    candidates = {(s['origin_sha256'],s['repetition']):s for s in samples if s['profile_id'] == profiles[2]}
    for sample in samples:
        peer=controls.get((sample['origin_sha256'],sample['repetition']))
        valid &= bool(peer and sample['database_sha256']==peer['database_sha256'] and sample['prefix_budget']==peer['prefix_budget'])
    if controls.keys() != candidates.keys():
        valid = False
    paired = []
    for key in controls.keys() & candidates.keys():
        left,right = controls[key],candidates[key]
        valid &= left['database_sha256'] == right['database_sha256'] and left['prefix_budget'] == right['prefix_budget']
        paired.append(dict(origin_sha256=key[0], repetition=key[1], control_passed=left['passed'],search_passed=right['passed']))
    return dict(status='COMPLETED' if valid else 'INVALID', metrics=metrics, pairs=sorted(paired,key=lambda p:(p['origin_sha256'],p['repetition'])),
        promotion_status='candidate', observed_improvement=False,
        comparison=dict(control_profile=profiles[1],candidate_profile=profiles[2]),
        note='同起点配对结果仍含模型随机性；本报告不自动作稳定收益或晋升结论。')


def evaluate(spec, dataset, output, *, execution_source='test_harness', on_progress=None):
    from eval.recovery_replay import capture_case, replay
    from src.harness import get_registry
    from src.model_gateway import load_gateway_config, ModelGatewayConfigurationError
    from src.runtime import BudgetLimits
    from src.llm import model_attempt_limit
    cases = spec.cases(dataset)
    profiles = tuple(spec.profiles)
    output.mkdir(parents=True, exist_ok=False)
    manifest = environment_manifest(dataset, 'adaptive', ROOT / spec.dataset)
    manifest['purpose_models'] = {}
    for purpose in ('goal','planner','agent','final_response'):
        try:
            config = load_gateway_config(purpose=purpose)
            manifest['purpose_models'][purpose] = [dict(provider=p.provider,model=p.model,
                thinking_mode=p.thinking_mode,reasoning_effort=p.reasoning_effort,strict_tools=p.strict_tools)
                for p in config.providers]
        except ModelGatewayConfigurationError:
            manifest['purpose_models'][purpose] = None
    limits = replace(BudgetLimits.adaptive(), max_model_calls=spec.max_model_calls,max_seconds=spec.max_seconds)
    contract = dict(schema_version=1, spec=spec.model_dump(), manifest=manifest,limits=limits.__dict__,
        execution_source=execution_source, profiles=[get_registry().resolve(p).provenance() for p in profiles],
        profile_labels={p:get_registry().resolve(p).profile.display_name for p in profiles},
        capture_profile=CAPTURE_PROFILE, comparison=dict(control_profile=profiles[1],candidate_profile=profiles[2]),
        max_success_regression=0, metrics=['task_success','suffix_tokens','suffix_steps'],
        metric_directions=['higher','lower','lower'], business_environment='frozen_demo_sqlite_per_replay',
        cost_scope='prefix reported separately; all search probes included in suffix', promotion_status='candidate')
    write_artifact(output/'contract.json',contract)
    samples, captures = [], []
    try:
        with model_attempt_limit(1):
            for case in cases:
                capture = capture_case(case,limits,source_manifest=manifest)
                name = f"point-{case['id']}.json"
                write_artifact(output/name,capture)
                captures.append(dict(case_id=case['id'],status=capture['status'],artifact=name))
                if on_progress:
                    on_progress(captures[-1])
                if capture['status'] != 'CAPTURED':
                    continue
                point = capture['point']
                for repetition in range(1,spec.repeats+1):
                    # 轮换顺序，减少固定总是最后执行搜索组造成的时间顺序影响。
                    order = profiles[repetition % len(profiles):] + profiles[:repetition % len(profiles)]
                    for profile in order:
                        row = replay(point,profile,source_manifest=manifest,repetition=repetition)
                        write_artifact(output/f"sample-{case['id']}-{repetition}-{profile}.json",row)
                        samples.append(row)
                        if on_progress:
                            on_progress(dict(case_id=case['id'],profile=profile,repetition=repetition,passed=row['passed']))
        result = summary(samples,expected_points=len(cases),repeats=spec.repeats,
                         profiles=profiles,
                         captures_complete=len(captures)==len(cases) and all(c['status']=='CAPTURED' for c in captures))
    except Exception as error:
        result = dict(status='INVALID',code=str(error) if str(error).startswith('REPLAY_') else 'REPLAY_EVALUATION_FAILED',
                      observed_improvement=False,promotion_status='candidate')
    result.update(execution_source=execution_source,contract_sha256=fingerprint(contract),captures=captures,
                  completed_samples=len(samples),expected_samples=len(cases)*spec.repeats*len(profiles))
    write_artifact(output/'summary.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,default=ROOT/'research/experiments/recovery-search-v1/smoke-spec.json')
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args(argv)
    spec=ReplaySpec.model_validate_json(args.spec.read_text())
    dataset=load_dataset(ROOT/spec.dataset)
    spec.cases(dataset)
    if not args.live:
        print(json.dumps(dict(status='VALIDATED',model_calls=0,max_task_runs=len(spec.case_ids)*(1+spec.repeats*len(spec.profiles)))))
        return 0
    if args.output_dir is None:
        parser.error('--live 必须指定新输出目录')
    output=args.output_dir.resolve()
    if output.exists() or not output.is_relative_to((ROOT/'research/recovery_search').resolve()):
        parser.error('输出必须是 research/recovery_search 下的新目录')
    from src.model_gateway import get_model_status
    if not get_model_status().get('configured'):
        print(json.dumps(dict(status='NOT_RUN',code='MODEL_NOT_CONFIGURED')))
        return 1
    result=evaluate(spec,dataset,output,execution_source='live_provider',on_progress=lambda row:print(json.dumps(row),flush=True))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['status']=='COMPLETED' else 1


if __name__=='__main__':
    raise SystemExit(main())
