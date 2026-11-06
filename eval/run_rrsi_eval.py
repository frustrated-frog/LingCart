"""RRSI 规划与恢复程序演化。默认只校验；--live 显式启用真实模型。"""
import argparse
import json
from pathlib import Path

from eval.run_live_eval import evaluate, load_dataset
from src.harness import get_registry
from src.harness.profile import HarnessProfile
from src.pg_learning import comparable, write_artifact
from src.rrsi_policy import HarnessProgram
from src.rrsi_learning import ModelProposer, RSISpec, evolve, measurement

ROOT = Path(__file__).resolve().parents[1]


def program_profile(program, path):
    payload = json.loads((ROOT / 'configs/harness/adaptive-v1.json').read_text())
    payload.update(profile_id=f'rrsi-{program.fingerprint[:24]}', display_name='RRSI 冻结规划与恢复程序',
                   description='基于执行反馈演化的规划与恢复候选；仍由既有 Kernel 完成工具和终止核验。')
    payload['research'] = dict(status='candidate', source_type='paper', paper_id='arXiv:2609.24972v3',
        hypothesis='规划或恢复组件经正则化演化后，可能在固定预算下改善未见任务的成功率或成本。')
    payload['mechanisms'] = [dict(slot='planning', id='rrsi-planning-recovery', version='1',
        configuration=dict(program_path=path.relative_to(ROOT).as_posix())) if m['slot'] == 'planning' else m
        for m in payload['mechanisms']]
    return HarnessProfile.model_validate_json(json.dumps(payload))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=ROOT / 'research/experiments/rrsi-recovery-v1/evolution-spec.json')
    parser.add_argument('--seed', type=Path, default=ROOT / 'research/rrsi/seed.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--heldout', action='store_true', help='演化结束冻结程序后，执行独立 test 对照；不反馈给学习器')
    args = parser.parse_args(argv)
    spec = RSISpec.model_validate_json(args.spec.read_text())
    dataset = load_dataset(ROOT / spec.dataset)
    spec.validate_dataset(dataset)
    seed = HarnessProgram.model_validate_json(args.seed.read_text())
    if not args.live:
        print(json.dumps(dict(status='VALIDATED', model_calls=0, seed_sha256=seed.fingerprint,
            rounds=spec.rounds, edit_budgets=[spec.edit_budget(t) for t in range(spec.rounds)])))
        return 0
    if args.output_dir is None:
        parser.error('--live 必须指定新输出目录')
    output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to((ROOT / 'research/rrsi').resolve()):
        parser.error('输出必须为 research/rrsi 内尚不存在的目录')
    registry = get_registry()
    profiles = {}
    def evaluator(program, cases, label, split='dev'):
        if program.fingerprint not in profiles:
            path = output / f'program-{program.fingerprint}.json'
            write_artifact(path, program.model_dump(mode='json'))
            profile = program_profile(program, path)
            registry.register_profile(profile)
            write_artifact(output / f'profile-{program.fingerprint}.json', profile.model_dump(mode='json'))
            profiles[program.fingerprint] = profile
        return evaluate(dataset, repeats=spec.repeats, split=split, case_ids=cases,
            execution_mode='adaptive', harness_profile=profiles[program.fingerprint].profile_id,
            dataset_path=ROOT / spec.dataset, max_model_calls_per_run=spec.max_model_calls_per_run,
            max_prompt_tokens_per_run=128000, max_completion_tokens_per_run=8000,
            max_seconds_per_run=spec.max_seconds_per_run, max_provider_attempts_per_call=1,
            on_sample=lambda sample: print(json.dumps(dict(stage=label, case=sample['case_id'],
                repetition=sample['repetition'], passed=sample['passed'])), flush=True))
    retained, report = evolve(seed, spec, evaluator=evaluator, proposer=ModelProposer(spec.target_module), output_dir=output,
        on_progress=lambda label: print(json.dumps(dict(stage=label)), flush=True))
    if report['status'] == 'COMPLETED':
        write_artifact(output / 'retained-profile.json', profiles[retained.fingerprint].model_dump(mode='json'))
    if report['status'] == 'COMPLETED' and args.heldout:
        write_artifact(output / 'heldout-contract.json', dict(
            baseline_program=seed.fingerprint, candidate_program=retained.fingerprint,
            cases=spec.test_cases, repeats=spec.repeats, source='live_provider',
            max_success_regression=0, max_stability_regression=0, max_token_regression=spec.beta0,
            promotion='manual_review_required'))
        # 保存选择结果以后才接触 test；测试结果不会进入提案、选择或再训练。
        comparisons = {}
        reports = {}
        try:
            for label, program in [('seed', seed), ('retained', retained)]:
                raw = evaluator(program, spec.test_cases, 'heldout-' + label, split='test')
                write_artifact(output / f'heldout-{label}.json', raw)
                reports[label] = raw
                comparisons[label] = measurement(raw, program, spec.test_cases, spec.repeats, split='test', target_module=spec.target_module)
            if not comparable(reports['seed'], reports['retained']):
                raise ValueError('RSI_ENVIRONMENT_CHANGED')
            left, right = comparisons['seed'], comparisons['retained']
            improved = (seed.fingerprint != retained.fingerprint and right['score'] >= left['score'] and right['pass_at_1'] >= left['pass_at_1'] and right['stable'] >= left['stable']
                and right['tokens'] <= left['tokens'] * (1 + spec.beta0)
                and not right['safety'] and not right['protocol']
                and (right['score'] > left['score'] or right['tokens'] < left['tokens']))
            result = dict(status='COMPLETED', measurements=comparisons, observed_improvement=improved,
                          promotion_status='candidate', note='小样本对照不能自动晋升或宣称稳定泛化收益')
        except ValueError as error:
            result = dict(status='INVALID', code=str(error) if str(error).startswith('RSI_') else 'RSI_HELDOUT_INVALID')
        write_artifact(output / 'heldout-summary.json', result)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'COMPLETED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
