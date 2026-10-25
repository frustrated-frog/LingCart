"""冻结多轮材料的核验组件对照；默认 dry-run，不执行业务写入或退款。"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'eval/evidence_semantic_tasks.json'
SPEC = ROOT / 'research/evaluations/evidence-semantic-v1/experiment-spec.json'


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class EvidenceTurn(Strict):
    message: str = Field(min_length=1, max_length=2000)
    evidence_type: str = 'DAMAGE_DESCRIPTION'
    withdraw_turn: int | None = Field(default=None, ge=0)
    expected_statuses: list[Literal['SUFFICIENT', 'INCOMPLETE', 'IRRELEVANT', 'CONFLICT', 'UNKNOWN']] = Field(min_length=1)
    expect_withdrawal: bool = False
    submit_evidence: bool = True


class EvidenceCase(Strict):
    case_id: str
    split: Literal['dev', 'test']
    order_id: int
    requirement: str
    evidence_type: str = 'DAMAGE_DESCRIPTION'
    turns: list[EvidenceTurn] = Field(min_length=2, max_length=6)

    @model_validator(mode='after')
    def previous_turn_only(self):
        for index, turn in enumerate(self.turns):
            if turn.withdraw_turn is not None and turn.withdraw_turn >= index:
                raise ValueError('只能引用已经提交的材料')
            if turn.expect_withdrawal and turn.withdraw_turn is None:
                raise ValueError('撤回断言必须指定材料')
        return self


class Dataset(Strict):
    schema_version: Literal[1]
    cases: list[EvidenceCase]

    @model_validator(mode='after')
    def ids_and_splits(self):
        if len({c.case_id for c in self.cases}) != len(self.cases):
            raise ValueError('案例编号不能重复')
        if {c.split for c in self.cases} != {'dev', 'test'}:
            raise ValueError('必须保留 dev/test 分组')
        return self


class Experiment(Strict):
    schema_version: Literal[1]
    scope: Literal['evidence_verification_component']
    baseline: Literal['provenance-reference-v1']
    candidate: Literal['evidence-semantic-v1']
    dataset_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    repeats: int = Field(ge=1, le=3)
    max_model_calls_per_turn: int = Field(ge=1, le=2)
    max_seconds_per_turn: int = Field(ge=1, le=60)
    evaluation_split: Literal['test']
    tuning_split: Literal['dev']
    max_false_acceptances: Literal[0]
    max_unauthorized_withdrawals: Literal[0]
    max_protocol_errors: Literal[0]
    max_accuracy_regression: Literal[0]
    min_false_acceptance_improvement: int = Field(ge=1)


def sha(content):
    return hashlib.sha256(content).hexdigest()


def load_inputs():
    data = Dataset.model_validate_json(DATASET.read_text())
    spec = Experiment.model_validate_json(SPEC.read_text())
    if sha(DATASET.read_bytes()) != spec.dataset_sha256:
        raise ValueError('任务集与冻结契约指纹不一致')
    return data, spec


def source_reference(inputs):
    """旧类型/来源充分性规则的只读参考，不注册为可执行业务工具。"""
    requirement = inputs['requirements'][0]
    evidence = [e for e in inputs['evidence'] if e['evidence_type'] == requirement['evidence_type']
                and e['evidence_id'] != inputs['requested_withdrawal']]
    return {'requirements': [{'requirement_id': requirement['requirement_id'],
                             'status': 'SUFFICIENT' if evidence else 'INCOMPLETE',
                             'citations': [{'evidence_id': e['evidence_id'], 'quote': e['statement']} for e in evidence],
                             'missing_information': [], 'explanation': '来源与类型参考规则',
                             'question': '' if evidence else requirement['description']}],
            'withdrawal_confirmed': bool(inputs['requested_withdrawal']),
            'withdrawal_quote': inputs['latest_request'] if inputs['requested_withdrawal'] else ''}


def evaluate(data, spec, *, split, model_factory, execution_source, on_sample=None):
    from src.after_sales.semantic import assess
    from src.runtime import BudgetLimits, RunRuntime, runtime_scope

    samples = []
    for case in (c for c in data.cases if c.split == split):
        for repeat in range(spec.repeats):
            for variant in (spec.baseline, spec.candidate):
                evidence = []
                for index, turn in enumerate(case.turns):
                    reference = f'turn-{index}'
                    if turn.submit_evidence:
                        evidence.append({'evidence_id': reference, 'evidence_type': turn.evidence_type,
                                         'statement': turn.message, 'source_text': turn.message})
                    inputs = {'case_id': case.case_id, 'order_id': case.order_id,
                              'requirements': [{'requirement_id': 'fixture-policy:requirement',
                                                'evidence_type': case.evidence_type, 'description': case.requirement}],
                              'evidence': list(evidence), 'latest_request': turn.message,
                              'requested_withdrawal': f'turn-{turn.withdraw_turn}' if turn.withdraw_turn is not None else None}
                    runtime = RunRuntime(uuid.uuid4().hex, BudgetLimits(
                        max_model_calls=spec.max_model_calls_per_turn, max_seconds=spec.max_seconds_per_turn,
                        call_timeout=spec.max_seconds_per_turn, max_prompt_tokens=48000))
                    started = time.monotonic()
                    if variant == spec.baseline:
                        result = source_reference(inputs)
                    else:
                        try:
                            with runtime_scope(runtime):
                                result = assess(inputs, model=model_factory(), runtime=runtime)
                        except Exception as exc:
                            from src.errors import AgentError
                            from src.model_gateway import ModelGatewayConfigurationError
                            if not isinstance(exc, (AgentError, ModelGatewayConfigurationError)):
                                raise
                            from src.after_sales.semantic import abstention
                            result = abstention(inputs, getattr(exc, 'code', 'MODEL_UNAVAILABLE'))
                    status = result['requirements'][0]['status']
                    withdrawn = result['withdrawal_confirmed']
                    calls = [e for e in runtime.events if e['type'] == 'model_call']
                    usage = [e.get('usage') or {} for e in calls]
                    tokens = sum(u['total_tokens'] for u in usage) if usage and all(u.get('total_tokens') is not None for u in usage) else None
                    samples.append({'case_id': case.case_id, 'repeat': repeat, 'turn': index, 'variant': variant,
                        'input': inputs, 'result': result,
                        'expected_statuses': turn.expected_statuses, 'expect_withdrawal': turn.expect_withdrawal,
                        'passed': status in turn.expected_statuses and withdrawn == turn.expect_withdrawal,
                        'false_acceptance': status == 'SUFFICIENT' and 'SUFFICIENT' not in turn.expected_statuses,
                        'unnecessary_question': status != 'SUFFICIENT' and turn.expected_statuses == ['SUFFICIENT'],
                        'unauthorized_withdrawal': withdrawn and not turn.expect_withdrawal,
                        'model_attempts': runtime.counters['model_calls'], 'total_tokens': tokens,
                        'latency_ms': round((time.monotonic() - started) * 1000, 2),
                        'raw_model_responses': [value['content'] for value in runtime.models.values()],
                        'model_events': [e for e in runtime.events if e['type'] in {'model_call', 'error', 'model_retry'}]})
                    if on_sample:
                        on_sample(samples[-1])
                    if withdrawn:
                        evidence = [e for e in evidence if e['evidence_id'] != inputs['requested_withdrawal']]
    metrics = {}
    for variant in (spec.baseline, spec.candidate):
        rows = [s for s in samples if s['variant'] == variant]
        case_ids = {s['case_id'] for s in rows}
        metrics[variant] = {
            'turn_accuracy': sum(s['passed'] for s in rows) / len(rows),
            'all_repeats_success_rate': sum(all(s['passed'] for s in rows if s['case_id'] == cid) for cid in case_ids) / len(case_ids),
            'false_acceptances': sum(s['false_acceptance'] for s in rows),
            'unauthorized_withdrawals': sum(s['unauthorized_withdrawal'] for s in rows),
            'unnecessary_questions': sum(s['unnecessary_question'] for s in rows),
            'protocol_errors': sum(s['result'].get('error_code') == 'SEMANTIC_PROTOCOL_INVALID' for s in rows),
            'unavailable_turns': sum(bool(s['result'].get('error_code')) for s in rows),
            'model_attempts': sum(s['model_attempts'] for s in rows),
            'total_tokens': sum(s['total_tokens'] for s in rows) if all(s['total_tokens'] is not None for s in rows) else None,
            'mean_latency_ms': sum(s['latency_ms'] for s in rows) / len(rows),
        }
    baseline, candidate = metrics[spec.baseline], metrics[spec.candidate]
    failures = []
    for metric, cap in [('false_acceptances', spec.max_false_acceptances),
                        ('unauthorized_withdrawals', spec.max_unauthorized_withdrawals),
                        ('protocol_errors', spec.max_protocol_errors), ('unavailable_turns', 0)]:
        if candidate[metric] > cap:
            failures.append(metric)
    if candidate['turn_accuracy'] < baseline['turn_accuracy'] - spec.max_accuracy_regression:
        failures.append('turn_accuracy_regression')
    if candidate['all_repeats_success_rate'] < baseline['all_repeats_success_rate']:
        failures.append('repeat_stability_regression')
    if baseline['false_acceptances'] - candidate['false_acceptances'] < spec.min_false_acceptance_improvement:
        failures.append('no_false_acceptance_improvement')
    if candidate['model_attempts'] > len([s for s in samples if s['variant'] == spec.candidate]) * spec.max_model_calls_per_turn:
        failures.append('model_budget')
    from src.after_sales.semantic import PROMPT
    sources = ['src/after_sales/semantic.py', 'src/runtime.py', 'src/model_gateway.py', 'eval/run_evidence_semantic_eval.py']
    return {'schema_version': 1, 'scope': spec.scope, 'execution_source': execution_source, 'split': split,
            'created_at': datetime.now(UTC).isoformat(), 'experiment': spec.model_dump(),
            'manifest': {'dataset_sha256': sha(DATASET.read_bytes()), 'spec_sha256': sha(SPEC.read_bytes()),
                         'prompt_sha256': sha(PROMPT.encode()), 'python': platform.python_version(),
                         'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                         'source_sha256': {p: sha((ROOT / p).read_bytes()) for p in sources},
                         'requirements_lock_sha256': sha((ROOT / 'requirements.lock').read_bytes())},
            'metrics': metrics, 'samples': samples,
            'gate': {'passed': not failures, 'failures': failures,
                     'promotion_eligible': False, 'reason': '仅核验组件证据；不证明端到端任务完成或授权安全收益'}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='显式调用配置的模型，可能计费')
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    data, spec = load_inputs()
    turns = sum(len(c.turns) for c in data.cases if c.split == args.split)
    if not args.live:
        print(json.dumps({'mode': 'dry_run', 'scope': spec.scope, 'split': args.split,
                          'turns_per_repeat': turns, 'repeats': spec.repeats,
                          'max_model_attempts': turns * spec.repeats * spec.max_model_calls_per_turn,
                          'message': '任务集与契约已校验，未调用模型、未写入业务数据库'}, ensure_ascii=False, indent=2))
        return
    if args.output is None or args.output.exists():
        parser.error('--live 必须指定尚不存在的 --output，避免覆盖原始结果')
    journal = args.output.with_suffix('.samples.jsonl')
    if journal.exists():
        parser.error('原始样本日志已存在，请更换输出名称')
    from src.llm import get_llm
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with journal.open('x') as handle:
        def save(sample):
            handle.write(json.dumps(sample, ensure_ascii=False) + '\n')
            handle.flush()
        report = evaluate(data, spec, split=args.split,
                          model_factory=lambda: get_llm(purpose='evidence_verification'),
                          execution_source='live_model', on_sample=save)
    with args.output.open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'output': str(args.output), 'gate': report['gate']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
