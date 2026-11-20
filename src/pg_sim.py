"""SIM：跨实验的问题与修改历史，仅供离线图优化器读取。"""
from __future__ import annotations

from typing import Literal
from pydantic import Field, field_validator, model_validator

from src.procedural_graph import FrozenModel, GraphPatch, fingerprint, strategy_text
from src.pg_hypotheses import EvolutionProposal

MemoryMode = Literal['none', 'problems', 'full']


class SIMProposal(EvolutionProposal):
    problem_id: str | None = Field(default=None, pattern=r'^sim-[0-9a-f]{20}$')
    predicted_fixes: tuple[str, ...] = Field(default=(), max_length=320,
        description='评测前预测能从失败变为通过的当前训练样本，格式 case_id:repetition；无修复预测时为空。')


    @model_validator(mode='after')
    def strategy_only(self):
        strategy_text(self.hypothesis.problem)
        strategy_text(self.patch.reason)
        return self


class SIMEnvironment(FrozenModel):
    dataset_sha256: str
    models_sha256: str
    protocol_sha256: str
    source_sha256: str
    limits_sha256: str
    train_cases: tuple[str, ...]


class SIMAttempt(FrozenModel):
    attempt_id: str
    experiment_id: str
    run_id: str
    iteration: int = Field(ge=1)
    parent_sha256: str
    candidate_sha256: str | None
    training_sha256: str
    candidate_training_sha256: str | None
    environment: SIMEnvironment
    change: str
    expected_change: str
    patch_sha256: str
    patch: GraphPatch
    status: Literal['ACCEPTED', 'REJECTED']
    reason: str
    predicted_fixes: tuple[str, ...]
    fixed: tuple[str, ...]
    missed: tuple[str, ...]
    regressed: tuple[str, ...]
    observation_status: Literal['MEASURED', 'NOT_EVALUATED']
    validation_summary: dict[str, int]

    _strategy = field_validator('change')(strategy_text)

    @model_validator(mode='after')
    def consistent_outcomes(self):
        if not set(self.patch.source_cases) <= set(self.environment.train_cases) or fingerprint(self.patch.model_dump(mode='json')) != self.patch_sha256:
            raise ValueError('SIM 修改内容或训练引用身份不符')
        strategy_text(self.patch.reason)
        predicted, fixed, missed = map(set, (self.predicted_fixes, self.fixed, self.missed))
        if len(predicted) != len(self.predicted_fixes):
            raise ValueError('SIM 预测样本不能重复')
        for ref in (*predicted, *fixed, *missed, *self.regressed):
            case, separator, repetition = ref.rpartition(':')
            if not separator or case not in self.environment.train_cases or not repetition.isdigit() or int(repetition) < 1:
                raise ValueError('SIM 样本必须来自训练分组')
        if self.observation_status == 'MEASURED':
            if not self.candidate_training_sha256 or fixed & missed or fixed | missed != predicted:
                raise ValueError('SIM 实测必须完整对应预期')
        elif fixed or missed or self.regressed or self.candidate_training_sha256:
            raise ValueError('未评测的 SIM 尝试不能生成效果')
        if set(self.validation_summary) - {'regressed_samples', 'improved_samples', 'compared_samples'}:
            raise ValueError('SIM 只能保留验证聚合反馈')
        if any(value < 0 for value in self.validation_summary.values()):
            raise ValueError('SIM 验证数量无效')
        return self


class SIMProblem(FrozenModel):
    problem_id: str = Field(pattern=r'^sim-[0-9a-f]{20}$')
    diagnosis: str = Field(min_length=1, max_length=600)
    failure_mode: str
    attempts: tuple[SIMAttempt, ...] = Field(default=(), max_length=128)

    _strategy = field_validator('diagnosis')(strategy_text)


class SIMMemory(FrozenModel):
    schema_version: Literal[1] = 1
    mechanism: Literal['pg-sim-v1'] = 'pg-sim-v1'
    online_profile: Literal['adaptive-pg-context-v4'] = 'adaptive-pg-context-v4'
    execution_source: Literal['live_provider', 'test_harness']
    retained_sha256: str
    problems: tuple[SIMProblem, ...] = Field(default=(), max_length=64)

    @model_validator(mode='after')
    def unique_history(self):
        ids = [p.problem_id for p in self.problems]
        attempts = [a.attempt_id for p in self.problems for a in p.attempts]
        if len(ids) != len(set(ids)) or len(attempts) != len(set(attempts)) or len(attempts) > 256:
            raise ValueError('SIM 历史身份重复或超过上限')
        return self

    @property
    def fingerprint(self):
        return fingerprint(self.model_dump(mode='json'))


def environment(report, train_cases):
    manifest = report['manifest']
    return SIMEnvironment(dataset_sha256=manifest['dataset_sha256'],
        models_sha256=fingerprint([manifest['purpose_models'], report.get('providers')]),
        protocol_sha256=fingerprint([manifest['evaluation_protocol_sha256'], manifest['business_protocol_sha256']]),
        source_sha256=manifest['source_tree_sha256'], limits_sha256=fingerprint(manifest['limits']),
        train_cases=tuple(train_cases))


def inherit(memory, *, graph, execution_source):
    if memory.execution_source != execution_source or memory.retained_sha256 != graph.fingerprint or graph.schema_version != 2:
        raise ValueError('SIM 来源或保留图身份不匹配')
    # 重新验证深层内容，不能信任 model_copy 或调用者传入的可变对象。
    return SIMMemory.model_validate_json(memory.model_dump_json())


def check_partition(memory, current_environment, validation_cases):
    if any(a.environment.dataset_sha256 == current_environment.dataset_sha256
           and set(a.environment.train_cases) & set(validation_cases)
           for p in memory.problems for a in p.attempts):
        raise ValueError('SIM 来源训练案例不能进入目标验证分组')


def memory_view(memory, mode, current_environment):
    if mode not in {'none', 'problems', 'full'}:
        raise ValueError('未知 SIM 对照条件')
    problems = []
    if mode != 'none':
        for problem in memory.problems:
            row = dict(problem_id=problem.problem_id, diagnosis=problem.diagnosis, failure_mode=problem.failure_mode)
            if mode == 'full':
                row['attempts'] = [dict(a.model_dump(mode='json'), environment_changed=a.environment != current_environment)
                                   for a in problem.attempts]
            problems.append(row)
    return dict(mode=mode, problems=problems, current_environment=current_environment.model_dump(mode='json'))


def bind_problem(memory, proposal, training, visible_mode):
    failed = {f"{s['case_id']}:{s['repetition']}" for s in training['samples'] if not s['passed']}
    if (not set(proposal.predicted_fixes) <= failed or len(set(proposal.predicted_fixes)) != len(proposal.predicted_fixes)):
        raise ValueError('SIM 只能预测修复当前训练失败样本')
    if proposal.problem_id:
        problem = next((p for p in memory.problems if p.problem_id == proposal.problem_id), None)
        if visible_mode == 'none' or problem is None or problem.failure_mode != proposal.hypothesis.failure_mode:
            raise ValueError('SIM 引用的问题未暴露或失败类型不一致')
        return problem.problem_id
    # 与论文一样，只做明确身份/精确诊断合并，不自动把语义相近的问题混为一谈。
    return 'sim-' + fingerprint([proposal.hypothesis.failure_mode, ' '.join(proposal.hypothesis.problem.split()).casefold()])[:20]


def append_attempt(memory, *, problem_id, proposal, record, before, after, current_environment, experiment_id, run_id):
    predicted = proposal.predicted_fixes
    observed = {f"{s['case_id']}:{s['repetition']}": s['passed'] for s in (after or {}).get('samples', [])}
    previous = {f"{s['case_id']}:{s['repetition']}": s['passed'] for s in before['samples']}
    measured = after is not None
    changes = record.get('validation_changes') or []
    attempt = SIMAttempt(attempt_id=fingerprint([record['training_sha256'], record['parent_sha256'],
        record['iteration'], proposal.model_dump(mode='json'), run_id]), experiment_id=experiment_id, run_id=run_id,
        iteration=record['iteration'], parent_sha256=record['parent_sha256'], candidate_sha256=record.get('candidate_sha256'),
        training_sha256=record['training_sha256'], candidate_training_sha256=fingerprint(after) if measured else None,
        environment=current_environment, change=proposal.patch.reason, expected_change=proposal.hypothesis.expected_change,
        patch_sha256=fingerprint(proposal.patch.model_dump(mode='json')), patch=proposal.patch, status=record['status'], reason=record['reason'],
        predicted_fixes=predicted, fixed=tuple(ref for ref in predicted if observed.get(ref)) if measured else (),
        missed=tuple(ref for ref in predicted if not observed.get(ref)) if measured else (),
        regressed=tuple(ref for ref, passed in previous.items() if passed and not observed.get(ref)) if measured else (),
        observation_status='MEASURED' if measured else 'NOT_EVALUATED', validation_summary=dict(
            regressed_samples=sum(c['before_passed'] and not c['after_passed'] for c in changes),
            improved_samples=sum(not c['before_passed'] and c['after_passed'] for c in changes), compared_samples=len(changes)))
    problems = list(memory.problems)
    existing = next((p for p in problems if p.problem_id == problem_id), None)
    if existing:
        problems[problems.index(existing)] = existing.model_copy(update={'attempts': (*existing.attempts, attempt)})
    else:
        problems.append(SIMProblem(problem_id=problem_id, diagnosis=proposal.hypothesis.problem,
                                  failure_mode=proposal.hypothesis.failure_mode, attempts=(attempt,)))
    updated = memory.model_copy(update={'problems': tuple(problems), 'retained_sha256': record['retained_sha256']})
    return SIMMemory.model_validate_json(updated.model_dump_json()), attempt
