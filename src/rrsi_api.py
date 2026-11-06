"""RRSI 不可变实验产物的只读、工程鉴权投影。"""
import json
import re
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.recovery_evidence import recovery_trace

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / 'research/rrsi'
router = APIRouter(prefix='/engineering/rrsi', tags=['RSI experiments'])


class RunSummary(BaseModel):
    run_id: str
    status: str
    target_module: str | None = None


class RunList(BaseModel):
    runs: list[RunSummary]


class SampleSide(BaseModel):
    passed: bool
    tokens: int | None
    source: str
    recovery: dict


class SamplePair(BaseModel):
    case_id: str
    repetition: int
    parent: SampleSide | None = None
    candidate: SampleSide | None = None


class Change(BaseModel):
    field: str
    before: str
    after: str


class CandidateView(BaseModel):
    round: int
    candidate: int
    status: str
    selected: bool = False
    hypothesis: str = ''
    parent_sha256: str
    program_sha256: str | None = None
    changes: list[Change] = Field(default_factory=list)
    measurement: dict | None = None
    samples: list[SamplePair] = Field(default_factory=list)


class RunDetail(RunSummary):
    source_revision: str | None
    historical: bool
    execution_source: str
    candidates: list[CandidateView]
    comparison: dict | None
    partial: bool
    warnings: list[str]
    spec: dict


def read(folder, name, warnings):
    path = folder / name
    if not path.exists():
        return None
    if path.is_symlink() or path.stat().st_size > 16_000_000:
        warnings.append(f'{name} 不可读取或过大')
        return None
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError()
        return value
    except (ValueError, OSError):
        warnings.append(f'{name} 尚未完整写入或格式损坏')
        return None


def folder_for(run_id):
    if not re.fullmatch(r'[a-z][a-z0-9-]{0,79}', run_id):
        raise HTTPException(404, detail={'code':'RSI_RUN_NOT_FOUND','message':'实验不存在'})
    folder = ARTIFACT_ROOT / run_id
    if folder.is_symlink() or not folder.is_dir():
        raise HTTPException(404, detail={'code':'RSI_RUN_NOT_FOUND','message':'实验不存在'})
    return folder


def side(sample, source):
    tokens = [(t.get('usage') or {}).get('total_tokens') for t in sample.get('turns', [])]
    return SampleSide(passed=sample.get('passed') is True,
        tokens=sum(tokens) if tokens and all(type(v) is int and v >= 0 for v in tokens) else None,
        source=source, recovery=recovery_trace(sample))


def pair_reports(parent, candidate, warnings):
    pairs = {}
    for label, report in [('parent',parent),('candidate',candidate)]:
        for sample in (report or {}).get('samples', []):
            if not isinstance(sample,dict) or not isinstance(sample.get('case_id'),str) or type(sample.get('repetition')) is not int:
                warnings.append('样本身份格式无效')
                continue
            key = (sample['case_id'],sample['repetition'])
            try:
                pairs.setdefault(key, dict(case_id=key[0], repetition=key[1]))[label] = side(sample, report.get('execution_source','unknown'))
            except (ValueError,TypeError,KeyError,AttributeError):
                warnings.append('样本轨迹格式无效')
    return [SamplePair(**pairs[key]) for key in sorted(pairs)]


@router.get('/runs', response_model=RunList)
def runs():
    rows = []
    if ARTIFACT_ROOT.is_dir():
        for folder in sorted(ARTIFACT_ROOT.iterdir()):
            if folder.is_symlink() or not folder.is_dir() or not re.fullmatch(r'[a-z][a-z0-9-]{0,79}',folder.name):
                continue
            warnings = []
            spec = read(folder,'spec.json',warnings)
            if not spec:
                continue
            report = read(folder,'evolution-report.json',warnings) or {}
            rows.append(RunSummary(run_id=folder.name,status=report.get('status','PARTIAL'),target_module=spec.get('target_module')))
    return RunList(runs=rows)


@router.get('/runs/{run_id}', response_model=RunDetail)
def detail(run_id: str):
    folder, warnings = folder_for(run_id), []
    spec = read(folder,'spec.json',warnings) or {}
    evolution = read(folder,'evolution-report.json',warnings) or {}
    history = evolution.get('history', [])
    if not evolution:
        selections = sorted(folder.glob('round-*-selection.json'))
        if selections:
            history = (read(folder,selections[-1].name,warnings) or {}).get('history', [])
    seed = read(folder,'seed.json',warnings) or {}
    seed_report = read(folder,'seed-validation.json',warnings) or {}
    revision = seed_report.get('manifest',{}).get('source_revision')
    try:
        current_revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        current_revision = None
    from src.rrsi_policy import HarnessProgram
    programs = {}
    if seed:
        try:
            programs[HarnessProgram.model_validate_json(json.dumps(seed)).fingerprint] = seed
        except ValueError:
            warnings.append('初始程序格式无效')
    validation = {}
    for path in sorted(folder.glob('*validation.json')):
        if not re.fullmatch(r'(seed|round-\d+-candidate-\d+)-validation.json',path.name):
            continue
        report = read(folder,path.name,warnings)
        if report:
            for m in report.get('manifest',{}).get('harness_profile',{}).get('mechanisms',[]):
                sha = m.get('configuration',{}).get('program_sha256')
                if sha:
                    validation.setdefault(sha,report)
    views = []
    for item in history:
        if (not isinstance(item,dict) or any(type(item.get(k)) is not int or not 0 <= item[k] <= 100 for k in ('round','candidate'))
                or not isinstance(item.get('parent_sha256'),str) or not isinstance(item.get('status'),str)
                or not isinstance(item.get('program',{}),dict)):
            warnings.append('候选记录格式无效')
            continue
        candidate = item.get('program') or {}
        parent = programs.get(item['parent_sha256'],{})
        changes = [Change(field=f'{module}.{key}', before=str(parent.get(module,{}).get(key,'尚无证据')),after=str(value))
            for module in ('planning','recovery') for key,value in candidate.get(module,{}).items()
            if parent.get(module,{}).get(key) != value]
        label = f"round-{item['round']}-candidate-{item['candidate']}-validation.json"
        report = read(folder,label,warnings)
        views.append(CandidateView(**{k:item[k] for k in ('round','candidate','status','parent_sha256')},
            selected=item.get('selected',False),hypothesis=item.get('hypothesis',''),program_sha256=item.get('program_sha256'),
            changes=changes,measurement=item.get('measurement'),
            samples=pair_reports(validation.get(item['parent_sha256']),report,warnings)))
        if item.get('program_sha256') and candidate:
            programs[item['program_sha256']] = candidate
    comparison = read(folder,'ablation-summary.json',warnings) or read(folder,'heldout-summary.json',warnings)
    if comparison:
        contract = read(folder,'ablation-contract.json',warnings) or {}
        identities = contract.get('arms', {})
        if not identities:
            heldout = read(folder,'heldout-contract.json',warnings) or {}
            identities = {'seed':heldout.get('baseline_program'),'retained':heldout.get('candidate_program')}
        groups = {}
        for arm, identity in identities.items():
            if isinstance(identity,str):
                groups.setdefault(identity,[]).append(arm)
        comparison['identical_arms'] = [arms for arms in groups.values() if len(arms)>1]

    return RunDetail(run_id=run_id,status=evolution.get('status','PARTIAL'),target_module=spec.get('target_module'),
        source_revision=revision,historical=bool(revision and revision != current_revision),
        execution_source=evolution.get('execution_source',seed_report.get('execution_source','unknown')),
        candidates=views,comparison=comparison,partial=not bool(evolution) or bool(warnings), warnings=warnings,spec=spec)
