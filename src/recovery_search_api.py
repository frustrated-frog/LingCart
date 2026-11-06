"""同状态回放产物的只读工程投影；不暴露数据库快照或模型输入。"""
from pathlib import Path
import re
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.rrsi_api import read
from src.recovery_usage import recovery_usage

ROOT = Path(__file__).resolve().parents[1] / 'research/recovery_search'
router = APIRouter(prefix='/engineering/recovery-replays', tags=['Recovery replay'])


class Summary(BaseModel):
    run_id: str
    status: str
    execution_source: str


class RunList(BaseModel):
    runs: list[Summary]


class Probe(BaseModel):
    branch_id: str
    probe_id: str
    status: str
    tool: str | None
    code: str | None
    product_records: int | None


class Search(BaseModel):
    search_id: str
    status: str = 'PARTIAL'
    chosen_branch: str | None = None
    reason: str | None = None
    probes: list[Probe] = Field(default_factory=list)


class Gate(BaseModel):
    gate_id: str
    status: str
    code: str
    failure_count: int
    threshold: int


class Profile(BaseModel):
    profile_id: str
    display_name: str
    role: Literal['reference', 'control', 'candidate', 'unknown']


class Sample(BaseModel):
    model_config = ConfigDict(strict=True)
    profile_id: str
    case_id: str
    repetition: int
    origin_sha256: str
    database_sha256: str
    passed: bool
    failures: list[str]
    terminal_reason: str | None
    tokens: int | None
    known_tokens: int | None
    usage_complete: bool
    model_calls: int
    tool_calls: int
    suffix_steps: int
    searches: list[Search]
    gates: list[Gate] = Field(default_factory=list)
    timeline: list[dict]


class Detail(Summary):
    source_revision: str | None
    partial: bool
    warnings: list[str]
    expected_samples: int | None
    samples: list[Sample]
    profiles: list[Profile] = Field(default_factory=list)


def folder_for(run_id):
    if not re.fullmatch(r'[a-z][a-z0-9-]{0,79}', run_id):
        raise HTTPException(404, detail={'code':'RECOVERY_REPLAY_NOT_FOUND','message':'回放记录不存在'})
    folder = ROOT/run_id
    if folder.is_symlink() or not folder.is_dir():
        raise HTTPException(404, detail={'code':'RECOVERY_REPLAY_NOT_FOUND','message':'回放记录不存在'})
    return folder


def project_sample(raw):
    result = raw['result']
    searches = {}
    for event in result.get('events', []):
        if not event.get('search_id') or not event.get('type','').startswith('recovery_search_'):
            continue
        item = searches.setdefault(event['search_id'],Search(search_id=event['search_id']))
        if event['type']=='recovery_search_probe':
            step = event.get('step') or {}
            products = (step.get('result') or {}).get('products')
            item.probes.append(Probe(branch_id=event['branch_id'],probe_id=event['probe_id'],status=event['status'],
                tool=step.get('tool'),code=event.get('code'),product_records=len(products) if isinstance(products,list) else None))
        elif event['type']=='recovery_search_completed':
            choice = event.get('selection') or {}
            item.status=event['status']
            item.chosen_branch=choice.get('branch_id')
            item.reason=choice.get('reason') or '；'.join(
                f"{issue.get('path','')}: {issue.get('message','')}" for issue in event.get('validation_issues', [])) or None
    steps = raw['suffix_steps']
    timeline = raw.get('recovery',{}).get('timeline',[])
    usage=recovery_usage(raw)
    return Sample(**{k:raw[k] for k in ('profile_id','case_id','repetition','origin_sha256','database_sha256','passed','failures','suffix_steps')},
        terminal_reason=result.get('terminal_reason'),tokens=(usage['usage'] or {}).get('total_tokens'),
        known_tokens=(usage['observed_usage'] or {}).get('total_tokens'),usage_complete=usage['complete'],
        model_calls=raw['suffix_budget']['model_calls'],tool_calls=raw['suffix_budget']['tool_calls'],
        searches=list(searches.values()),timeline=timeline[-steps:] if steps else [],
        gates=[Gate(**{key:event[key] for key in Gate.model_fields}) for event in result.get('events',[])
               if event.get('type')=='recovery_search_gate'])


@router.get('',response_model=RunList)
def runs():
    rows=[]
    if ROOT.is_dir():
        for folder in sorted(ROOT.iterdir()):
            if folder.is_symlink() or not folder.is_dir() or not re.fullmatch(r'[a-z][a-z0-9-]{0,79}',folder.name):
                continue
            contract=read(folder,'contract.json',[])
            if not contract:
                continue
            result=read(folder,'summary.json',[]) or {}
            rows.append(Summary(run_id=folder.name,status=result.get('status','PARTIAL'),
                execution_source=contract.get('execution_source','unknown')))
    return RunList(runs=rows)


@router.get('/{run_id}',response_model=Detail)
def detail(run_id: str):
    folder=folder_for(run_id)
    warnings=[]
    contract=read(folder,'contract.json',warnings) or {}
    summary=read(folder,'summary.json',warnings) or {}
    samples=[]
    for path in sorted(folder.glob('sample-*.json')):
        raw=read(folder,path.name,warnings)
        if raw:
            try:
                samples.append(project_sample(raw))
            except (ValueError,TypeError,KeyError,AttributeError):
                warnings.append(f'{path.name} 的样本协议无效')
    expected=summary.get('expected_samples')
    if expected is not None and (type(expected) is not int or expected != len(samples)):
        warnings.append('尚未取得合同要求的全部回放样本')
    profiles=[]
    declared=contract.get('profiles',[])
    labels=contract.get('profile_labels',{})
    if not isinstance(declared,list) or not isinstance(labels,dict):
        warnings.append('策略合同格式无效')
        declared,labels=[],{}
    for index,profile in enumerate(declared):
        profile_id=profile.get('profile_id') if isinstance(profile,dict) else None
        if not isinstance(profile_id,str) or any(p.profile_id==profile_id for p in profiles):
            warnings.append('策略合同身份缺失或重复')
            continue
        label=labels.get(profile_id)
        profiles.append(Profile(profile_id=profile_id,
            display_name=label if isinstance(label,str) and label else profile_id,
            role=('reference','control','candidate')[index] if index < 3 else 'unknown'))
    for sample in samples:
        if sample.profile_id not in {p.profile_id for p in profiles}:
            warnings.append('样本策略缺少对应合同身份')
            profiles.append(Profile(profile_id=sample.profile_id,display_name=sample.profile_id,role='unknown'))
    return Detail(run_id=run_id,status=summary.get('status','PARTIAL'),
        execution_source=contract.get('execution_source','unknown'),source_revision=contract.get('manifest',{}).get('source_revision'),
        partial=not bool(summary) or bool(warnings),warnings=warnings,expected_samples=expected if type(expected) is int else None,
        samples=samples,profiles=profiles)
