"""受工程鉴权保护的 PG 实验边界；执行逻辑不放进 HTTP 线程。"""
import json
import os
import re
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.pg_jobs import JobStore, LabError, StartExperiment, catalog, detail

router = APIRouter(prefix='/engineering/pg', tags=['PG experiments'])


class JobResponse(BaseModel):
    job_id: str
    created_at: str
    preset: Literal['smoke', 'full']
    status: Literal['QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED', 'INTERRUPTED']
    cancel_requested: bool
    frozen: dict
    identity: dict
    stage: str
    stage_completed: int
    stage_total: int
    completed_samples: int
    history: list[dict]
    arms: dict
    comparison: dict | None
    error: str | None
    error_type: str | None = None
    execution_source: str
    seed_graph: dict
    cases: list[dict]
    learning_cost: dict
    budget_usage: dict
    retained_graph: dict | None = None
    evolution: dict | None = None
    contracts: dict | None = None
    stale: bool = False


class LabCatalog(BaseModel):
    dataset: dict
    plans: dict
    seed_graph: dict
    model_configured: bool
    current_identity: dict


class JobList(BaseModel):
    jobs: list[dict]


class SampleList(BaseModel):
    samples: list[dict]


def checked(action):
    try:
        return action()
    except LabError as error:
        raise HTTPException(error.status, detail={'code': error.code, 'message': error.message}) from error


def write_origin(request):
    origin = request.headers.get('origin')
    allowed = {str(request.base_url).rstrip('/'), *os.getenv('LINGCART_WEB_ORIGINS',
               'http://127.0.0.1:5173,http://localhost:5173').split(',')}
    if origin is not None and origin not in allowed:
        raise HTTPException(403, detail={'code': 'PG_ORIGIN_DENIED', 'message': '实验写操作不接受此网页来源'})


@router.get('/catalog', response_model=LabCatalog)
def pg_catalog():
    return catalog()


@router.get('/jobs', response_model=JobList)
def pg_jobs():
    return {'jobs': [{k: j[k] for k in ('job_id', 'created_at', 'preset', 'status', 'stage', 'completed_samples')}
                     for j in JobStore().list()]}


@router.post('/jobs', response_model=JobResponse, status_code=202)
def pg_start(payload: StartExperiment, request: Request):
    write_origin(request)
    return checked(lambda: JobStore().create(payload))


@router.get('/jobs/{job_id}', response_model=JobResponse)
def pg_job(job_id: str):
    return checked(lambda: detail(job_id))


@router.post('/jobs/{job_id}/cancel', response_model=JobResponse)
def pg_cancel(job_id: str, request: Request):
    write_origin(request)
    return checked(lambda: JobStore().cancel(job_id))


def sample_files(store, job_id):
    store.get(job_id)
    return sorted((store.directory(job_id) / 'samples').glob('*/*.json'))


@router.get('/jobs/{job_id}/samples', response_model=SampleList)
def pg_samples(job_id: str):
    def read():
        store = JobStore()
        records = []
        for path in sample_files(store, job_id):
            sample = json.loads(path.read_text())
            evidence = sample.get('pg_evidence', {}).get('summary', {})
            records.append(dict(stage=path.parent.name, case_id=sample['case_id'], repetition=sample['repetition'],
                passed=sample['passed'], failures=[f for t in sample.get('turns', []) for f in t.get('failures', [])],
                interventions=evidence.get('interventions', 0), failure_stage=evidence.get('failure_stage')))
        return {'samples': records}
    return checked(read)


@router.get('/jobs/{job_id}/samples/{stage}/{case_id}/{repetition}')
def pg_sample(job_id: str, stage: str, case_id: str, repetition: int):
    def read():
        store = JobStore()
        job = store.get(job_id)
        stages = r'(seed-validation|iteration-[0-9]{2}-(training|validation)|test-(baseline|seed|evolved))'
        if (not re.fullmatch(stages, stage) or case_id not in sum(job['frozen']['groups'].values(), [])
                or not 1 <= repetition <= job['frozen']['spec']['repeats']):
            raise LabError('PG_SAMPLE_NOT_FOUND', '该样本尚未完成或不存在', 404)
        path = store.directory(job_id) / 'samples' / stage / f'{case_id}-{repetition}.json'
        if not path.is_file() or path.is_symlink() or path.parent.is_symlink():
            raise LabError('PG_SAMPLE_NOT_FOUND', '该样本尚未完成或不存在', 404)
        return json.loads(path.read_text())
    return checked(read)
