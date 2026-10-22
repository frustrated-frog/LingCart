"""独立 SQLite 实验任务账本。幂等启动、单活动任务、心跳与中断证据。"""
from contextlib import contextmanager
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
import uuid

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from src.pg_experiment import ROOT, DATASET, SEED, challenge_dataset, plan
from src.procedural_graph import ProceduralGraph, canonical, fingerprint

ACTIVE = ('QUEUED', 'RUNNING')


class StartExperiment(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    request_id: str = Field(pattern=r'^[a-zA-Z0-9_-]{8,80}$')
    preset: Literal['smoke', 'full']


class LabError(Exception):
    def __init__(self, code, message, status=409):
        self.code, self.message, self.status = code, message, status
        super().__init__(message)


def identity():
    from eval.run_live_eval import environment_manifest
    from src.model_gateway import load_gateway_config, ModelGatewayConfigurationError
    manifest = environment_manifest(challenge_dataset(), 'adaptive', DATASET)
    models = {}
    for purpose in ('router', 'goal', 'planner', 'agent', 'final_response', 'liveplan_advisor'):
        try:
            config = load_gateway_config(purpose=purpose)
            models[purpose] = [{'provider': p.provider, 'model': p.model, 'thinking_mode': p.thinking_mode,
                                'reasoning_effort': p.reasoning_effort, 'strict_tools': p.strict_tools,
                                'endpoint_sha256': fingerprint(p.base_url)} for p in config.providers]
        except ModelGatewayConfigurationError:
            models[purpose] = None
    return {key: manifest[key] for key in ('dataset_sha256', 'source_tree_sha256', 'dependency_lock_sha256',
            'evaluation_protocol_sha256', 'business_protocol_sha256', 'python_runtime')} | {
            'purpose_models': models, 'seed_sha256': ProceduralGraph.from_path(SEED).fingerprint,
            'plan_sha256': fingerprint(plan('full'))}


class JobStore:
    def __init__(self, root=None):
        self.root = Path(root or os.getenv('LINGCART_PG_LAB_DIR', str(ROOT / 'output/pg-lab')))
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, request_id TEXT UNIQUE NOT NULL, '
                       'preset TEXT NOT NULL, status TEXT NOT NULL, heartbeat REAL NOT NULL, cancel INTEGER NOT NULL DEFAULT 0, '
                       'data TEXT NOT NULL)')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.root / 'jobs.db', timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def directory(self, job_id):
        if not re.fullmatch(r'pg-[0-9a-f]{32}', job_id):
            raise LabError('PG_JOB_NOT_FOUND', '实验不存在', 404)
        path = self.root / job_id
        if path.is_symlink():
            raise LabError('PG_INVALID_PATH', '实验产物路径无效', 400)
        return path

    def _expire(self, db):
        db.execute("UPDATE jobs SET status='INTERRUPTED' WHERE status IN ('QUEUED','RUNNING') AND heartbeat < ?",
                   (time.time() - 90,))

    def get(self, job_id):
        self.directory(job_id)
        with self.connect() as db:
            self._expire(db)
            row = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        if not row:
            raise LabError('PG_JOB_NOT_FOUND', '实验不存在', 404)
        return {**json.loads(row['data']), 'status': row['status'], 'cancel_requested': bool(row['cancel'])}

    def list(self):
        with self.connect() as db:
            self._expire(db)
            rows = db.execute('SELECT id FROM jobs ORDER BY rowid DESC LIMIT 50').fetchall()
        return [self.get(row['id']) for row in rows]

    def create(self, request, *, launch=True):
        # 重试先返回原任务；不因代码更新或模型暂时不可用而再次计费。
        with self.connect() as db:
            row = db.execute('SELECT id,preset FROM jobs WHERE request_id=?', (request.request_id,)).fetchone()
        if row:
            if row['preset'] != request.preset:
                raise LabError('PG_IDEMPOTENCY_CONFLICT', '同一请求不能更换实验配置')
            return self.get(row['id'])
        from src.llm import get_model_status
        if launch and not get_model_status().get('configured'):
            raise LabError('MODEL_NOT_CONFIGURED', '请先配置 Model Gateway，再启动真实实验', 422)
        frozen = plan(request.preset)
        job_id = 'pg-' + uuid.uuid4().hex
        data = dict(job_id=job_id, created_at=datetime.now(UTC).isoformat(), preset=request.preset,
                    frozen=frozen, identity=identity(), stage='queued', stage_completed=0, stage_total=0,
                    completed_samples=0, history=[], arms={}, comparison=None, error=None,
                    execution_source='live_provider', seed_graph=ProceduralGraph.from_path(SEED).model_dump(mode='json'),
                    cases=[{k: c[k] for k in ('id', 'learning_partition', 'scenario_family', 'category', 'description')}
                           for c in challenge_dataset()['cases']],
                    learning_cost={'rollout_model_attempts': 0, 'refiner_model_attempts': 0, 'rollout_tokens': None})
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            self._expire(db)
            # 并发相同请求也必须返回同一任务。
            row = db.execute('SELECT id,preset FROM jobs WHERE request_id=?', (request.request_id,)).fetchone()
            if row:
                if row['preset'] != request.preset:
                    raise LabError('PG_IDEMPOTENCY_CONFLICT', '同一请求不能更换实验配置')
                job_id = row['id']
                launch = False
            else:
                if db.execute("SELECT 1 FROM jobs WHERE status IN ('QUEUED','RUNNING')").fetchone():
                    raise LabError('PG_EXPERIMENT_BUSY', '已有实验执行中，请等待完成或停止后再启动')
                db.execute('INSERT INTO jobs(id,request_id,preset,status,heartbeat,data) VALUES (?,?,?,?,?,?)',
                    (job_id, request.request_id, request.preset, 'QUEUED', time.time(), canonical(data)))
        if launch:
            try:
                subprocess.Popen([sys.executable, '-m', 'eval.run_pg_lab', '--job-id', job_id], cwd=ROOT,
                    env={**os.environ, 'LINGCART_PG_LAB_DIR': str(self.root.resolve())},
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            except OSError:
                self.update(job_id, status='FAILED', error='PG_WORKER_START_FAILED')
        return self.get(job_id)

    def claim(self, job_id):
        with self.connect() as db:
            result = db.execute("UPDATE jobs SET status='RUNNING',heartbeat=? WHERE id=? AND status='QUEUED' AND cancel=0",
                                (time.time(), job_id))
            return result.rowcount == 1

    def update(self, job_id, *, status=None, **changes):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
            if not row or row['status'] not in ACTIVE:
                return False
            data = json.loads(row['data']) | changes
            db.execute('UPDATE jobs SET data=?,status=?,heartbeat=? WHERE id=?',
                       (canonical(data), status or row['status'], time.time(), job_id))
            return True

    def heartbeat(self, job_id):
        with self.connect() as db:
            db.execute("UPDATE jobs SET heartbeat=? WHERE id=? AND status='RUNNING'", (time.time(), job_id))

    def cancel(self, job_id):
        self.get(job_id)
        with self.connect() as db:
            db.execute("UPDATE jobs SET cancel=1,status=CASE WHEN status='QUEUED' THEN 'CANCELLED' ELSE status END "
                       "WHERE id=? AND status IN ('QUEUED','RUNNING')", (job_id,))
        return self.get(job_id)


def catalog():
    from src.llm import get_model_status
    return dict(dataset=challenge_dataset(), plans={key: plan(key) for key in ('smoke', 'full')},
                seed_graph=ProceduralGraph.from_path(SEED).model_dump(mode='json'),
                model_configured=bool(get_model_status().get('configured')), current_identity=identity())


def detail(job_id, store=None):
    store = store or JobStore()
    job = store.get(job_id)
    job['stale'] = job['identity'] != identity()
    return job
