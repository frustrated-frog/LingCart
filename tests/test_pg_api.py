import json
from fastapi.testclient import TestClient
import pytest
import api
from src.pg_jobs import JobStore, StartExperiment


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('LINGCART_PG_LAB_DIR', str(tmp_path))
    monkeypatch.setenv('LINGCART_ENGINEERING_KEY', 'pg-test-key')
    return TestClient(api.app, headers={'X-Engineering-Key': 'pg-test-key'})


def test_catalog_and_empty_state_are_real_and_private(client):
    payload = client.get('/engineering/pg/catalog').json()
    assert len(payload['dataset']['cases']) == 18
    assert payload['plans']['smoke']['max_model_attempts'] == 50
    assert payload['plans']['smoke']['max_samples'] == 6
    assert client.get('/engineering/pg/jobs').json() == {'jobs': []}
    assert TestClient(api.app).get('/engineering/pg/catalog').status_code == 403
    assert client.get('/engineering/pg/jobs/absent').json()['detail']['code'] == 'PG_JOB_NOT_FOUND'


def test_start_idempotency_errors_and_cross_origin_writes(client, monkeypatch):
    import src.llm
    monkeypatch.setattr(src.llm, 'get_model_status', lambda: {'configured': False})
    payload = {'request_id': 'api-experiment', 'preset': 'smoke'}
    assert client.post('/engineering/pg/jobs', json=payload).json()['detail']['code'] == 'MODEL_NOT_CONFIGURED'
    assert client.post('/engineering/pg/jobs', json={**payload, 'preset': 'arbitrary'}).status_code == 422
    assert client.post('/engineering/pg/jobs', json=payload, headers={'Origin': 'https://untrusted.test'}).status_code == 403
    job = JobStore().create(StartExperiment(**payload), launch=False)
    response = client.post('/engineering/pg/jobs', json=payload)
    assert response.status_code == 202 and response.json()['job_id'] == job['job_id']
    assert client.post('/engineering/pg/jobs', json={**payload, 'preset': 'full'}).status_code == 409
    assert client.post(f"/engineering/pg/jobs/{job['job_id']}/cancel").json()['status'] == 'CANCELLED'


def test_samples_remain_inspectable_after_interruption(client):
    store = JobStore()
    job = store.create(StartExperiment(request_id='partial-samples', preset='smoke'), launch=False)
    folder = store.directory(job['job_id']) / 'samples/seed-validation'
    folder.mkdir(parents=True)
    sample = dict(case_id='PG07', repetition=1, passed=False,
                  turns=[{'failures': ['脚本验证失败'], 'result': {'events': [{'type': 'test_fixture'}]}}])
    (folder / 'PG07-1.json').write_text(json.dumps(sample))
    store.update(job['job_id'], status='FAILED')
    prefix = f"/engineering/pg/jobs/{job['job_id']}"
    assert client.get(prefix).json()['status'] == 'FAILED'
    assert client.get(prefix + '/samples').json()['samples'][0]['failures'] == ['脚本验证失败']
    assert client.get(prefix + '/samples/seed-validation/PG07/1').json() == sample
    assert client.get(prefix + '/samples/seed-validation/PG99/1').status_code == 404


def test_credit_policy_freezes_dev_scope_and_is_part_of_idempotency(client):
    catalog = client.get('/engineering/pg/catalog').json()
    plan = catalog['learning_plans']['smoke']
    assert plan['max_samples'] == 3 and plan['max_model_attempts'] == 26
    assert plan['groups']['test'] == [] and plan['spec']['online_profile'] == 'adaptive-pg-context-v4'
    payload = {'request_id': 'credit-api-check', 'preset': 'smoke', 'evolution_policy': 'credit-memory-v1'}
    job = JobStore().create(StartExperiment(**payload), launch=False)
    assert client.post('/engineering/pg/jobs', json=payload).json()['job_id'] == job['job_id']
    assert client.post('/engineering/pg/jobs', json={**payload, 'evolution_policy': 'basic-v1'}).status_code == 409
    assert client.post('/engineering/pg/jobs', json={**payload, 'evolution_policy': 'unknown'}).status_code == 422
