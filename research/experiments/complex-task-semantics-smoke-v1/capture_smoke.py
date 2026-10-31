"""Run real evaluator unchanged, and retain its demo SQLite after the two turns for UI inspection."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from contextlib import contextmanager
import hashlib
import sqlite3
from eval import run_live_eval as live
original = live.offline._isolated_db
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--label', required=True)
args = parser.parse_args()
if not args.label.replace('-', '').isalnum():
    raise ValueError('label must contain only letters, digits and hyphens')
snapshot = ROOT / f'output/complex-task-semantics/{args.label}.db'
snapshot.parent.mkdir(parents=True, exist_ok=True)
if snapshot.exists():
    raise RuntimeError('Refusing to replace a previous smoke database')
@contextmanager
def captured(runtime, fixture):
    with original(runtime, fixture) as value:
        try:
            yield value
        finally:
            with runtime['tools']._engine.connect() as source, sqlite3.connect(snapshot) as target:
                source.connection.driver_connection.backup(target)
live.offline._isolated_db = captured
path = ROOT / 'research/experiments/complex-task-semantics-smoke-v1/raw/followup-cases.json'
report = live.evaluate(live.load_dataset(path), repeats=1, split='dev', execution_mode='adaptive',
                      dataset_path=path, max_model_calls_per_run=16)
report['manifest']['database_capture_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
report['manifest']['retained_demo_database'] = str(snapshot.relative_to(ROOT))
output = path.with_name(f'{args.label}-live-report.json')
if output.exists():
    raise RuntimeError('Refusing to replace a previous smoke report')
output.write_text(live._report_json_dumps(report, indent=2)+'\n')
for sample in report['samples']:
    for turn in sample['turns']:
        r=turn['result']
        print(sample['case_id'],r['run_id'],r['status'],r.get('terminal_reason'),turn['failures'])
