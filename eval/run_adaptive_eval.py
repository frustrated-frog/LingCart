"""独立的自主任务评测：20项、开发/测试各十项、默认三次，明确调用adaptive入口。"""
from pathlib import Path
import argparse
import sys
try:
    from eval import run_live_eval as live
except ModuleNotFoundError:
    import run_live_eval as live

DATASET = Path(__file__).with_name('adaptive_tasks.json')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--execution-mode', default='adaptive')
    known, _ = parser.parse_known_args(argv)
    if known.execution_mode != 'adaptive':
        raise ValueError('自主专项评测必须使用 adaptive，auto 请使用通用评测入口')
    if live.load_dataset(known.dataset).get('suite') != 'adaptive':
        raise ValueError('自主专项入口不能将通用任务集作为自主评测报告')
    if not any(a == '--dataset' or a.startswith('--dataset=') for a in argv):
        argv += ['--dataset', str(DATASET)]
    if not any(a == '--execution-mode' or a.startswith('--execution-mode=') for a in argv):
        argv += ['--execution-mode', 'adaptive']
    return live.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
