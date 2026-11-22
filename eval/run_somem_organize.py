"""离线组织记忆；默认只验证。--live 显式调用模型，--proposal 用于离线夹具。"""
import argparse
import json
from pathlib import Path
from pydantic import Field
from src.somem import Memory, Source, Organization, organize, views
from src.procedural_graph import FrozenModel, fingerprint
from typing import Literal
from src.pg_learning import write_artifact
from src.somem_learning import ModelOrganizer


class SourceBatch(FrozenModel):
    split: Literal['dev']
    sources: tuple[Source, ...] = Field(min_length=1, max_length=128)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('research/memories/somem-empty-v1.json'))
    parser.add_argument('--sources', type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--live', action='store_true')
    group.add_argument('--proposal', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    base = Memory.model_validate_json(args.base.read_text())
    sources = SourceBatch.model_validate_json(args.sources.read_text()).sources
    # 在计费前检查 Source 不变性与容量；空记忆进入 fixture 仅用于校验。
    organize(base, sources, base.organization, origin='fixture')
    if not args.live and not args.proposal:
        print('VALIDATED：来源与容量已校验，未调用模型；尚未生成主题。')
        return 0
    if args.output_dir is None:
        parser.error('组织记忆须提供新的 --output-dir，不覆盖已有版本')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    write_artifact(output / 'source-input.json', dict(base=base.model_dump(mode='json'), sources=[s.model_dump(mode='json') for s in sources]))
    response = ModelOrganizer().propose(base, sources) if args.live else dict(proposal=args.proposal.read_text(),
        error=None, events=[], writer_sha256=fingerprint(args.proposal.read_text()))
    write_artifact(output / 'writer.json', response)
    try:
        if response.get('error'):
            raise ValueError(response['error'])
        proposal = Organization.model_validate_json(response['proposal'])
        memory = organize(base, sources, proposal, origin='model' if args.live else 'fixture', writer_sha256=response['writer_sha256'])
        from src.harness.mechanisms import ROOT
        profile = None
        memory_path = (output / 'memory.json').resolve()
        if memory_path.is_relative_to((ROOT / 'research/memories').resolve()):
            from src.harness.profile import HarnessProfile
            profile = json.loads((ROOT / 'configs/harness/adaptive-somem-context-v1.json').read_text())
            profile['profile_id'] = 'somem-' + memory.fingerprint[:16]
            next(m for m in profile['mechanisms'] if m['slot'] == 'context')['configuration']['memory_path'] = memory_path.relative_to(ROOT).as_posix()
            profile = HarnessProfile.model_validate_json(json.dumps(profile)).model_dump(mode='json')
        write_artifact(output / 'memory.json', memory.model_dump(mode='json'))
        if profile:
            write_artifact(output / 'harness-profile.json', profile)
        write_artifact(output / 'views.json', views(memory))
        result = dict(status='ACCEPTED', memory_sha256=memory.fingerprint, parent_sha256=base.fingerprint,
                      execution_source='live_provider' if args.live else 'test_harness', effect_evaluated=False)
    except (ValueError, TypeError, KeyError) as error:
        result = dict(status='REJECTED', code=response.get('error') or 'SOMEM_INVALID_ORGANIZATION',
                      error_type=type(error).__name__, retained_sha256=base.fingerprint, effect_evaluated=False)
    write_artifact(output / 'report.json', result)
    print(result['status'])
    return 0 if result['status'] == 'ACCEPTED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
