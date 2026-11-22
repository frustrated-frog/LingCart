"""SOMEM 受限文档模型：原始记录不可变，组织结果与派生视图分离。"""
from __future__ import annotations
from collections import Counter
from datetime import date
import math
import re
from typing import Annotated, Literal
from pydantic import Field, field_validator, model_validator
from src.procedural_graph import FrozenModel, fingerprint
from src.errors import AgentError

Identifier = Annotated[str, Field(pattern=r'^[a-z][a-z0-9_-]{0,63}$')]


def validate_event_time(value):
    if value is not None:
        if not re.fullmatch(r'\d{4}(-\d{2}){0,2}', value):
            raise ValueError('时间必须保留年、月或日精度')
        date.fromisoformat(value + ('-01-01' if len(value) == 4 else '-01' if len(value) == 7 else ''))
    return value


class Source(FrozenModel):
    source_id: Identifier
    text: str = Field(min_length=1, max_length=8000)
    event_time: str | None = None
    _time = field_validator('event_time')(validate_event_time)


class Citation(FrozenModel):
    source_id: Identifier
    quote: str = Field(min_length=1, max_length=512)


class Statement(FrozenModel):
    text: str = Field(min_length=1, max_length=1000)
    event_time: str | None = None
    citations: tuple[Citation, ...] = Field(min_length=1, max_length=8)
    _time = field_validator('event_time')(validate_event_time)


class Paragraph(FrozenModel):
    paragraph_id: Identifier
    topic: str = Field(min_length=1, max_length=100)
    statements: tuple[Statement, ...] = Field(min_length=1, max_length=8)
    links: tuple[Identifier, ...] = Field(default=(), max_length=16)


class Organization(FrozenModel):
    paragraphs: tuple[Paragraph, ...] = Field(default=(), max_length=64)
    core: tuple[Identifier, ...] = Field(default=(), max_length=8)


class Memory(FrozenModel):
    schema_version: Literal[1] = 1
    memory_id: Identifier
    version: int = Field(ge=0)
    parent_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    split: Literal['dev'] = 'dev'
    origin: Literal['empty', 'fixture', 'model']
    sources: tuple[Source, ...] = Field(default=(), max_length=128)
    organization: Organization = Field(default_factory=Organization)
    creation_order: tuple[Identifier, ...] = Field(default=(), max_length=1024)
    writer_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')

    @model_validator(mode='after')
    def consistent(self):
        source = {s.source_id: s for s in self.sources}
        paragraphs = {p.paragraph_id: p for p in self.organization.paragraphs}
        if len(source) != len(self.sources) or len(paragraphs) != len(self.organization.paragraphs):
            raise ValueError('Source 与段落身份必须唯一')
        if len(set(self.creation_order)) != len(self.creation_order) or not set(paragraphs) <= set(self.creation_order):
            raise ValueError('段落创建顺序无效')
        if len(set(self.organization.core)) != len(self.organization.core) or not set(self.organization.core) <= paragraphs.keys():
            raise ValueError('Core 必须引用已存在段落')
        if sum(len(s.text) for key in self.organization.core for s in paragraphs[key].statements) > 4000:
            raise ValueError('Core 超过容量上限')
        for p in paragraphs.values():
            if len(set(p.links)) != len(p.links) or not set(p.links) <= paragraphs.keys() or p.paragraph_id in p.links:
                raise ValueError('段落链接必须可解析且不重复')
            for statement in p.statements:
                for citation in statement.citations:
                    original = source.get(citation.source_id)
                    if original is None or citation.quote not in original.text:
                        raise ValueError('引用必须准确对应原始记录')
                    if statement.event_time is not None and statement.event_time != original.event_time:
                        raise ValueError('不能提升时间精度或虚构事件时间')
        if self.origin == 'empty' and (self.sources or self.organization.paragraphs):
            raise ValueError('空记忆不能包含数据')
        if self.origin == 'model' and not self.writer_sha256:
            raise ValueError('模型组织必须绑定 Writer 身份')
        if len(self.model_dump_json()) > 200000:
            raise ValueError('记忆产物超过容量上限')
        return self

    @property
    def fingerprint(self):
        return fingerprint(self.model_dump(mode='json'))


def organize(base, new_sources, proposal, *, origin, writer_sha256=None):
    """先构造完整候选再校验；不修改输入，也不允许删除或重写原始记录。"""
    old = {s.source_id: s for s in base.sources}
    additions = []
    for source in new_sources:
        if source.source_id in old:
            if source != old[source.source_id]:
                raise ValueError('原始记录不可改写')
        else:
            additions.append(source)
            old[source.source_id] = source
    current = {p.paragraph_id for p in base.organization.paragraphs}
    if any(p.paragraph_id in base.creation_order and p.paragraph_id not in current for p in proposal.paragraphs):
        raise ValueError('已删除段落身份不能重新分配')
    order = (*base.creation_order, *(p.paragraph_id for p in proposal.paragraphs if p.paragraph_id not in base.creation_order))
    candidate = Memory(memory_id=base.memory_id, version=base.version + 1, parent_sha256=base.fingerprint,
        origin=origin, sources=(*base.sources, *additions), organization=proposal, creation_order=order,
        writer_sha256=writer_sha256)
    return Memory.model_validate_json(candidate.model_dump_json())


def views(memory):
    paragraphs = {p.paragraph_id: p for p in memory.organization.paragraphs}
    graph = {key: dict(links=list(p.links), backlinks=[]) for key, p in paragraphs.items()}
    timeline = {}
    for key, p in paragraphs.items():
        for link in p.links:
            graph[link]['backlinks'].append(key)
        for statement in p.statements:
            if statement.event_time:
                timeline.setdefault(statement.event_time, []).append(dict(paragraph_id=key, **statement.model_dump(mode='json')))
    return dict(graph=graph, timeline=timeline, recent=[key for key in reversed(memory.creation_order) if key in paragraphs][:5])


def terms(text):
    # 英文词与中文双字片段；确定性 BM25，不依赖远端 embedding。
    result = re.findall(r'[a-z0-9_]+', text.casefold())
    for phrase in re.findall(r'[\u4e00-\u9fff]+', text):
        result.extend(phrase[i:i + 2] for i in range(max(1, len(phrase) - 1)))
    return result


def search(memory, query):
    docs = [(f'somem:paragraph/{p.paragraph_id}', ' '.join(s.text for s in p.statements), p.topic)
            for p in memory.organization.paragraphs]
    docs += [(f'somem:source/{s.source_id}', s.text, 'Source') for s in memory.sources]
    if not docs:
        return []
    counters = [Counter(terms(text)) for _, text, _ in docs]
    lengths = [sum(c.values()) for c in counters]
    average = sum(lengths) / len(docs) or 1
    scores = [0.0] * len(docs)
    for word in set(terms(query)):
        frequency = sum(word in c for c in counters)
        idf = math.log(1 + (len(docs) - frequency + .5) / (frequency + .5))
        for index, counts in enumerate(counters):
            tf = counts[word]
            scores[index] += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * lengths[index] / average))
    return [dict(reference=docs[i][0], topic=docs[i][2], excerpt=docs[i][1][:300], score=round(scores[i], 6))
            for i in sorted(range(len(docs)), key=lambda i: (-scores[i], docs[i][0])) if scores[i] > 0]


def read(memory, reference, *, offset=0, limit=1):
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 3:
        raise AgentError('SOMEM_INVALID_PAGE', '记忆分页参数无效', retryable=False)
    if not isinstance(reference, str) or len(reference) > 600 or not reference.startswith('somem:'):
        raise AgentError('SOMEM_REFERENCE_INVALID', '记忆地址无效', retryable=False)
    address = reference[6:]
    derived = views(memory)
    paragraphs = {p.paragraph_id: p for p in memory.organization.paragraphs}
    rows = None
    if address == 'inventory':
        rows = [dict(reference=f'somem:paragraph/{key}', topic=p.topic) for key, p in paragraphs.items()]
    elif address in {'core', 'recent'}:
        rows = [paragraphs[key].model_dump(mode='json') for key in (memory.organization.core if address == 'core' else derived['recent'])]
    elif address.startswith('search/') and address[7:].strip():
        rows = search(memory, address[7:])
    elif address.startswith('paragraph/') and address[10:] in paragraphs:
        p = paragraphs[address[10:]]
        rows = [dict(paragraph_id=p.paragraph_id, topic=p.topic, links=list(p.links), **s.model_dump(mode='json')) for s in p.statements]
    elif address.startswith('source/'):
        source = next((s for s in memory.sources if s.source_id == address[7:]), None)
        rows = [dict(source_id=source.source_id, event_time=source.event_time, text=source.text[start:start+1000],
                     char_start=start, char_end=min(start+1000, len(source.text)))
                for start in range(0, len(source.text), 1000)] if source else None
    elif address.startswith('graph/') and address[6:] in derived['graph']:
        rows = [dict(paragraph_id=address[6:], **derived['graph'][address[6:]])]
    elif address.startswith('timeline/'):
        rows = derived['timeline'].get(address[9:])
    elif address == 'timeline':
        rows = [dict(event_time=key, reference=f'somem:timeline/{key}') for key in sorted(derived['timeline'])]
    if rows is None:
        raise AgentError('SOMEM_NOT_FOUND', '记忆地址不存在', retryable=False)
    if offset > len(rows):
        raise AgentError('SOMEM_INVALID_PAGE', '记忆分页超出范围', retryable=False)
    end = min(len(rows), offset + limit)
    return dict(memory_only=True, authority='ARCHIVAL_UNTRUSTED', memory_sha256=memory.fingerprint,
                reference=reference, items=rows[offset:end], total=len(rows), next_offset=end if end < len(rows) else None,
                refreshed=False)
