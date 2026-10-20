from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Event, Lock

import pytest

from src.catalog_metadata import (
    CATALOG_FACETS_DEPENDENCY,
    CatalogMetadataProvider,
)
from src.capability_registry import get_capability_registry
from src.errors import AgentError
from src.intake import (
    DIRECT,
    ORDER_STATUS,
    PRODUCT_COMPARE,
    PRODUCT_DISCOVERY,
    PRODUCT_SEARCH,
    analyze_request,
)


def _facets(*categories: str) -> dict:
    return {
        "categories": list(categories),
        "category_aliases": {category: [] for category in categories},
        "search_aliases": {},
        "price_range": {"min": 100, "max": 900},
        "spec_keys": {category: [] for category in categories},
        "searchable_values": ["主动降噪", "通勤"],
        "available_products": len(categories),
        "data_source": "TEST_CATALOG",
    }


def test_non_catalog_intake_never_resolves_catalog_metadata(monkeypatch):
    def unexpected_read(_self):
        raise AssertionError("non-catalog intake must not read catalog metadata")

    monkeypatch.setattr(CatalogMetadataProvider, "get_snapshot", unexpected_read)

    for message in (
        "你是谁",
        "现在几点",
        "计算 (18 + 6) / 3",
        "查询订单 1004 的当前状态",
    ):
        result = analyze_request(message)
        assert result.request == message


def test_product_intake_resolves_declared_metadata_dependency(monkeypatch):
    snapshot = CatalogMetadataProvider(
        lambda: _facets("耳机"),
        source="TEST_CATALOG",
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    ).get_snapshot()
    reads = []

    def metadata_snapshot(self, objective_kind, dependency):
        reads.append((objective_kind, dependency))
        return snapshot

    monkeypatch.setattr(type(get_capability_registry()), "metadata_snapshot", metadata_snapshot)

    result = analyze_request("推荐 500 元以内的通勤降噪耳机")

    assert result.execution_mode == DIRECT
    assert result.intents == (PRODUCT_SEARCH,)
    assert result.entities == {"category": "耳机", "query": "通勤 降噪", "max_price": 500.0}
    assert reads == [(PRODUCT_SEARCH, CATALOG_FACETS_DEPENDENCY)]


def test_product_capabilities_declare_catalog_metadata_and_others_cannot_read_it():
    registry = get_capability_registry()
    specifications = {item["objective_kind"]: item for item in registry.prompt_catalog()}

    for kind in (PRODUCT_SEARCH, PRODUCT_DISCOVERY, PRODUCT_COMPARE):
        assert specifications[kind]["metadata_dependencies"] == [CATALOG_FACETS_DEPENDENCY]
    assert specifications[ORDER_STATUS]["metadata_dependencies"] == []
    with pytest.raises(AgentError) as error:
        registry.metadata_snapshot(ORDER_STATUS, CATALOG_FACETS_DEPENDENCY)
    assert error.value.code == "UNDECLARED_METADATA_DEPENDENCY"


def test_provider_cache_hit_and_explicit_invalidation_observe_data_change():
    state = _facets("耳机")
    loads = []
    timestamps = iter((
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=UTC),
    ))

    def load():
        loads.append(tuple(state["categories"]))
        return state

    provider = CatalogMetadataProvider(
        load,
        source="TEST_CATALOG",
        clock=lambda: next(timestamps),
    )

    first = provider.get_snapshot()
    cached = provider.get_snapshot()
    state["categories"].append("显示器")

    assert cached is first
    assert loads == [("耳机",)]
    assert first.source == "TEST_CATALOG"
    assert first.version.endswith(first.fingerprint[:16])
    assert first.generated_at == "2026-01-01T00:00:00+00:00"
    mutated_copy = first.facets
    mutated_copy["categories"].append("不能污染缓存")
    assert first.facets["categories"] == ["耳机"]

    assert provider.invalidate() is first
    changed = provider.get_snapshot()

    assert loads == [("耳机",), ("耳机", "显示器")]
    assert changed.facets["categories"] == ["耳机", "显示器"]
    assert changed.fingerprint != first.fingerprint
    assert changed.version != first.version
    assert changed.generated_at == "2026-01-02T00:00:00+00:00"


def test_provider_refresh_replaces_a_cached_snapshot_without_invalidation():
    state = _facets("耳机")
    provider = CatalogMetadataProvider(lambda: state, source="TEST_CATALOG")
    first = provider.get_snapshot()
    state["categories"].append("键盘")

    refreshed = provider.refresh()

    assert provider.get_snapshot() is refreshed
    assert refreshed.fingerprint != first.fingerprint
    assert refreshed.facets["categories"] == ["耳机", "键盘"]


def test_provider_serializes_concurrent_cache_misses():
    loader_started = Event()
    allow_loader_to_finish = Event()
    calls = 0
    calls_lock = Lock()

    def load():
        nonlocal calls
        with calls_lock:
            calls += 1
        loader_started.set()
        assert allow_loader_to_finish.wait(timeout=2)
        return _facets("耳机")

    provider = CatalogMetadataProvider(load, source="TEST_CATALOG")
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(provider.get_snapshot) for _ in range(8)]
        assert loader_started.wait(timeout=2)
        allow_loader_to_finish.set()
        snapshots = [future.result(timeout=2) for future in futures]

    assert calls == 1
    assert all(snapshot is snapshots[0] for snapshot in snapshots)
