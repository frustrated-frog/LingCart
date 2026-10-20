"""Versioned, immutable catalog metadata snapshots for request understanding.

Catalog metadata is read-only routing context, not a business tool result.  The
provider keeps that boundary explicit: callers receive a content-addressed
snapshot and must opt in through a declared capability dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import json
from threading import RLock
from typing import Callable, Mapping


CATALOG_FACETS_DEPENDENCY = "catalog.facets"
CATALOG_METADATA_SCHEMA_VERSION = "catalog-metadata/v1"


@dataclass(frozen=True)
class CatalogMetadataSnapshot:
    """An immutable, auditable view of catalog language and schema metadata."""

    schema_version: str
    source: str
    version: str
    fingerprint: str
    generated_at: str
    _encoded_facets: str = field(repr=False, compare=False)

    @property
    def facets(self) -> dict:
        """Return an isolated copy so a consumer cannot mutate the cache."""

        return json.loads(self._encoded_facets)

    def audit_identity(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "version": self.version,
            "fingerprint": self.fingerprint,
            "generated_at": self.generated_at,
        }

    def to_dict(self) -> dict:
        return {**self.audit_identity(), "facets": self.facets}


class CatalogMetadataProvider:
    """Thread-safe, explicitly invalidated cache of catalog metadata.

    Reads and refreshes are serialized.  A failed refresh never replaces a
    previously valid snapshot, while ``invalidate`` guarantees that the next
    ``get_snapshot`` performs a physical load.
    """

    def __init__(
        self,
        loader: Callable[[], Mapping[str, object]],
        *,
        source: str,
        schema_version: str = CATALOG_METADATA_SCHEMA_VERSION,
        clock: Callable[[], datetime] | None = None,
    ):
        if not callable(loader):
            raise TypeError("catalog metadata loader must be callable")
        if not source.strip() or not schema_version.strip():
            raise ValueError("catalog metadata source and schema version are required")
        self._loader = loader
        self._source = source
        self._schema_version = schema_version
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = RLock()
        self._snapshot: CatalogMetadataSnapshot | None = None

    def get_snapshot(self) -> CatalogMetadataSnapshot:
        with self._lock:
            if self._snapshot is None:
                self._snapshot = self._load_snapshot()
            return self._snapshot

    def invalidate(self) -> CatalogMetadataSnapshot | None:
        """Drop the cached snapshot and return the value that was invalidated."""

        with self._lock:
            previous = self._snapshot
            self._snapshot = None
            return previous

    def refresh(self) -> CatalogMetadataSnapshot:
        """Load and atomically publish a fresh snapshot, even when cached."""

        with self._lock:
            refreshed = self._load_snapshot()
            self._snapshot = refreshed
            return refreshed

    def _load_snapshot(self) -> CatalogMetadataSnapshot:
        raw = self._loader()
        if not isinstance(raw, Mapping):
            raise TypeError("catalog metadata loader must return a mapping")
        try:
            encoded_facets = json.dumps(
                dict(raw),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("catalog metadata must be finite JSON data") from exc
        facets = json.loads(encoded_facets)
        source = str(facets.get("data_source") or self._source).strip()
        if not source:
            raise ValueError("catalog metadata snapshot has no source")
        identity_payload = json.dumps(
            {
                "schema_version": self._schema_version,
                "source": source,
                "facets": facets,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        fingerprint = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()
        generated_at = self._clock()
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=UTC)
        return CatalogMetadataSnapshot(
            schema_version=self._schema_version,
            source=source,
            version=f"{self._schema_version}:{fingerprint[:16]}",
            fingerprint=fingerprint,
            generated_at=generated_at.astimezone(UTC).isoformat(),
            _encoded_facets=encoded_facets,
        )
