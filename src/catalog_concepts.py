"""Versioned semantic concepts and lossless codecs for catalog predicates.

The model may identify *what* a user means, but it must not invent the physical
catalog representation.  This module is the trusted adapter between those two
layers.  A reference is useful only when its registry revision, operator,
canonical value and observed catalog representation all agree.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


CATALOG_CONCEPT_SCHEMA_REVISION = "demo-catalog/product-concepts/v1"
CATALOG_CONCEPT_REGISTRY_ID = "lingcart.catalog-concepts"

PredicateOperator = Literal["eq", "ne", "contains", "not_contains", "lte", "gte"]
Scalar = str | float | int


class CatalogConceptError(ValueError):
    """A concept cannot be resolved without changing or guessing its meaning."""


class CatalogConceptRef(BaseModel):
    """Frozen semantic reference emitted during intent declaration."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)

    schema_revision: str = Field(min_length=1, max_length=100)
    concept_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{1,79}$")
    operator: PredicateOperator
    canonical_value: Scalar

    @model_validator(mode="after")
    def registered_reference(self):
        get_catalog_concept_registry().validate_reference(self)
        return self


class ResolvedCatalogPredicate(BaseModel):
    """Executable predicate produced only by a trusted registry codec."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)

    field: str = Field(pattern=r"^(category|specs\.[^.]+)$")
    op: PredicateOperator
    value: Scalar
    codec_id: str
    schema_revision: str


class ValueCodec(Protocol):
    codec_id: str

    def accepts(self, canonical_value: Scalar) -> bool: ...

    def encode(self, canonical_value: Scalar) -> Scalar: ...

    def validate_observations(self, values: tuple[Scalar, ...]) -> None: ...

    def manifest(self) -> dict: ...


@dataclass(frozen=True)
class EnumCodec:
    codec_id: str
    values: tuple[tuple[str, str], ...]

    def accepts(self, canonical_value: Scalar) -> bool:
        return isinstance(canonical_value, str) and canonical_value in dict(self.values)

    def encode(self, canonical_value: Scalar) -> Scalar:
        if not self.accepts(canonical_value):
            raise CatalogConceptError(f"codec {self.codec_id} 不接受 canonical_value={canonical_value!r}")
        return dict(self.values)[canonical_value]

    def validate_observations(self, values: tuple[Scalar, ...]) -> None:
        encoded = set(dict(self.values).values())
        if not values or any(not isinstance(value, str) or value not in encoded for value in values):
            raise CatalogConceptError(f"codec {self.codec_id} 与已观察目录值不兼容")

    def manifest(self) -> dict:
        return {"codec_id": self.codec_id, "kind": "enum", "values": dict(self.values)}


@dataclass(frozen=True)
class QuantityCodec:
    codec_id: str
    unit: str
    stored_pattern: str
    allow_stored_number: bool = False

    def accepts(self, canonical_value: Scalar) -> bool:
        return (type(canonical_value) in {int, float} and math.isfinite(canonical_value)
                and canonical_value >= 0)

    def encode(self, canonical_value: Scalar) -> Scalar:
        if not self.accepts(canonical_value):
            raise CatalogConceptError(f"codec {self.codec_id} 只接受非负有限数值")
        return canonical_value

    def validate_observations(self, values: tuple[Scalar, ...]) -> None:
        if not values:
            raise CatalogConceptError(f"codec {self.codec_id} 缺少已观察目录值")
        pattern = re.compile(self.stored_pattern)
        for value in values:
            if self.allow_stored_number and type(value) in {int, float} and math.isfinite(value):
                continue
            if not isinstance(value, str) or pattern.fullmatch(value) is None:
                raise CatalogConceptError(f"codec {self.codec_id} 无法无损解释已观察目录值 {value!r}")

    def manifest(self) -> dict:
        return {"codec_id": self.codec_id, "kind": "quantity", "unit": self.unit,
                "stored_pattern": self.stored_pattern, "allow_stored_number": self.allow_stored_number}


@dataclass(frozen=True)
class ConceptDefinition:
    concept_id: str
    description: str
    field: str
    operators: tuple[PredicateOperator, ...]
    codec: ValueCodec

    def manifest(self) -> dict:
        return {"concept_id": self.concept_id, "description": self.description,
                "field": self.field, "operators": list(self.operators),
                "codec": self.codec.manifest()}


class CatalogConceptRegistry:
    """Immutable registry whose identity covers every concept and codec rule."""

    def __init__(self, definitions: tuple[ConceptDefinition, ...], *,
                 registry_id: str = CATALOG_CONCEPT_REGISTRY_ID,
                 schema_revision: str = CATALOG_CONCEPT_SCHEMA_REVISION):
        if not definitions or len({item.concept_id for item in definitions}) != len(definitions):
            raise ValueError("catalog concept IDs must be nonempty and unique")
        if len({item.codec.codec_id for item in definitions}) != len(definitions):
            raise ValueError("each catalog concept must own a stable codec identity")
        self._definitions = {item.concept_id: item for item in definitions}
        self.registry_id = registry_id
        self.schema_revision = schema_revision
        encoded = json.dumps(self.manifest(include_fingerprint=False), ensure_ascii=False,
                             sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def manifest(self, *, include_fingerprint: bool = True) -> dict:
        result = {"registry_id": self.registry_id, "schema_revision": self.schema_revision,
                  "concepts": [self._definitions[key].manifest() for key in sorted(self._definitions)]}
        if include_fingerprint:
            result["fingerprint"] = getattr(self, "fingerprint", None)
        return result

    def prompt_catalog(self) -> dict:
        """Compact semantic menu; physical mappings remain a kernel concern."""
        concepts = []
        for identity in sorted(self._definitions):
            definition = self._definitions[identity]
            codec = definition.codec.manifest()
            concepts.append({"concept_id": identity, "meaning": definition.description,
                             "operators": list(definition.operators),
                             "canonical_values": sorted(codec.get("values", {})) or
                                                 ({"type": "number", "unit": codec.get("unit")} if codec.get("kind") == "quantity" else [])})
        return {"schema_revision": self.schema_revision, "concepts": concepts}

    def validate_reference(self, reference: CatalogConceptRef) -> ConceptDefinition:
        if reference.schema_revision != self.schema_revision:
            raise CatalogConceptError(
                f"catalog concept schema revision {reference.schema_revision!r} is not {self.schema_revision!r}")
        definition = self._definitions.get(reference.concept_id)
        if definition is None:
            raise CatalogConceptError(f"unknown catalog concept {reference.concept_id!r}")
        if reference.operator not in definition.operators:
            raise CatalogConceptError(
                f"operator {reference.operator!r} is not valid for {reference.concept_id!r}")
        if not definition.codec.accepts(reference.canonical_value):
            raise CatalogConceptError(
                f"canonical value {reference.canonical_value!r} is not valid for {reference.concept_id!r}")
        return definition

    def resolve(self, reference: CatalogConceptRef, observed_values: tuple[Scalar, ...]) -> ResolvedCatalogPredicate:
        definition = self.validate_reference(reference)
        definition.codec.validate_observations(observed_values)
        return self.encode(reference)

    def encode(self, reference: CatalogConceptRef) -> ResolvedCatalogPredicate:
        """Return the registry encoding; callers still need observed-value validation."""
        definition = self.validate_reference(reference)
        return ResolvedCatalogPredicate(field=definition.field, op=reference.operator,
            value=definition.codec.encode(reference.canonical_value), codec_id=definition.codec.codec_id,
            schema_revision=self.schema_revision)

    def field_for(self, reference: CatalogConceptRef) -> str:
        return self.validate_reference(reference).field

    def constrain_schema(self, schema: dict) -> dict:
        """Constrain generated JSON Schema without pretending cross-field JSON Schema support."""
        concept_ids = sorted(self._definitions)
        operators = sorted({operator for item in self._definitions.values() for operator in item.operators})

        def visit(value):
            if isinstance(value, list):
                for item in value:
                    visit(item)
                return
            if not isinstance(value, dict):
                return
            properties = value.get("properties", {})
            if {"schema_revision", "concept_id", "operator", "canonical_value"} <= set(properties):
                properties["schema_revision"]["enum"] = [self.schema_revision]
                properties["concept_id"]["enum"] = concept_ids
                properties["operator"]["enum"] = operators
            for item in value.values():
                visit(item)

        visit(schema)
        return schema


_REGISTRY = CatalogConceptRegistry((
    ConceptDefinition("audio.battery_hours", "耳机综合续航小时数", "specs.综合续航", ("gte", "lte", "eq"),
        QuantityCodec("hours-string/v1", "hour", r"\s*\d+(?:\.\d+)?\s*小时\s*")),
    ConceptDefinition("audio.noise_cancellation", "耳机降噪模式", "specs.降噪",
        ("eq", "ne", "contains", "not_contains"), EnumCodec("audio-noise-mode/v1", (
            ("none", "无"), ("active", "主动降噪"), ("call", "通话降噪")))),
    ConceptDefinition("audio.wearing_style", "耳机佩戴方式", "specs.佩戴方式",
        ("eq", "ne", "contains", "not_contains"), EnumCodec("audio-wearing-style/v1", (
            ("in_ear", "入耳式"), ("over_ear", "头戴式"), ("open_ear", "开放式")))),
    ConceptDefinition("hub.port_count", "扩展坞接口数量", "specs.接口数量", ("gte", "lte", "eq"),
        QuantityCodec("integer-scalar/v1", "count", r"\d+", allow_stored_number=True)),
    ConceptDefinition("hub.video_output", "扩展坞视频输出规格", "specs.视频输出",
        ("eq", "ne", "contains", "not_contains"), EnumCodec("hub-video-output/v1", (
            ("none", "无"), ("4k_30hz", "4K 30Hz"), ("4k_60hz", "4K 60Hz"),
            ("dual_hdmi_4k_60hz", "双 HDMI 4K 60Hz")))),
    ConceptDefinition("keyboard.connection_mode", "键盘连接方式", "specs.连接方式",
        ("eq", "ne", "contains", "not_contains"), EnumCodec("keyboard-connection-mode/v1", (
            ("wired", "有线"), ("dual_mode", "双模"), ("tri_mode", "三模")))),
    ConceptDefinition("keyboard.key_count", "键盘按键数量", "specs.键数", ("gte", "lte", "eq"),
        QuantityCodec("integer-scalar/v2", "key", r"\d+", allow_stored_number=True)),
    ConceptDefinition("keyboard.switch_type", "键盘轴体类型", "specs.轴体",
        ("eq", "ne", "contains", "not_contains"), EnumCodec("keyboard-switch-type/v1", (
            ("red", "红轴"), ("tea", "茶轴"), ("silent", "静音轴"),
            ("magnetic", "磁轴"), ("low_profile", "矮轴")))),
))


def get_catalog_concept_registry() -> CatalogConceptRegistry:
    return _REGISTRY

