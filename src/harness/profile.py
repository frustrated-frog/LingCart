"""Versioned, serializable descriptions of runnable harness configurations."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EngineRef(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9._-]*$", min_length=1, max_length=80)
    version: str = Field(min_length=1, max_length=40)


class MechanismRef(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    slot: str = Field(pattern=r"^[a-z][a-z0-9._-]*$", min_length=1, max_length=80)
    id: str = Field(pattern=r"^[a-z][a-z0-9._-]*$", min_length=1, max_length=80)
    version: str = Field(min_length=1, max_length=40)
    configuration: dict[str, str | int | bool] = Field(default_factory=dict, max_length=16)

    @field_validator("configuration")
    @classmethod
    def configuration_uses_identifiers(cls, configuration):
        if any(not re.fullmatch(r"[a-z][a-z0-9_]*", key) for key in configuration):
            raise ValueError("机制配置名称必须使用英文标识")
        return configuration

    def canonical_payload(self) -> dict:
        payload = self.model_dump(mode="json")
        if not payload["configuration"]:
            payload.pop("configuration")
        return payload


class ResearchOrigin(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    status: Literal["baseline", "candidate", "promoted"]
    source_type: Literal["native", "paper", "hybrid"]
    paper_id: str | None = Field(default=None, max_length=160)
    hypothesis: str | None = Field(default=None, max_length=500)

    @field_validator("paper_id")
    @classmethod
    def normalize_paper_id(cls, value):
        return value or None

    @model_validator(mode="after")
    def research_claims_are_traceable(self):
        if self.source_type in {"paper", "hybrid"} and not self.paper_id:
            raise ValueError("论文或混合来源必须提供 paper_id")
        if self.status == "candidate" and not self.hypothesis:
            raise ValueError("候选机制必须声明可验证假设")
        return self


class HarnessProfile(BaseModel):
    """A reproducible selection of an engine and its active mechanisms."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1]
    profile_id: str = Field(pattern=r"^[a-z][a-z0-9-]*$", min_length=1, max_length=80)
    version: str = Field(min_length=1, max_length=40)
    display_name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=500)
    engine: EngineRef
    mechanisms: tuple[MechanismRef, ...] = ()
    capabilities: tuple[str, ...] = ()
    research: ResearchOrigin

    @field_validator("mechanisms")
    @classmethod
    def mechanisms_have_unique_slots(cls, mechanisms):
        slots = [item.slot for item in mechanisms]
        if len(slots) != len(set(slots)):
            raise ValueError("同一个扩展槽位不能配置多个机制")
        return mechanisms

    @field_validator("capabilities")
    @classmethod
    def capabilities_are_unique_identifiers(cls, capabilities):
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("能力标识不能重复")
        if any(not item or not item.replace("_", "").isalnum() or not item[0].isalpha() for item in capabilities):
            raise ValueError("能力必须使用英文标识")
        return capabilities

    def canonical_payload(self) -> dict:
        payload = self.model_dump(mode="json")
        payload["mechanisms"] = [item.canonical_payload() for item in self.mechanisms]
        return payload

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.canonical_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def provenance(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "profile_version": self.version,
            "profile_sha256": self.fingerprint,
            "engine": self.engine.model_dump(),
            "mechanisms": [item.canonical_payload() for item in self.mechanisms],
            "research": self.research.model_dump(),
        }

