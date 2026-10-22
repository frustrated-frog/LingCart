"""Policy-aware after-sales domain primitives."""

from src.after_sales.contracts import (
    ActorType,
    CaseCreateCommand,
    CaseStatus,
    CaseType,
    EvidenceCommand,
    PolicyEvaluation,
    PolicyOutcome,
    RequestedResolution,
)
from src.after_sales.policy import AfterSalesPolicyEngine
from src.after_sales.service import AfterSalesCaseService, AfterSalesDomainError

__all__ = [
    "ActorType",
    "AfterSalesCaseService",
    "AfterSalesDomainError",
    "AfterSalesPolicyEngine",
    "CaseCreateCommand",
    "CaseStatus",
    "CaseType",
    "EvidenceCommand",
    "PolicyEvaluation",
    "PolicyOutcome",
    "RequestedResolution",
]
