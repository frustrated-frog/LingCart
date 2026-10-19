"""Trusted entity normalization with explicit provenance.

The dialogue model may identify an entity mention, but canonical values used by
tools are resolved here.  This keeps model interpretation separate from
deterministic validation and makes every normalization inspectable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.errors import AgentError


@dataclass(frozen=True)
class ResolvedEntity:
    name: str
    raw_text: str
    normalized_value: str
    resolver: str
    resolver_version: str
    source_quote: str
    confidence: float

    def payload(self) -> dict:
        return asdict(self)


# This is entity metadata, not an intent taxonomy.  Aliases may grow without
# changing routing or tool code.  IANA identifiers are accepted directly.
_TIMEZONE_ALIASES: dict[str, str] = {
    "上海": "Asia/Shanghai",
    "北京": "Asia/Shanghai",
    "中国": "Asia/Shanghai",
    "北京时间": "Asia/Shanghai",
    "shanghai": "Asia/Shanghai",
    "beijing": "Asia/Shanghai",
    "纽约": "America/New_York",
    "纽约时间": "America/New_York",
    "new york": "America/New_York",
    "洛杉矶": "America/Los_Angeles",
    "los angeles": "America/Los_Angeles",
    "伦敦": "Europe/London",
    "london": "Europe/London",
    "巴黎": "Europe/Paris",
    "paris": "Europe/Paris",
    "东京": "Asia/Tokyo",
    "tokyo": "Asia/Tokyo",
    "首尔": "Asia/Seoul",
    "seoul": "Asia/Seoul",
    "新加坡": "Asia/Singapore",
    "singapore": "Asia/Singapore",
    "悉尼": "Australia/Sydney",
    "sydney": "Australia/Sydney",
    "迪拜": "Asia/Dubai",
    "dubai": "Asia/Dubai",
}


def _valid_iana(value: str) -> bool:
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError):
        return False
    return True


def resolve_timezone(raw_value: object, source_quote: str) -> ResolvedEntity | None:
    """Resolve a time-zone mention to an IANA identifier.

    ``None`` means no location was stated and the capability may use its
    documented runtime default.  Conflicting or unknown explicit entities are
    rejected rather than guessed.
    """

    text = source_quote.strip()
    folded = text.casefold()
    matches = [(alias, zone) for alias, zone in _TIMEZONE_ALIASES.items() if alias.casefold() in folded]
    zones = {zone for _, zone in matches}
    if len(zones) > 1:
        raise AgentError("AMBIGUOUS_TIMEZONE", "请求中包含多个时区，请明确要查询哪一个", retryable=False)

    raw = str(raw_value).strip() if raw_value not in (None, "") else ""
    if raw and _valid_iana(raw):
        if zones and raw not in zones:
            raise AgentError("UNGROUNDED_PLAN_ENTITY", "模型给出的时区与用户地点不一致", retryable=False)
        if raw.casefold() not in folded and not zones:
            raise AgentError("UNGROUNDED_PLAN_ENTITY", "IANA 时区必须来自用户原文或受信地点解析", retryable=False)
        alias = next((name for name, zone in matches if zone == raw), raw)
        return ResolvedEntity("timezone", alias, raw, "trusted_timezone", "1", source_quote, 1.0)

    if raw:
        direct = _TIMEZONE_ALIASES.get(raw.casefold()) or _TIMEZONE_ALIASES.get(raw)
        if direct is None:
            raise AgentError("INVALID_TIMEZONE_ENTITY", f"无法把地点“{raw}”解析为受支持的 IANA 时区", retryable=False)
        if raw.casefold() not in folded:
            raise AgentError("UNGROUNDED_PLAN_ENTITY", "时区地点必须来自用户原文", retryable=False)
        return ResolvedEntity("timezone", raw, direct, "trusted_timezone", "1", source_quote, 1.0)

    if zones:
        zone = next(iter(zones))
        alias = next(name for name, candidate in matches if candidate == zone)
        return ResolvedEntity("timezone", alias, zone, "trusted_timezone", "1", source_quote, 0.99)
    return None


def normalize_clock_objective(objective: dict, sources: tuple[str, ...]) -> tuple[dict, dict[str, dict]]:
    normalized = dict(objective)
    quote = str(objective.get("source_quote") or "")
    try:
        resolved = resolve_timezone(objective.get("timezone"), quote)
    except AgentError as error:
        if error.code != "UNGROUNDED_PLAN_ENTITY" or not sources or sources[0] == quote:
            raise
        resolved = resolve_timezone(objective.get("timezone"), sources[0])
    if resolved is None and sources and sources[0] != quote:
        resolved = resolve_timezone(objective.get("timezone"), sources[0])
    if resolved is None:
        return normalized, {}
    normalized["timezone"] = resolved.normalized_value
    return normalized, {"timezone": resolved.payload()}
