"""Shared source-grounding primitives for model-authored structured values."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Iterable


_NUMBER_TOKEN = re.compile(
    r"(?<![\d.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\d)(?!\.\d)"
)

_CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CHINESE_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_CHINESE_BIG_UNITS = {"万": 10_000, "亿": 100_000_000}
_CHINESE_NUMBER_TOKEN = re.compile(r"[负+-]?[零〇一二两三四五六七八九十百千万亿]+(?:点[零〇一二两三四五六七八九]+)?")
# A single Chinese digit is numeric only when syntax around it makes the
# cardinal meaning explicit. This avoids treating the 一 in ordinary words
# such as “一致” as grounded numeric evidence.
_CHINESE_NUMBER_CLASSIFIERS = frozenset(
    "个件款只台套条种份本张把双对支颗枚位名次项目组批箱包瓶杯块片辆部间家年月天时分秒元角米克斤升度"
)


def _parse_chinese_integer(token: str) -> int | None:
    if not token:
        return None
    if not any(char in _CHINESE_SMALL_UNITS or char in _CHINESE_BIG_UNITS for char in token):
        if any(char not in _CHINESE_DIGITS for char in token):
            return None
        return int("".join(str(_CHINESE_DIGITS[char]) for char in token))

    total = 0
    section = 0
    number = 0
    for char in token:
        if char in _CHINESE_DIGITS:
            number = _CHINESE_DIGITS[char]
        elif char in _CHINESE_SMALL_UNITS:
            section += (number or 1) * _CHINESE_SMALL_UNITS[char]
            number = 0
        elif char in _CHINESE_BIG_UNITS:
            section += number
            total += (section or 1) * _CHINESE_BIG_UNITS[char]
            section = 0
            number = 0
        else:
            return None
    return total + section + number


def _parse_chinese_number(token: str) -> Decimal | None:
    sign = -1 if token[:1] in {"负", "-"} else 1
    if token[:1] in {"负", "-", "+"}:
        token = token[1:]
    integer, separator, fraction = token.partition("点")
    parsed_integer = _parse_chinese_integer(integer)
    if parsed_integer is None:
        return None
    value = Decimal(parsed_integer)
    if separator:
        if not fraction or any(char not in _CHINESE_DIGITS for char in fraction):
            return None
        decimal_text = "".join(str(_CHINESE_DIGITS[char]) for char in fraction)
        value += Decimal("0." + decimal_text)
    return value * sign


def _explicit_chinese_number(match: re.Match[str], source: str) -> bool:
    token = match.group(0).lstrip("负+-")
    integer = token.split("点", 1)[0]
    if len(integer) > 1 or any(char in _CHINESE_SMALL_UNITS or char in _CHINESE_BIG_UNITS for char in integer):
        return True
    previous = source[match.start() - 1] if match.start() else ""
    following = source[match.end()] if match.end() < len(source) else ""
    return previous == "第" or not following or following.isspace() or following in _CHINESE_NUMBER_CLASSIFIERS or following in "，。！？；、,.!?;:：()（）"


def number_is_grounded(value: int | float, sources: Iterable[str]) -> bool:
    """Compare complete numeric tokens by value, including sentence-final numbers."""

    if isinstance(value, bool):
        return False
    try:
        expected = Decimal(str(value))
    except InvalidOperation:
        return False
    for source in sources:
        for match in _NUMBER_TOKEN.finditer(source):
            try:
                observed = Decimal(match.group(0).replace(",", ""))
            except InvalidOperation:
                continue
            if observed == expected:
                return True
        for match in _CHINESE_NUMBER_TOKEN.finditer(source):
            if not _explicit_chinese_number(match, source):
                continue
            observed = _parse_chinese_number(match.group(0))
            if observed is not None and observed == expected:
                return True
    return False
