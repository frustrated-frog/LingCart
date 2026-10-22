"""Shared source-grounding primitives for model-authored structured values."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Iterable


_NUMBER_TOKEN = re.compile(
    r"(?<![\d.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\d)(?!\.\d)"
)


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
    return False
