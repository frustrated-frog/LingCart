"""Versioned semantic taxonomy for policy-relevant after-sales reasons.

The taxonomy is descriptive metadata for dialogue understanding. It does not
classify text with keywords and it does not make policy decisions. The model
may select one of these stable codes only when the user's exact quote supports
the meaning; policy evaluation remains deterministic and fail-closed.
"""

from __future__ import annotations

import hashlib
import json


REASON_TAXONOMY_VERSION = "after-sales-reason/2026-01"
REASON_TAXONOMY_SOURCE = "LINGCART_AFTER_SALES_POLICY"
REASON_CODE_DESCRIPTIONS = {
    "DAMAGED": "商品存在明确的外观、结构或运输损坏，例如破裂、变形、碎裂或明显磕碰",
    "NOT_AS_DESCRIBED": "商品的功能、规格、质量或实际表现与商品描述或正常预期不一致",
    "CHANGED_MIND": "商品本身没有已陈述的质量问题，但用户因主观原因不再需要",
}


def _fingerprint() -> str:
    encoded = json.dumps(
        {
            "version": REASON_TAXONOMY_VERSION,
            "source": REASON_TAXONOMY_SOURCE,
            "values": REASON_CODE_DESCRIPTIONS,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


REASON_TAXONOMY_FINGERPRINT = _fingerprint()


def reason_taxonomy_identity() -> dict[str, str]:
    return {
        "version": REASON_TAXONOMY_VERSION,
        "source": REASON_TAXONOMY_SOURCE,
        "fingerprint": REASON_TAXONOMY_FINGERPRINT,
    }
