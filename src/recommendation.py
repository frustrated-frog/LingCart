"""Evidence-bound candidate selection for under-specified catalog discovery."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import re


def _text(product: dict) -> str:
    return " ".join((
        str(product.get("product_id", "")),
        str(product.get("name", "")),
        str(product.get("category", "")),
        str(product.get("summary", "")),
        json.dumps(product.get("specs") or {}, ensure_ascii=False, sort_keys=True),
    )).casefold()


def _terms(value: str) -> list[str]:
    return [term for term in re.split(r"[\s,，、;；]+", value.casefold().strip()) if term]


def _stable_fraction(seed: str, product_id: str) -> float:
    digest = hashlib.sha256(f"{seed}\n{product_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _price_bucket(price: float, minimum: float, maximum: float) -> str:
    if maximum <= minimum:
        return "single"
    ratio = (price - minimum) / (maximum - minimum)
    return "low" if ratio < 1 / 3 else "mid" if ratio < 2 / 3 else "high"


def _features(product: dict, minimum: float, maximum: float) -> set[str]:
    features = {f"category:{product.get('category')}",
                f"price:{_price_bucket(float(product.get('price', 0)), minimum, maximum)}"}
    for key, value in (product.get("specs") or {}).items():
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            features.add(f"spec:{key}={value}")
    return features


def _similarity(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _snapshot_id(products: list[dict]) -> str:
    payload = [
        {
            "product_id": product.get("product_id"),
            "category": product.get("category"),
            "price": product.get("price"),
            "stock": product.get("stock"),
            "summary": product.get("summary"),
            "specs": product.get("specs"),
        }
        for product in sorted(products, key=lambda item: str(item.get("product_id")))
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def _follow_up(products: list[dict], category: str | None) -> str | None:
    if not products:
        return None
    if not category:
        counts = Counter(str(product.get("category")) for product in products if product.get("category"))
        choices = [name for name, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:4]]
        if len(choices) > 1:
            return "你更偏向" + "、".join(choices) + "中的哪一类，或者更看重预算和使用场景？"
    keys = Counter(
        str(key)
        for product in products
        for key in (product.get("specs") or {})
        if isinstance(key, str) and key
    )
    dimensions = [name for name, _count in sorted(keys.items(), key=lambda item: (-item[1], item[0]))[:3]]
    if dimensions:
        return "如果继续缩小范围，你更看重" + "、".join(dimensions) + "中的哪一项？"
    return "如果继续缩小范围，你更看重预算、便携性还是具体使用场景？"


def select_discovery_slate(products: list[dict], *, query: str = "", category: str | None = None,
                           max_price: float | None = None, soft_preferences: list[str] | None = None,
                           limit: int = 4, context_seed: str = "") -> dict:
    """Select a reproducible, diverse slate from already-authorized catalog rows.

    Hard filters are applied before this function.  Preference text affects a
    soft relevance score; it never silently excludes a candidate.
    """

    if not 1 <= limit <= 12:
        return {"error": "探索推荐数量必须在 1 到 12 之间"}
    if max_price is not None and (not math.isfinite(max_price) or max_price < 0):
        return {"error": "预算不能为负数"}

    candidates = [dict(product) for product in products]
    snapshot = _snapshot_id(candidates)
    if not candidates:
        return {
            "products": [],
            "total": 0,
            "candidate_set": {
                "candidate_set_id": snapshot,
                "catalog_snapshot": snapshot,
                "candidate_ids": [],
                "selected_ids": [],
                "score_components": {},
            },
            "interaction_policy": "NO_CANDIDATE_EVIDENCE",
            "personalization": "NONE",
            "follow_up_question": None,
            "filters": {"query": query, "category": category, "max_price": max_price},
            "data_source": "DEMO_CATALOG",
        }

    prices = [float(product["price"]) for product in candidates]
    minimum, maximum = min(prices), max(prices)
    query_terms = _terms(query)
    preference_terms = [term for preference in (soft_preferences or []) for term in _terms(preference)]
    product_features = {str(product["product_id"]): _features(product, minimum, maximum) for product in candidates}
    selected: list[dict] = []
    scores: dict[str, dict[str, float]] = {}
    remaining = list(candidates)
    seed = context_seed or snapshot

    while remaining and len(selected) < min(limit, len(candidates)):
        selected_features = [product_features[str(item["product_id"])] for item in selected]
        selected_categories = {str(item.get("category")) for item in selected}
        best: tuple[float, str, dict, dict[str, float]] | None = None
        for product in remaining:
            document = _text(product)
            query_match = sum(term in document for term in query_terms) / max(1, len(query_terms))
            preference_match = sum(term in document for term in preference_terms) / max(1, len(preference_terms))
            evidence = min(1.0, (len(product.get("specs") or {}) + bool(product.get("summary"))) / 5.0)
            coverage = 1.0 if str(product.get("category")) not in selected_categories else 0.0
            novelty = 1.0
            if selected_features:
                novelty = 1.0 - max(_similarity(product_features[str(product["product_id"])], other)
                                    for other in selected_features)
            stable_tiebreak = _stable_fraction(seed, str(product["product_id"]))
            total_score = (2.5 * query_match + 1.5 * preference_match + 0.25 * evidence
                           + 1.5 * coverage + 0.9 * novelty + 0.01 * stable_tiebreak)
            components = {
                "query_relevance": round(query_match, 4),
                "preference_relevance": round(preference_match, 4),
                "evidence_completeness": round(evidence, 4),
                "coverage_gain": round(coverage, 4),
                "novelty": round(novelty, 4),
                "total": round(total_score, 4),
            }
            rank_key = (total_score, str(product["product_id"]), product, components)
            if best is None or rank_key[:2] > best[:2]:
                best = rank_key
        assert best is not None
        _score, _product_id, chosen, components = best
        selected.append(chosen)
        scores[str(chosen["product_id"])] = components
        remaining.remove(chosen)

    exploratory = not query_terms and not preference_terms and category is None and max_price is None
    return {
        "products": selected,
        "total": len(candidates),
        "candidate_set": {
            "candidate_set_id": hashlib.sha256(f"{snapshot}:{seed}".encode("utf-8")).hexdigest()[:20],
            "catalog_snapshot": snapshot,
            "candidate_ids": [str(product["product_id"]) for product in candidates],
            "selected_ids": [str(product["product_id"]) for product in selected],
            "score_components": scores,
            "covered_categories": sorted({str(product.get("category")) for product in selected}),
        },
        "interaction_policy": "RECOMMEND_AND_ASK" if exploratory else "RECOMMEND",
        "personalization": "EXPLORATORY" if exploratory else "CONSTRAINT_GUIDED",
        "follow_up_question": _follow_up(candidates, category),
        "filters": {
            "query": query,
            "category": category,
            "max_price": max_price,
            "soft_preferences": list(soft_preferences or []),
        },
        "data_source": "DEMO_CATALOG",
    }

