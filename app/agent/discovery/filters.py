"""
app/agent/discovery/filters.py
================================
Filtering and Ranking layer for Discovery results.

This module sits between Discovery Tools and the LLM context.
It applies deterministic business rules — no LLM reasoning involved.

Architecture position:
    Database → Service → Discovery Tools → [THIS MODULE] → LLM Context

Philosophy:
    - Database decides what data EXISTS.
    - Tools decide what data is RELEVANT (discovery).
    - This module decides what data is IMPORTANT (filtering + ranking).
    - LLM decides what the data MEANS.

Public API:
    filter_candidates(candidates, filters)   → filtered list
    rank_candidates(candidates, strategy)    → ranked list
    build_compact_context(candidates, limit) → top-N compact list

Supported filter keys:
    stock_quantity_max    → float  — keep only products with stock <= this
    stock_quantity_min    → float  — keep only products with stock >= this
    avg_daily_sales_min   → float  — keep only products selling >= this/day
    days_to_stockout_max  → float  — keep only products running out <= this days
    category              → str    — keep only products in this category
    limit                 → int    — maximum results to return

Supported ranking strategies:
    "urgency"         → sort by days_to_stockout asc (most urgent first)
    "sales_velocity"  → sort by avg_daily_sales desc (fastest moving first)
    "stock_level"     → sort by current_quantity asc (lowest stock first)
    "tied_up_value"   → sort by tied_up_value desc (most capital tied up first)
    "restock_priority"→ sort by priority score (RED > YELLOW > GREEN)
"""
from __future__ import annotations

from typing import Any

import structlog

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.discovery.filters")

# Priority rank for string-based priority fields (restock)
_PRIORITY_RANK = {"RED": 0, "YELLOW": 1, "GREEN": 2}


def filter_candidates(
    candidates: list[dict[str, Any]],
    *,
    stock_quantity_max: float | None = None,
    stock_quantity_min: float | None = None,
    avg_daily_sales_min: float | None = None,
    days_to_stockout_max: float | None = None,
    category: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Apply deterministic business filters to a list of candidate products.

    Args:
        candidates:           List of product candidate dicts from a discovery tool.
        stock_quantity_max:   Only include products with stock <= this value.
        stock_quantity_min:   Only include products with stock >= this value.
        avg_daily_sales_min:  Only include products with avg daily sales >= this value.
        days_to_stockout_max: Only include products with days_to_stockout <= this value.
        category:             Only include products in this category (case-insensitive).
        limit:                Maximum results to return.

    Returns:
        Filtered list, preserving the original order.
    """
    before = len(candidates)

    result = []
    for c in candidates:
        # stock_quantity_max filter
        qty = c.get("current_quantity", c.get("quantity", None))
        if stock_quantity_max is not None and qty is not None:
            if qty > stock_quantity_max:
                continue

        # stock_quantity_min filter
        if stock_quantity_min is not None and qty is not None:
            if qty < stock_quantity_min:
                continue

        # avg_daily_sales_min filter
        ads = c.get("avg_daily_sales", None)
        if avg_daily_sales_min is not None and ads is not None:
            if ads < avg_daily_sales_min:
                continue

        # days_to_stockout_max filter
        dts = c.get("days_to_stockout", None)
        if days_to_stockout_max is not None and dts is not None:
            if dts > days_to_stockout_max:
                continue

        # category filter
        if category is not None:
            cat = c.get("category", "")
            if cat.lower() != category.lower():
                continue

        result.append(c)

    after = len(result)

    LOGGER.debug(
        "candidates_filtered",
        before=before,
        after=after,
        filters_applied={
            "stock_quantity_max": stock_quantity_max,
            "stock_quantity_min": stock_quantity_min,
            "avg_daily_sales_min": avg_daily_sales_min,
            "days_to_stockout_max": days_to_stockout_max,
            "category": category,
        },
    )

    # Apply limit last
    if limit is not None:
        result = result[:limit]

    return result


def rank_candidates(
    candidates: list[dict[str, Any]],
    strategy: str = "urgency",
) -> list[dict[str, Any]]:
    """
    Rank candidate products by a named business strategy.

    Ranking is deterministic — no LLM involved.

    Strategies:
        "urgency"          → days_to_stockout asc (None goes last)
        "sales_velocity"   → avg_daily_sales desc
        "stock_level"      → current_quantity / quantity asc
        "tied_up_value"    → tied_up_value desc
        "restock_priority" → RED → YELLOW → GREEN, then days_to_stockout asc

    Args:
        candidates: List of candidate product dicts.
        strategy:   Ranking strategy name (default "urgency").

    Returns:
        New sorted list. Original list is not mutated.
    """
    if not candidates:
        return []

    before_count = len(candidates)

    if strategy == "urgency":
        ranked = sorted(
            candidates,
            key=lambda c: (
                c.get("days_to_stockout") is None,
                c.get("days_to_stockout", 9999),
            ),
        )
    elif strategy == "sales_velocity":
        ranked = sorted(
            candidates,
            key=lambda c: c.get("avg_daily_sales", 0.0),
            reverse=True,
        )
    elif strategy == "stock_level":
        ranked = sorted(
            candidates,
            key=lambda c: c.get("current_quantity", c.get("quantity", 0)),
        )
    elif strategy == "tied_up_value":
        ranked = sorted(
            candidates,
            key=lambda c: c.get("tied_up_value", 0.0),
            reverse=True,
        )
    elif strategy == "restock_priority":
        ranked = sorted(
            candidates,
            key=lambda c: (
                _PRIORITY_RANK.get(c.get("priority", "GREEN"), 3),
                c.get("days_to_stockout") is None,
                c.get("days_to_stockout", 9999),
            ),
        )
    else:
        LOGGER.warning("rank_candidates_unknown_strategy", strategy=strategy)
        ranked = list(candidates)

    LOGGER.debug(
        "candidates_ranked",
        strategy=strategy,
        count=before_count,
    )

    return ranked


def build_compact_context(
    candidates: list[dict[str, Any]],
    limit: int = 20,
    include_fields: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build a compact context dict from candidates for the LLM.

    This is the Context Builder step:
        Graph State (may contain many candidates)
            ↓
        build_compact_context
            ↓
        Compact LLM Context (top-N, trimmed fields)

    Args:
        candidates:     Ranked/filtered candidate list.
        limit:          Maximum products to include in LLM context (default 20).
        include_fields: If provided, only include these keys per product.
                        None = include all fields.

    Returns:
        Dict with:
            "total_candidates": int — total before limit
            "top_candidates":   list — compact candidate list (up to limit)
            "context_note":     str — human-readable summary for LLM
    """
    total = len(candidates)
    top = candidates[:limit]

    if include_fields:
        top = [
            {k: v for k, v in c.items() if k in include_fields}
            for c in top
        ]

    context_note = (
        f"{total} candidate products found. "
        f"Showing top {len(top)} for analysis."
    )
    if total > limit:
        context_note += f" {total - limit} additional products exist but were not included to keep context concise."

    LOGGER.info(
        "context_built",
        total_candidates=total,
        context_candidates=len(top),
        limit=limit,
    )

    return {
        "total_candidates": total,
        "top_candidates": top,
        "context_note": context_note,
    }
