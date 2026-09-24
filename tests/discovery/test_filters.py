"""
tests/discovery/test_filters.py
=================================
Unit tests for the Discovery filtering and ranking layer.

Tests:
    - filter_candidates: threshold, category, sales velocity, stockout days
    - rank_candidates: urgency, sales_velocity, stock_level, restock_priority
    - build_compact_context: top-N selection, field projection, notes
    - Edge cases: empty lists, no matching filters, unknown strategy
"""
from __future__ import annotations

import pytest

import sys
import os

# The filter module has no DB dependencies — import it directly
# without going through the full app.agent chain (which pulls in motor/pymongo).
_DISCOVERY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "app", "agent", "discovery"
)
sys.path.insert(0, os.path.dirname(_DISCOVERY_PATH))

from app.agent.discovery.filters import (

    filter_candidates,
    rank_candidates,
    build_compact_context,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CANDIDATES = [
    {
        "product_id": "prod-001",
        "name": "Rice 5kg",
        "category": "Grains",
        "current_quantity": 3,
        "avg_daily_sales": 2.5,
        "days_to_stockout": 1.2,
        "price": 250.0,
        "priority": "RED",
    },
    {
        "product_id": "prod-002",
        "name": "Wheat Flour 10kg",
        "category": "Grains",
        "current_quantity": 15,
        "avg_daily_sales": 1.8,
        "days_to_stockout": 8.3,
        "price": 400.0,
        "priority": "YELLOW",
    },
    {
        "product_id": "prod-003",
        "name": "Cooking Oil 1L",
        "category": "Oils",
        "current_quantity": 50,
        "avg_daily_sales": 0.5,
        "days_to_stockout": 100.0,
        "price": 120.0,
        "priority": "GREEN",
    },
    {
        "product_id": "prod-004",
        "name": "Sugar 2kg",
        "category": "Sweeteners",
        "current_quantity": 5,
        "avg_daily_sales": 3.0,
        "days_to_stockout": 1.7,
        "price": 90.0,
        "priority": "RED",
    },
    {
        "product_id": "prod-005",
        "name": "Dal 1kg",
        "category": "Grains",
        "current_quantity": 8,
        "avg_daily_sales": 0.0,
        "days_to_stockout": None,
        "price": 80.0,
        "priority": "YELLOW",
    },
]


# ---------------------------------------------------------------------------
# filter_candidates tests
# ---------------------------------------------------------------------------

class TestFilterCandidates:
    def test_filter_by_stock_quantity_max(self):
        result = filter_candidates(SAMPLE_CANDIDATES, stock_quantity_max=10)
        assert all(c["current_quantity"] <= 10 for c in result)
        assert len(result) == 3  # prod-001(3), prod-004(5), prod-005(8)

    def test_filter_by_stock_quantity_min(self):
        result = filter_candidates(SAMPLE_CANDIDATES, stock_quantity_min=10)
        assert all(c["current_quantity"] >= 10 for c in result)
        assert len(result) == 2  # prod-002(15), prod-003(50)

    def test_filter_by_avg_daily_sales_min(self):
        result = filter_candidates(SAMPLE_CANDIDATES, avg_daily_sales_min=1.0)
        assert all(c["avg_daily_sales"] >= 1.0 for c in result)
        # prod-001(2.5), prod-002(1.8), prod-004(3.0) = 3
        assert len(result) == 3

    def test_filter_by_days_to_stockout_max(self):
        result = filter_candidates(SAMPLE_CANDIDATES, days_to_stockout_max=7)
        # prod-001(1.2), prod-004(1.7) have days_to_stockout <= 7
        # prod-005 has None → excluded since None is not <= 7
        for c in result:
            assert c["days_to_stockout"] is not None
            assert c["days_to_stockout"] <= 7
        assert len(result) == 2

    def test_filter_by_category(self):
        result = filter_candidates(SAMPLE_CANDIDATES, category="Grains")
        assert all(c["category"] == "Grains" for c in result)
        assert len(result) == 3

    def test_filter_category_case_insensitive(self):
        result_lower = filter_candidates(SAMPLE_CANDIDATES, category="grains")
        result_upper = filter_candidates(SAMPLE_CANDIDATES, category="GRAINS")
        assert len(result_lower) == len(result_upper) == 3

    def test_filter_with_limit(self):
        result = filter_candidates(SAMPLE_CANDIDATES, limit=2)
        assert len(result) == 2

    def test_filter_empty_input(self):
        result = filter_candidates([], stock_quantity_max=10)
        assert result == []

    def test_filter_no_matches(self):
        result = filter_candidates(SAMPLE_CANDIDATES, stock_quantity_max=0)
        assert result == []

    def test_filter_combined(self):
        """stock_quantity_max + category."""
        result = filter_candidates(
            SAMPLE_CANDIDATES,
            stock_quantity_max=10,
            category="Grains",
        )
        # prod-001(3, Grains), prod-005(8, Grains) but prod-002 is 15 > 10
        assert len(result) == 2
        assert all(c["category"] == "Grains" for c in result)
        assert all(c["current_quantity"] <= 10 for c in result)


# ---------------------------------------------------------------------------
# rank_candidates tests
# ---------------------------------------------------------------------------

class TestRankCandidates:
    def test_rank_urgency(self):
        result = rank_candidates(SAMPLE_CANDIDATES, strategy="urgency")
        # None goes last; among non-None, ascending days_to_stockout
        non_none = [c for c in result if c["days_to_stockout"] is not None]
        days = [c["days_to_stockout"] for c in non_none]
        assert days == sorted(days)
        # None-valued items are last
        assert result[-1]["days_to_stockout"] is None

    def test_rank_sales_velocity(self):
        result = rank_candidates(SAMPLE_CANDIDATES, strategy="sales_velocity")
        velocities = [c["avg_daily_sales"] for c in result]
        assert velocities == sorted(velocities, reverse=True)

    def test_rank_stock_level(self):
        result = rank_candidates(SAMPLE_CANDIDATES, strategy="stock_level")
        qtys = [c["current_quantity"] for c in result]
        assert qtys == sorted(qtys)

    def test_rank_restock_priority(self):
        result = rank_candidates(SAMPLE_CANDIDATES, strategy="restock_priority")
        # First two should be RED
        assert result[0]["priority"] == "RED"
        assert result[1]["priority"] == "RED"

    def test_rank_unknown_strategy_returns_original_order(self):
        ids_before = [c["product_id"] for c in SAMPLE_CANDIDATES]
        result = rank_candidates(SAMPLE_CANDIDATES, strategy="nonexistent")
        ids_after = [c["product_id"] for c in result]
        assert ids_before == ids_after

    def test_rank_empty_input(self):
        result = rank_candidates([], strategy="urgency")
        assert result == []

    def test_rank_does_not_mutate_input(self):
        original_first = SAMPLE_CANDIDATES[0]["product_id"]
        rank_candidates(SAMPLE_CANDIDATES, strategy="sales_velocity")
        assert SAMPLE_CANDIDATES[0]["product_id"] == original_first


# ---------------------------------------------------------------------------
# build_compact_context tests
# ---------------------------------------------------------------------------

class TestBuildCompactContext:
    def test_returns_top_n(self):
        result = build_compact_context(SAMPLE_CANDIDATES, limit=2)
        assert result["total_candidates"] == 5
        assert len(result["top_candidates"]) == 2

    def test_full_list_within_limit(self):
        result = build_compact_context(SAMPLE_CANDIDATES, limit=100)
        assert len(result["top_candidates"]) == 5

    def test_context_note_present(self):
        result = build_compact_context(SAMPLE_CANDIDATES, limit=2)
        assert "context_note" in result
        assert len(result["context_note"]) > 0
        assert "5" in result["context_note"]  # total candidates

    def test_include_fields_projection(self):
        result = build_compact_context(
            SAMPLE_CANDIDATES,
            limit=5,
            include_fields=["product_id", "name", "days_to_stockout"],
        )
        for item in result["top_candidates"]:
            assert set(item.keys()) == {"product_id", "name", "days_to_stockout"}

    def test_empty_candidates(self):
        result = build_compact_context([], limit=20)
        assert result["total_candidates"] == 0
        assert result["top_candidates"] == []


# ---------------------------------------------------------------------------
# Store isolation test (service-level contract, checked via filter)
# ---------------------------------------------------------------------------

class TestStoreIsolation:
    """
    The store isolation contract is enforced at the service/database layer.
    Here we verify that filter_candidates never modifies or ignores the
    store context embedded in candidate records (e.g., no cross-contamination
    of store_id values if present in the candidate dict).
    """

    def test_candidates_store_id_preserved(self):
        candidates_with_store = [
            {**c, "store_id": "store-A"}
            for c in SAMPLE_CANDIDATES
        ]
        result = filter_candidates(candidates_with_store, stock_quantity_max=20)
        assert all(c.get("store_id") == "store-A" for c in result)

    def test_no_cross_store_contamination(self):
        store_a = [{"product_id": "a1", "current_quantity": 5, "store_id": "store-A"}]
        store_b = [{"product_id": "b1", "current_quantity": 5, "store_id": "store-B"}]
        combined = store_a + store_b
        # filter_candidates doesn't filter by store_id (that's done at service layer)
        # but it must not corrupt or lose the store_id field
        result = filter_candidates(combined, stock_quantity_max=10)
        store_ids = {c.get("store_id") for c in result}
        assert "store-A" in store_ids
        assert "store-B" in store_ids


# ---------------------------------------------------------------------------
# Limit guard tests
# ---------------------------------------------------------------------------

class TestLimitHandling:
    def test_limit_zero_returns_empty(self):
        result = filter_candidates(SAMPLE_CANDIDATES, limit=0)
        assert result == []

    def test_limit_larger_than_input(self):
        result = filter_candidates(SAMPLE_CANDIDATES, limit=1000)
        assert len(result) == len(SAMPLE_CANDIDATES)

    def test_compact_context_limit_zero(self):
        result = build_compact_context(SAMPLE_CANDIDATES, limit=0)
        assert result["top_candidates"] == []
        assert result["total_candidates"] == 5
