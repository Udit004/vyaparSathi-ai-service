"""
app/agent/tools/sales/fast_moving.py
======================================
Discovery tool: fast-moving products by sales velocity.

This is a Stage 1 discovery tool — it identifies the top-N products
by average daily sales, ranked by sales velocity descending.

Use this before detailed analysis to narrow the question
"which products are selling fast?" to a compact candidate list.

Example flow:
    User: "Which are my fastest moving products?"
    Think → get_fast_moving_products(lookback_days=30, limit=20)
           → [20 candidates]
    Observe → Context Builder → LLM
"""
from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime

from app.agent.service.sales.fast_moving import fetch_fast_moving_products


class FastMovingInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    lookback_days: int = Field(
        30,
        description=(
            "Number of past days to consider for sales velocity calculation. "
            "Default: 30. Range: 1–365."
        ),
    )
    min_daily_sales: float = Field(
        0.1,
        description=(
            "Minimum average daily sales threshold. Products below this are excluded. "
            "Default: 0.1 units/day."
        ),
    )
    limit: int = Field(
        30,
        description="Maximum number of products to return. Default: 30. Max: 100.",
    )


class FastMovingItem(BaseModel):
    product_id: str
    name: str
    category: str
    current_quantity: float
    avg_daily_sales: float
    total_qty_sold: float
    price: float
    sales_velocity_rank: int


class FastMovingOutput(BaseModel):
    items: List[FastMovingItem]
    count: int
    lookback_days: int
    fetched_at: str


@tool("get_fast_moving_products", args_schema=FastMovingInput)
async def get_fast_moving_products(
    store_id: str,
    lookback_days: int = 30,
    min_daily_sales: float = 0.1,
    limit: int = 30,
) -> FastMovingOutput:
    """
    [DISCOVERY TOOL] Find products with the highest sales velocity (fast-moving).

    Use this as a Stage 1 discovery step to identify which products are selling
    fastest. Results are sorted by average daily sales descending (rank 1 = fastest).

    Key behaviour:
    - Uses MongoDB aggregation — no full product scan in Python
    - Products below min_daily_sales threshold are excluded (removes noise)
    - Returns compact candidate data: name, velocity, current stock
    - Store isolation: always scoped to the authenticated store_id

    When to use:
    - "Which products are selling fastest?"
    - "What's moving quickly in my store?"
    - "I want to know my top performers by velocity"
    - "Which products need frequent restocking due to high demand?"

    Formula: avg_daily_sales = total_qty_sold_in_lookback_days / lookback_days
    """
    items = await fetch_fast_moving_products(
        store_id,
        lookback_days=lookback_days,
        min_daily_sales=min_daily_sales,
        limit=limit,
    )
    return FastMovingOutput(
        items=[FastMovingItem(**i) for i in items],
        count=len(items),
        lookback_days=lookback_days,
        fetched_at=datetime.utcnow().isoformat(),
    )
