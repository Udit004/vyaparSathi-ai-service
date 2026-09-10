from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class SaleItemModel(BaseModel):
    productId: str
    quantity: float
    lineTotal: float

class SaleModel(BaseModel):
    id: str = Field(alias="_id")
    completedAt: datetime
    items: List[SaleItemModel] = Field(default_factory=list)

    class Config:
        populate_by_name = True
