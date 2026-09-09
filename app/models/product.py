from typing import Optional
from pydantic import BaseModel, Field

class ProductModel(BaseModel):
    id: str = Field(alias="_id")
    name: str
    category: Optional[str] = "General"
    quantity: Optional[float] = 0
    price: Optional[float] = 0

    class Config:
        populate_by_name = True
