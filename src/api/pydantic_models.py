from pydantic import BaseModel, Field
from typing import List, Optional

class TransactionInput(BaseModel):
    """
    Schema for incoming prediction requests.
    Includes numerical features used by the model.
    """
    Amount: float
    Value: float
    Total_Transaction_Amount: float
    Average_Transaction_Amount: float
    Transaction_Count: int
    Std_Transaction_Amount: float
    Transaction_Hour: int
    Transaction_Day: int
    Transaction_Month: int
    Transaction_Year: int
    Transaction_DayOfWeek: int
    Transaction_IsWeekend: int
    ProductCategory_data_bundles: int = Field(default=0)
    ProductCategory_financial_services: int = Field(default=0)
    ProductCategory_movies: int = Field(default=0)
    ProductCategory_tv: int = Field(default=0)
    ProductCategory_utilities: int = Field(default=0)
    ChannelId_ChannelId_2: int = Field(default=0)
    ChannelId_ChannelId_3: int = Field(default=0)
    ChannelId_ChannelId_5: int = Field(default=0)
    PricingStrategy_1: int = Field(default=0)
    PricingStrategy_2: int = Field(default=0)
    PricingStrategy_4: int = Field(default=0)

class RiskPrediction(BaseModel):
    """
    Schema for API response.
    """
    is_high_risk: int
    probability: float
    status: str
