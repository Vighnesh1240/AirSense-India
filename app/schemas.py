from typing import List
from pydantic import BaseModel, Field


class AQIRequest(BaseModel):
    City: str = Field(min_length=1, max_length=100)
    Date: str

    PM2_5: float = Field(ge=0)
    PM10: float = Field(ge=0)
    NO: float = Field(ge=0)
    NO2: float = Field(ge=0)
    NOx: float = Field(ge=0)
    NH3: float = Field(ge=0)
    CO: float = Field(ge=0)
    SO2: float = Field(ge=0)
    O3: float = Field(ge=0)
    Benzene: float = Field(ge=0)
    Toluene: float = Field(ge=0)
    Xylene: float = Field(ge=0)


class PredictionResponse(BaseModel):
    success: bool
    prediction: dict


class CitiesResponse(BaseModel):
    cities: List[str]
