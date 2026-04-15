"""Pydantic v2 request/response schemas for the prediction API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PatientRecord(BaseModel):
    """Input features for a single patient admission."""

    age_group: Literal["0 to 17", "18 to 29", "30 to 49", "50 to 69", "70 or Older"] = Field(
        ..., description="Patient age bracket"
    )
    gender: Literal["M", "F", "U"] = Field(..., description="Patient gender")
    race: str = Field("Unknown", description="Patient race")
    ethnicity: str = Field("Unknown", description="Patient ethnicity")
    length_of_stay: int = Field(..., ge=1, le=365, description="Length of stay in days")
    type_of_admission: Literal[
        "Emergency", "Urgent", "Elective", "Newborn", "Trauma", "Not Available"
    ] = Field(..., description="Type of admission")
    apr_severity_code: Literal[1, 2, 3, 4] = Field(
        ..., description="APR Severity of Illness Code (1=Minor … 4=Extreme)"
    )
    apr_severity_desc: Literal["Minor", "Moderate", "Major", "Extreme"] = Field(
        ..., description="APR Severity of Illness Description"
    )
    apr_risk_of_mortality: Literal["Minor", "Moderate", "Major", "Extreme"] = Field(
        ..., description="APR Risk of Mortality"
    )
    apr_medical_surgical: Literal["Medical", "Surgical", "Not Applicable"] = Field(
        ..., description="Medical or Surgical case"
    )
    payment_typology: str = Field("Medicare", description="Primary payer type")
    health_service_area: str = Field("New York City", description="Health service area")
    hospital_county: str = Field("New York", description="Hospital county")
    birth_weight: int = Field(0, ge=0, description="Birth weight in grams (0 if not newborn)")
    ccs_diagnosis_code: int = Field(1, ge=1, le=260, description="CCS diagnosis code")
    apr_drg_code: int = Field(1, ge=1, le=950, description="APR DRG code")
    apr_mdc_code: int = Field(0, ge=0, le=25, description="APR MDC code")

    @field_validator("length_of_stay", mode="before")
    @classmethod
    def coerce_los(cls, v: object) -> int:
        if isinstance(v, str):
            return int(v.replace("+", "").strip())
        return int(v)  # type: ignore[arg-type]


class PredictionResponse(BaseModel):
    predicted_cost: float = Field(..., description="Predicted total cost in USD")
    model_used: str = Field(..., description="Name of the model that made the prediction")
    confidence_interval_lower: float = Field(..., description="95% CI lower bound")
    confidence_interval_upper: float = Field(..., description="95% CI upper bound")
    features_used: list[str] = Field(..., description="Feature names used for prediction")


class BatchRequest(BaseModel):
    records: list[PatientRecord] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[float]
    model_used: str
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None


class MetricsResponse(BaseModel):
    metrics: dict[str, dict[str, float]]
    best_model: str
