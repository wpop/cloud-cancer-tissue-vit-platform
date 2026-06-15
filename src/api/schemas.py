"""
Pydantic response schemas for the inference API.
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """
    Basic API health response.
    """

    status: str
    gpu: bool


class ModelStatusResponse(BaseModel):
    """
    Report whether the model is loaded and which device it uses.
    """

    model_loaded: bool
    device: str | None
    checkpoint_path: str


class PredictionResponse(BaseModel):
    """
    Prediction result returned by the ViT classifier.
    """

    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
