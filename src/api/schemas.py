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


class PredictionArtifacts(BaseModel):
    """
    Paths to files created for a prediction.
    """

    prediction_json: str | None = None
    probability_plot: str | None = None


class ExplainArtifacts(PredictionArtifacts):
    """
    Paths to files created for an explanation response.
    """

    attention_overlay: str


class PredictionResponse(BaseModel):
    """
    Prediction result returned by the ViT classifier.
    """

    filename: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    artifacts: PredictionArtifacts


class S3PredictionRequest(BaseModel):
    """S3 location of one image to classify."""

    bucket: str
    key: str


class S3Source(BaseModel):
    """S3 source metadata returned with a prediction."""

    bucket: str
    key: str


class S3PredictionResponse(PredictionResponse):
    """Prediction result for an image downloaded from S3."""

    source: S3Source


class BatchPredictionResult(PredictionResponse):
    """
    Prediction result for one uploaded file in a batch.
    """

    filename: str


class BatchPredictionResponse(BaseModel):
    """
    Prediction results for a batch of uploaded images.
    """

    num_files: int
    results: list[BatchPredictionResult]


class ExplainResponse(PredictionResponse):
    """
    Prediction result with an explainability artifact.
    """

    artifacts: ExplainArtifacts
    warning: str
