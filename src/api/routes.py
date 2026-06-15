"""
API routes for ViT tissue image inference.
"""

import torch
from fastapi import APIRouter, File, HTTPException, UploadFile

from src.api.file_utils import save_uploaded_image
from src.api.model_loader import (
    CLASS_NAMES,
    get_inference_config,
    get_model_status,
    load_model,
)
from src.api.schemas import HealthResponse, ModelStatusResponse, PredictionResponse
from src.models.inference import predict_image


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def read_health() -> HealthResponse:
    """
    Return basic API health and GPU availability.
    """

    return HealthResponse(
        status="ok",
        gpu=torch.cuda.is_available(),
    )


@router.get("/model/status", response_model=ModelStatusResponse)
def read_model_status() -> ModelStatusResponse:
    """
    Return whether the model has been loaded into memory.
    """

    return ModelStatusResponse(**get_model_status())


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Save an uploaded tissue image and run ViT classification inference.
    """

    try:
        image_path = await save_uploaded_image(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        model = load_model()
        config = get_inference_config()
        image_size = int(config.get("model", {}).get("image_size", 224))
        device = next(model.parameters()).device

        result = predict_image(
            model=model,
            image_path=image_path,
            class_names=CLASS_NAMES,
            image_size=image_size,
            device=device,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(**result)
