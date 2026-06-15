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
from src.api.schemas import (
    BatchPredictionResponse,
    BatchPredictionResult,
    HealthResponse,
    ModelStatusResponse,
    PredictionResponse,
)
from src.models.inference import predict_image


router = APIRouter()


def _predict_saved_image(image_path) -> dict:
    """
    Run model inference for an image that has already been saved to disk.
    """

    model = load_model()
    config = get_inference_config()
    image_size = int(config.get("model", {}).get("image_size", 224))
    device = next(model.parameters()).device

    return predict_image(
        model=model,
        image_path=image_path,
        class_names=CLASS_NAMES,
        image_size=image_size,
        device=device,
    )


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
        result = _predict_saved_image(image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(**result)


@router.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(files: list[UploadFile] = File(...)) -> BatchPredictionResponse:
    """
    Save multiple uploaded tissue images and return one prediction per image.
    """

    if not files:
        raise HTTPException(status_code=400, detail="At least one image file is required.")

    results = []

    try:
        for file in files:
            image_path = await save_uploaded_image(file)
            prediction = _predict_saved_image(image_path)
            results.append(
                BatchPredictionResult(
                    filename=image_path.name,
                    **prediction,
                )
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}") from exc

    return BatchPredictionResponse(
        num_files=len(results),
        results=results,
    )
