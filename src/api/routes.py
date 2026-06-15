"""
API routes for ViT tissue image inference.
"""

import json
from pathlib import Path

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


DEFAULT_PREDICTION_DIR = Path("outputs") / "predictions"
DEFAULT_PROBABILITY_PLOT_DIR = Path("outputs") / "figures"


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


def _build_prediction_response(image_path: Path, prediction: dict) -> dict:
    """
    Add filename and saved artifact paths to a prediction result.
    """

    artifacts = _save_prediction_artifacts(image_path=image_path, prediction=prediction)

    return {
        "filename": image_path.name,
        **prediction,
        "artifacts": artifacts,
    }


def _save_probability_plot(probabilities: dict[str, float], save_path: Path) -> None:
    """
    Save the probability plot, importing Matplotlib only when needed.
    """

    from src.visualization.probability_plot import save_probability_plot

    save_probability_plot(
        probabilities=probabilities,
        save_path=save_path,
    )


def _save_prediction_artifacts(image_path: Path, prediction: dict) -> dict:
    """
    Save configured prediction artifacts and return their paths.
    """

    config = get_inference_config()
    output_config = config.get("output", {})

    artifacts = {
        "prediction_json": None,
        "probability_plot": None,
    }

    if output_config.get("save_predictions", True):
        prediction_dir = Path(
            output_config.get("prediction_dir", DEFAULT_PREDICTION_DIR)
        )
        prediction_dir.mkdir(parents=True, exist_ok=True)
        prediction_json_path = prediction_dir / f"{image_path.stem}_prediction.json"

        payload = {
            "filename": image_path.name,
            **prediction,
        }
        with prediction_json_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4)

        artifacts["prediction_json"] = str(prediction_json_path)

    if output_config.get("save_probability_plot", True):
        probability_plot_dir = Path(
            output_config.get("probability_plot_dir", DEFAULT_PROBABILITY_PLOT_DIR)
        )
        probability_plot_path = probability_plot_dir / f"{image_path.stem}_probabilities.png"
        _save_probability_plot(
            probabilities=prediction["probabilities"],
            save_path=probability_plot_path,
        )
        artifacts["probability_plot"] = str(probability_plot_path)

    return artifacts


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
        prediction = _predict_saved_image(image_path)
        result = _build_prediction_response(
            image_path=Path(image_path),
            prediction=prediction,
        )
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
            result = _build_prediction_response(
                image_path=Path(image_path),
                prediction=prediction,
            )
            results.append(
                BatchPredictionResult(
                    **result,
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
