"""
FastAPI application entrypoint for tissue image inference.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.model_loader import load_model
from src.api.routes import router


app = FastAPI(
    title="Cloud Cancer Tissue ViT API",
    version="0.1.0",
)

app.include_router(router)

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_DIR / "outputs"
FRONTEND_DIR = PROJECT_DIR / "frontend"

app.mount(
    "/outputs",
    StaticFiles(directory=OUTPUTS_DIR, check_dir=False),
    name="outputs",
)

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


@app.on_event("startup")
def load_trained_model_on_startup() -> None:
    """
    Load the trained model once when the API starts.
    """

    try:
        load_model()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Model checkpoint not found. /predict will return an error until it exists.")
