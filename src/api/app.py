"""
FastAPI application entrypoint for tissue image inference.
"""

from fastapi import FastAPI

from src.api.model_loader import load_model
from src.api.routes import router


app = FastAPI(
    title="Cloud Cancer Tissue ViT API",
    version="0.1.0",
)

app.include_router(router)


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
