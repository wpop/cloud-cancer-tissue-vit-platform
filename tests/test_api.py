from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from src.api.routes import router


SAMPLE_IMAGE_PATH = Path("outputs/predictions/sample_pcam_image.png")


def get_test_image_path(tmp_path: Path) -> Path:
    """
    Return the sample PCam image, or create a tiny temporary RGB PNG.
    """

    if SAMPLE_IMAGE_PATH.exists():
        return SAMPLE_IMAGE_PATH

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(image_path)
    return image_path


def fake_prediction(image_path: Path) -> dict:
    """
    Return a stable prediction payload for API tests.
    """

    return {
        "predicted_class": "cancer",
        "confidence": 0.91,
        "probabilities": {
            "benign": 0.09,
            "cancer": 0.91,
        },
    }


@pytest.fixture
def api_test_config(monkeypatch, tmp_path: Path) -> dict:
    """
    Use temporary API output directories during upload and artifact tests.
    """

    config = {
        "model": {
            "checkpoint": "models/checkpoints/best_model.pt",
            "image_size": 224,
            "num_classes": 2,
        },
        "input": {
            "supported_formats": ["png", "jpg", "jpeg", "tif", "tiff"],
        },
        "output": {
            "api_upload_dir": str(tmp_path / "api_uploads"),
            "save_predictions": True,
            "prediction_dir": str(tmp_path / "predictions"),
            "save_probability_plot": True,
            "probability_plot_dir": str(tmp_path / "figures"),
        },
    }

    monkeypatch.setattr("src.api.file_utils.get_inference_config", lambda: config)
    monkeypatch.setattr("src.api.routes.get_inference_config", lambda: config)

    def fake_probability_plot(
        probabilities: dict[str, float],
        save_path: str | Path,
    ) -> None:
        """
        Write a small placeholder file instead of rendering a real plot.
        """

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(str(probabilities), encoding="utf-8")

    monkeypatch.setattr("src.api.routes._save_probability_plot", fake_probability_plot)

    return config


def upload_file_tuple(path: Path, media_type: str = "image/png") -> tuple:
    """
    Build a FastAPI TestClient file tuple from a local path.
    """

    return (path.name, path.open("rb"), media_type)


@pytest.fixture
def client() -> TestClient:
    """
    Create a small test app with the production API router.
    """

    test_app = FastAPI()
    test_app.include_router(router)
    return TestClient(test_app)


def test_health(client: TestClient) -> None:
    """
    The health endpoint should return service status and GPU availability.
    """

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert isinstance(payload["gpu"], bool)


def test_model_status(client: TestClient) -> None:
    """
    The model status endpoint should expose model load metadata.
    """

    response = client.get("/model/status")

    assert response.status_code == 200
    payload = response.json()
    assert "model_loaded" in payload
    assert "device" in payload
    assert payload["checkpoint_path"] == "models/checkpoints/best_model.pt"


def test_predict_with_valid_png(
    client: TestClient,
    monkeypatch,
    api_test_config: dict,
    tmp_path: Path,
) -> None:
    """
    A valid PNG upload should return one prediction response.
    """

    monkeypatch.setattr("src.api.routes._predict_saved_image", fake_prediction)
    image_path = get_test_image_path(tmp_path)

    file_tuple = upload_file_tuple(image_path)
    try:
        response = client.post("/predict", files={"file": file_tuple})
    finally:
        file_tuple[1].close()

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == image_path.name
    assert payload["predicted_class"] == "cancer"
    assert payload["confidence"] == 0.91
    assert payload["probabilities"] == {"benign": 0.09, "cancer": 0.91}

    artifacts = payload["artifacts"]
    assert artifacts["prediction_json"].startswith(api_test_config["output"]["prediction_dir"])
    assert artifacts["probability_plot"].startswith(
        api_test_config["output"]["probability_plot_dir"]
    )
    assert Path(artifacts["prediction_json"]).exists()
    assert Path(artifacts["probability_plot"]).exists()


def test_predict_with_unsupported_file(
    client: TestClient,
    api_test_config: dict,
    tmp_path: Path,
) -> None:
    """
    An unsupported upload extension should return a clear HTTP 400 error.
    """

    text_path = tmp_path / "not_supported.txt"
    text_path.write_text("not an image", encoding="utf-8")

    file_tuple = upload_file_tuple(text_path, media_type="text/plain")
    try:
        response = client.post("/predict", files={"file": file_tuple})
    finally:
        file_tuple[1].close()

    assert response.status_code == 400
    assert "Unsupported image type" in response.json()["detail"]


def test_predict_batch_with_valid_png_files(
    client: TestClient,
    monkeypatch,
    api_test_config: dict,
    tmp_path: Path,
) -> None:
    """
    Multiple valid PNG uploads should return one result per file.
    """

    monkeypatch.setattr("src.api.routes._predict_saved_image", fake_prediction)
    image_path = get_test_image_path(tmp_path)

    first_file = upload_file_tuple(image_path)
    second_file = upload_file_tuple(image_path)
    try:
        response = client.post(
            "/predict-batch",
            files=[
                ("files", first_file),
                ("files", second_file),
            ],
        )
    finally:
        first_file[1].close()
        second_file[1].close()

    assert response.status_code == 200
    payload = response.json()
    assert payload["num_files"] == 2
    assert len(payload["results"]) == 2

    for result in payload["results"]:
        assert result["filename"] == image_path.name
        assert result["predicted_class"] == "cancer"
        assert result["confidence"] == 0.91
        assert result["probabilities"] == {"benign": 0.09, "cancer": 0.91}
        assert result["artifacts"]["prediction_json"].startswith(
            api_test_config["output"]["prediction_dir"]
        )
        assert result["artifacts"]["probability_plot"].startswith(
            api_test_config["output"]["probability_plot_dir"]
        )
        assert Path(result["artifacts"]["prediction_json"]).exists()
        assert Path(result["artifacts"]["probability_plot"]).exists()


def test_predict_batch_with_unsupported_file(
    client: TestClient,
    monkeypatch,
    api_test_config: dict,
    tmp_path: Path,
) -> None:
    """
    A batch containing an unsupported file should return a clear HTTP 400 error.
    """

    monkeypatch.setattr("src.api.routes._predict_saved_image", fake_prediction)
    image_path = get_test_image_path(tmp_path)
    text_path = tmp_path / "not_supported.txt"
    text_path.write_text("not an image", encoding="utf-8")

    image_file = upload_file_tuple(image_path)
    text_file = upload_file_tuple(text_path, media_type="text/plain")
    try:
        response = client.post(
            "/predict-batch",
            files=[
                ("files", image_file),
                ("files", text_file),
            ],
        )
    finally:
        image_file[1].close()
        text_file[1].close()

    assert response.status_code == 400
    assert "Unsupported image type" in response.json()["detail"]
