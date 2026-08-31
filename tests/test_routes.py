"""Endpoints del servicio de motor.

El contrato tiene que ser exactamente el mismo que el de
`ocr_service_tesseract` y `ocr_service_easyocr`: es lo que le permite al
coordinador tratar a los tres motores igual.
"""

import json
from unittest.mock import patch

import pytest

from app.core.config import settings
from app.schemas import Transform
from tests.conftest import box, detection, render_image


def files(content_type: str = "image/jpeg"):
    return {"file": ("scan.jpg", render_image(), content_type)}


DETECTIONS = detection(["LECHE"], [0.93], [box(10, 10, 100, 30)])


# ---------------------------------------------------------------------------
# /engine/run
# ---------------------------------------------------------------------------

async def test_run_returns_words_with_confidence(client, fake_predictor):
    fake_predictor(DETECTIONS)

    with patch("app.engine.is_available", return_value=True):
        r = await client.post("/engine/run", files=files(), data={"lang": "spa"})

    body = r.json()

    assert r.status_code == 200
    assert body["engine"] == "paddleocr"
    assert body["words"][0]["text"] == "LECHE"
    assert body["words"][0]["confidence"] == pytest.approx(0.93)


async def test_run_accepts_the_transform_as_json(client, fake_predictor):
    fake_predictor(DETECTIONS)
    spec = Transform(rotate=270).model_dump(mode="json")

    with patch("app.engine.is_available", return_value=True):
        r = await client.post(
            "/engine/run", files=files(), data={"transform": json.dumps(spec)}
        )

    assert r.status_code == 200


async def test_run_rejects_a_malformed_transform(client):
    r = await client.post("/engine/run", files=files(), data={"transform": "{no es json"})

    assert r.status_code == 400


async def test_run_rejects_an_unsupported_format(client):
    r = await client.post("/engine/run", files=files("application/pdf"))

    assert r.status_code == 400


async def test_run_rejects_an_empty_file(client):
    r = await client.post("/engine/run", files={"file": ("x.jpg", b"", "image/jpeg")})

    assert r.status_code == 400


async def test_run_rejects_something_that_is_not_an_image(client):
    with patch("app.engine.is_available", return_value=True):
        r = await client.post(
            "/engine/run", files={"file": ("x.jpg", b"no soy una imagen", "image/jpeg")}
        )

    assert r.status_code == 400


async def test_run_rejects_an_oversized_upload(client, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 10)

    r = await client.post("/engine/run", files=files())

    assert r.status_code == 413


async def test_run_reports_503_when_the_dependency_is_missing(client):
    """Sin `paddleocr` instalado el servicio lo dice, no simula una lectura vacía."""
    with patch("app.engine.is_available", return_value=False):
        r = await client.post("/engine/run", files=files())

    assert r.status_code == 503


async def test_run_reports_an_engine_failure_as_500(client):
    with patch("app.engine.is_available", return_value=True), \
         patch("app.engine.run", side_effect=RuntimeError("el modelo falló")):
        r = await client.post("/engine/run", files=files())

    assert r.status_code == 500


# ---------------------------------------------------------------------------
# Metadatos y salud
# ---------------------------------------------------------------------------

async def test_info_declares_that_preprocessing_does_not_help(client):
    """Es el dato con el que el coordinador decide no gastarle variantes."""
    body = (await client.get("/engine/info")).json()

    assert body["engine"] == "paddleocr"
    assert body["supports_preprocessing"] is False


async def test_livez_has_no_dependencies(client):
    r = await client.get("/engine/livez")

    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_readyz_is_ok_when_the_model_loads(client):
    with patch("app.engine.is_available", return_value=True), \
         patch("app.engine.load", return_value=object()):
        body = (await client.get("/engine/readyz")).json()

    assert body["status"] == "ok"


async def test_readyz_reports_error_without_the_dependency(client):
    with patch("app.engine.is_available", return_value=False):
        body = (await client.get("/engine/readyz")).json()

    assert body["status"] == "error"


async def test_readyz_reports_error_when_the_model_fails_to_load(client):
    with patch("app.engine.is_available", return_value=True), \
         patch("app.engine.load", side_effect=RuntimeError("sin memoria")):
        body = (await client.get("/engine/readyz")).json()

    assert body["status"] == "error"
    assert "sin memoria" in body["detail"]


# ---------------------------------------------------------------------------
# Clave compartida
# ---------------------------------------------------------------------------

async def test_without_a_configured_key_no_header_is_required(client, fake_predictor):
    fake_predictor(DETECTIONS)

    with patch("app.engine.is_available", return_value=True):
        r = await client.post("/engine/run", files=files())

    assert r.status_code == 200


async def test_a_configured_key_is_enforced(client, monkeypatch):
    monkeypatch.setattr(settings, "ENGINE_API_KEY", "s3cr3t")

    r = await client.post("/engine/run", files=files())

    assert r.status_code == 401


async def test_the_right_key_is_accepted(client, fake_predictor, monkeypatch):
    monkeypatch.setattr(settings, "ENGINE_API_KEY", "s3cr3t")
    fake_predictor(DETECTIONS)

    with patch("app.engine.is_available", return_value=True):
        r = await client.post(
            "/engine/run", files=files(), headers={"X-Engine-Key": "s3cr3t"}
        )

    assert r.status_code == 200
