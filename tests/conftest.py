"""Fixtures compartidas.

Los tests no cargan los pesos de PP-OCR —son varios segundos y cientos de MB—:
se dobla el predictor. Lo que sí se ejercita de verdad es el giro, el acotado de
ancho y la traducción de las detecciones a palabras con línea, que es lo que
este servicio aporta además de invocar al modelo.
"""

import io
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw

os.environ.setdefault("DEBUG", "true")

from app.main import app  # noqa: E402


@pytest.fixture
async def client():
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as instance:
        yield instance


def render_image(text: str = "LECHE ENTERA", size: tuple[int, int] = (600, 300)) -> bytes:
    """JPEG con texto oscuro sobre fondo claro."""
    image = Image.new("RGB", size, color=(255, 255, 255))
    ImageDraw.Draw(image).text((30, 60), text, fill=(0, 0, 0))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


@pytest.fixture
def image_bytes() -> bytes:
    return render_image()


@pytest.fixture
def color_image() -> np.ndarray:
    image = np.full((200, 600, 3), 240, dtype=np.uint8)
    image[80:120, 50:400] = (30, 30, 30)
    return image


def box(left: int, top: int, width: int, height: int) -> np.ndarray:
    """Cuadrilátero en el formato que devuelve PP-OCR (cuatro vértices)."""
    return np.array(
        [
            [left, top],
            [left + width, top],
            [left + width, top + height],
            [left, top + height],
        ],
        dtype=np.float32,
    )


def detection(texts, scores, polygons, key: str = "rec_polys") -> list[dict]:
    """Salida de `predict`: una entrada por imagen, con listas paralelas dentro."""
    return [{"rec_texts": list(texts), "rec_scores": list(scores), key: list(polygons)}]


class FakePredictor:
    """Doble del predictor de PaddleOCR: devuelve detecciones fijas."""

    def __init__(self, detections):
        self.detections = detections
        self.calls: list[np.ndarray] = []

    def predict(self, image):
        self.calls.append(image)
        return self.detections


@pytest.fixture
def fake_predictor(monkeypatch):
    def install(detections):
        predictor = FakePredictor(detections)
        monkeypatch.setattr("app.engine.load", lambda *_args, **_kwargs: predictor)
        return predictor

    return install
