"""Preparación geométrica y traducción de la salida de PaddleOCR."""

import numpy as np
import pytest

from app import engine, prepare
from app.core.config import settings
from app.schemas import Binarize, Transform
from tests.conftest import box, detection

IMAGE = np.full((200, 600, 3), 240, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Preparación
# ---------------------------------------------------------------------------

def test_decode_returns_colour(image_bytes):
    """PP-OCR trabaja en color: convertir a grises acá lo degradaría."""
    image = prepare.decode(image_bytes)

    assert image.ndim == 3
    assert image.shape[2] == 3


def test_decode_rejects_something_that_is_not_an_image():
    with pytest.raises(prepare.DecodeError):
        prepare.decode(b"esto no es una imagen")


@pytest.mark.parametrize("degrees", [90, 270])
def test_rotate_a_quarter_turn_swaps_the_axes(degrees):
    assert prepare.rotate(IMAGE, degrees).shape[:2] == IMAGE.shape[:2][::-1]


def test_rotate_zero_is_a_no_op():
    assert np.array_equal(prepare.rotate(IMAGE, 0), IMAGE)


def test_bound_width_shrinks_and_reports_the_scale():
    wide = np.full((400, 2000, 3), 240, dtype=np.uint8)

    result, scale = prepare.bound_width(wide, 800)

    assert result.shape[1] == 800
    assert scale == pytest.approx(0.4)


def test_bound_width_leaves_a_small_image_untouched():
    result, scale = prepare.bound_width(IMAGE, 800)

    assert result is IMAGE
    assert scale == 1.0


def test_the_service_ceiling_wins_over_the_requested_width(monkeypatch):
    """El coordinador pide resolución; el techo de memoria del motor manda."""
    monkeypatch.setattr(settings, "MAX_WIDTH", 400)
    wide = np.full((400, 2000, 3), 240, dtype=np.uint8)

    result, _ = prepare.run(wide, Transform(max_width=1600))

    assert result.shape[1] == 400


def test_the_requested_width_wins_when_it_is_smaller(monkeypatch):
    monkeypatch.setattr(settings, "MAX_WIDTH", 1600)
    wide = np.full((400, 2000, 3), 240, dtype=np.uint8)

    result, _ = prepare.run(wide, Transform(max_width=800))

    assert result.shape[1] == 800


def test_the_filters_of_the_contract_are_ignored():
    """Se aceptan por contrato, pero degradan a este motor: no se aplican."""
    spec = Transform(grayscale=True, invert=True, contrast=True, binarize=Binarize.OTSU)

    result, _ = prepare.run(IMAGE, spec)

    assert result.ndim == 3  # sigue en color


# ---------------------------------------------------------------------------
# Traducción de detecciones
# ---------------------------------------------------------------------------

def run(detections, fake_predictor, transform: Transform | None = None):
    fake_predictor(detections)
    return engine.run(IMAGE, "spa", "block", transform or Transform())


def test_words_carry_text_box_and_confidence(fake_predictor):
    result = run(detection(["LECHE"], [0.93], [box(10, 10, 100, 30)]), fake_predictor)

    word = result.words[0]

    assert word.text == "LECHE"
    assert word.confidence == pytest.approx(0.93)
    assert (word.left, word.top, word.width, word.height) == (10, 10, 100, 30)


def test_empty_detections_are_dropped(fake_predictor):
    result = run(
        detection(
            ["LECHE", "   "], [0.9, 0.4], [box(10, 10, 100, 30), box(0, 0, 5, 5)]
        ),
        fake_predictor,
    )

    assert len(result.words) == 1


def test_words_on_the_same_height_share_a_line(fake_predictor):
    """PP-OCR no numera líneas: se agrupan por proximidad vertical."""
    result = run(
        detection(
            ["LECHE", "ENTERA", "1L"],
            [0.9, 0.9, 0.9],
            [box(10, 10, 100, 30), box(130, 12, 110, 30), box(10, 90, 60, 28)],
        ),
        fake_predictor,
    )

    assert result.text == "LECHE ENTERA\n1L"


def test_boxes_come_back_in_the_original_coordinates(fake_predictor, monkeypatch):
    """El recorte interno no debe filtrarse al coordinador."""
    monkeypatch.setattr(settings, "MAX_WIDTH", 300)
    wide = np.full((400, 600, 3), 240, dtype=np.uint8)

    fake_predictor(detection(["LECHE"], [0.9], [box(50, 50, 100, 20)]))
    result = engine.run(wide, "spa", "block", Transform(max_width=300))

    # La imagen se redujo a la mitad: la caja se reexpande al doble.
    assert result.words[0].left == 100


def test_confidence_is_clamped(fake_predictor):
    result = run(detection(["X"], [1.4], [box(0, 0, 10, 10)]), fake_predictor)

    assert result.words[0].confidence == 1.0


def test_a_slanted_polygon_becomes_its_bounding_box(fake_predictor):
    """El contrato es una caja recta; PP-OCR puede devolver el cuadrilátero girado."""
    slanted = np.array([[20, 10], [120, 30], [116, 60], [16, 40]], dtype=np.float32)

    result = run(detection(["VTO"], [0.9], [slanted]), fake_predictor)

    word = result.words[0]

    assert (word.left, word.top) == (16, 10)
    assert (word.width, word.height) == (104, 50)


def test_the_response_reports_the_engine_and_the_elapsed_time(fake_predictor):
    result = run(detection(["X"], [0.9], [box(0, 0, 10, 10)]), fake_predictor)

    assert result.engine == "paddleocr"
    assert result.elapsed_ms >= 0


def test_an_empty_reading_returns_no_words(fake_predictor):
    result = run(detection([], [], []), fake_predictor)

    assert result.words == []
    assert result.text == ""


def test_a_predict_that_returns_nothing_is_survivable(fake_predictor):
    """Sobre una imagen sin texto PP-OCR puede no devolver ninguna entrada."""
    result = run([None], fake_predictor)

    assert result.words == []


def test_detection_polygons_are_used_when_there_are_no_recognition_ones(fake_predictor):
    """`rec_polys` no está en todas las versiones; `dt_polys` es el respaldo."""
    result = run(
        detection(["LECHE"], [0.9], [box(10, 10, 100, 30)], key="dt_polys"),
        fake_predictor,
    )

    assert result.words[0].left == 10


def test_a_text_without_its_polygon_still_comes_back(fake_predictor):
    """Un texto sin caja se ofrece igual: la confianza es lo que decide, no la caja."""
    result = run(detection(["LECHE", "ENTERA"], [0.9, 0.8], [box(10, 10, 100, 30)]), fake_predictor)

    assert [w.text for w in result.words] == ["LECHE", "ENTERA"]


def test_the_rotation_reaches_the_model(fake_predictor):
    predictor = fake_predictor(detection(["X"], [0.9], [box(0, 0, 10, 10)]))

    engine.run(IMAGE, "spa", "block", Transform(rotate=90))

    assert predictor.calls[0].shape[:2] == IMAGE.shape[:2][::-1]


# ---------------------------------------------------------------------------
# Disponibilidad
# ---------------------------------------------------------------------------

def test_languages_are_reported_with_tesseract_codes():
    """El resto del sistema habla en `spa`/`eng`; la traducción es interna."""
    assert engine.languages() == ["eng", "spa"]
