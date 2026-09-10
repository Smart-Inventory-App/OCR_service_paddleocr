"""Ejecución de PaddleOCR.

La propuesta lo describe como el motor de mayor precisión sobre "imágenes
complejas o de baja calidad", robusto frente a rotaciones y ruido, a cambio de
mayor complejidad de instalación, dependencia de PaddlePaddle y un consumo de
recursos "significativamente mayor". Este servicio existe para poder verificar
esa descripción sobre las fotos reales del proyecto en lugar de darla por buena.

Como el de Tesseract, este módulo hace una sola cosa: correr el motor
y devolver las palabras con su caja y su confianza. No interpreta el resultado,
no extrae campos y no decide reintentos: de eso se ocupa `ocr_service`.
"""

import logging
import threading
import time

import numpy as np

from app.core.config import settings
from app.prepare import run as prepare
from app.schemas import EngineResponse, Transform, Word

logger = logging.getLogger(__name__)

NAME = "paddleocr"

# PaddleOCR usa códigos ISO-639-1; el resto del sistema habla en códigos de
# Tesseract (`spa`, `eng`).
_LANG_CODES = {"spa": "es", "eng": "en"}

# Alto en píxeles por debajo del cual dos cajas se consideran de la misma línea.
_LINE_TOLERANCE_RATIO = 0.6

_predictor = None
_predictor_lang: str | None = None
# Construir el predictor no es reentrante y el servidor atiende en varios
# threads: sin el lock, dos peticiones simultáneas al arrancar cargan los pesos
# dos veces y duplican el pico de memoria.
_lock = threading.Lock()


def is_available() -> bool:
    try:
        import paddleocr  # noqa: F401
        return True
    except ImportError:
        return False


def version() -> str:
    try:
        import paddleocr
        return getattr(paddleocr, "__version__", "desconocida")
    except ImportError:
        return "no instalado"


def languages() -> list[str]:
    return sorted(_LANG_CODES)


def load(lang: str | None = None):
    """Construye el predictor (costoso) y lo memoiza. Es idempotente."""
    global _predictor, _predictor_lang

    code = _LANG_CODES.get(lang or settings.DEFAULT_LANG, lang or "es")

    with _lock:
        if _predictor is None or _predictor_lang != code:
            from paddleocr import PaddleOCR

            started = time.perf_counter()
            _predictor = PaddleOCR(
                lang=code,
                # Los tres módulos opcionales van apagados: el giro en múltiplos
                # de 90° lo decide el coordinador —que prueba las cuatro
                # orientaciones y se queda con la que lee—, y dejarlos
                # encendidos haría que el motor deshaga el giro que le acaban de
                # pedir. Ver `core/config.py`.
                use_doc_orientation_classify=settings.USE_DOC_ORIENTATION,
                use_doc_unwarping=settings.USE_DOC_UNWARPING,
                use_textline_orientation=settings.USE_TEXTLINE_ORIENTATION,
                text_det_box_thresh=settings.DET_BOX_THRESHOLD,
                enable_mkldnn=settings.ENABLE_MKLDNN,
                # Sin tope, cada instancia abre tantos hilos como núcleos ve; con
                # el motor replicado eso hace que las réplicas se peleen la CPU.
                **({"cpu_threads": settings.CPU_THREADS} if settings.CPU_THREADS > 0 else {}),
            )
            _predictor_lang = code
            logger.info(
                "Predictor PaddleOCR '%s' cargado en %.1f s",
                code,
                time.perf_counter() - started,
            )

    return _predictor


def run(image: np.ndarray, lang: str, layout: str, transform: Transform) -> EngineResponse:
    """Reconoce sobre la imagen en color.

    `layout` se acepta por contrato pero se ignora: PP-OCR detecta las regiones
    de texto con su propia red y no expone nada equivalente al *page
    segmentation mode* de Tesseract.
    """
    started = time.perf_counter()

    predictor = load(lang)
    prepared, scale = prepare(image, transform)

    detections = predictor.predict(prepared)

    words = _assign_lines(
        [
            # Las cajas se devuelven en coordenadas de la imagen recibida, para
            # que el coordinador no sepa del recorte interno.
            _to_word(polygon, text, confidence, scale)
            for polygon, text, confidence in _flatten(detections)
            if text.strip()
        ]
    )

    height, width = prepared.shape[:2]

    return EngineResponse(
        engine=NAME,
        text=render_text(words),
        words=words,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        image_width=width,
        image_height=height,
        lang=_LANG_CODES.get(lang, lang),
    )


def _flatten(detections) -> list[tuple]:
    """Aplana la salida de `predict` a `(polígono, texto, confianza)`.

    PP-OCR devuelve **una entrada por imagen** —acá siempre una— y dentro de
    ella los resultados vienen en listas paralelas: los textos por un lado, sus
    puntajes por otro y sus polígonos por otro. Se recorren juntas.

    `rec_polys` son los polígonos ya alineados con el texto reconocido;
    `dt_polys` son los que salieron de la detección, antes de descartar las
    regiones que el reconocedor no pudo leer. Se prefiere el primero y se cae al
    segundo, porque de una versión a otra de PaddleOCR no siempre están los dos.
    """
    flattened: list[tuple] = []

    for result in detections or []:
        if result is None:
            continue

        texts = _get(result, "rec_texts") or []
        scores = _get(result, "rec_scores") or []
        polygons = _get(result, "rec_polys")
        if polygons is None or len(polygons) != len(texts):
            polygons = _get(result, "dt_polys") or []

        for index, text in enumerate(texts):
            score = scores[index] if index < len(scores) else 0.0
            polygon = polygons[index] if index < len(polygons) else None
            flattened.append((polygon, str(text), float(score)))

    return flattened


def _get(result, key: str):
    """Lee un campo del resultado, que según la versión es dict o un objeto."""
    if isinstance(result, dict):
        return result.get(key)

    try:
        return result[key]
    except (KeyError, TypeError, IndexError):
        return getattr(result, key, None)


def render_text(words: list[Word]) -> str:
    grouped: dict[int, list[Word]] = {}
    for word in words:
        if word.text.strip():
            grouped.setdefault(word.line_num, []).append(word)

    return "\n".join(
        " ".join(w.text for w in sorted(line, key=lambda w: w.left))
        for _, line in sorted(grouped.items())
    )


def _to_word(polygon, text: str, confidence: float, scale: float) -> Word:
    """Convierte una detección en una palabra con su caja rectangular.

    El polígono de PP-OCR es un cuadrilátero que puede venir inclinado; el
    contrato del coordinador es una caja recta, así que se toma la envolvente.
    """
    factor = 1 / scale if scale else 1.0

    points = np.asarray(polygon, dtype=float).reshape(-1, 2) if polygon is not None else None
    if points is None or points.size == 0:
        left = top = width = height = 0
    else:
        xs = [int(x * factor) for x in points[:, 0]]
        ys = [int(y * factor) for y in points[:, 1]]
        left, top = min(xs), min(ys)
        width, height = max(xs) - left, max(ys) - top

    return Word(
        text=text,
        confidence=min(max(float(confidence), 0.0), 1.0),
        left=left,
        top=top,
        width=width,
        height=height,
    )


def _assign_lines(words: list[Word]) -> list[Word]:
    """PP-OCR no devuelve número de línea: se agrupa por proximidad vertical."""
    if not words:
        return []

    ordered = sorted(words, key=lambda w: (w.top, w.left))
    tolerance = max(1, int(_median_height(ordered) * _LINE_TOLERANCE_RATIO))

    result: list[Word] = []
    line_num = 0
    line_top = ordered[0].top

    for word in ordered:
        if word.top - line_top > tolerance:
            line_num += 1
            line_top = word.top
        result.append(word.model_copy(update={"line_num": line_num}))

    return result


def _median_height(words: list[Word]) -> float:
    heights = sorted(w.height for w in words)
    return heights[len(heights) // 2] or 1.0
