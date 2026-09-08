"""Preparación de la imagen para PaddleOCR.

Es la misma cadena corta que la de `ocr_service_easyocr`, y por el mismo motivo
medido: PP-OCR hace su propio preprocesamiento (normaliza, reescala a múltiplo
de 32 y ecualiza dentro de la red) y sus modelos están entrenados sobre imágenes
en color. La cadena de filtros que hace rendir a Tesseract —grises, inversión,
CLAHE, binarización— le **quita** información al detector en vez de agregársela.
Por eso este servicio declara `supports_preprocessing=False` y el coordinador no
le gasta reintentos variando filtros que va a ignorar.

Lo que sí aplica son las dos operaciones puramente geométricas que el
coordinador necesita poder pedir: el giro en múltiplos de 90° y el tope de ancho.
"""

import logging

import cv2
import numpy as np

from app.core.config import settings
from app.schemas import Transform

logger = logging.getLogger(__name__)

_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class DecodeError(ValueError):
    """El binario recibido no es una imagen decodificable."""


def decode(content: bytes) -> np.ndarray:
    """Decodifica bytes a una matriz BGR en color (PP-OCR trabaja en color)."""
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise DecodeError("El archivo no es una imagen válida")
    return image


def rotate(image: np.ndarray, degrees: int) -> np.ndarray:
    """Gira la imagen en múltiplos de 90°, como pidió el coordinador."""
    code = _ROTATIONS.get(degrees % 360)
    return image if code is None else cv2.rotate(image, code)


def bound_width(image: np.ndarray, limit: int) -> tuple[np.ndarray, float]:
    """Acota el ancho. Devuelve la imagen y el factor de escala aplicado.

    El factor se usa después para devolver las cajas en coordenadas de la imagen
    que recibió el servicio, de modo que el coordinador no tenga que saber nada
    de este recorte interno.
    """
    width = image.shape[1]
    if width <= limit or limit <= 0:
        return image, 1.0

    scale = limit / width
    resized = cv2.resize(
        image,
        (limit, max(1, int(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    logger.debug("Imagen reducida de %d a %d px de ancho", width, limit)

    return resized, scale


# Recorte a una banda de texto. Ver `crop_to_band`.
#
# Núcleos proporcionales al ancho de la imagen: un renglón de texto ocupa una
# fracción del cuadro, no una cantidad fija de píxeles.
_BAND_MIN_WIDTH_RATIO = 0.06
_BAND_MIN_HEIGHT_RATIO = 0.004
_BAND_MAX_HEIGHT_RATIO = 0.35
_BAND_MIN_ASPECT = 3.0
_BAND_MARGIN = 0.5


def text_bands(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Franjas horizontales con textura de texto, de más a menos prometedora.

    Existe para el sello de vencimiento. En una foto de envase el sello es un
    renglón diminuto dentro de un cuadro de 4000 px: ampliado junto con todo lo
    demás sigue midiendo unos pocos píxeles de alto, y ningún motor lo levanta.
    Recortado, el modelo recibe el renglón y no el envase entero: deja de competir
    con la etiqueta impresa, con el fondo de la mesada y con la mano que sostiene
    el producto. Lo que **no** se hace es ampliarlo — `bound_width` sólo achica,
    y está medido: llevar la banda a más ancho con interpolación no agrega
    información y el reconocimiento empeora (un lote que sale `LM85131262` a
    891 px sale `LNSP131212` llevado a 1280).

    **No intenta adivinar cuál de las franjas es el sello.** Esa discriminación
    es poco confiable —la etiqueta impresa también es texto oscuro sobre claro, y
    medido sobre las fotos reales el criterio de "la más marcada" elige el
    teléfono de atención al consumidor— así que devuelve varias ordenadas y deja
    que el coordinador pruebe las primeras. Elegir es su trabajo, no el del motor.

    El realce es un blackhat y no un umbral: resalta lo oscuro sobre fondo claro
    sin depender del brillo absoluto, que es lo que hace fallar a un umbral fijo
    sobre un envase brillante.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if height == 0 or width == 0:
        return []

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(9, width // 60), max(3, height // 200))
    )
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Une los caracteres de un mismo renglón sin unir renglones distintos.
    joiner = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, width // 40), 3))
    lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, joiner)

    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scored: list[tuple[float, tuple[int, int, int, int]]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * _BAND_MIN_WIDTH_RATIO or h < height * _BAND_MIN_HEIGHT_RATIO:
            continue
        if h > height * _BAND_MAX_HEIGHT_RATIO:  # un bloque alto es la etiqueta
            continue
        if w / h < _BAND_MIN_ASPECT:  # un renglón es mucho más ancho que alto
            continue

        density = float(binary[y:y + h, x:x + w].mean()) / 255.0
        scored.append((density * (w / h) ** 0.5, (x, y, w, h)))

    return [box for _, box in sorted(scored, key=lambda item: -item[0])]


def crop_to_band(image: np.ndarray, index: int) -> np.ndarray:
    """Recorta a la banda `index`. Si no existe, devuelve la imagen entera.

    Devolver la imagen sin tocar —en vez de fallar— hace que un índice de más
    cueste sólo un intento repetido y nunca una lectura perdida.
    """
    if index < 0:
        return image

    bands = text_bands(image)
    if index >= len(bands):
        return image

    x, y, w, h = bands[index]
    margin_y, margin_x = int(h * _BAND_MARGIN), int(w * 0.04)
    height, width = image.shape[:2]

    return image[
        max(0, y - margin_y):min(height, y + h + margin_y),
        max(0, x - margin_x):min(width, x + w + margin_x),
    ]


def run(image: np.ndarray, transform: Transform) -> tuple[np.ndarray, float]:
    """Gira y acota. Devuelve la imagen preparada y el factor de escala."""
    rotated = rotate(image, transform.normalized_rotation)
    rotated = crop_to_band(rotated, transform.crop_band)

    # `max_width` del contrato es el techo que pide el coordinador; el del
    # servicio es el techo que impone la memoria disponible. Manda el más chico.
    limit = min(transform.max_width, settings.MAX_WIDTH) if transform.rescale else settings.MAX_WIDTH

    return bound_width(rotated, limit)
