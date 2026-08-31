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


def run(image: np.ndarray, transform: Transform) -> tuple[np.ndarray, float]:
    """Gira y acota. Devuelve la imagen preparada y el factor de escala."""
    rotated = rotate(image, transform.normalized_rotation)

    # `max_width` del contrato es el techo que pide el coordinador; el del
    # servicio es el techo que impone la memoria disponible. Manda el más chico.
    limit = min(transform.max_width, settings.MAX_WIDTH) if transform.rescale else settings.MAX_WIDTH

    return bound_width(rotated, limit)
