"""Servicio de motor OCR — PaddleOCR.

El tercer motor de la comparativa que pide la propuesta. Vive en su propio
repositorio porque arrastra PaddlePaddle y los pesos de PP-OCR: es el más caro
de los tres en imagen y en memoria. Tenerlo separado permite levantar el
coordinador y el motor liviano sin pagar ese costo, y encenderlo sólo cuando se
lo quiere usar o comparar.

La propuesta lo describe como el de mayor precisión sobre imágenes complejas o
de baja calidad, "robusto frente a rotaciones, ruido y variaciones visuales", a
cambio de mayor complejidad de instalación y consumo de recursos. Este servicio
es lo que permite verificar esa afirmación sobre las fotos reales del proyecto
en lugar de darla por buena.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.routes import router
from app import engine

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not engine.is_available():
        logger.error("`paddleocr` no está instalado: /engine/run va a responder 503")
    else:
        if settings.PRELOAD_MODEL:
            # Cargarlo acá y no en la primera petición: son varios segundos que,
            # si no, los paga el primer usuario.
            engine.load()
        logger.info(
            "Motor PaddleOCR %s listo (ancho máximo %d px)", engine.version(), settings.MAX_WIDTH
        )

    yield


app = FastAPI(
    title="Smart Inventory — Motor OCR PaddleOCR",
    description=(
        "Ejecuta PaddleOCR sobre una imagen y devuelve las palabras reconocidas "
        "con su caja y su confianza. Mismo contrato que los servicios de "
        "Tesseract y EasyOCR: la interpretación del resultado es "
        "responsabilidad de `ocr_service`."
    ),
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    lifespan=lifespan,
)

app.include_router(router)
