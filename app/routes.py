"""Endpoints del servicio de motor.

Mismo contrato que `ocr_service_tesseract`: es lo que le permite al 
coordinador tratar a los tres motores igual y compararlos sobre la
misma imagen.
"""

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError

from app.core.config import settings
from app.core.security import verify_engine_key
from app import engine, prepare
from app.schemas import EngineInfo, EngineResponse, HealthResponse, Transform

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/engine", tags=["engine"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/run", response_model=EngineResponse)
async def run(
    file: UploadFile = File(..., description="Imagen original, sin preprocesar"),
    lang: str = Form(default=""),
    layout: str = Form(default="block", description="Se acepta por contrato; PaddleOCR lo ignora"),
    transform: str = Form(
        default="",
        description="Transformación a aplicar, en JSON. Sólo se usan `rotate` y el tope de ancho.",
    ),
    _: None = Depends(verify_engine_key),
):
    """Reconoce el texto de la imagen y devuelve las palabras con su confianza."""
    content = await _read_image(file)
    spec = _parse_transform(transform)

    if not engine.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El paquete `paddleocr` no está instalado en este servicio",
        )

    try:
        image = prepare.decode(content)
    except prepare.DecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    # La inferencia es CPU-bound y bloqueante: va a un thread aparte.
    try:
        result = await asyncio.to_thread(
            engine.run, image, lang or settings.DEFAULT_LANG, layout, spec
        )
    except Exception as exc:
        logger.exception("Falló la ejecución del motor")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"El motor falló al procesar la imagen: {exc}",
        )

    logger.info(
        "OCR: %d palabras, rot=%d (%d ms)",
        len(result.words),
        spec.normalized_rotation,
        result.elapsed_ms,
    )

    return result


@router.get("/info", response_model=EngineInfo)
async def info():
    return EngineInfo(
        engine=engine.NAME,
        version=engine.version(),
        languages=engine.languages(),
        # PP-OCR normaliza y reescala dentro de la red, y sus modelos esperan
        # color: variarle los filtros no mejora nada y cada corrida cuesta
        # segundos. El coordinador lee esto para no gastarle reintentos.
        supports_preprocessing=False,
    )


@router.get("/livez", response_model=HealthResponse)
async def livez():
    return HealthResponse(status="ok", engine=engine.NAME)


@router.get("/readyz", response_model=HealthResponse)
async def readyz():
    """Verifica que la dependencia esté instalada y el modelo se pueda cargar."""
    if not engine.is_available():
        return HealthResponse(
            status="error",
            engine=engine.NAME,
            detail="El paquete `paddleocr` no está instalado",
        )

    try:
        engine.load()
    except Exception as exc:
        return HealthResponse(
            status="error", engine=engine.NAME, detail=f"No se pudo cargar el modelo: {exc}"
        )

    return HealthResponse(status="ok", engine=engine.NAME)


def _parse_transform(raw: str) -> Transform:
    if not raw.strip():
        return Transform()

    try:
        return Transform(**json.loads(raw))
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"`transform` no es un JSON válido para el contrato: {exc}",
        )


async def _read_image(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        accepted = ", ".join(sorted(ALLOWED_CONTENT_TYPES))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato '{file.content_type}' no permitido. Formatos aceptados: {accepted}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="El archivo está vacío"
        )
    if len(content) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"La imagen supera los {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )

    return content
