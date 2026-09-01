FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        # PaddlePaddle enlaza contra OpenMP: sin libgomp el import falla.
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Los dos: `requirements.txt` referencia al base con `-r`, así que sin él el
# install falla.
COPY requirements.txt requirements-base.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Los pesos de PP-OCR se descargan una sola vez, en build: así el contenedor
# arranca sin depender de la red y la primera petición no paga la descarga.
# Se instancia con la misma configuración que usa el servicio para que los
# modelos que quedan en caché sean exactamente los que se van a cargar.
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='es', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)"

COPY . .

EXPOSE 8007

# `HOST` y `PORT` salen del entorno para no tener que rearmar la imagen al
# desplegar. En Railway `PORT` lo inyecta la plataforma; `HOST` sólo hay que
# tocarlo si el servicio tiene que escuchar en IPv6 (`::`) para la red privada.
CMD ["sh", "-c", "uvicorn app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8007}"]
