# Motor OCR — PaddleOCR

Microservicio que hace **una sola cosa**: correr PaddleOCR sobre una imagen y devolver las palabras reconocidas con su caja y su confianza.

Habla **el mismo contrato** que [`ocr_service_tesseract`](../ocr_service_tesseract) y [`ocr_service_easyocr`](../ocr_service_easyocr): es lo que le permite a [`ocr_service`](../ocr_service) tratar a los tres motores igual, compararlos sobre la misma foto y combinarlos en el modo híbrido.

```
ocr_service  ──POST /engine/run──►  este servicio
             ◄── palabras + confianza ──┘
```

Con este servicio queda implementado el **tercer motor** que nombra la propuesta, y la evaluación comparativa que pide pasa a cubrir a los tres.

---

## Por Qué Existe

La propuesta describe a PaddleOCR como el motor de mayor precisión sobre imágenes complejas o de baja calidad, "robusto frente a rotaciones, ruido y variaciones visuales", a cambio de mayor complejidad de instalación, dependencia de PaddlePaddle y un consumo de recursos "significativamente mayor".

Medido sobre las fotos reales del proyecto, esa descripción **se cumple**, y con un margen que cambia la conclusión del módulo: es el único de los tres motores que alcanza por sí solo los dos criterios de aceptación. Los números están en el [README del coordinador](../ocr_service#resultados-medidos---benchmark); el resumen es que lee el sello de vencimiento en las cuatro fotos —incluida la que ningún otro motor levanta— y con confianzas que no obligan a mandar el documento a revisión manual.

Vive en su propio repositorio por la misma razón que EasyOCR: arrastra PaddlePaddle y los pesos de PP-OCR, **2,68 GB de imagen**. Separado, el motor caro se enciende sólo cuando se lo quiere usar o comparar, y si está apagado el coordinador lo reporta como no disponible y el modo híbrido degrada a su motor principal en lugar de fallar.

---

## Qué Recibe y Qué Devuelve

### `POST /engine/run`

**Body:** `multipart/form-data` — idéntico al de los otros dos motores.

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `file` | archivo (`jpeg`/`png`/`webp`) | la imagen **original**, en color |
| `lang` | string | idioma (`spa` por defecto; se traduce a `es`) |
| `layout` | string | se acepta por contrato, pero **se ignora** |
| `transform` | JSON | de la cadena, sólo se aplican `rotate` y el tope de ancho |

**Qué se ignora y por qué.** Es la misma decisión que en EasyOCR y por el mismo motivo: PP-OCR normaliza, reescala a múltiplo de 32 y ecualiza **dentro de la red**, y sus modelos están entrenados sobre imágenes en color. La cadena de filtros que hace rendir a Tesseract —grises, inversión, CLAHE, binarización— le quita información al detector en vez de agregársela. Aplicarla sería empeorar la lectura a pedido. Lo que sí se aplica es lo puramente geométrico: el giro y la resolución, que son decisiones del coordinador sobre el documento, no sobre el motor.

`layout` se ignora porque PP-OCR detecta las regiones de texto con su propia red y no expone nada equivalente al *page segmentation mode* de Tesseract.

**Respuesta `200`:** el mismo esquema que los otros motores.

```jsonc
{
  "engine": "paddleocr",
  "text": "HARINA\nPUREZA\nESPECIAL\nPARA PIZZAS CASERAS CON",
  "words": [
    { "text": "HARINA", "confidence": 1.0,
      "left": 235, "top": 228, "width": 239, "height": 53, "line_num": 0 }
  ],
  "elapsed_ms": 7243,
  "image_width": 800,
  "image_height": 1422,
  "lang": "es"
}
```

Las cajas se devuelven en **coordenadas de la imagen recibida**, no de la imagen recortada internamente: el coordinador no tiene por qué saber de ese recorte.

Dos diferencias de forma con la salida cruda de PP-OCR, resueltas acá para que el coordinador vea lo mismo en los tres motores:

- **El polígono se vuelve caja recta.** PP-OCR devuelve un cuadrilátero de cuatro vértices que puede venir inclinado; el contrato es una caja, así que se toma la envolvente.
- **Las líneas se numeran.** PP-OCR no numera, así que se agrupan por proximidad vertical, igual que en EasyOCR.

> **Sobre la granularidad.** PP-OCR detecta **regiones de texto**, no palabras: una
> línea entera puede volver como una sola "palabra" del contrato, y a veces dos
> palabras vecinas vuelven pegadas (`CAFÉINSTANTÁNEO`). Es una diferencia real
> con Tesseract, que sí segmenta por palabra. El coordinador la absorbe —su
> parser de etiquetas reconoce la categoría aunque venga pegada— pero conviene
> tenerla presente al comparar las salidas crudas de los tres motores.

**Errores:** `400` formato no soportado, archivo vacío, imagen ilegible o `transform` mal formado · `401` clave compartida inválida · `413` imagen demasiado grande · `503` el paquete `paddleocr` no está instalado · `500` el modelo falló.

### `GET /engine/info`

```json
{ "engine": "paddleocr", "version": "3.7.0",
  "languages": ["eng", "spa"], "supports_preprocessing": false }
```

`supports_preprocessing: false` le dice a `ocr_service` que no gaste reintentos variando filtros que este motor va a ignorar: cada corrida cuesta segundos y no cambiaría nada.

### `GET /engine/livez` · `GET /engine/readyz`

- `livez` → `{ "status": "ok" }`. Sin dependencias.
- `readyz` → verifica que la dependencia esté instalada **y que el predictor se construya**.

---

## Módulos del Pipeline de PP-OCR

PP-OCR trae tres etapas opcionales antes del reconocimiento. **Las tres van apagadas**, y cada una tiene su motivo:

| Módulo | Variable | Por qué apagado |
|---|---|---|
| Orientación del documento | `USE_DOC_ORIENTATION` | decide el giro en múltiplos de 90°, que es una decisión del **coordinador**: prueba las cuatro orientaciones y conserva la que lee. Encendido, el motor deshace el giro que le acaban de pedir. |
| Desdoblado (*unwarping*) | `USE_DOC_UNWARPING` | endereza páginas curvas fotografiadas. Está pensado para hojas, no para envases: sobre una etiqueta cilíndrica deforma, y agrega casi un segundo por corrida. |
| Orientación de renglón | `USE_TEXTLINE_ORIENTATION` | corrige textos invertidos 180°. Sobre un sello de vencimiento de una sola línea confunde más de lo que arregla, y el giro de 180° ya está en el plan de intentos del coordinador. |

Quedan configurables para poder medir el efecto de cada una, pero el reparto de responsabilidades es deliberado: **la estrategia es del coordinador y la ejecución es del motor**, igual que con los filtros.

---

## Manejo de Memoria y Latencia

Medido con `docker stats` sobre el contenedor en reposo y en proceso:

| | Valor |
|---|---|
| Imagen | 2,68 GB |
| Memoria en reposo | ~350 MB |
| Latencia por corrida | 4–10 s (etiqueta), hasta 15 s (sello a máxima resolución) |
| Carga del predictor al arrancar | ~2,6 s |

`MAX_WIDTH` es un **techo de seguridad**, no una preferencia de calidad: la red de detección reescala a múltiplo de 32 y su memoria crece con el cuadrado del lado. Qué resolución conviene para cada documento lo decide el coordinador y lo pide en `transform.max_width`; acá se le impone este techo.

> **Docker Desktop** reparte por defecto ~2 GB entre *todos* los contenedores.
> Con este servicio y el de EasyOCR arriba a la vez conviene subirlo
> (Settings → Resources → Memory, recomendado 4 GB) o bajar `MAX_WIDTH`.

---

## Ejecución

```bash
docker network create tpp-net
```

```bash
cp .env.example .env
```

```bash
docker-compose up -d --build
```

La primera build instala PaddlePaddle y descarga los pesos de PP-OCR **en tiempo de build**, con la misma configuración con la que después se los carga: así el contenedor arranca sin depender de la red y la primera petición no paga la descarga. Tarda varios minutos.

```bash
curl http://localhost:8007/engine/readyz
```

Con `PRELOAD_MODEL=true` el predictor se construye al arrancar, para que el costo no lo pague el primer usuario. Por eso el healthcheck del compose tiene `start_period`.

### Local (sin Docker)

```bash
pip install -r requirements.txt
```

```bash
uvicorn app.main:app --reload --port 8007
```

> **En Windows hay que apagar oneDNN.** Con `paddlepaddle` 3.3.1 los modelos
> PP-OCRv6 abortan ahí con `ConvertPirAttribute2RuntimeAttribute not support`,
> que es un fallo del backend de aceleración y no del modelo ni de este
> servicio. El `.env.example` trae `ENABLE_MKLDNN=true`, que es lo que
> corresponde dentro del contenedor Linux; para correr directamente sobre
> Windows hay que ponerlo en `false`. El resultado es el mismo, más lento.

---

## Configuración

| Variable | Default | Para qué |
|----------|---------|----------|
| `PORT` | `8007` | puerto del servicio |
| `DEFAULT_LANG` | `spa` | idioma cuando la petición no lo especifica |
| `MAX_WIDTH` | `1600` | **techo** de ancho de entrada (límite de memoria) |
| `PRELOAD_MODEL` | `true` | construir el predictor al arrancar |
| `USE_DOC_ORIENTATION` | `false` | orientación del documento (la decide el coordinador) |
| `USE_DOC_UNWARPING` | `false` | desdoblado de página (pensado para hojas, no envases) |
| `USE_TEXTLINE_ORIENTATION` | `false` | orientación de renglón |
| `DET_BOX_THRESHOLD` | `0.6` | puntaje mínimo para considerar texto una región |
| `ENABLE_MKLDNN` | `true` | aceleración oneDNN — **apagar en Windows** |
| `ENGINE_API_KEY` | *(vacía)* | clave compartida con el coordinador |
| `MAX_UPLOAD_BYTES` | `15728640` | tope del binario recibido |

---

## Tests

```bash
pytest
```

**40 tests, 91 % de cobertura.** No construyen el predictor —son segundos y cientos de MB—: se lo dobla. Lo que sí se ejercita es el giro, el acotado de ancho, la reexpansión de las cajas a las coordenadas originales, la conversión del polígono a caja y el agrupamiento por líneas, que es todo lo que este servicio aporta además de invocar al modelo.

| Archivo | Qué cubre |
|---------|-----------|
| `test_engine.py` | preparación geométrica y traducción de las detecciones |
| `test_routes.py` | el contrato HTTP, validaciones, salud y clave compartida |
