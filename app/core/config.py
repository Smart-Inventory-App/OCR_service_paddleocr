from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Servidor
    HOST: str = "0.0.0.0"
    PORT: int = 8007
    DEBUG: bool = False

    DEFAULT_LANG: str = "spa"

    # Techo duro de ancho de entrada. La red de detección de PaddleOCR reescala
    # la imagen a un múltiplo de 32 antes de inferir, y su memoria crece con el
    # cuadrado del lado: este valor no es una preferencia de calidad sino un
    # límite de seguridad.
    #
    # Qué resolución conviene para cada documento lo decide el coordinador y lo
    # pide en `transform.max_width`; acá sólo se le impone este techo.
    MAX_WIDTH: int = 1600

    # Cargar el modelo al arrancar evita que la primera petición pague la
    # inicialización del predictor. Se puede apagar para que el contenedor
    # levante liviano.
    PRELOAD_MODEL: bool = True

    # ── Módulos del pipeline de PaddleOCR ─────────────────────────────────────
    # PP-OCR trae tres etapas opcionales antes del reconocimiento. Las tres
    # están apagadas a propósito y cada una tiene su motivo:
    #
    # - **Orientación del documento** decide el giro en múltiplos de 90°. Esa
    #   decisión es del coordinador, que prueba las cuatro y se queda con la que
    #   lee; dejarla también acá haría que el motor deshiciera el giro que le
    #   acaban de pedir.
    # - **Desdoblado (unwarping)** endereza páginas curvas fotografiadas. Está
    #   pensado para hojas, no para envases: sobre una etiqueta cilíndrica
    #   introduce deformaciones y agrega casi un segundo por corrida.
    # - **Orientación de renglón** corrige textos invertidos 180°. Sobre un
    #   sello de vencimiento de una sola línea confunde más de lo que arregla, y
    #   el giro de 180° ya está en el plan de intentos del coordinador.
    #
    # Se dejan configurables para poder medir el efecto de cada una.
    USE_DOC_ORIENTATION: bool = False
    USE_DOC_UNWARPING: bool = False
    USE_TEXTLINE_ORIENTATION: bool = False

    # Puntaje mínimo para que una región detectada se considere texto. Es el
    # default de PaddleOCR; bajarlo recupera texto de bajo contraste —el caso de
    # los sellos de vencimiento sobre envases brillantes— a costa de más ruido.
    DET_BOX_THRESHOLD: float = 0.6

    # Aceleración oneDNN (MKL-DNN) de PaddlePaddle en CPU. Encendida es la ruta
    # rápida y es lo que corresponde en el contenedor Linux.
    #
    # Hay que apagarla para correr el servicio **directamente sobre Windows**:
    # con `paddlepaddle` 3.3.1 los modelos PP-OCRv6 abortan ahí con
    # `ConvertPirAttribute2RuntimeAttribute not support`, que es un fallo del
    # backend y no de la imagen ni del modelo. Sin oneDNN el motor da el mismo
    # resultado, más lento. El `.env` de desarrollo la deja apagada por eso.
    ENABLE_MKLDNN: bool = True

    # Clave compartida con el coordinador. Vacía = sin control de acceso.
    ENGINE_API_KEY: str = ""

    MAX_UPLOAD_BYTES: int = 15 * 1024 * 1024

settings = Settings()
