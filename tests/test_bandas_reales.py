"""La detección de bandas, sobre una foto real y sin motor.

El resto de `test_engine.py` prueba `prepare` con imágenes dibujadas en memoria:
un rectángulo negro sobre fondo blanco alcanza para verificar que el giro gira y
que el acotado acota. Lo que no puede decir es si `text_bands` encuentra un
renglón en una **foto**, con su iluminación despareja, su envase curvo y su fondo
de mesada — y ese es justo el intento que el coordinador le pide a este motor
para el sello de vencimiento.

Por eso hay una sola foto en el repositorio (`tests/fixtures/imagenes/`), de
1400 px: un sello de matriz de puntos impreso en el canto de un envase. No se
ejecuta el modelo, así que corre en el CI como cualquier otro test — que es la
razón por la que las afirmaciones son geométricas y no de contenido. Cuál de las
bandas es el sello no lo decide este servicio: las ordena y el coordinador
prueba las primeras.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from app import prepare
from app.schemas import Transform

FOTO = Path(__file__).parent / "fixtures" / "imagenes" / "vencimiento_sello.jpg"


@pytest.fixture(scope="module")
def envase() -> np.ndarray:
    return prepare.decode(FOTO.read_bytes())


def test_la_foto_del_conjunto_de_prueba_se_decodifica(envase):
    assert envase.shape[:2] == (788, 1400)


def test_encuentra_renglones_en_una_foto_de_verdad(envase):
    """Sobre una foto, no sobre un rectángulo dibujado."""
    bandas = prepare.text_bands(envase)

    assert bandas, "no encontró un solo renglón en la foto"

    alto, ancho = envase.shape[:2]
    for x, y, w, h in bandas:
        assert 0 <= x and 0 <= y and x + w <= ancho and y + h <= alto
        # Un renglón es mucho más ancho que alto, y nunca es media foto.
        assert w / h >= 3.0
        assert h < alto * 0.35


def test_el_recorte_deja_una_fracción_de_la_foto(envase):
    """El sello es un renglón dentro del cuadro: aislarlo tiene que achicarlo."""
    recorte = prepare.crop_to_band(envase, 0)

    assert recorte.size < envase.size / 4


def test_una_banda_que_no_existe_devuelve_la_foto_entera(envase):
    """Un índice de más cuesta un intento repetido, nunca una lectura perdida."""
    assert prepare.crop_to_band(envase, 999).shape == envase.shape
    assert prepare.crop_to_band(envase, -1).shape == envase.shape


def test_la_cadena_que_pide_el_coordinador_para_el_sello(envase):
    """Giro + recorte + acotado, tal como llega el intento desde el coordinador."""
    preparada, escala = prepare.run(envase, Transform(rotate=90, crop_band=0, max_width=1280))

    assert preparada.shape[1] <= 1280
    assert 0 < escala <= 1.0
    # El giro se aplica antes del recorte, así que la banda sale de la foto ya
    # girada y no de la original.
    derecha, _ = prepare.run(envase, Transform(rotate=0, crop_band=0, max_width=1280))
    assert preparada.shape[:2] != derecha.shape[:2]


def test_el_recorte_no_encoge_una_foto_que_ya_entra(envase):
    """`bound_width` sólo achica: una banda angosta llega al modelo tal cual.

    Está medido y es deliberado — ampliarla con interpolación no agrega
    información y el reconocimiento empeora: sobre esta misma carpeta, un lote
    que sale `LM85131262` a 891 px de ancho sale `LNSP131212` llevado a 1280.
    """
    recorte = prepare.crop_to_band(envase, 0)
    preparada, escala = prepare.run(
        cv2.copyMakeBorder(recorte, 0, 0, 0, 0, cv2.BORDER_CONSTANT),
        Transform(crop_band=-1, max_width=4000),
    )

    assert escala == 1.0
    assert preparada.shape[1] == recorte.shape[1]
