"""Orden mascara OSD vs recorte ROI: la mascara esta calibrada sobre el
encuadre ORIGINAL. Si se aplica despues del corte (como hacia detectar),
la zona de arriba-izquierda cae sobre la placa del recorte y la tapa:
medido con GT-29, AFH-559 cumple la franja pero el detector no la veia
(con_osd=0, sin_osd=1). Con ROI activo: mascara ANTES de recortar y el
detector entra ya enmascarado (enmascarar=False).
"""
import numpy as np
import pytest

import modelo
import worker

SUP, INF = 0.45, 0.75


@pytest.fixture
def roi_on(monkeypatch):
    monkeypatch.setattr(worker, "_cfg", {
        "roi_habilitado": True,
        "roi_linea_verde_sup": SUP,
        "roi_linea_verde_inf": INF,
    })


@pytest.fixture
def roi_off(monkeypatch):
    monkeypatch.setattr(worker, "_cfg", {"roi_habilitado": False})


def _hay(color, img):
    """Cuenta pixeles cercanos a un color BGR puro (tolerancia 60)."""
    b, g, r = (img[..., 0].astype(int), img[..., 1].astype(int),
               img[..., 2].astype(int))
    return int(((np.abs(b - color[0]) < 60)
                & (np.abs(g - color[1]) < 60)
                & (np.abs(r - color[2]) < 60)).sum())


ROJO = (0, 0, 255)
AZUL = (255, 0, 0)


def _foto_con_marcas():
    """Blanco 160x120: marca ROJA en la zona roja (y 10..30) y marca
    AZUL en la esquina inferior-derecha (zona OSD 'Camera 01')."""
    img = np.full((160, 120, 3), 255, np.uint8)
    img[10:30, 10:50] = ROJO          # placa detras de la linea (rojo)
    img[140:155, 95:115] = AZUL       # texto OSD original abajo-dcha
    return img


# ------------------------------------------------- lo rojo no se detecta ---

def test_lo_que_esta_en_rojo_no_llega_al_detector(roi_on):
    img = _foto_con_marcas()
    salida, _y = worker._preparar_imagen(img)
    # el recorte empieza en la fila 72 (0.45*160): la marca roja (10..30)
    # quedo EN la zona cortada. El detector nunca la ve = no la detecta.
    assert salida.shape[0] == 88
    assert _hay(ROJO, salida) == 0


def test_lo_de_la_franja_si_llega_al_detector(roi_on):
    img = np.full((160, 120, 3), 255, np.uint8)
    img[90:110, 30:90] = ROJO          # placa DENTRO de la franja
    salida, _y = worker._preparar_imagen(img)
    assert _hay(ROJO, salida) > 0


# ------------------------------------------- mascara antes del recorte ---

def test_mascara_osd_no_tapa_la_placa_del_recorte(roi_on):
    img = np.full((160, 120, 3), 255, np.uint8)
    # placa cumplidora: justo DEBAJO del corte (y 76..96) y con x < 0.34
    # (columnas 5..45): exactamente donde caeria la mascara de arriba si
    # se aplicara en coords de la imagen recortada.
    img[76:96, 5:45] = ROJO
    salida, _y = worker._preparar_imagen(img)
    # con mascara-antes-de-recortar la placa sobrevive intacta
    assert _hay(ROJO, salida) > 400
    # (con el orden viejo crop->mascara la tapaba: 0 pixeles rojos)


def test_mascara_osd_sigue_ocultando_el_texto_de_abajo(roi_on):
    img = _foto_con_marcas()
    salida, _y = worker._preparar_imagen(img)
    # la marca azul original (y 140..155, dentro de la zona OSD de la
    # esquina) queda negra tras mascara+recorte: el detector no la ve
    assert _hay(AZUL, salida) == 0


def test_sin_roi_no_se_enmascara_antes_de_recortar(roi_off):
    img = _foto_con_marcas()
    salida, y_sup = worker._preparar_imagen(img)
    # rollback: imagen completa, sin tocar (la mascara la sigue aplicando
    # detectar por su cuenta, como siempre)
    assert salida.shape == img.shape
    assert y_sup == 0.0
    assert _hay(ROJO, salida) > 0


# --------------------------------------------------------- enmascarar ---

class _SesionFalsa:
    def run(self, *_a, **_k):
        return [np.zeros((1, 1), np.float32)]


def _detector_falso():
    d = modelo.DetectorPlacas.__new__(modelo.DetectorPlacas)
    d.umbral = 0.25
    d.lado = 640
    d.nombre_entrada = "x"
    d.sesion = _SesionFalsa()
    d._postprocesar = lambda *a, **k: []
    return d


def test_detectar_con_enmascarar_false_no_vuelve_a_enmascarar(monkeypatch):
    llamadas = []
    monkeypatch.setattr(modelo, "enmascarar_osd",
                        lambda img: llamadas.append(1) or img)
    d = _detector_falso()
    img = np.zeros((40, 40, 3), np.uint8)

    d.detectar(img, enmascarar=False)
    assert llamadas == []           # la imagen ya viene enmascarada

    d.detectar(img)                 # default: comportamiento de siempre
    assert len(llamadas) == 1


def test_candidatos_placa_con_roi_pasa_enmascarar_false(roi_on, monkeypatch):
    class _Registra:
        def __init__(self):
            self.kwargs = []

        def detectar(self, img, enmascarar=True):
            self.kwargs.append(enmascarar)
            return []

    fake = _Registra()
    monkeypatch.setattr(worker, "_detector", fake)
    worker._candidatos_placa(np.zeros((40, 40, 3), np.uint8))
    assert fake.kwargs == [False]   # ROI activo: ya viene enmascarada


def test_candidatos_placa_sin_roi_enmascara_como_antes(roi_off, monkeypatch):
    class _Registra:
        def __init__(self):
            self.kwargs = []

        def detectar(self, img, enmascarar=True):
            self.kwargs.append(enmascarar)
            return []

    fake = _Registra()
    monkeypatch.setattr(worker, "_detector", fake)
    worker._candidatos_placa(np.zeros((40, 40, 3), np.uint8))
    assert fake.kwargs == [True]    # sin ROI: mascara interna de siempre
