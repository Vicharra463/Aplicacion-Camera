"""Linea de PRIORIDAD: la toma mas cerca de la linea gana con tolerancia.

Reglas (pedido FELIPE 2026-11-09):
  * la linea azul PRIORIDAD (config [roi] linea_prioridad) es la
    referencia de MEJOR FOTO en vez del centro de la franja;
  * gana la mas CERCA de la linea con una tolerancia chica
    (roi_tolerancia_prioridad, 0.02): "un poquito" mas cerca ya cuenta;
  * sin linea calibrada -> rollback al centro de la franja (criterio
    viejo) y no se dibuja nada azul.
"""
import numpy as np
import pytest

import worker

SUP, INF, LIN, TOL = 0.35, 0.85, 0.60, 0.02


@pytest.fixture
def roi_linea():
    previo = getattr(worker, "_cfg", None)
    worker._cfg = {
        "roi_habilitado": True,
        "roi_linea_verde_sup": SUP,
        "roi_linea_verde_inf": INF,
        "roi_linea_prioridad": LIN,
        "roi_tolerancia_prioridad": TOL,
    }
    yield worker._cfg
    worker._cfg = previo


@pytest.fixture
def roi_sin_linea(roi_linea):
    worker._cfg["roi_linea_prioridad"] = 0.0
    return worker._cfg


# ------------------------------------------------------------- distancia ---

def test_distancia_se_mide_desde_la_linea(roi_linea):
    assert worker._distancia_franja(LIN, SUP, INF) == pytest.approx(0.0)
    assert worker._distancia_franja(0.50, SUP, INF) == pytest.approx(0.10)


def test_distancia_sin_linea_usa_el_centro_de_la_franja(roi_sin_linea):
    # centro = (0.35+0.85)/2 = 0.60 (mismo valor aca, pero el camino
    # distinto: si la linea estuviera en 0.70 mandaria ella)
    assert worker._distancia_franja(0.70, SUP, INF) == pytest.approx(0.10)


def test_linea_fuera_de_rango_se_ignora(roi_linea):
    worker._cfg["roi_linea_prioridad"] = 1.5
    assert worker._linea_prioridad() is None
    worker._cfg["roi_linea_prioridad"] = -0.2
    assert worker._linea_prioridad() is None


# -------------------------------------------------------------- mejora ---

def test_en_la_linea_gana_sobre_la_que_estaba_atras(roi_linea):
    # nueva cy 0.60 (dist 0) vs vieja 0.38 (dist 0.22)
    assert worker._mejora_por_posicion(LIN, 0.38) is True


def test_la_atras_no_gana_sobre_la_de_la_linea(roi_linea):
    assert worker._mejora_por_posicion(0.38, LIN) is False


def test_mejora_chica_ya_cuenta_con_tolerancia(roi_linea):
    # dist 0.20 -> 0.16: mejora 0.04 > tolerancia 0.02 -> GANA
    # (con el viejo umbral del 25% esto NO ganaba)
    assert worker._mejora_por_posicion(0.76, 0.80) is True


def test_mejora_dentro_de_la_tolerancia_no_voltea(roi_linea):
    # dist 0.20 -> 0.19: mejora 0.01 < 0.02 -> no cuenta (anti-parpadeo)
    assert worker._mejora_por_posicion(0.79, 0.80) is False


def test_igual_distancia_no_gana(roi_linea):
    assert worker._mejora_por_posicion(0.50, 0.50) is False


def test_pegar_la_linea_por_detras_no_gana(roi_linea):
    # cy 0.65: dist 0.05 pero del OTRO lado de la linea; la vieja esta
    # EN la linea (dist 0) -> la que ya esta en la linea no se degrada
    assert worker._mejora_por_posicion(0.65, LIN) is False


def test_sin_roi_es_false(sin_roi_cfg=None):
    previo = getattr(worker, "_cfg", None)
    worker._cfg = {"roi_habilitado": False}
    try:
        assert worker._mejora_por_posicion(0.60, 0.38) is False
    finally:
        worker._cfg = previo


def test_sin_etiqueta_previa_es_false(roi_linea):
    assert worker._mejora_por_posicion(LIN, None) is False


# -------------------------------------------------------------- dibujo ---

def test_dibuja_la_linea_azul_con_etiqueta(roi_linea):
    img = np.full((100, 160, 3), 60, dtype=np.uint8)
    out = worker._dibujar_roi(img, SUP, INF)
    fila = out[int(LIN * 100)]
    # azul BGR puro en al menos 50 pixeles de la fila
    azul = (fila[:, 0].astype(int) > 200) & (fila[:, 1] < 60) \
        & (fila[:, 2] < 60)
    assert int(azul.sum()) >= 50


def test_sin_linea_no_pinta_azul(roi_sin_linea):
    img = np.full((100, 160, 3), 60, dtype=np.uint8)
    out = worker._dibujar_roi(img, SUP, INF)
    fila = out[int(LIN * 100)]
    azul = (fila[:, 0].astype(int) > 200) & (fila[:, 1] < 60) \
        & (fila[:, 2] < 60)
    assert int(azul.sum()) == 0


# -------------------------------------------------------------- config ---

def test_config_ini_expone_linea_y_tolerancia():
    import configparser
    from pathlib import Path
    raiz = Path(__file__).resolve().parents[1]
    p = configparser.ConfigParser()
    p.read(raiz / "config.ini", encoding="utf-8-sig")
    lin = p.getfloat("roi", "linea_prioridad")
    tol = p.getfloat("roi", "tolerancia_prioridad")
    assert 0.0 < lin < 1.0
    assert 0.0 < tol < 0.10


def test_cargar_config_pasa_la_linea_al_worker():
    # el gotcha conocido: sin la clave en cargar_config() el ini no
    # hace nada (default silencioso)
    import ftp_server
    from pathlib import Path
    ini_dir = Path(ftp_server.__file__).resolve().parent
    cfg = ftp_server.cargar_config()
    assert "roi_linea_prioridad" in cfg
    assert "roi_tolerancia_prioridad" in cfg
    assert 0.0 < cfg["roi_linea_prioridad"] < 1.0
    assert 0.0 < cfg["roi_tolerancia_prioridad"] < 0.10
    assert (ini_dir / "config.ini").is_file()
