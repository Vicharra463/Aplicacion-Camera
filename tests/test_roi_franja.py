"""ROI franja verde: recorte arriba, placa dentro de la franja y
MEJOR FOTO por posicion. Ver seccion [roi] de config.ini.
"""
from pathlib import Path
import re

import pytest

import worker

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture
def roi(monkeypatch):
    """ROI activo: franja 0.40-0.80 sobre la imagen original."""
    monkeypatch.setattr(worker, "_cfg", {
        "roi_habilitado": True,
        "roi_linea_verde_sup": 0.40,
        "roi_linea_verde_inf": 0.80,
    })


@pytest.fixture
def sin_roi(monkeypatch):
    monkeypatch.setattr(worker, "_cfg", {"roi_habilitado": False})


# --------------------------------------------------------------- recorte ---

def test_recortar_roi_apagado_pasa_la_imagen_completa(sin_roi):
    import numpy as np
    img = np.zeros((100, 80, 3), np.uint8)
    salida, y_sup = worker._recortar_roi(img)
    assert salida.shape == img.shape
    assert salida is img
    assert y_sup == 0.0


def test_recortar_roi_corta_todo_lo_de_arriba(roi):
    import numpy as np
    img = np.arange(100 * 80 * 3, dtype=np.uint8).reshape(100, 80, 3)
    salida, y_sup = worker._recortar_roi(img)
    # sup 0.40 de 100 filas -> 40 cortadas, quedan 60
    assert salida.shape == (60, 80, 3)
    assert y_sup == pytest.approx(40.0)
    # la primera fila de la salida es la fila 40 del original (lo de
    # arriba, lo rojo, ya no existe)
    assert (salida[0] == img[40]).all()
    assert (salida[-1] == img[99]).all()


def test_recortar_roi_config_invalida_no_recorta(monkeypatch):
    import numpy as np
    monkeypatch.setattr(worker, "_cfg", {
        "roi_habilitado": True,
        "roi_linea_verde_sup": 0.80,   # sup > inf: config rota
        "roi_linea_verde_inf": 0.30,
    })
    img = np.zeros((100, 80, 3), np.uint8)
    salida, y_sup = worker._recortar_roi(img)
    assert salida is img
    assert y_sup == 0.0


def test_recortar_roi_franja_minima_no_recorta(monkeypatch):
    import numpy as np
    monkeypatch.setattr(worker, "_cfg", {
        "roi_habilitado": True,
        "roi_linea_verde_sup": 0.50,
        "roi_linea_verde_inf": 0.52,   # franja de 2% (< 5% minimo)
    })
    img = np.zeros((100, 80, 3), np.uint8)
    salida, _ = worker._recortar_roi(img)
    assert salida is img


# --------------------------------------------------------------- franja ---

def test_franja_inf_px_roi_apagado_es_none(sin_roi):
    assert worker._franja_inf_px(60) is None


def test_franja_inf_px_calcula_sobre_la_imagen_recortada(roi):
    # sup 0.40, inf 0.80 -> la linea de abajo cae a mitad de la
    # imagen recortada (que arranca en 0.40): (0.80-0.40)/(1-0.40)=0.667
    # 0.667 * 60 filas = 40
    assert worker._franja_inf_px(60) == 40


def test_franja_inf_px_sin_linea_inferior_es_none(monkeypatch):
    monkeypatch.setattr(worker, "_cfg", {
        "roi_habilitado": True,
        "roi_linea_verde_sup": 0.40,
        "roi_linea_verde_inf": 1.00,   # solo corte arriba
    })
    assert worker._franja_inf_px(60) is None


def test_placa_dentro_de_franja_pasa(roi):
    # y+h = 30 <= 40+4 -> totalmente visible dentro de la franja
    assert worker._placa_en_franja(y=10, h=20, alto=60) is True


def test_placa_cruzando_la_linea_verde_falla(roi):
    # y+h = 55 > 44 -> la placa paso de la linea verde de abajo
    assert worker._placa_en_franja(y=35, h=20, alto=60) is False


def test_placa_en_el_limite_de_tolerancia(roi):
    # tolerancia 4px igual que toca_borde: 44 pasa, 45 no
    assert worker._placa_en_franja(y=24, h=20, alto=60) is True
    assert worker._placa_en_franja(y=25, h=20, alto=60) is False


def test_placa_en_franja_sin_roi_pasa(sin_roi):
    assert worker._placa_en_franja(y=90, h=20, alto=60) is True


# ------------------------------------------------------------- posicion ---

def test_distancia_franja_centro_de_franja_es_cero(roi):
    # cy viene en coords de la foto ORIGINAL (el .txt guarda cy sobre
    # la foto completa): centro de la franja (0.40+0.80)/2 = 0.60
    dist = worker._distancia_franja(0.60, 0.40, 0.80)
    assert dist == pytest.approx(0.0, abs=1e-6)


def test_distancia_franja_sin_roi_es_distancia_al_centro(sin_roi):
    assert worker._distancia_franja(0.5, 0.0, 1.0) == pytest.approx(0.0)
    assert worker._distancia_franja(0.8, 0.0, 1.0) == pytest.approx(0.3)


def test_mejora_posicion_clara(roi):
    # centro de franja 0.60; vieja cy 0.94 -> dist 0.34
    # nueva cy 0.70 -> dist 0.10 (< 75% de 0.34)
    assert worker._mejora_por_posicion(0.70, 0.94) is True


def test_mejora_posicion_marginal_gana_con_tolerancia(roi):
    # dist 0.20 -> 0.16: mejora 0.04 > tolerancia 0.02 -> GANA.
    # (El viejo umbral del 25% la dejaba fuera; pedido 2026-11-09:
    # la mas cerca de la linea gana con "un poquito" de tolerancia.)
    assert worker._mejora_por_posicion(0.76, 0.80) is True


def test_mejora_posicion_casi_igual_no_voltea(roi):
    # dist 0.20 -> 0.19: mejora 0.01 < 0.02 -> no voltea (anti-parpadeo)
    assert worker._mejora_por_posicion(0.79, 0.80) is False


def test_mejora_posicion_igual_no_cuenta(roi):
    assert worker._mejora_por_posicion(0.50, 0.50) is False


def test_mejora_posicion_sin_roi_es_false(sin_roi):
    # rollback total: con ROI apagado manda solo el area (como antes)
    assert worker._mejora_por_posicion(0.10, 0.90) is False


def test_mejora_posicion_sin_etiqueta_previa(roi):
    assert worker._mejora_por_posicion(0.50, None) is False


# ---------------------------------------------------------------- config ---

def test_config_ini_expone_las_claves_roi():
    import configparser
    p = configparser.ConfigParser()
    p.read(RAIZ / "config.ini", encoding="utf-8-sig")
    assert p.has_section("roi")
    assert p.get("roi", "habilitado").strip().lower() in ("si", "no")
    sup = p.getfloat("roi", "linea_verde_sup")
    inf = p.getfloat("roi", "linea_verde_inf")
    assert 0.0 <= sup < inf <= 1.0


def test_cargar_config_roi_se_pasa_al_worker(monkeypatch):
    # El gotcha conocido: claves que existen en el ini pero que
    # cargar_config() no carga quedan en default silencioso.
    # (tempfile y no tmp_path: el pytest-of-FELIPE de este equipo esta
    # con permisos rotos y tmp_path revienta con PermissionError)
    import tempfile
    texto = (RAIZ / "config.ini").read_text(encoding="utf-8-sig")
    texto = texto.replace("habilitado = no", "habilitado = si")
    texto = re.sub(r"(?m)^linea_verde_sup = .*$",
                   "linea_verde_sup = 0.33", texto)
    texto = re.sub(r"(?m)^linea_verde_inf = .*$",
                   "linea_verde_inf = 0.66", texto)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "config.ini").write_text(texto, encoding="utf-8")
        import ftp_server
        monkeypatch.setattr(ftp_server, "BASE", Path(tmp))
        cfg = ftp_server.cargar_config()

    assert cfg["roi_habilitado"] is True
    assert cfg["roi_linea_verde_sup"] == pytest.approx(0.33)
    assert cfg["roi_linea_verde_inf"] == pytest.approx(0.66)
