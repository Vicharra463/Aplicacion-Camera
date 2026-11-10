"""Franja vertical del ROI: linea_verde_izq / linea_verde_der (2026-11-07).

El corte horizontal (sup/inf) manda el vehiculo de frente; el vertical
mata la placa contra el borde izquierdo/derecho del encuadre (mitad
fuera, "depende de la suerte" leerla). La placa debe quedar TOTALMENTE
entre izq y der, con la misma tolerancia de 4px que toca_borde y la
franja y. Sin claves en el ini -> defaults 0.0/1.0 = sin corte.
"""
import configparser
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

import worker

RAIZ = Path(__file__).resolve().parents[1]


def _cfg(**extra):
    base = {"roi_habilitado": True,
            "roi_linea_verde_izq": 0.15,
            "roi_linea_verde_der": 0.95}
    base.update(extra)
    return base


# ------------------------------------------------- _placa_en_franja_x ---

def test_placa_contra_el_borde_izquierdo_no_pasa(monkeypatch):
    monkeypatch.setattr(worker, "_cfg", _cfg())
    # izq = 0.15 de 1000 -> limite 150 (con tolerancia: x >= 146);
    # la placa pegada al borde empieza en x = 0 y no llega ni ahi.
    assert worker._placa_en_franja_x(0, 40, 1000) is False


def test_placa_totalmente_dentro_de_la_franja_vertical_pasa(monkeypatch):
    monkeypatch.setattr(worker, "_cfg", _cfg())
    assert worker._placa_en_franja_x(300, 200, 1000) is True


def test_placa_cruzando_la_linea_derecha_no_pasa(monkeypatch):
    monkeypatch.setattr(worker, "_cfg", _cfg())
    # x 930..980 de 1000: der = 0.95 -> limite 950 (+4 = 954); 980 lo cruza.
    assert worker._placa_en_franja_x(930, 50, 1000) is False


def test_placa_entera_mas_a_la_derecha_que_der_no_pasa(monkeypatch):
    monkeypatch.setattr(worker, "_cfg", _cfg())
    assert worker._placa_en_franja_x(960, 30, 1000) is False


def test_sin_claves_verticales_pasa_cualquiera(monkeypatch):
    # defaults 0.0/1.0: sin las claves en _cfg no hay corte (rollback).
    monkeypatch.setattr(worker, "_cfg", {"roi_habilitado": True})
    assert worker._placa_en_franja_x(0, 40, 1000) is True
    assert worker._placa_en_franja_x(960, 40, 1000) is True


def test_roi_apagado_no_corta_vertical(monkeypatch):
    monkeypatch.setattr(worker, "_cfg", _cfg(roi_habilitado=False))
    assert worker._placa_en_franja_x(0, 40, 1000) is True


def test_tolerancia_4px_igual_que_la_franja_y(monkeypatch):
    monkeypatch.setattr(worker, "_cfg", _cfg())
    # izq = 0.15 de 1000 -> limite 150; con tolerancia pasa desde x = 146.
    assert worker._placa_en_franja_x(146, 10, 1000) is True
    assert worker._placa_en_franja_x(145, 10, 1000) is False


# ---------------------------------------------------------------- config ---

def test_config_ini_expone_las_claves_verticales():
    p = configparser.ConfigParser()
    p.read(RAIZ / "config.ini", encoding="utf-8-sig")
    izq = p.getfloat("roi", "linea_verde_izq")
    der = p.getfloat("roi", "linea_verde_der")
    assert 0.0 <= izq < der <= 1.0


def test_cargar_config_carga_las_lineas_verticales(monkeypatch):
    # Mismo gotcha que sup/inf: la clave tiene que estar en _cfg o el
    # worker la lee en default silencioso y editar el ini no hace nada.
    texto = (RAIZ / "config.ini").read_text(encoding="utf-8-sig")
    texto = re.sub(r"(?m)^linea_verde_izq = .*$",
                   "linea_verde_izq = 0.10", texto)
    texto = re.sub(r"(?m)^linea_verde_der = .*$",
                   "linea_verde_der = 0.90", texto)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "config.ini").write_text(texto, encoding="utf-8")
        import ftp_server
        monkeypatch.setattr(ftp_server, "BASE", Path(tmp))
        cfg = ftp_server.cargar_config()

    assert cfg["roi_linea_verde_izq"] == pytest.approx(0.10)
    assert cfg["roi_linea_verde_der"] == pytest.approx(0.90)


# ------------------------------------------------------- _dibujar_roi ---

def test_dibujar_roi_pinta_las_verticales():
    img = np.full((100, 80, 3), 128, np.uint8)
    out = worker._dibujar_roi(img, 0.45, 0.75, izq=0.2, der=0.8)

    # linea vertical blanca en x = 0.2*80 = 16 (sobre la franja verde)
    assert int(out[60, 16, 0]) > 230
    assert int(out[60, 16, 1]) > 230
    assert int(out[60, 16, 2]) > 230

    # tinte rojo de "se corta" en la franja izquierda (x < 16)
    assert int(out[60, 5, 2]) > int(out[60, 5, 1]) + 40

    # derecha: linea en x = 0.8*80 = 64 y tinte rojo despues
    assert int(out[60, 64, 0]) > 230
    assert int(out[60, 75, 2]) > int(out[60, 75, 1]) + 40


def test_dibujar_roi_sin_verticales_no_cambia_las_bandas():
    # Sin izq/der (defaults 0.0/1.0) el dibujo es el viejo: sin lineas
    # ni tintas verticales.
    img = np.full((100, 80, 3), 128, np.uint8)
    out = worker._dibujar_roi(img, 0.45, 0.75)

    assert int(out[60, 5, 1]) > int(out[60, 5, 2]) + 30   # verde, sin rojo
    assert not (int(out[60, 16, 0]) > 230
                and int(out[60, 16, 1]) > 230
                and int(out[60, 16, 2]) > 230)             # sin linea blanca
