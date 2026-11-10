"""vista_roi: ver en la IMAGEN que corta el ROI y donde debe estar el
vehiculo. El veredicto es la verdad visual que despues guia los tests de
produccion: rojo = no se detecta, verde = cumple, naranja = fuera.
"""
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture
def vista():
    spec = importlib.util.spec_from_file_location(
        "vista_roi", RAIZ / "scripts" / "vista_roi.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------ veredicto ---

def test_placa_en_rojo_no_se_detecta(vista):
    # B1V-756 (vista real): placa y 0.14..0.17 con sup 0.45
    assert "EN ROJO" in vista.veredicto(0.140, 0.174, 0.45, 0.75)


def test_placa_cruzando_la_linea_de_arriba_es_rojo(vista):
    assert "EN ROJO" in vista.veredicto(0.40, 0.50, 0.45, 0.75)


def test_placa_dentro_de_la_franja_cumple(vista):
    # AFH-559 (vista real): placa y 0.522..0.580 con sup 0.45 inf 0.75
    assert "CUMPLE" in vista.veredicto(0.522, 0.580, 0.45, 0.75)


def test_placa_cruzando_la_linea_de_abajo_fuera(vista):
    assert "FUERA" in vista.veredicto(0.70, 0.80, 0.45, 0.75)


def test_placa_demasiado_cerca_fuera(vista):
    # CF635 (vista real): placa y 0.805..0.890
    assert "FUERA" in vista.veredicto(0.805, 0.890, 0.45, 0.75)


# ------------------------------------------------- franja vertical ---

def test_placa_fuera_de_la_vertical_es_fuera(vista):
    # corte izq = 0.15: placa x 0.02..0.12 no llega a la linea
    assert "FUERA" in vista.veredicto(0.50, 0.60, 0.35, 0.85,
                                      x0=0.02, x1=0.12,
                                      izq=0.15, der=0.95)


def test_placa_cruzando_la_vertical_derecha_es_fuera(vista):
    assert "FUERA" in vista.veredicto(0.50, 0.60, 0.35, 0.85,
                                      x0=0.90, x1=0.98,
                                      izq=0.15, der=0.95)


def test_placa_totalmente_dentro_de_las_verticales_cumple(vista):
    assert "CUMPLE" in vista.veredicto(0.50, 0.60, 0.35, 0.85,
                                       x0=0.30, x1=0.50,
                                       izq=0.15, der=0.95)


def test_veredicto_sin_x_sigue_funcionando(vista):
    # llamada vieja (solo y): los tests de arriba la siguen cubriendo.
    assert "CUMPLE" in vista.veredicto(0.522, 0.580, 0.45, 0.75)


# -------------------------------------------------------------- gt_bbox ---

def test_gt_bbox_lee_el_txt_yolo_hermano(vista):
    with tempfile.TemporaryDirectory() as tmp:
        foto = Path(tmp) / "placa.jpg"
        foto.write_bytes(b"x")
        # cls cx cy w h normalizados: caja de (0.40,0.20) a (0.60,0.60)
        (Path(tmp) / "placa.txt").write_text("0 0.50 0.40 0.20 0.40\n",
                                             encoding="utf-8")
        g = vista.gt_bbox(foto, ancho=100, alto=50)
    assert g == pytest.approx((40.0, 10.0, 60.0, 30.0))


def test_gt_bbox_sin_txt_devuelve_none(vista):
    with tempfile.TemporaryDirectory() as tmp:
        foto = Path(tmp) / "placa.jpg"
        foto.write_bytes(b"x")
        assert vista.gt_bbox(foto, 100, 50) is None


# --------------------------------------------------------------- pintar ---

def test_pintar_dibuja_las_bandas_y_lineas(vista):
    img = np.full((100, 80, 3), 200, np.uint8)
    out = vista.pintar(img, 0.45, 0.75)
    assert out.shape == img.shape

    # zona roja (fila 10): tinte rojizo -> canal R domina sobre B
    assert int(out[10, 40, 2]) > int(out[10, 40, 0]) + 40

    # franja verde (fila 60): tinte verdoso -> G domina
    assert int(out[60, 40, 1]) > int(out[60, 40, 0]) + 40
    assert int(out[60, 40, 1]) > int(out[60, 40, 2]) + 40

    # linea verde superior pura en la fila del corte (fila 45)
    assert int(out[45, 40, 1]) > 230
    assert int(out[45, 40, 0]) < 60

    # linea naranja inferior (fila 75): BGR (0,200,255) -> R y G altos
    assert int(out[75, 40, 2]) > 200 and int(out[75, 40, 1]) > 150
    assert int(out[75, 40, 0]) < 80

    # bajo la linea: tinte naranja (R > B)
    assert int(out[90, 40, 2]) > int(out[90, 40, 0]) + 30


def test_pintar_dibuja_las_verticales(vista):
    img = np.full((100, 80, 3), 200, np.uint8)
    out = vista.pintar(img, 0.45, 0.75, izq=0.2, der=0.8)

    # linea vertical blanca en x = 0.2*80 = 16
    assert int(out[60, 16, 0]) > 230
    assert int(out[60, 16, 1]) > 230
    assert int(out[60, 16, 2]) > 230

    # tinte "se corta" a la izquierda (x < 16): R domina sobre G
    assert int(out[60, 5, 2]) > int(out[60, 5, 1]) + 40


def test_pintar_sin_verticales_no_toca_las_bandas(vista):
    img = np.full((100, 80, 3), 200, np.uint8)
    out = vista.pintar(img, 0.45, 0.75)

    # sin izq/der: la banda verde sigue intacta a la izquierda
    assert int(out[60, 5, 1]) > int(out[60, 5, 2]) + 30
    assert not (int(out[60, 16, 0]) > 230
                and int(out[60, 16, 1]) > 230
                and int(out[60, 16, 2]) > 230)
