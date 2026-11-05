"""Cadena dual-API: Roboflow (recorte) -> PlateRecognizer -> motor local.

El detector local corre SIEMPRE primero; Roboflow sobre la foto completa
es solo red de seguridad cuando el modelo no ve nada. Estos tests cubren
el eslabon nuevo: Roboflow como LECTOR del recorte del modelo, con la
regla "primera valida gana".
"""
import numpy as np
import roboflow
import worker

IMG = np.zeros((80, 120, 3), np.uint8)
BBOX = (10, 10, 60, 30)


def _activo(monkeypatch, umbral=0.4):
    monkeypatch.setattr(worker, "_ROBOFLOW_ACTIVO", True)
    monkeypatch.setattr(worker, "_cfg", {"umbral_modelo": umbral})


def test_inactivo_no_llama_la_api(monkeypatch):
    monkeypatch.setattr(worker, "_ROBOFLOW_ACTIVO", False)
    llamadas = []
    monkeypatch.setattr(roboflow, "detectar_leer",
                        lambda img: llamadas.append(img) or [])
    assert worker._rf_leer_recorte(IMG, BBOX) == ("", 0.0)
    assert llamadas == []


def test_manda_el_recorte_y_no_la_foto_completa(monkeypatch):
    _activo(monkeypatch)
    captura = {}
    monkeypatch.setattr(roboflow, "detectar_leer",
                        lambda img: captura.update(shape=img.shape) or [])
    worker._rf_leer_recorte(IMG, BBOX)
    assert captura["shape"][:2] == (30, 60)   # (h, w) del bbox


def test_acepta_placa_valida_de_6(monkeypatch):
    _activo(monkeypatch)
    monkeypatch.setattr(
        roboflow, "detectar_leer",
        lambda img: [((0, 0, 10, 10), "PERU\nAWI-935", 0.91)])
    texto, conf = worker._rf_leer_recorte(IMG, BBOX)
    assert texto == "AWI-935"
    assert conf == 0.91


def test_texto_invalido_no_gana_y_deja_siguiente_api(monkeypatch):
    _activo(monkeypatch)
    monkeypatch.setattr(
        roboflow, "detectar_leer",
        lambda img: [((0, 0, 10, 10), "PERU", 0.99)])
    assert worker._rf_leer_recorte(IMG, BBOX) == ("", 0.0)


def test_confianza_bajo_umbral_no_gana(monkeypatch):
    _activo(monkeypatch, umbral=0.4)
    monkeypatch.setattr(
        roboflow, "detectar_leer",
        lambda img: [((0, 0, 10, 10), "AWI-935", 0.31)])
    assert worker._rf_leer_recorte(IMG, BBOX) == ("", 0.0)


def test_de_todas_las_validas_gana_la_de_mayor_confianza(monkeypatch):
    _activo(monkeypatch)
    monkeypatch.setattr(
        roboflow, "detectar_leer",
        lambda img: [((0, 0, 10, 10), "AWI-935", 0.50),
                     ((0, 0, 10, 10), "AWI935", 0.80)])
    texto, conf = worker._rf_leer_recorte(IMG, BBOX)
    assert texto == "AWI-935"
    assert conf == 0.80


def test_excepcion_de_red_no_propaga(monkeypatch):
    _activo(monkeypatch)
    monkeypatch.setattr(roboflow, "detectar_leer",
                        lambda img: (_ for _ in ()).throw(
                            RuntimeError("sin red")))
    assert worker._rf_leer_recorte(IMG, BBOX) == ("", 0.0)


def test_respuesta_vacia_sigue_a_la_siguiente_api(monkeypatch):
    _activo(monkeypatch)
    monkeypatch.setattr(roboflow, "detectar_leer", lambda img: [])
    assert worker._rf_leer_recorte(IMG, BBOX) == ("", 0.0)
