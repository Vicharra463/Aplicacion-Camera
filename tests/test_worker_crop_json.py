"""Tests del crop de placa: _guardar lo escribe y el reporte JSON lo incluye."""
import json
from pathlib import Path

import numpy as np
import pytest

import worker


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "DetectorPlacas", None)
    monkeypatch.setattr(worker, "leer_placa", None)
    monkeypatch.setattr(worker, "leer_placa_6", None)
    procesadas = tmp_path / "procesadas"
    cfg = {
        "registro": tmp_path / "registro.db",
        "procesadas": procesadas,
        "reportes": tmp_path / "reportes",
        "descartes": tmp_path / "descartadas",
        "reposo_minutos": 0,
        "umbral": 8.0,
        "replay_inicio": "no",
        "modo_descarte": "retener",
    }
    worker.iniciar(cfg)
    return cfg


def _foto_con_placa(entorno, texto="ABC-123"):
    img = np.full((200, 400, 3), 128, np.uint8)
    bbox = (50, 60, 200, 80)
    destino = worker._guardar(img, bbox, "placa", texto)
    crop = destino.with_name(destino.stem + "_crop.jpg")
    return destino, crop


def test_guardar_escribe_crop_al_lado_de_la_foto(entorno):
    destino, crop = _foto_con_placa(entorno)
    assert destino.is_file()
    assert crop.is_file()


def test_guardar_panoramica_no_escribe_crop(entorno):
    img = np.full((200, 400, 3), 128, np.uint8)
    destino = worker._guardar(img, None, "vehiculo", "ABC-123")
    crop = destino.with_name(destino.stem + "_crop.jpg")
    assert destino.is_file()
    assert not crop.exists()


def test_json_dia_incluye_recorte(entorno):
    destino, crop = _foto_con_placa(entorno)
    rel = str(destino.relative_to(entorno["procesadas"]))
    worker.registrar(rel, placa="ABC-123", confianza=0.9, estado="guardado")
    ruta = worker.generar_json_dia(worker._dia_dedup)
    datos = json.loads(Path(ruta).read_text("utf-8"))
    entrada = datos["placas"][0]
    assert entrada["placa"] == "ABC-123"
    assert entrada["imagenes"]["recorte"] == [str(crop)]
    assert entrada["imagenes"]["placa"] == [str(destino)]


def test_json_dia_sin_crop_no_rompe(entorno):
    img = np.full((200, 400, 3), 128, np.uint8)
    destino = worker._guardar(img, None, "placa", "DEF-456")
    rel = str(destino.relative_to(entorno["procesadas"]))
    worker.registrar(rel, placa="DEF-456", confianza=0.9, estado="guardado")
    ruta = worker.generar_json_dia(worker._dia_dedup)
    datos = json.loads(Path(ruta).read_text("utf-8"))
    assert datos["placas"][0]["imagenes"]["recorte"] == []
