"""Estructura de salida simplificada: solo carpeta placa, sin pendientes,
sin panoramica en el JSON, y traza de envios a la AOI."""
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


def test_canal_01_y_02_van_a_placa():
    assert worker._carpeta_de("foto_01_20261028.jpg") == "placa"
    assert worker._carpeta_de("foto_02_20261028.jpg") == "placa"
    assert worker._carpeta_de("sin_canal.jpg") == "placa"


def test_guardar_sin_subcarpeta_pendientes(entorno):
    worker._detector = None
    img = np.full((200, 400, 3), 128, np.uint8)
    destino = worker._guardar(img, (50, 60, 200, 80), "placa", "ABC-123")
    assert "pendientes" not in destino.parts
    assert destino.parent.parent.name == "placa"


def test_json_sin_panoramica(entorno):
    img = np.full((200, 400, 3), 128, np.uint8)
    destino = worker._guardar(img, (50, 60, 200, 80), "placa", "ABC-123")
    rel = str(destino.relative_to(entorno["procesadas"]))
    worker.registrar(rel, placa="ABC-123", confianza=0.9, estado="guardado")
    ruta = worker.generar_json_dia(worker._dia_dedup)
    datos = json.loads(Path(ruta).read_text("utf-8"))
    imagenes = datos["placas"][0]["imagenes"]
    assert "panoramica" not in imagenes
    assert set(imagenes) == {"placa", "recorte"}


def test_traza_aoi_graba_jsonl_y_crop(entorno, monkeypatch):
    monkeypatch.setattr(worker, "_AOI_ACTIVA", True)
    img = np.full((200, 400, 3), 128, np.uint8)
    ruta_foto = entorno["procesadas"] / "ftp" / "foto_01_x.jpg"
    ruta_foto.parent.mkdir(parents=True, exist_ok=True)
    ruta_foto.write_bytes(b"x")
    intentos = [("api", "ABC-123", 0.9), ("crnn", "ABC-123", 0.8)]
    worker._trazar_aoi(ruta_foto, img, (50, 60, 200, 80), "ABC-123",
                        0.9, intentos)
    jsonl = entorno["reportes"] / "aoi" / worker._dia_dedup \
        / f"aoi-{worker._dia_dedup}.jsonl"
    assert jsonl.is_file()
    linea = json.loads(jsonl.read_text("utf-8").splitlines()[-1])
    assert linea["texto"] == "ABC-123"
    assert linea["via_api"] is True
    assert linea["foto"] == "foto_01_x.jpg"
    crop = Path(linea["crop"])
    assert crop.is_file()


def test_traza_aoi_off_no_escribe_nada(entorno, monkeypatch):
    monkeypatch.setattr(worker, "_AOI_ACTIVA", False)
    img = np.full((200, 400, 3), 128, np.uint8)
    ruta_foto = entorno["procesadas"] / "ftp" / "foto_01_y.jpg"
    ruta_foto.parent.mkdir(parents=True, exist_ok=True)
    ruta_foto.write_bytes(b"x")
    worker._trazar_aoi(ruta_foto, img, (10, 10, 100, 40), "ABC-123",
                        0.9, [("api", "ABC-123", 0.9)])
    carpeta_aoi = entorno["reportes"] / "aoi"
    assert not carpeta_aoi.exists()


def test_traza_aoi_falla_local(entorno, monkeypatch):
    monkeypatch.setattr(worker, "_AOI_ACTIVA", True)
    img = np.full((200, 400, 3), 128, np.uint8)
    ruta_foto = entorno["procesadas"] / "ftp" / "foto_01_z.jpg"
    ruta_foto.parent.mkdir(parents=True, exist_ok=True)
    ruta_foto.write_bytes(b"x")
    intentos = [("crnn", "", 0.0)]
    worker._trazar_aoi(ruta_foto, img, (10, 10, 100, 40), "", 0.0, intentos)
    jsonl = entorno["reportes"] / "aoi" / worker._dia_dedup \
        / f"aoi-{worker._dia_dedup}.jsonl"
    linea = json.loads(jsonl.read_text("utf-8").splitlines()[-1])
    assert linea["texto"] == ""
    assert linea["via_api"] is False
