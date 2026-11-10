"""Barrido de la cola de entrada en segundos (2026-11-07).

El bucle_barrido original dormia 60 s: tras reiniciar el servicio, las
fotos estancadas en ftp_recibidos tardaban hasta ~90 s (60 s de pausa +
30 s de mtime) en re-encolarse. FELIPE pidio recuperacion "en segundos".
"""
import os
import queue
import time
from pathlib import Path

import pytest

import ftp_server
import worker


def _vaciar_cola() -> None:
    while True:
        try:
            worker.COLA.get_nowait()
        except queue.Empty:
            break


@pytest.fixture
def entorno(monkeypatch):
    monkeypatch.setattr(worker, "_encolados", set())
    _vaciar_cola()
    yield
    _vaciar_cola()


def _archivo_viejo(carpeta: Path, nombre: str = "foto.jpg",
                   antiguedad: float = 60.0) -> Path:
    p = carpeta / nombre
    p.write_bytes(b"jpg")
    t = time.time() - antiguedad
    os.utime(p, (t, t))
    return p


def test_barrido_encola_archivo_viejo(entorno, tmp_path):
    p = _archivo_viejo(tmp_path)
    assert ftp_server._barrido_una_vez(tmp_path) == 1
    assert worker.COLA.get_nowait() == str(p)


def test_barrido_no_encola_archivo_fresco(entorno, tmp_path):
    p = tmp_path / "recien.jpg"
    p.write_bytes(b"jpg")
    assert ftp_server._barrido_una_vez(tmp_path) == 0
    assert worker.COLA.empty()


def test_barrido_no_encola_directorios(entorno, tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    t = time.time() - 120
    os.utime(sub, (t, t))
    assert ftp_server._barrido_una_vez(tmp_path) == 0
    assert worker.COLA.empty()


def test_barrido_no_duplica_lo_ya_encolado(entorno, tmp_path):
    p = _archivo_viejo(tmp_path)
    worker._encolados.add(str(p))
    assert ftp_server._barrido_una_vez(tmp_path) == 0
    assert worker.COLA.empty()


def test_barrido_regresa_en_segundos():
    # Recuperacion tras reinicio: el intervalo tiene que ser de SEGUNDOS,
    # no el minuto original.
    assert ftp_server.BARRIDO_SEG <= 5.0
