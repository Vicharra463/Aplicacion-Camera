"""Cableado AOI del worker: la activacion viene de config.ini [aoi] (cfg)."""
import numpy as np
import pytest

import platerecognizer
import worker


def _cfg(tmp_path, aoi_tokens=""):
    return {
        "registro": tmp_path / "registro.db",
        "procesadas": tmp_path / "procesadas",
        "reportes": tmp_path / "reportes",
        "descartes": tmp_path / "descartadas",
        "reposo_minutos": 0,
        "umbral": 8.0,
        "replay_inicio": "no",
        "modo_descarte": "retener",
        "aoi_tokens": aoi_tokens,
    }


@pytest.fixture
def base(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "DetectorPlacas", None)
    monkeypatch.setattr(worker, "leer_placa", None)
    monkeypatch.setattr(worker, "leer_placa_6", None)
    monkeypatch.setattr(worker, "_AOI_ACTIVA", False)
    monkeypatch.setattr(platerecognizer, "_tokens", [])
    monkeypatch.setattr(platerecognizer, "_cuota", {})
    monkeypatch.delenv("PLATERECOGNIZER_TOKEN", raising=False)
    return tmp_path


def test_iniciar_activa_aoi_con_claves_de_config(base, monkeypatch):
    monkeypatch.setattr(worker, "leer_placa_6", lambda *a, **k: ("", 0.0, []))
    worker.iniciar(_cfg(base, "A, B"))
    assert worker._AOI_ACTIVA is True
    assert worker.leer_placa_6 is platerecognizer.leer_placa_6_api
    assert platerecognizer._tokens == ["A", "B"]


def test_iniciar_sin_claves_sequeda_en_motor_local(base, monkeypatch):
    local = lambda *a, **k: ("", 0.0, [])  # noqa: E731
    monkeypatch.setattr(worker, "leer_placa_6", local)
    worker.iniciar(_cfg(base, ""))
    assert worker._AOI_ACTIVA is False
    assert worker.leer_placa_6 is local


def test_iniciar_toma_clave_de_variable_de_entorno(base, monkeypatch):
    monkeypatch.setattr(worker, "leer_placa_6", lambda *a, **k: ("", 0.0, []))
    monkeypatch.setenv("PLATERECOGNIZER_TOKEN", "de_env")
    worker.iniciar(_cfg(base, ""))
    assert worker._AOI_ACTIVA is True
    assert platerecognizer._tokens == ["de_env"]
