"""Cooldowns y reposo ceden ante cola pendiente (2026-11-07).

Con 281 fotos esperando en ftp_recibidos, los sleep de 5/15 s de los
cooldowns y el reposo a los 10 min dejaban al worker "dormido" mientras
la cola no terminaba: ahora solo duermen/reposan con la cola vacia.
"""
import queue
import time

import pytest

import worker


def _vaciar_cola() -> None:
    while True:
        try:
            worker.COLA.get_nowait()
        except queue.Empty:
            break


@pytest.fixture
def entorno(monkeypatch):
    cfg = {
        "cooldown_placa_seg": 5.0,
        "cooldown_sin_placa_seg": 5.0,
    }
    monkeypatch.setattr(worker, "_cfg", cfg)
    monkeypatch.setattr(worker, "_reposo_seg", 600.0)
    monkeypatch.setattr(worker, "_en_reposo", False)
    monkeypatch.setattr(worker, "_t_ultima_actividad", time.time())
    _vaciar_cola()
    yield cfg
    _vaciar_cola()


@pytest.fixture
def sleeps(monkeypatch):
    llamadas: list[float] = []
    monkeypatch.setattr(worker.time, "sleep", lambda s: llamadas.append(s))
    return llamadas


# ------------------------------------------- cooldown sin placa ---

def test_cooldown_sin_placa_duerme_con_cola_vacia(entorno, sleeps):
    worker._cooldown_sin_placa()
    assert sleeps == [5.0]


def test_cooldown_sin_placa_no_duerme_con_cola_pendiente(entorno, sleeps):
    worker.COLA.put("foto_pendiente.jpg")
    worker._cooldown_sin_placa()
    assert sleeps == []


def test_cooldown_sin_placa_apagado_nunca_duerme(entorno, sleeps):
    entorno["cooldown_sin_placa_seg"] = 0.0
    worker._cooldown_sin_placa()
    assert sleeps == []


# ---------------------------------------------- cooldown placa ---

def test_cooldown_placa_duerme_con_cola_vacia(entorno, sleeps):
    worker._cooldown_placa()
    assert sleeps == [5.0]


def test_cooldown_placa_no_duerme_con_cola_pendiente(entorno, sleeps):
    worker.COLA.put("foto_pendiente.jpg")
    worker._cooldown_placa()
    assert sleeps == []


def test_cooldown_placa_apagado_nunca_duerme(entorno, sleeps):
    entorno["cooldown_placa_seg"] = 0.0
    worker._cooldown_placa()
    assert sleeps == []


# ----------------------------------------------------- reposo ---

def test_reposo_no_entra_con_cola_pendiente(entorno, monkeypatch):
    monkeypatch.setattr(worker, "_t_ultima_actividad", time.time() - 700)
    worker.COLA.put("foto_pendiente.jpg")
    assert worker._debe_entrar_en_reposo() is False


def test_reposo_entra_con_cola_vacia_y_actividad_vieja(entorno,
                                                        monkeypatch):
    monkeypatch.setattr(worker, "_t_ultima_actividad", time.time() - 700)
    assert worker._debe_entrar_en_reposo() is True


def test_reposo_no_entra_si_actividad_reciente(entorno, monkeypatch):
    monkeypatch.setattr(worker, "_t_ultima_actividad", time.time())
    assert worker._debe_entrar_en_reposo() is False


def test_reposo_no_entra_si_ya_en_reposo(entorno, monkeypatch):
    monkeypatch.setattr(worker, "_en_reposo", True)
    monkeypatch.setattr(worker, "_t_ultima_actividad", time.time() - 700)
    assert worker._debe_entrar_en_reposo() is False


def test_reposo_apagado_nunca_entra(entorno, monkeypatch):
    monkeypatch.setattr(worker, "_reposo_seg", 0.0)
    monkeypatch.setattr(worker, "_t_ultima_actividad", time.time() - 700)
    assert worker._debe_entrar_en_reposo() is False
