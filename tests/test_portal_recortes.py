"""Portal: listado de recortes AOI (eficacia) y control de pasadas."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

import portal


@pytest.fixture
def base(tmp_path):
    registro = tmp_path / "registro.db"
    con = sqlite3.connect(registro)
    con.execute("""
        CREATE TABLE pasadas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            placa TEXT NOT NULL,
            pasada INTEGER NOT NULL,
            fecha_hora TEXT NOT NULL,
            excedida INTEGER NOT NULL DEFAULT 0
        )""")
    con.execute("""
        CREATE TABLE registros (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha_hora TEXT NOT NULL,
            archivo TEXT NOT NULL,
            placa TEXT,
            confianza REAL,
            estado TEXT NOT NULL DEFAULT 'guardado',
            evidencia INTEGER,
            semaforo TEXT,
            phash TEXT
        )""")
    con.commit()
    con.close()
    return {"registro": registro, "reportes": tmp_path / "reportes",
            "procesadas": tmp_path / "procesadas"}


def _sembrar_pasadas(base, placa="ABC-123", veces=2):
    con = sqlite3.connect(base["registro"])
    for i in range(veces):
        con.execute(
            "INSERT INTO pasadas (placa, pasada, fecha_hora, excedida) "
            "VALUES (?, ?, ?, 0)",
            (placa, i + 1, f"2026-10-28T10:0{i}:00"))
    con.commit()
    con.close()


def test_estado_pasadas_agrupa_por_placa(base):
    _sembrar_pasadas(base, "ABC-123", 2)
    _sembrar_pasadas(base, "XYZ-789", 1)
    filas = portal.estado_pasadas(base["registro"])
    por_placa = {f["placa"]: f for f in filas}
    assert por_placa["ABC-123"]["pasadas"] == 2
    assert por_placa["XYZ-789"]["pasadas"] == 1
    assert por_placa["ABC-123"]["ultima"] == "2026-10-28T10:01:00"


def test_reiniciar_pasadas_borra_contador(base):
    _sembrar_pasadas(base, "ABC-123", 3)
    portal.reiniciar_pasadas(base["registro"], "ABC-123")
    filas = portal.estado_pasadas(base["registro"])
    assert all(f["placa"] != "ABC-123" for f in filas)


def test_reiniciar_pasadas_no_toca_otras_placas(base):
    _sembrar_pasadas(base, "ABC-123", 2)
    _sembrar_pasadas(base, "XYZ-789", 1)
    portal.reiniciar_pasadas(base["registro"], "ABC-123")
    filas = portal.estado_pasadas(base["registro"])
    assert {f["placa"] for f in filas} == {"XYZ-789"}


def test_listar_traza_aoi_lee_jsonl(base, tmp_path):
    dia = "2026-10-28"
    carpeta = base["reportes"] / "aoi" / dia
    carpeta.mkdir(parents=True)
    crop = tmp_path / "crop.jpg"
    crop.write_bytes(b"\xff\xd8")
    entradas = [
        {"ts": f"{dia}T10:00:0{i}", "foto": f"foto_{i}.jpg",
         "bbox": [10, 10, 100, 40], "texto": "ABC-123" if i else "",
         "conf": 0.9, "via_api": True, "crop": str(crop)}
        for i in range(3)
    ]
    (carpeta / f"aoi-{dia}.jsonl").write_text(
        "\n".join(json.dumps(e) for e in entradas), encoding="utf-8")
    resumen, filas = portal.listar_traza_aoi(base["reportes"], dia)
    assert resumen["total"] == 3
    assert resumen["con_lectura"] == 2
    assert resumen["via_api"] == 3
    assert resumen["eficacia_api_pct"] == pytest.approx(66.67, abs=0.1)
    assert len(filas) == 3


def test_listar_traza_aoi_dia_vacio(base):
    resumen, filas = portal.listar_traza_aoi(base["reportes"], "2020-01-01")
    assert resumen["total"] == 0
    assert filas == []


def test_listar_traza_aoi_sin_dia_devuelve_hoy(base, monkeypatch):
    class _FalsoDatetime:
        @staticmethod
        def now():
            return datetime(2026, 10, 28, 12, 0, 0)

    carpeta = base["reportes"] / "aoi" / "2026-10-28"
    carpeta.mkdir(parents=True)
    (carpeta / "aoi-2026-10-28.jsonl").write_text(
        json.dumps({"ts": "2026-10-28T09:00:00", "foto": "x.jpg",
                    "bbox": [0, 0, 10, 10], "texto": "AA-11",
                    "conf": 0.8, "via_api": False, "crop": ""}) + "\n",
        encoding="utf-8")
    monkeypatch.setattr(portal, "datetime", _FalsoDatetime)
    resumen, filas = portal.listar_traza_aoi(base["reportes"], None)
    assert resumen["dia"] == "2026-10-28"
    assert resumen["total"] == 1
