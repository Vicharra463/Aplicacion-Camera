"""La memoria del dedup del dia (_hashes/_placas) se carga desde
registros AUNQUE el archivo ya no este en disco (movido o borrado).

La DB es la memoria del dedup; el disco es solo evidencia. El filtro
is_file() que tenia el loader hacia que una placa "se olvidara" al
moverse la foto (p. ej. a FOTOS CAMARA) y la segunda toma del mismo
carro entraba como si fuera nueva: repeticiones como C2Q-618.
"""
import sqlite3

import worker


def _crear_registro(ruta) -> sqlite3.Connection:
    con = sqlite3.connect(ruta)
    con.execute(
        "CREATE TABLE registros ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, fecha_hora TEXT,"
        "archivo TEXT, placa TEXT, confianza REAL, estado TEXT,"
        "phash TEXT)")
    return con


def test_carga_placas_cuyo_archivo_no_esta_en_disco(tmp_path):
    con = _crear_registro(tmp_path / "registro.db")
    # La foto ya NO existe en procesadas (fue movida/borrada): la placa
    # igual debe quedar en la memoria del dedup.
    con.execute(
        "INSERT INTO registros (fecha_hora, archivo, placa, phash) "
        "VALUES (?, ?, ?, ?)",
        ("2026-11-06T12:43:00", "placa\\2026-11-06\\C2Q-618\\C2Q-618.jpg",
         "C2Q-618", "abcdef0123456789"))
    con.commit()

    worker._cargar_dedup_dia(con, "2026-11-06T00:00:00")

    assert any("C2Q-618" in placas for placas in worker._placas.values())
    assert any(0xABCDEF0123456789 in h
               for h in worker._hashes.values())


def test_carga_solo_lo_del_dia(tmp_path):
    con = _crear_registro(tmp_path / "registro.db")
    con.execute(
        "INSERT INTO registros (fecha_hora, archivo, placa, phash) "
        "VALUES (?, ?, ?, ?)",
        ("2026-11-05T12:43:00", "placa\\2026-11-05\\OLD-111\\old.jpg",
         "OLD-111", "1111111111111111"))
    con.execute(
        "INSERT INTO registros (fecha_hora, archivo, placa, phash) "
        "VALUES (?, ?, ?, ?)",
        ("2026-11-06T09:00:00", "placa\\2026-11-06\\NEW-222\\new.jpg",
         "NEW-222", "2222222222222222"))
    con.commit()

    worker._cargar_dedup_dia(con, "2026-11-06T00:00:00")

    assert not any("OLD-111" in p for p in worker._placas.values())
    assert any("NEW-222" in p for p in worker._placas.values())


def test_archivo_vacio_no_revuelve_la_memoria(tmp_path):
    con = _crear_registro(tmp_path / "registro.db")
    con.execute(
        "INSERT INTO registros (fecha_hora, archivo, placa, phash) "
        "VALUES (?, ?, ?, ?)",
        ("2026-11-06T09:00:00", "", "SIN-RUTA", "3333333333333333"))
    con.commit()

    worker._cargar_dedup_dia(con, "2026-11-06T00:00:00")

    assert worker._placas == {}
    assert worker._hashes == {}
