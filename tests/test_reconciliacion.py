import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

import worker

HOY = datetime.now().strftime("%Y-%m-%d")


@contextmanager
def _entorno():
    """Base tmp: procesadas + reportes + registro.db con el schema real."""
    tmp = Path(tempfile.mkdtemp(prefix="cytv_reconc_"))
    previo = (worker._cfg, worker._ruta_registro, worker._dia_dedup,
              worker._hashes, worker._placas, worker._json_pendiente)
    try:
        worker._cfg = {"procesadas": tmp / "proc",
                       "reportes": tmp / "reportes"}
        worker._ruta_registro = tmp / "registro.db"
        worker._dia_dedup = HOY
        worker._hashes = {}
        worker._placas = {}
        worker._json_pendiente = False
        con = worker._conexion()
        try:
            con.execute(
                """
                CREATE TABLE registros (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    fecha_hora TEXT    NOT NULL,
                    archivo    TEXT    NOT NULL,
                    placa      TEXT,
                    confianza  REAL,
                    estado     TEXT    NOT NULL DEFAULT 'guardado',
                    evidencia  INTEGER,
                    semaforo   TEXT,
                    phash      TEXT,
                    fuente     TEXT,
                    provisional INTEGER,
                    placa_api  TEXT,
                    placa_modelo TEXT,
                    motivo_sin_api TEXT
                )
                """
            )
            con.execute(
                """
                CREATE TABLE pasadas (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    placa     TEXT    NOT NULL,
                    pasada    INTEGER NOT NULL,
                    fecha_hora TEXT   NOT NULL,
                    excedida  INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            con.commit()
        finally:
            con.close()
        yield tmp
    finally:
        (worker._cfg, worker._ruta_registro, worker._dia_dedup,
         worker._hashes, worker._placas, worker._json_pendiente) = previo
        shutil.rmtree(tmp, ignore_errors=True)


def _foto_en(proc: Path, placa: str, nombre: str = "foto.jpg",
             dia: str = HOY) -> Path:
    carpeta = proc / "placa" / dia / placa
    carpeta.mkdir(parents=True, exist_ok=True)
    foto = carpeta / nombre
    foto.write_bytes(b"jpg")
    return foto


def _json_del_dia() -> dict:
    ruta = worker._cfg["reportes"] / f"placas-{HOY}.json"
    return json.loads(ruta.read_text(encoding="utf-8"))


def test_falta_fila_se_anota_y_llega_al_json():
    """Carpeta de placa sin fila en la DB -> se anota y aparece en el JSON."""
    with _entorno() as tmp:
        foto = _foto_en(tmp / "proc", "ABR-820")
        con = worker._conexion()
        try:
            assert worker._reconciliar_carpetas_dia(con, HOY) == 1
        finally:
            con.close()
        con = worker._conexion()
        try:
            fila = con.execute(
                "SELECT placa, estado, archivo FROM registros").fetchone()
        finally:
            con.close()
        assert fila == ("ABR-820", "guardado",
                        str(foto.relative_to(tmp / "proc")))
        datos = json.loads(
            worker.generar_json_dia(HOY).read_text(encoding="utf-8"))
        assert datos["total_placas"] == 1
        assert datos["placas"][0]["placa"] == "ABR-820"
        assert datos["placas"][0]["imagenes"]["placa"] == [str(foto)]


def test_ya_guardado_no_se_toca():
    """Fila ya 'guardado' con archivo vivo: reconciliar no la modifica."""
    with _entorno() as tmp:
        proc = tmp / "proc"
        viva = _foto_en(proc, "ABR-820", "viva.jpg")
        con = worker._conexion()
        try:
            con.execute(
                "INSERT INTO registros (fecha_hora, archivo, placa, estado, "
                "evidencia, fuente) VALUES (?, ?, 'ABR-820', 'guardado', "
                "4, 'roboflow')",
                (f"{HOY}T09:00:00",
                 str(viva.relative_to(proc))))
            con.commit()
            assert worker._reconciliar_carpetas_dia(con, HOY) == 0
            fila = con.execute(
                "SELECT id, archivo, estado, evidencia, fuente "
                "FROM registros").fetchone()
        finally:
            con.close()
        assert fila[1] == str(viva.relative_to(proc))
        assert fila[2] == "guardado"
        assert fila[3] == 4
        assert fila[4] == "roboflow"


def test_pendiente_se_promueve():
    """Fila 'pendiente' con carpeta viva -> se promueve a 'guardado'."""
    with _entorno() as tmp:
        proc = tmp / "proc"
        foto = _foto_en(proc, "ABR-820")
        con = worker._conexion()
        try:
            con.execute(
                "INSERT INTO registros (fecha_hora, archivo, placa, estado) "
                "VALUES (?, ?, 'ABR-820', 'pendiente')",
                (f"{HOY}T08:00:00", "placa/otro-layout/muerta.jpg"))
            con.commit()
            assert worker._reconciliar_carpetas_dia(con, HOY) == 1
            fila = con.execute(
                "SELECT estado, archivo FROM registros").fetchone()
        finally:
            con.close()
        assert fila == ("guardado", str(foto.relative_to(proc)))
        datos = json.loads(
            worker.generar_json_dia(HOY).read_text(encoding="utf-8"))
        assert datos["total_placas"] == 1


def test_archivo_muerto_se_repunta():
    """Ya 'guardado' pero el archivo de la fila ya no existe: se repunta
    a la foto viva de la carpeta (el JSON deja de listar una ruta muerta)."""
    with _entorno() as tmp:
        proc = tmp / "proc"
        foto = _foto_en(proc, "ABR-820", "viva.jpg")
        con = worker._conexion()
        try:
            con.execute(
                "INSERT INTO registros (fecha_hora, archivo, placa, estado) "
                "VALUES (?, ?, 'ABR-820', 'guardado')",
                (f"{HOY}T07:00:00",
                 "placa/" + HOY + "/ABR-820/muerta.jpg"))
            con.commit()
            assert worker._reconciliar_carpetas_dia(con, HOY) == 1
            fila = con.execute("SELECT archivo FROM registros").fetchone()
        finally:
            con.close()
        assert fila[0] == str(foto.relative_to(proc))
        datos = json.loads(
            worker.generar_json_dia(HOY).read_text(encoding="utf-8"))
        assert datos["placas"][0]["imagenes"]["placa"] == [str(foto)]


def test_carpeta_sin_forma_de_placa_se_ignora():
    """Subcarpetas que no son placas ('recortes', 'CROP') no generan filas."""
    with _entorno() as tmp:
        proc = tmp / "proc"
        _foto_en(proc, "recortes", "algo.jpg")
        _foto_en(proc, "CROP", "algo.jpg")
        con = worker._conexion()
        try:
            assert worker._reconciliar_carpetas_dia(con, HOY) == 0
            total = con.execute(
                "SELECT COUNT(*) FROM registros").fetchone()[0]
        finally:
            con.close()
        assert total == 0


def test_json_excluye_pendiente_sin_reconciliar():
    """CAUSA RAIZ documentada: el JSON solo lee estado='guardado'; una
    fila pendiente NO llega al reporte aunque su foto este en disco."""
    with _entorno() as tmp:
        _foto_en(tmp / "proc", "ABR-820")
        worker.registrar("placa/" + HOY + "/ABR-820/foto.jpg",
                         placa="ABR-820", estado="pendiente")
        datos = json.loads(
            worker.generar_json_dia(HOY).read_text(encoding="utf-8"))
        assert datos["total_placas"] == 0


def test_fuente_de_intentos():
    assert worker.fuente_de_intentos(
        [("roboflow", "ABC-123", 0.9)]) == "roboflow"
    assert worker.fuente_de_intentos(
        [("api", "ABC-123", 0.9)]) == "platerecognizer"
    assert worker.fuente_de_intentos(
        [("crnn", "ABC-123", 0.9)]) == "modelo"
    assert worker.fuente_de_intentos([]) == "modelo"
    assert worker.fuente_de_intentos(None) == "modelo"


def test_reemplazo_mejor_foto_no_tira_nameerror():
    """Regresion: _reemplazar_si_mejor lee 'fuente' (era NameError ->
    rollback del UPDATE mientras el archivo nuevo ya estaba en disco)."""
    with _entorno() as tmp:
        proc = tmp / "proc"
        carpeta = proc / "placa" / HOY / "ABR-820"
        carpeta.mkdir(parents=True)
        viejo = carpeta / "ABR-820.jpg"
        viejo.write_bytes(b"vieja")
        viejo.with_suffix(".txt").write_text("0 0.5 0.5 0.1 0.1\n",
                                             encoding="utf-8")
        con = worker._conexion()
        try:
            con.execute(
                "INSERT INTO registros (fecha_hora, archivo, placa, estado, "
                "evidencia) VALUES (?, ?, 'ABR-820', 'guardado', 1)",
                (f"{HOY}T10:00:00",
                 str(viejo.relative_to(proc))))
            con.commit()
        finally:
            con.close()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        ok = worker._reemplazar_si_mejor(
            "placa", 1111, "ABR-820", img, (10, 10, 50, 50), "ABR-820",
            evidencia=5, color="verde", confianza=0.9, huella=999,
            etiqueta="", fuente="roboflow")
        assert ok is True
        con = worker._conexion()
        try:
            fila = con.execute(
                "SELECT archivo, fuente, estado FROM registros").fetchone()
        finally:
            con.close()
        assert fila[1] == "roboflow"
        assert fila[2] == "guardado"
        nuevo = proc / fila[0]
        assert nuevo.is_file()
        # _guardar reutiliza el mismo nombre: el viejo se borro primero y
        # el contenido ahora es la foto nueva (jpeg real, no b"vieja").
        assert nuevo.read_bytes() != b"vieja"
        assert viejo.with_suffix(".txt").is_file()


def test_carpetas_vacias_se_eliminan():
    """REGLA: carpeta de placa sin foto principal -> se elimina al
    arrancar. Las carpetas sanas y los layout viejos se conservan."""
    with _entorno() as tmp:
        proc = tmp / "proc"
        # Residuos rotos: vacia, solo .txt, solo _crop, solo _original.
        (proc / "placa" / HOY / "ABR-820").mkdir(parents=True)
        solo_txt = proc / "placa" / HOY / "XYZ-123"
        solo_txt.mkdir(parents=True)
        (solo_txt / "XYZ-123.txt").write_text("0 0.5 0.5 0.1 0.1\n",
                                              encoding="utf-8")
        solo_crop = proc / "placa" / HOY / "DEF-789"
        solo_crop.mkdir(parents=True)
        (solo_crop / "DEF-789_crop.jpg").write_bytes(b"jpg")
        solo_original = proc / "placa" / HOY / "GHI-321"
        solo_original.mkdir(parents=True)
        (solo_original / "GHI-321_original.jpg").write_bytes(b"jpg")
        # Sana: tiene foto principal.
        sana = proc / "placa" / HOY / "ABC-456"
        sana.mkdir(parents=True)
        (sana / "ABC-456.jpg").write_bytes(b"jpg")
        # Fecha vacia -> fuera; fecha con foto suelta (layout viejo) ->
        # se conserva.
        (proc / "placa" / "2026-10-01").mkdir(parents=True)
        layout_viejo = proc / "placa" / "2026-10-02"
        layout_viejo.mkdir(parents=True)
        (layout_viejo / "placa_10-00-00_123456.jpg").write_bytes(b"jpg")

        assert worker._depurar_carpetas_vacias() == 4

        assert not (proc / "placa" / HOY / "ABR-820").exists()
        assert not solo_txt.exists()
        assert not solo_crop.exists()
        assert not solo_original.exists()
        assert (sana / "ABC-456.jpg").is_file()
        assert not (proc / "placa" / "2026-10-01").exists()
        assert (layout_viejo / "placa_10-00-00_123456.jpg").is_file()
        # La carpeta de vista jamas se toca (la verifica el FTP).
        assert (proc / "placa").is_dir()


def test_reconciliar_toma_la_principal_no_la_original():
    """La carpeta de placa tiene principal + _original; el reporte debe
    anotar la principal aunque la original tenga mtime mayor."""
    with _entorno() as tmp:
        proc = tmp / "proc"
        carpeta = proc / "placa" / HOY / "ABR-820"
        carpeta.mkdir(parents=True)
        principal = carpeta / "ABR-820.jpg"
        principal.write_bytes(b"jpg")
        original = carpeta / "ABR-820_original.jpg"
        original.write_bytes(b"jpg")
        base = datetime.now().timestamp() - 3600
        os.utime(principal, (base, base))
        os.utime(original, (base + 60, base + 60))   # la original, mas nueva
        con = worker._conexion()
        try:
            assert worker._reconciliar_carpetas_dia(con, HOY) == 1
            fila = con.execute("SELECT archivo FROM registros").fetchone()
        finally:
            con.close()
        assert fila[0] == str(principal.relative_to(proc))


def test_reemplazo_con_texto_nuevo_no_deja_carpetas_fantasma():
    """_reemplazar_si_mejor desarma el set viejo COMPLETO (main, txt,
    _crop y _original) antes del rmdir: si el _original queda suelto, la
    carpeta vieja sobrevive aunque el texto cambio."""
    with _entorno() as tmp:
        proc = tmp / "proc"
        carpeta = proc / "placa" / HOY / "ABR-820"
        carpeta.mkdir(parents=True)
        viejo = carpeta / "ABR-820.jpg"
        viejo.write_bytes(b"vieja")
        viejo.with_suffix(".txt").write_text("0 0.5 0.5 0.1 0.1\n",
                                             encoding="utf-8")
        (carpeta / "ABR-820_crop.jpg").write_bytes(b"crop")
        (carpeta / "ABR-820_original.jpg").write_bytes(b"original")
        con = worker._conexion()
        try:
            con.execute(
                "INSERT INTO registros (fecha_hora, archivo, placa, estado, "
                "evidencia) VALUES (?, ?, 'ABR-820', 'guardado', 1)",
                (f"{HOY}T10:00:00",
                 str(viejo.relative_to(proc))))
            con.commit()
        finally:
            con.close()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        ok = worker._reemplazar_si_mejor(
            "placa", 1111, "ABR-820", img, (10, 10, 50, 50), "QRS-999",
            evidencia=5, color="verde", confianza=0.9, huella=999,
            etiqueta="", fuente=None)
        assert ok is True
        # el texto cambio -> _guardar creo OTRA carpeta; la vieja quedo
        # sin archivos y debe haberse eliminado (sin _original suelto)
        assert not carpeta.exists()
        nueva = proc / "placa" / HOY / "QRS-999"
        assert (nueva / "QRS-999.jpg").is_file()
        assert (nueva / "QRS-999_original.jpg").is_file()


def test_guardar_falla_no_deja_carpeta_vacia(monkeypatch):
    """REGLA en el origen: si la foto no se puede escribir, no queda la
    carpeta recien creada y el error es ruidoso (no se registra un
    archivo inexistente)."""
    with _entorno() as tmp:
        monkeypatch.setattr(worker.cv2, "imwrite",
                            lambda *args, **kwargs: False)
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        try:
            worker._guardar(img, (5, 5, 20, 20), "placa", "ABR-820")
        except OSError:
            pass
        else:
            raise AssertionError("_guardar debio fallar si no escribe")
        assert not (tmp / "proc" / "placa" / HOY / "ABR-820").exists()
