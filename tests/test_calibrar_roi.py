"""calibrar_roi: escritura de [roi] en config.ini sin romper el archivo.

La reescritura es lo delicado del script: configparser reescribiria el
ini entero y borraria TODOS los comentarios, asi que se opera sobre el
texto crudo con regex acotada a las tres claves de [roi].
"""
import importlib.util
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture
def calibrar():
    spec = importlib.util.spec_from_file_location(
        "calibrar_roi", RAIZ / "scripts" / "calibrar_roi.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ini_roi() -> str:
    return (
        "; comentario original\n"
        "[ftp]\npuerto = 2121\n"
        "[roi]\n"
        "; ROI franja verde (camara fija)\n"
        "habilitado = no\n"
        "linea_verde_sup = 0.45\n"
        "linea_verde_inf = 0.75\n"
        "[log]\nnivel = INFO\n"
    )


def test_escribir_roi_preserva_comentarios_y_demas_secciones(calibrar,
                                                              monkeypatch):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / "config.ini"
        ruta.write_text(_ini_roi(), encoding="utf-8")
        monkeypatch.setattr(calibrar, "RUTA_CONFIG", ruta)

        calibrar.escribir_roi(0.33, 0.66)

        texto = ruta.read_text(encoding="utf-8")
    # lo que no se toca, queda
    assert "; comentario original" in texto
    assert "; ROI franja verde (camara fija)" in texto
    assert "puerto = 2121" in texto
    assert "nivel = INFO" in texto
    # las tres claves se actualizan
    assert "linea_verde_sup = 0.3300" in texto
    assert "linea_verde_inf = 0.6600" in texto
    assert "habilitado = si" in texto
    # y no se duplica la seccion
    assert texto.count("[roi]") == 1


def test_escribir_roi_agrega_la_seccion_si_falta(calibrar, monkeypatch):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / "config.ini"
        ruta.write_text("[ftp]\npuerto = 2121\n", encoding="utf-8")
        monkeypatch.setattr(calibrar, "RUTA_CONFIG", ruta)

        calibrar.escribir_roi(0.40, 0.80)

        texto = ruta.read_text(encoding="utf-8")
    assert "[roi]" in texto
    assert "linea_verde_sup = 0.4000" in texto
    assert "linea_verde_inf = 0.8000" in texto
    assert "habilitado = si" in texto
    assert "puerto = 2121" in texto


def test_foto_mas_reciente_elige_la_mas_nueva(calibrar):
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmp:
        carpeta = Path(tmp)
        vieja = carpeta / "a.jpg"
        nueva = carpeta / "b.jpg"
        (carpeta / "no_foto.txt").write_text("x", encoding="utf-8")
        vieja.write_bytes(b"x")
        nueva.write_bytes(b"x")
        os.utime(vieja, (1_600_000_000, 1_600_000_000))
        os.utime(nueva, (1_700_000_000, 1_700_000_000))

        assert calibrar.foto_mas_reciente(carpeta) == nueva


def test_foto_mas_reciente_sin_fotos_devuelve_none(calibrar):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        assert calibrar.foto_mas_reciente(Path(tmp)) is None
    assert calibrar.foto_mas_reciente(Path(tmp) / "que_no_existe") is None
