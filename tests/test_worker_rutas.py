import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

import worker


@contextmanager
def _carpeta_tmp():
    ruta = Path(tempfile.mkdtemp(prefix="cytv_test_"))
    cfg_previo = worker._cfg
    try:
        yield ruta
    finally:
        worker._cfg = cfg_previo
        shutil.rmtree(ruta, ignore_errors=True)


def test_guardar_crea_subcarpeta_por_placa():
    with _carpeta_tmp() as tmp:
        worker._cfg = {"procesadas": tmp / "proc"}
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        destino = worker._guardar(img, (10, 20, 100, 40), "placa",
                                  "ABR-820", "_api-Roboflow")
        dia = datetime.now().strftime("%Y-%m-%d")
        carpeta = tmp / "proc" / "placa" / dia / "ABR-820"
        assert destino.parent == carpeta
        assert destino.name == "ABR-820_api-Roboflow.jpg"
        assert destino.is_file()
        assert destino.with_suffix(".txt").is_file()
        assert destino.with_name(destino.stem + "_crop.jpg").is_file()


def test_guardar_dudosa_va_a_la_misma_subcarpeta():
    with _carpeta_tmp() as tmp:
        worker._cfg = {"procesadas": tmp / "proc"}
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        destino = worker._guardar(img, (10, 20, 100, 40), "placa",
                                  "ABR-820", "_api-Roboflow", dudosa=True)
        dia = datetime.now().strftime("%Y-%m-%d")
        assert destino.parent == tmp / "proc" / "placa" / dia / "ABR-820"
        assert "(DUDOSA)" in destino.name


def test_guardar_sin_placa_se_queda_en_la_carpeta_del_dia():
    with _carpeta_tmp() as tmp:
        worker._cfg = {"procesadas": tmp / "proc"}
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        destino = worker._guardar(img, None, "placa", "")
        dia = datetime.now().strftime("%Y-%m-%d")
        assert destino.parent == tmp / "proc" / "placa" / dia
