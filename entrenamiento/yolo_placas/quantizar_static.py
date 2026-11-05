"""Cuantiza el small ONNX a INT8 ESTATICO (QDQ) con calibracion.

El INT8 dinámico dio 3795 ms (kernels de referencia lentos); QDQ estatico con
calibracion es el recipe estandar de YOLO en CPU.

Uso: venv\\Scripts\\python.exe quantizar_static.py
Salida: runs\\yolov8s\\weights\\best_int8_static.onnx
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

sys.path.insert(0, r"D:\Aplicacion-Camera\app")
from modelo import FONDO, enmascarar_osd  # noqa: E402

BASE = Path(__file__).resolve().parent
ORIG = BASE / "runs" / "yolov8s" / "weights" / "best.onnx"
SALIDA = BASE / "runs" / "yolov8s" / "weights" / "best_int8_static.onnx"
TRAIN = Path(r"C:/Users/FELIPE/Downloads/Peru License Plate.v9i.yolov8/train/images")
LADO = 640
N_CALIB = 100


def prep(ruta: Path) -> np.ndarray:
    """Mismo preproceso que DetectorPlacas.detectar (sin OSD)."""
    img = cv2.imread(str(ruta))
    entrada = enmascarar_osd(img)
    alto, ancho = entrada.shape[:2]
    escala = min(LADO / ancho, LADO / alto)
    nw, nh = max(1, int(round(ancho * escala))), max(1, int(round(alto * escala)))
    redim = cv2.resize(entrada, (nw, nh), interpolation=cv2.INTER_LINEAR)
    lienzo = np.full((LADO, LADO, 3), FONDO, dtype=np.uint8)
    dx, dy = (LADO - nw) // 2, (LADO - nh) // 2
    lienzo[dy:dy + nh, dx:dx + nw] = redim
    blob = lienzo[:, :, ::-1].transpose(2, 0, 1)
    blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
    return blob[None]


class Reader(CalibrationDataReader):
    def __init__(self, rutas: list[Path]) -> None:
        self._it = iter(rutas)

    def get_next(self):
        ruta = next(self._it, None)
        if ruta is None:
            return None
        return {"images": prep(ruta)}


def main() -> int:
    rutas = sorted(TRAIN.glob("*.jpg"))
    random.Random(42).shuffle(rutas)
    muestra = rutas[:N_CALIB]
    print(f"calibracion: {len(muestra)} imagenes de {TRAIN}")
    print(f"origen : {ORIG} ({ORIG.stat().st_size / 1e6:.1f} MB)")
    quantize_static(
        model_input=str(ORIG),
        model_output=str(SALIDA),
        calibration_data_reader=Reader(muestra),
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        per_channel=True,
        activation_type=QuantType.QInt8,
    )
    print(f"salida : {SALIDA} ({SALIDA.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
