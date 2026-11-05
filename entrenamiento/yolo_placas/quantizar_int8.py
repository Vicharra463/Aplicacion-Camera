"""Cuantiza el small ONNX a INT8 (dynamic, per-channel) para CPU.

Uso: venv\\Scripts\\python.exe quantizar_int8.py
Salida: runs\\yolov8s\\weights\\best_int8.onnx
"""
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

BASE = Path(__file__).resolve().parent
ORIG = BASE / "runs" / "yolov8s" / "weights" / "best.onnx"
SALIDA = BASE / "runs" / "yolov8s" / "weights" / "best_int8.onnx"


def main() -> int:
    print(f"origen : {ORIG} ({ORIG.stat().st_size / 1e6:.1f} MB)")
    quantize_dynamic(
        model_input=str(ORIG),
        model_output=str(SALIDA),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )
    print(f"salida : {SALIDA} ({SALIDA.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
