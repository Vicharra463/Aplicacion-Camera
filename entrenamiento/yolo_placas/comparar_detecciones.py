"""Compara detecciones nano (candidato) vs prod en las fotos que fallan GT-29.

Responde: el nano NO detecta la placa, o detecta pero el crop peor?

Uso: venv\\Scripts\\python.exe comparar_detecciones.py
"""
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Aplicacion-Camera\app")
import cv2

from gt_placas import RAIZ
from modelo import DetectorPlacas

NANO = Path(
    r"D:\Aplicacion-Camera\entrenamiento\yolo_placas\runs\yolov8n\weights\best.onnx"
)
PROD = Path(r"D:\Aplicacion-Camera\modelo\placas.onnx")
FALLAS = [
    "2026-10-13/AFH-559.jpg",
    "2026-10-14/4O-820.jpg",
    "2026-10-15/1G-777.jpg",
    "2026-10-15/E5A947.jpg",
]


def main() -> int:
    umbral = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
    nano = DetectorPlacas(NANO, umbral=umbral)
    prod = DetectorPlacas(PROD, umbral=umbral)
    print(f"umbral={umbral}")
    for rel in FALLAS:
        ruta = RAIZ / rel
        img = cv2.imread(str(ruta))
        print(f"\n== {rel} ==")
        if img is None:
            print("   FOTO NO LEIDA")
            continue
        print(f"   img {img.shape[1]}x{img.shape[0]}")
        for nombre, det in (("nano", nano), ("prod", prod)):
            dets = det.detectar(img)
            if not dets:
                print(f"   {nombre}: SIN DETECCION")
            for caja, conf in dets[:3]:
                x, y, w, h = caja
                print(
                    f"   {nombre}: conf={conf:.3f} caja=({x},{y},{w},{h}) "
                    f"rel_w={w / img.shape[1]:.3f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
