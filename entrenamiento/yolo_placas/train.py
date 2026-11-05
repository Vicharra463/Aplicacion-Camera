"""Entrena un detector de placas (YOLOv8) sobre el dataset Roboflow v9.

Uso:
    python train.py yolov8n.pt yolov8n
    python train.py yolov8s.pt yolov8s

Correr con el python que tenga torch CUDA (anaconda base). Los runs quedan en
entrenamiento\\yolo_placas\\runs\\<nombre>.
"""
import sys
from pathlib import Path

from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
DATA = BASE / "data_placas.yaml"


def main() -> None:
    if len(sys.argv) != 3:
        print("Uso: python train.py <yolov8n.pt|yolov8s.pt> <nombre_run>")
        raise SystemExit(2)
    modelo, nombre = sys.argv[1], sys.argv[2]
    YOLO(modelo).train(
        data=str(DATA),
        imgsz=640,
        epochs=100,
        batch=32,
        device=0,
        workers=4,
        project=str(BASE / "runs"),
        name=nombre,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
