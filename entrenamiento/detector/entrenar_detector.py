#!/usr/bin/env python
"""Fase 1: entrena el detector de placas (YOLOv8n) y lo exporta al servicio.

Transfer learning sobre los pesos COCO: se congela poco y se reentrena la
cabeza sobre el dataset publico de placas (8.823 imagenes, clase unica
``license_plate``). Despues, en la fase 5, se suman las imagenes propias del
carril de Ancon y se reentrena con lo aprendido.

Se corre CON EL ENTORNO DEL ML, nunca desde el servicio:

    C:\\Users\\felip\\anpr\\venv\\Scripts\\python.exe entrenar_detector.py --epocas 60

Argumentos:
    --epocas N        epocas de entrenamiento (defecto 60)
    --imgsz N         lado cuadrado de entrada (defecto 640)
    --batch N         imagenes por lote (defecto 16)
    --data RUTA       data.yaml de ultralytics (defecto: dataset_fuente)
    --salida RUTA     carpeta donde publica placas.pt/onnx
    --runs RUTA       carpeta de runs de ultralytics
    --solo-export R   exporta a ONNX un best.pt ya entrenado, sin entrenar
    --sin-onnx        no exporta a ONNX al terminar

Salida:
    C:\\Users\\felip\\anpr\\runs\\placas\\weights\\{best,last}.pt
    <proyecto>\\modelo\\placas.pt + placas.onnx   <- lo que lee el servicio

Nota de hardware: sin GPU utilizable se entrena en CPU. Un i5-9300H tarda
minutos por epoca; el script imprime el tiempo medido para que sepas cuánto
te tomará el plan completo.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

RAIZ_ANPR = Path(r"C:\Users\felip\anpr")
DATA_YAML = RAIZ_ANPR / "dataset_fuente" / "raw" / "data.yaml"
DIR_RUNS = RAIZ_ANPR / "runs"
DIR_SALIDA = Path(r"C:\Users\felip\OneDrive\Desktop\Motorgas\Camara CYTV ANCON\modelo")

MODELO_BASE = "yolov8n.pt"   
NOMBRE_RUN = "placas"


def exportar_onnx(ruta_pt: Path, imgsz: int) -> Path | None:
    """best.pt -> best.onnx (lo que el servicio carga con onnxruntime)."""
    from ultralytics import YOLO

    print(f"[export] {ruta_pt} -> ONNX (imgsz={imgsz})")
    modelo = YOLO(str(ruta_pt))
    ruta_onnx = Path(modelo.export(format="onnx", imgsz=imgsz, opset=12))
    print(f"[export] listo: {ruta_onnx} "
          f"({ruta_onnx.stat().st_size / 1024 / 1024:.1f} MB)")
    return ruta_onnx


def publicar(ruta_pt: Path, ruta_onnx: Path | None, destino: Path) -> None:
    """Copia los pesos al proyecto, que es donde el servicio los busca."""
    destino.mkdir(parents=True, exist_ok=True)
    destino_pt = destino / "placas.pt"
    shutil.copy2(ruta_pt, destino_pt)
    print(f"[publica] {destino_pt} ({destino_pt.stat().st_size / 1024 / 1024:.1f} MB)")

    if ruta_onnx and ruta_onnx.is_file():
        destino_onnx = destino / "placas.onnx"
        shutil.copy2(ruta_onnx, destino_onnx)
        print(f"[publica] {destino_onnx} "
              f"({destino_onnx.stat().st_size / 1024 / 1024:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--data", type=str, default=str(DATA_YAML),
                    help="data.yaml de ultralytics")
    ap.add_argument("--salida", type=str, default=str(DIR_SALIDA),
                    help="carpeta destino de placas.pt/onnx")
    ap.add_argument("--runs", type=str, default=str(DIR_RUNS),
                    help="carpeta de runs de ultralytics")
    ap.add_argument("--solo-export", type=str, default=None,
                    help="Ruta a un best.pt ya entrenado: exporta y publica")
    ap.add_argument("--sin-onnx", action="store_true")
    args = ap.parse_args()

    data_yaml = Path(args.data)
    dir_salida = Path(args.salida)

    if not data_yaml.is_file():
        print(f"[ERROR] Falta {data_yaml}. Corre antes convertir_coco_a_yolo.py")
        return 1

    if args.solo_export:
        ruta_pt = Path(args.solo_export)
        if not ruta_pt.is_file():
            print(f"[ERROR] No existe {ruta_pt}")
            return 1
        ruta_onnx = None if args.sin_onnx else exportar_onnx(ruta_pt, args.imgsz)
        publicar(ruta_pt, ruta_onnx, dir_salida)
        return 0

    from ultralytics import YOLO

    print("=" * 60)
    print(f"  Detector de placas - {args.epocas} epocas, imgsz {args.imgsz}, "
          f"batch {args.batch}, CPU")
    print(f"  data: {data_yaml}")
    print("=" * 60)

    modelo = YOLO(MODELO_BASE)
    t0 = time.time()

    modelo.train(
        data=str(data_yaml),
        epochs=args.epocas,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        workers=0,
        project=str(Path(args.runs)),
        name=NOMBRE_RUN,
        exist_ok=True,
        patience=10,         
        cache=False,
        verbose=True,
        plots=True,
    )

    total = time.time() - t0
    mejor = Path(args.runs) / NOMBRE_RUN / "weights" / "best.pt"
    if not mejor.is_file():
        print(f"[ERROR] No se genero {mejor}")
        return 1

    print("-" * 60)
    print(f"Entrenamiento terminado en {total / 60:.1f} min "
          f"({total / max(args.epocas, 1):.1f} s/epoca)")
    print(f"Mejores pesos: {mejor}")

    ruta_onnx = None if args.sin_onnx else exportar_onnx(mejor, args.imgsz)
    publicar(mejor, ruta_onnx, dir_salida)
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
