#!/usr/bin/env python
"""Convierte las anotaciones COCO del dataset publico a formato YOLO.

Entrada : C:\\Users\\felip\\anpr\\dataset_fuente\\raw\\{train,valid,test}\\
          *.jpg + _annotations.coco.json   (export de Roboflow)
Salida  : misma carpeta, un .txt por imagen  (YOLO: clase cx cy w h, 0..1)
          + raw/data.yaml que ultralytics usa para entrenar.

Uso (con el entorno del ML, NO con el Python del servicio):
    C:\\Users\\felip\\anpr\\venv\\Scripts\\python.exe convertir_coco_a_yolo.py

El dataset publico es la fase 0: sirve para el entrenamiento base. Las
imagenes de la camara se suman despues (fase 5) con el mismo formato.
"""

from __future__ import annotations

import json
from pathlib import Path

RAIZ_DATASET = Path(r"C:\Users\felip\anpr\dataset_fuente\raw")
SPLITS = ("train", "valid", "test")
NOMBRE_CLASE = "license_plate"

DATA_YAML = """\
path: {raiz}
train: train
val: valid
test: test
names:
  0: {clase}
"""


def convertir_split(carpeta: Path) -> tuple[int, int, int]:
    """Devuelve (imagenes escritas, anotaciones, imagenes sin caja)."""
    ruta_json = carpeta / "_annotations.coco.json"
    if not ruta_json.is_file():
        raise FileNotFoundError(f"Falta {ruta_json}")

    coco = json.loads(ruta_json.read_text(encoding="utf-8"))

    # Mapa categoria COCO -> indice YOLO contiguo (0, 1, 2...). Nunca se
    # usa el id de COCO tal cual: a veces arranca en 1 y romperia el .txt.
    categorias = sorted(c["id"] for c in coco["categories"])
    indice = {cid: i for i, cid in enumerate(categorias)}
    imagenes = {im["id"]: im for im in coco["images"]}

    por_imagen: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd"):
            continue
        por_imagen.setdefault(ann["image_id"], []).append(ann)

    escritas = 0
    anotaciones = 0
    sin_caja = 0

    for imagen_id, im in imagenes.items():
        ancho, alto = float(im["width"]), float(im["height"])
        lineas: list[str] = []

        for ann in por_imagen.get(imagen_id, []):
            x, y, w, h = (float(v) for v in ann["bbox"])
            if w <= 0 or h <= 0:
                continue
            cx = (x + w / 2.0) / ancho
            cy = (y + h / 2.0) / alto
            nw, nh = w / ancho, h / alto
            # Recorte defensivo: un bbox fuera de cuadro rompe el entrenamiento.
            if not (0.0 < nw <= 1.0 and 0.0 < nh <= 1.0):
                continue
            clase = indice[ann["category_id"]]
            lineas.append(f"{clase} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            anotaciones += 1

        etiqueta = carpeta / (Path(im["file_name"]).stem + ".txt")
        etiqueta.write_text("\n".join(lineas) + ("\n" if lineas else ""),
                            encoding="utf-8")
        escritas += 1
        if not lineas:
            sin_caja += 1

    return escritas, anotaciones, sin_caja


def main() -> int:
    if not RAIZ_DATASET.is_dir():
        print(f"[ERROR] No existe {RAIZ_DATASET}")
        return 1

    for split in SPLITS:
        carpeta = RAIZ_DATASET / split
        if not carpeta.is_dir():
            print(f"[ERROR] Falta la carpeta {carpeta}")
            return 1
        escritas, anotaciones, sin_caja = convertir_split(carpeta)
        print(f"{split:6s}: {escritas:5d} imagenes | {anotaciones:5d} cajas"
              f" | {sin_caja} sin placa")

    (RAIZ_DATASET / "data.yaml").write_text(
        DATA_YAML.format(raiz=str(RAIZ_DATASET), clase=NOMBRE_CLASE),
        encoding="utf-8")
    print(f"data.yaml -> {RAIZ_DATASET / 'data.yaml'}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
