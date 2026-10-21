#!/usr/bin/env python3
"""Depuracion del motor OCR sobre fallos conocidos.

Subcomandos:
    crudos     lecturas CRUDAS por variante (recorte/clahe/umbral/upscale)
               + top-3 del detector ONNX  [antes debug_crudos.py]
    ocr        OCR crudo sin normalizar sobre los crops  [antes debug_ocr.py]
    variantes  experimentos de preprocesamiento nuevos  [antes debug_variantes.py]

Uso:
    python entrenamiento/herramientas/depurar.py crudos
    python entrenamiento/herramientas/depurar.py ocr
    python entrenamiento/herramientas/depurar.py variantes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parents[1] / "app"))

from ocr import (_obtener_motor, _normalizar, _recorte, _clahe_gamma,
                 _umbral_invertido)

RAIZ = BASE.parents[1] / "imagenes_procesadas" / "placa"

FALLOS_CRUDOS = [
    ("2026-10-13", "B1V-756.jpg",  "B1Y-756"),
    ("2026-10-13", "BIY-756.jpg",  "B1Y-756"),
    ("2026-10-13", "CF635.jpg",    "ACF-635"),
    ("2026-10-13", "C6Y-488.jpg",  "C6Y-488"),
    ("2026-10-14", "4O-820.jpg",   "AOR-820"),
    ("2026-10-15", "1G-777.jpg",   "WIG-774"),
    ("2026-10-15", "C21-303.jpg",  "C2I-303"),
    ("2026-10-15", "E5A947.jpg",   "F5A-947"),
    ("2026-10-15", "NTA703.jpg",   "WIA-703"),
    ("2026-10-15", "NTG-774.jpg",  "WIG-774"),
    ("2026-10-15", "Z1S-508.jpg",  "ZIS-508"),
    ("2026-10-16", "A7-935.jpg",   "AWI-935"),
    ("2026-10-16", "AHL-935.jpg",  "AWI-935"),
    ("2026-10-16", "EZX165.jpg",   "E7X-165"),
    ("2026-10-16", "F1-953.jpg",   "FIH-953"),
]

CASOS_OCR = [
    ("2026-10-13", "12-790.jpg"),
    ("2026-10-13", "AFH-559.jpg"),
    ("2026-10-16", "BTU394.jpg"),
    ("2026-10-16", "F8E876.jpg"),
    ("2026-10-15", "ZIS-508.jpg"),
    ("2026-10-13", "A3E726.jpg"),
]

CASOS_VARIANTES = [
    ("2026-10-13", "CF635.jpg",   "ACF-635"),
    ("2026-10-14", "4O-820.jpg",  "AOR-820"),
    ("2026-10-15", "1G-777.jpg",  "WIG-774"),
    ("2026-10-15", "NTA703.jpg",  "WIA-703"),
    ("2026-10-15", "NTG-774.jpg", "WIG-774"),
    ("2026-10-16", "A7-935.jpg",  "AWI-935"),
    ("2026-10-16", "F1-953.jpg",  "FIH-953"),
]


def _bbox_txt(ruta_txt: Path, ancho: int, alto: int):
    linea = ruta_txt.read_text(encoding="utf-8").strip().splitlines()[0]
    _c, cx, cy, w, h = (float(v) for v in linea.split())
    bw, bh = int(w * ancho), int(h * alto)
    return (int(cx * ancho - bw / 2), int(cy * alto - bh / 2), bw, bh)


def _raws(motor, crop):
    if crop is None or getattr(crop, "size", 0) == 0:
        return []
    resultado, _ = motor(crop)
    return [(str(t), round(float(s), 2)) for _b, t, s in (resultado or [])]


def _banda_media(img):
    alto = img.shape[0]
    y0, y1 = int(alto * 0.25), int(alto * 0.75)
    return img[y0:y1]


def _sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def cmd_crudos() -> int:
    from modelo import DetectorPlacas, RUTA_ONNX
    motor = _obtener_motor()
    detector = DetectorPlacas(RUTA_ONNX)
    for dia, nombre, real in FALLOS_CRUDOS:
        ruta = RAIZ / dia / nombre
        img = cv2.imread(str(ruta))
        alto, ancho = img.shape[:2]
        bb = _bbox_txt(ruta.with_suffix(".txt"), ancho, alto)
        print(f"\n=== {dia}/{nombre} (real {real}) bbox_txt={bb} ===")
        base = _recorte(img, bb, 0.35)
        variantes = [("recorte", base),
                     ("clahe", _clahe_gamma(base)),
                     ("umbral_inv", _umbral_invertido(base))]
        grande = cv2.resize(base, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        variantes.append(("upscale2x", grande))
        for nom, crop in variantes:
            print(f"  txt/{nom}: {_raws(motor, crop)}")
        det = detector.detectar(img)
        for i, (bbb, conf) in enumerate(det[:3]):
            print(f"  modelo[{i}] conf={conf} caja={bbb}: "
                  f"{_raws(motor, _recorte(img, bbb, 0.35))}")
    return 0


def cmd_ocr() -> int:
    motor = _obtener_motor()
    for dia, nombre in CASOS_OCR:
        ruta = RAIZ / dia / nombre
        img = cv2.imread(str(ruta))
        alto, ancho = img.shape[:2]
        bb = _bbox_txt(ruta.with_suffix(".txt"), ancho, alto)
        crop = _recorte(img, bb, 0.35)
        resultado, _ = motor(crop) if crop is not None and crop.size else (None, None)
        crudos = [(t, round(float(s), 2)) for _b, t, s in (resultado or [])]
        print(f"{dia}/{nombre}: img={ancho}x{alto} bbox={bb} "
              f"crop={None if crop is None else crop.shape[:2]}")
        print(f"  crudos: {crudos}")
        print(f"  normaliza: {[(_normalizar(t), t) for t, _s in crudos]}")
    return 0


def cmd_variantes() -> int:
    motor = _obtener_motor()
    for dia, nombre, real in CASOS_VARIANTES:
        ruta = RAIZ / dia / nombre
        img = cv2.imread(str(ruta))
        alto, ancho = img.shape[:2]
        bb = _bbox_txt(ruta.with_suffix(".txt"), ancho, alto)
        base = _recorte(img, bb, 0.35)
        print(f"\n=== {dia}/{nombre} (real {real}) bbox={bb} ===")
        print(f"  banda_media:   {_raws(motor, _banda_media(base))}")
        print(f"  sharpen:       {_raws(motor, _sharpen(base))}")
        print(f"  sharpen+up3x:  {_raws(motor, cv2.resize(_sharpen(base), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))}")
        print(f"  up3x:          {_raws(motor, cv2.resize(base, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))}")
        print(f"  margen 0.6:    {_raws(motor, _recorte(img, bb, 0.6))}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="comando", required=True)
    sub.add_parser("crudos", help="lecturas crudas por variante + ONNX")
    sub.add_parser("ocr", help="OCR crudo sin normalizar")
    sub.add_parser("variantes", help="experimentos de preprocesamiento")
    args = ap.parse_args()
    return {"crudos": cmd_crudos, "ocr": cmd_ocr,
            "variantes": cmd_variantes}[args.comando]()


if __name__ == "__main__":
    raise SystemExit(main())
