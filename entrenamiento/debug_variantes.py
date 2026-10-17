#!/usr/bin/env python3
"""Experimento: variantes nuevas de preprocesamiento sobre los casos ilegibles."""
import sys
from pathlib import Path
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent / "app"))
from ocr import _obtener_motor, _recorte  # noqa: E402

RAIZ = BASE.parent / "imagenes_procesadas" / "placa"
CASOS = [
    ("2026-10-13", "CF635.jpg",   "ACF-635"),
    ("2026-10-14", "4O-820.jpg",  "AOR-820"),
    ("2026-10-15", "1G-777.jpg",  "WIG-774"),
    ("2026-10-15", "NTA703.jpg",  "WIA-703"),
    ("2026-10-15", "NTG-774.jpg", "WIG-774"),
    ("2026-10-16", "A7-935.jpg",  "AWI-935"),
    ("2026-10-16", "F1-953.jpg",  "FIH-953"),
]

def banda_media(img):
    """Placa ancha y baja: si el bbox es casi cuadrado, la mitad vertical
    central suele contener solo la placa."""
    alto = img.shape[0]
    y0, y1 = int(alto * 0.25), int(alto * 0.75)
    return img[y0:y1]

def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def raws(motor, crop):
    if crop is None or getattr(crop, "size", 0) == 0:
        return []
    resultado, _ = motor(crop)
    return [(str(t), round(float(s), 2)) for _b, t, s in (resultado or [])]

motor = _obtener_motor()
for dia, nombre, real in CASOS:
    ruta = RAIZ / dia / nombre
    img = cv2.imread(str(ruta))
    alto, ancho = img.shape[:2]
    linea = ruta.with_suffix(".txt").read_text(encoding="utf-8").strip().splitlines()[0]
    _c, cx, cy, w, h = (float(v) for v in linea.split())
    bw, bh = int(w * ancho), int(h * alto)
    bb = (int(cx * ancho - bw / 2), int(cy * alto - bh / 2), bw, bh)
    base = _recorte(img, bb, 0.35)
    print(f"\n=== {dia}/{nombre} (real {real}) bbox={bb} ===")
    print(f"  banda_media:   {raws(motor, banda_media(base))}")
    print(f"  sharpen:       {raws(motor, sharpen(base))}")
    print(f"  sharpen+up3x:  {raws(motor, cv2.resize(sharpen(base), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))}")
    print(f"  up3x:          {raws(motor, cv2.resize(base, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))}")
    print(f"  margen 0.6:    {raws(motor, _recorte(img, bb, 0.6))}")
