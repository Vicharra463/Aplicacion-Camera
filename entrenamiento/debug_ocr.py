#!/usr/bin/env python3
"""Depuracion: que devuelve el OCR CRUDO (sin normalizar) sobre los crops."""
import sys
from pathlib import Path
import cv2

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent / "app"))
from ocr import _obtener_motor, _normalizar, _recorte  # noqa: E402

RAIZ = BASE.parent / "imagenes_procesadas" / "placa"
CASOS = [
    ("2026-10-13", "12-790.jpg"),
    ("2026-10-13", "AFH-559.jpg"),   # este SI pasa
    ("2026-10-16", "BTU394.jpg"),    # este SI pasa
    ("2026-10-16", "F8E876.jpg"),    # historico OK, ahora sin lectura
    ("2026-10-15", "ZIS-508.jpg"),   # historico OK, ahora sin lectura
    ("2026-10-13", "A3E726.jpg"),    # historico OK, ahora sin lectura
]

motor = _obtener_motor()
for dia, nombre in CASOS:
    ruta = RAIZ / dia / nombre
    img = cv2.imread(str(ruta))
    alto, ancho = img.shape[:2]
    linea = ruta.with_suffix(".txt").read_text(encoding="utf-8").strip().splitlines()[0]
    _c, cx, cy, w, h = (float(v) for v in linea.split())
    bw, bh = int(w * ancho), int(h * alto)
    x, y = int(cx * ancho - bw / 2), int(cy * alto - bh / 2)
    bbox = (x, y, bw, bh)
    crop = _recorte(img, bbox, 0.35)
    resultado, _ = motor(crop) if crop is not None and crop.size else (None, None)
    crudos = [(t, round(float(s), 2)) for _b, t, s in (resultado or [])]
    print(f"{dia}/{nombre}: img={ancho}x{alto} bbox={bbox} crop={None if crop is None else crop.shape[:2]}")
    print(f"  crudos: {crudos}")
    print(f"  normaliza: {[(_normalizar(t), t) for t, _s in crudos]}")
