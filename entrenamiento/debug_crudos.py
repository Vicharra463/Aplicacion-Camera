#!/usr/bin/env python3
"""Depuracion: lecturas CRUDAS del motor por variante, en los fallos restantes."""
import sys
from pathlib import Path
import cv2

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent / "app"))
from ocr import (_obtener_motor, _normalizar, _recorte, _clahe_gamma,
                 _umbral_invertido)  
from modelo import DetectorPlacas, RUTA_ONNX  

RAIZ = BASE.parent / "imagenes_procesadas" / "placa"
FALLOS = [
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

def raws(motor, crop):
    if crop is None or crop.size == 0:
        return []
    resultado, _ = motor(crop)
    return [(str(t), round(float(s), 2)) for _b, t, s in (resultado or [])]

def bbox_txt(ruta_txt, ancho, alto):
    linea = ruta_txt.read_text(encoding="utf-8").strip().splitlines()[0]
    _c, cx, cy, w, h = (float(v) for v in linea.split())
    bw, bh = int(w * ancho), int(h * alto)
    return (int(cx * ancho - bw / 2), int(cy * alto - bh / 2), bw, bh)

motor = _obtener_motor()
detector = DetectorPlacas(RUTA_ONNX)
for dia, nombre, real in FALLOS:
    ruta = RAIZ / dia / nombre
    img = cv2.imread(str(ruta))
    alto, ancho = img.shape[:2]
    bb = bbox_txt(ruta.with_suffix(".txt"), ancho, alto)
    print(f"\n=== {dia}/{nombre} (real {real}) bbox_txt={bb} ===")
    base = _recorte(img, bb, 0.35)
    variantes = [("recorte", base),
                 ("clahe", _clahe_gamma(base)),
                 ("umbral_inv", _umbral_invertido(base))]
    grande = cv2.resize(base, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    variantes.append(("upscale2x", grande))
    for nom, crop in variantes:
        print(f"  txt/{nom}: {raws(motor, crop)}")
    det = detector.detectar(img)
    for i, (bbb, conf) in enumerate(det[:3]):
        print(f"  modelo[{i}] conf={conf} caja={bbb}: "
              f"{raws(motor, _recorte(img, bbb, 0.35))}")
