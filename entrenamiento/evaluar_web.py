#!/usr/bin/env python
"""Evaluacion rapida de una imagen para la web (JSON por stdout).

Corre el MISMO pipeline de deteccion + OCR que el worker (modelo ONNX
placas.onnx, OSD enmascarado, RapidOCR) y devuelve JSON para que el
front muestre al instante si hay placa y donde esta.

Uso (lo llama server.js):
    python evaluar_web.py ruta\\a\\imagen.jpg
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import cv2

RUTA_ONNX = Path(__file__).resolve().parents[1] / "modelo" / "placas.onnx"
UMBRAL = 0.40  


def b64(img) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def main(ruta: str) -> int:
    
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
    from modelo import DetectorPlacas, RUTA_ONNX as ONNX
    from ocr import leer_placa

    img = cv2.imread(ruta)
    if img is None:
        print(json.dumps({"ok": False, "error": f"No se pudo leer {ruta}"}))
        return 1

    if not ONNX.is_file():
        print(json.dumps({"ok": False, "error": f"No existe el modelo {ONNX}"}))
        return 1

    detector = DetectorPlacas(ONNX, umbral=UMBRAL)
    dets = detector.detectar(img)

    if not dets:
        print(json.dumps({
            "ok": True,
            "detectado": False,
            "mensaje": "Sin detectar placa",
            "anotada": b64(img),
        }))
        return 0

    
    (x, y, w, h), conf = dets[0]
    recorte = img[y:y + h, x:x + w]
    texto = leer_placa(recorte) if recorte.size else ""

    
    anotada = img.copy()
    color = (0, 200, 0) if texto else (0, 200, 255)
    cv2.rectangle(anotada, (x, y), (x + w, y + h), color, 3)
    etiqueta = texto or f"conf {conf:.2f}"
    cv2.putText(anotada, etiqueta, (x, max(24, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    print(json.dumps({
        "ok": True,
        "detectado": True,
        "placa": texto or None,
        "confianza": round(float(conf), 3),
        "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        "x0": int(x), "x1": int(x + w),
        "y0": int(y), "y1": int(y + h),
        "detecciones": len(dets),
        "recorte": b64(recorte),
        "anotada": b64(anotada),
    }))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"ok": False, "error": "Uso: python evaluar_web.py imagen.jpg"}))
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
