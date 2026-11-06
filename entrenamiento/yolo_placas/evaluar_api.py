"""Evalua la cadena PRODUCCION detectar -> recortar -> PlateRecognizer
con un ONNX candidato, contra GT-29.

Usa las mismas piezas que produccion:
  - DetectorPlacas (app/modelo.py) para la deteccion
  - leer_placa_6_api (app/platerecognizer.py) para recorte+API+fallback local
  - DIAGNOSTICOS/RAIZ (app/gt_placas.py) como verdad

Ademas del % total, reporta cuantas salieron por la API (fuente 'api') y
cuantas cayeron al motor local (fallback).

Uso:
    venv\\Scripts\\python.exe evaluar_api.py <candidato.onnx>
"""
from __future__ import annotations

import configparser
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Aplicacion-Camera")
sys.path.insert(0, r"D:\Aplicacion-Camera\app")
import cv2

from gt_placas import DIAGNOSTICOS, RAIZ
from modelo import DetectorPlacas
import platerecognizer
import worker
from ftp_server import cargar_config

CONFIG = Path(r"D:\Aplicacion-Camera\config.ini")


def _configurar_tokens() -> None:
    cfg = configparser.ConfigParser()
    cfg.read(CONFIG, encoding="utf-8")
    platerecognizer.configurar(cfg.get("aoi", "tokens", fallback=""))


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    ruta = Path(sys.argv[1]).resolve()
    if not ruta.is_file():
        print(f"[ERROR] No existe {ruta}")
        return 1

    _configurar_tokens()
    if not platerecognizer.hay_claves():
        print("[ERROR] sin tokens PlateRecognizer en config.ini [aoi]")
        return 1

    detector = DetectorPlacas(ruta)
    # ROI franja verde de produccion: mismo codigo (_recortar_roi) y
    # mismas claves (cargar_config) que el worker. Con habilitado = no
    # la imagen pasa identico que siempre.
    worker._cfg = cargar_config()
    roi = worker._cfg.get("roi_habilitado", False)
    print(f"[eval] modelo={ruta.name}  claves={len(platerecognizer._tokens)}"
          f"  roi={'si' if roi else 'no'}")

    aciertos = 0
    por_api = 0
    por_local = 0
    fallos: list[str] = []

    for dia, nombre, real, nota in DIAGNOSTICOS:
        img = cv2.imread(str(RAIZ / dia / nombre))
        if img is None:
            fallos.append(f"{dia}/{nombre}: FOTO NO ENCONTRADA")
            continue
        img, _y_roi = worker._preparar_imagen(img)

        detecciones = detector.detectar(
            img, enmascarar=not worker._cfg.get("roi_habilitado", False))
        bboxes = [bb for (bb, _c) in
                  sorted(detecciones, key=lambda par: par[0][2] * par[0][3],
                         reverse=True)]

        placa, conf, intentos = "", 0.0, []
        for bb in bboxes:
            placa, conf, intentos = platerecognizer.leer_placa_6_api(
                img, bb, enmascarar=not roi)
            if placa:
                break

        fuente = (intentos[0][0] if intentos else "sin")
        if fuente == "api":
            por_api += 1
        elif placa:
            por_local += 1

        ok = placa == real
        if ok:
            aciertos += 1
        else:
            fallos.append(f"{dia}/{nombre}: real={real} motor={placa or 'SIN LECTURA'} "
                          f"(fuente={fuente}, cand={len(bboxes)})")
        marca = "OK  " if ok else "FALLO"
        print(f"  {marca} {dia}/{nombre}: {placa or 'SIN LECTURA'} "
              f"[{fuente}] (real={real})")

    total = len(DIAGNOSTICOS)
    pct = 100 * aciertos / total if total else 0
    print(f"\n  TOTAL: {aciertos}/{total} = {pct:.1f}%")
    print(f"  via API: {por_api}  |  via motor local: {por_local}  "
          f"|  sin lectura: {total - por_api - por_local}")
    if fallos:
        print(f"  Fallos ({len(fallos)}):")
        for f in fallos:
            print(f"    {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
