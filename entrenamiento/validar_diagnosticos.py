#!/usr/bin/env python3
"""Validacion de Fase 0 contra los diagnosticos hechos a mano.

Corre el motor de 6 intentos sobre las fotos con veredicto visual conocido
(memorias #610 y #624) y mide el % de acierto contra la placa REAL visible,
no contra el nombre del archivo (el nombre es la lectura historica, ~60%
errada). Meta del plan: >= 90%.

Uso:
    python validar_diagnosticos.py
    python validar_diagnosticos.py --verbose   # detalle de cada intento
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent / "app"))

from ocr import leer_placa_6  
from gt_placas import DIAGNOSTICOS, RAIZ  


def _bbox_desde_txt(ruta_txt: Path, ancho: int, alto: int):
    try:
        linea = ruta_txt.read_text(encoding="utf-8").strip().splitlines()[0]
        _c, cx, cy, w, h = linea.split()
        cx, cy, w, h = (float(v) for v in (cx, cy, w, h))
    except (OSError, ValueError, IndexError):
        return None
    bw, bh = w * ancho, h * alto
    return int(cx * ancho - bw / 2), int(cy * alto - bh / 2), int(bw), int(bh)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true",
                    help="imprime los 6 intentos de cada fallo")
    ap.add_argument("--margen", type=float, default=0.35)
    ap.add_argument("--detector", choices=("txt", "modelo"), default="txt",
                    help="txt = bbox guardados (pipeline viejo); "
                         "modelo = ONNX placas.onnx (camino de produccion)")
    args = ap.parse_args()

    detector = None
    if args.detector == "modelo":
        from modelo import DetectorPlacas, RUTA_ONNX
        if not RUTA_ONNX.is_file():
            print(f"[ERROR] No existe {RUTA_ONNX}")
            return 1
        detector = DetectorPlacas(RUTA_ONNX)

    aciertos = 0
    fallos: list[str] = []
    por_dia: dict[str, list[int]] = {}

    for dia, nombre, real, nota in DIAGNOSTICOS:
        ruta = RAIZ / dia / nombre
        img = cv2.imread(str(ruta)) if ruta.is_file() else None
        if img is None:
            print(f"[FALTA] {dia}/{nombre}")
            fallos.append(f"{dia}/{nombre}: FOTO NO ENCONTRADA")
            continue
        if detector is not None:
            
            detecciones = detector.detectar(img)
            bboxes = [bb for (bb, _c) in
                      sorted(detecciones, key=lambda par: par[0][2] * par[0][3],
                             reverse=True)]
        else:
            alto, ancho = img.shape[:2]
            bbox = _bbox_desde_txt(ruta.with_suffix(".txt"), ancho, alto)
            bboxes = [bbox] if bbox else []
        placa, conf, intentos = "", 0.0, []
        for bb in bboxes:
            placa, conf, intentos = leer_placa_6(img, bb, args.margen)
            if placa:
                break

        ok = placa == real
        contador = por_dia.setdefault(dia, [0, 0])
        contador[0] += 1
        if ok:
            aciertos += 1
            contador[1] += 1
            print(f"  OK   {dia}/{nombre}: {placa}  (historico tenia: {nota})")
        else:
            detalle = ""
            if args.verbose:
                detalle = " | " + " | ".join(
                    f"{n}={(t or '-')}({c:.2f})" for n, t, c in intentos)
            print(f"  FALLO {dia}/{nombre}: real={real} motor={placa or 'SIN LECTURA'} "
                  f"conf={conf:.2f}  <- {nota}{detalle}")
            fallos.append(f"{dia}/{nombre}: real={real} motor={placa or '(nada)'}")

    total = sum(n for n, _ok in por_dia.values())
    print("\n=== RESUMEN por dia ===")
    for d, (n, ok) in sorted(por_dia.items()):
        print(f"  {d}: {ok}/{n} ({100 * ok / n if n else 0:.0f}%)")
    pct = 100 * aciertos / total if total else 0
    print(f"\n  TOTAL: {aciertos}/{total} = {pct:.1f}% de acierto "
          f"(meta >= 90%) -> {'PASA' if pct >= 90 else 'NO PASA'}")
    if fallos:
        print(f"  Fallos ({len(fallos)}):")
        for f in fallos:
            print(f"    {f}")
    return 0 if pct >= 90 else 2


if __name__ == "__main__":
    raise SystemExit(main())
