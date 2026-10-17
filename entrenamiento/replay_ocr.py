#!/usr/bin/env python3
"""Replay del motor OCR (6 intentos) sobre las fotos de los 4 dias.

Validacion de Fase 0: corre el pipeline de lectura sobre cada foto ya
guardada y compara contra el nombre del archivo (lectura historica). El
desacuerdo se lista para revision visual; el resumen da el % de coincidencia.

Modos:
  --modo guardado  usa el bbox guardado en el .txt YOLO (aisla al OCR)
  --modo modelo    redetecta con el ONNX y gana el de MAYOR AREA (camino
                   de produccion completo, ejercita el fix E5A947)

Uso (desde la raiz del proyecto):
    python replay_ocr.py                       # modo guardado, 4 dias
    python replay_ocr.py --dia 2026-10-14      # un solo dia
    python replay_ocr.py --modo modelo         # con deteccion ONNX
    python replay_ocr.py --solo-diferencias    # no imprime acuerdos
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent / "app"))

from ocr import leer_placa_6, _normalizar  # noqa: E402

RAIZ = BASE.parent / "imagenes_procesadas" / "placa"


def _leer_bbox_txt(ruta_txt: Path, ancho: int, alto: int):
    """Etiqueta YOLO normalizada -> (x, y, w, h) en pixeles."""
    try:
        linea = ruta_txt.read_text(encoding="utf-8").strip().splitlines()[0]
        _clase, cx, cy, w, h = linea.split()
        cx, cy, w, h = (float(v) for v in (cx, cy, w, h))
    except (OSError, ValueError, IndexError):
        return None
    bw, bh = w * ancho, h * alto
    x, y = cx * ancho - bw / 2, cy * alto - bh / 2
    return int(x), int(y), int(bw), int(bh)


def _esperado_canonico(stem: str) -> str:
    """Nombre del archivo (lectura historica) -> forma canonica comparable."""
    if stem.startswith("placa_"):
        return ""  # guardado sin lectura (nombre por hora)
    return _normalizar(stem) or re.sub(r"[^A-Za-z0-9]", "", stem).upper()


def _detector_onnx():
    try:
        from modelo import DetectorPlacas, RUTA_ONNX
        if RUTA_ONNX.is_file():
            return DetectorPlacas(RUTA_ONNX)
    except Exception as exc:
        print(f"[aviso] modelo ONNX no disponible: {exc}", file=sys.stderr)
    return None


def _bbox_mayor_area(detector, img):
    """Candidatos del modelo ordenados por AREA descendente (produccion)."""
    detecciones = detector.detectar(img)
    return [bbox for bbox, _conf in
            sorted(detecciones, key=lambda par: par[0][2] * par[0][3],
                   reverse=True)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modo", choices=("guardado", "modelo"), default="guardado")
    ap.add_argument("--dia", help="solo una fecha, p. ej. 2026-10-14")
    ap.add_argument("--margen", type=float, default=0.35)
    ap.add_argument("--csv", default=str(BASE / "replay_resultado.csv"))
    ap.add_argument("--solo-diferencias", action="store_true")
    args = ap.parse_args()

    if not RAIZ.is_dir():
        print(f"[ERROR] no existe {RAIZ}", file=sys.stderr)
        return 1

    detector = _detector_onnx() if args.modo == "modelo" else None
    if args.modo == "modelo" and detector is None:
        print("[ERROR] modo modelo requiere modelo\\placas.onnx", file=sys.stderr)
        return 1

    dias = sorted(d for d in RAIZ.iterdir() if d.is_dir())
    if args.dia:
        dias = [d for d in dias if d.name == args.dia]
        if not dias:
            print(f"[ERROR] dia no encontrado: {args.dia}", file=sys.stderr)
            return 1

    filas_csv = []
    total = aciertos = con_lectura = sin_esperado = 0
    diferencias: list[str] = []

    for dia in dias:
        for jpg in sorted(dia.glob("*.jpg")):
            txt = jpg.with_suffix(".txt")
            img = cv2.imread(str(jpg))
            if img is None or not txt.is_file():
                continue
            alto, ancho = img.shape[:2]
            esperado = _esperado_canonico(jpg.stem)

            if args.modo == "guardado":
                bbox = _leer_bbox_txt(txt, ancho, alto)
                bboxes = [bbox] if bbox else []
            else:
                bboxes = _bbox_mayor_area(detector, img)

            placa, conf, intentos = "", 0.0, []
            for bbox in bboxes:
                placa, conf, intentos = leer_placa_6(img, bbox, args.margen)
                if placa:
                    break

            total += 1
            if placa:
                con_lectura += 1
            coincide = bool(esperado) and placa == esperado
            if not esperado:
                sin_esperado += 1
            elif coincide:
                aciertos += 1
            else:
                detalle = " | ".join(
                    f"{n}={(t or '-')}({c:.2f})" for n, t, c in intentos)
                diferencias.append(
                    f"{dia.name}/{jpg.name}: esperado={esperado or '(sin nombre)'} "
                    f"motor={placa or 'SIN LECTURA'} conf={conf:.2f} "
                    f"[{detalle}]")

            filas_csv.append({
                "dia": dia.name, "archivo": jpg.name, "esperado": esperado,
                "motor": placa, "confianza": round(conf, 3),
                "coincide": int(coincide), "modo": args.modo,
            })
            if coincide and not args.solo_diferencias:
                print(f"  OK   {dia.name}/{jpg.name}: {placa}")

    # Resumen por dia
    print("\n=== RESUMEN ===")
    por_dia: dict[str, list[int]] = {}
    for f in filas_csv:
        c = por_dia.setdefault(f["dia"], [0, 0])
        c[0] += 1
        c[1] += f["coincide"]
    for d, (n, ok) in sorted(por_dia.items()):
        pct = 100 * ok / n if n else 0
        print(f"  {d}: {ok}/{n} coinciden con el nombre ({pct:.0f}%)")

    con_nombre = total - sin_esperado
    pct = 100 * aciertos / con_nombre if con_nombre else 0
    print(f"\n  Total fotos: {total} | con lectura del motor: {con_lectura} "
          f"({100 * con_lectura / total if total else 0:.0f}%)")
    print(f"  Con nombre de placa: {con_nombre} | coinciden: {aciertos} "
          f"({pct:.1f}%) | sin nombre (placa_*): {sin_esperado}")
    print(f"  Diferencias a revisar: {len(diferencias)}")
    for linea in diferencias:
        print(f"    {linea}")

    ruta_csv = Path(args.csv)
    with ruta_csv.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(filas_csv[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas_csv)
    print(f"\n  CSV: {ruta_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
