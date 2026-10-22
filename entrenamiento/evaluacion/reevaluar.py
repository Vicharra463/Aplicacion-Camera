#!/usr/bin/env python3
"""Reevalua TODAS las imagenes de TODAS las carpetas de dias.

Corre el motor de 6 intentos sobre cada imagen bajo
imagenes_procesadas\\placa\\<dia>\\ y compara contra la placa REAL:

  * Con etiqueta (gt_placas.DIAGNOSTICOS verificado visualmente, y
    entrenamiento\\etiquetas_reales.csv) -> OK / FALLO y % de acierto.
  * Sin etiqueta -> se muestra la lectura del motor SIN dar veredicto
    (sirve aunque la data de etiquetas este incompleta o se actualice).

Uso (normalmente via scripts\reevaluar.bat):
    python reevaluar.py
    python reevaluar.py --dia 2026-10-14
    python reevaluar.py --modo modelo      # redetecta con ONNX
    python reevaluar.py --raiz C:\\ruta\\placa --limite 5
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parents[1] / "app"))

from ocr import leer_placa_6  
from gt_placas import DIAGNOSTICOS  

RAIZ_DEFECTO = BASE.parents[1] / "imagenes_procesadas" / "placa"
CSV_ETIQUETAS = BASE.parent / "etiquetas_reales.csv"


def _bbox_desde_txt(ruta_txt: Path, ancho: int, alto: int):
    try:
        linea = ruta_txt.read_text(encoding="utf-8").strip().splitlines()[0]
        _c, cx, cy, w, h = linea.split()
        cx, cy, w, h = (float(v) for v in (cx, cy, w, h))
    except (OSError, ValueError, IndexError):
        return None
    bw, bh = w * ancho, h * alto
    return int(cx * ancho - bw / 2), int(cy * alto - bh / 2), int(bw), int(bh)


def _cargar_etiquetas() -> dict[tuple[str, str], str]:
    """(dia, archivo) en minusculas -> placa real.

    Primero etiquetas_reales.csv; despues DIAGNOSTICOS pisana lo que
    tenga (son los verificados visualmente, van arriba).
    """
    etiquetas: dict[tuple[str, str], str] = {}
    if CSV_ETIQUETAS.is_file():
        with CSV_ETIQUETAS.open(encoding="utf-8", newline="") as fh:
            for fila in csv.DictReader(fh):
                if fila.get("dia") and fila.get("archivo") and fila.get("placa"):
                    clave = (fila["dia"].strip().lower(),
                             fila["archivo"].strip().lower())
                    etiquetas[clave] = fila["placa"].strip()
    for dia, nombre, real, _nota in DIAGNOSTICOS:
        etiquetas[(dia.lower(), nombre.lower())] = real
    return etiquetas


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
    ap.add_argument("--raiz", default=str(RAIZ_DEFECTO),
                    help="carpeta con las carpetas de dias "
                         f"(defecto: {RAIZ_DEFECTO})")
    ap.add_argument("--dia", help="solo una fecha, p. ej. 2026-10-14")
    ap.add_argument("--modo", choices=("auto", "txt", "modelo"), default="auto",
                    help="auto = usa el .txt si existe, si no ONNX")
    ap.add_argument("--margen", type=float, default=0.35)
    ap.add_argument("--limite", type=int, default=0,
                    help="procesa solo las primeras N imagenes (prueba)")
    args = ap.parse_args()

    raiz = Path(args.raiz)
    if not raiz.is_dir():
        print(f"[ERROR] no existe la carpeta: {raiz}", file=sys.stderr)
        return 1

    etiquetas = _cargar_etiquetas()
    detector = None  

    dias = sorted(d for d in raiz.iterdir() if d.is_dir())
    if args.dia:
        dias = [d for d in dias if d.name == args.dia]
        if not dias:
            print(f"[ERROR] dia no encontrado: {args.dia}", file=sys.stderr)
            return 1

    print("=" * 60)
    print(" REEVALUACION GENERAL - todas las imagenes")
    print("=" * 60)
    print(f"Raiz: {raiz}")
    print(f"Etiquetas: {len(etiquetas)} cargadas "
          f"(csv + DIAGNOSTICOS verificados)")

    total = 0
    con_etiqueta = 0
    aciertos = 0
    sin_lectura = 0
    fallos: list[str] = []
    por_dia: dict[str, list[int]] = {}
    sin_etiqueta: list[str] = []

    for dia in dias:
        imagenes = sorted(list(dia.glob("*.jpg")) + list(dia.glob("*.png")))
        for img_ruta in imagenes:
            if args.limite and total >= args.limite:
                break
            total += 1

            clave = (dia.name.lower(), img_ruta.name.lower())
            real = etiquetas.get(clave)

            img = cv2.imread(str(img_ruta))
            if img is None:
                print(f"  FALTA {dia.name}/{img_ruta.name}: no se pudo leer")
                fallos.append(f"{dia.name}/{img_ruta.name}: IMAGEN CORRUPTA")
                continue

            alto, ancho = img.shape[:2]
            txt = img_ruta.with_suffix(".txt")
            if args.modo == "txt":
                bboxes = [b for b in [_bbox_desde_txt(txt, ancho, alto)] if b]
            elif args.modo == "modelo":
                if detector is None:
                    detector = _detector_onnx()
                if detector is None:
                    print("[ERROR] modo modelo requiere modelo\\placas.onnx",
                          file=sys.stderr)
                    return 1
                bboxes = _bbox_mayor_area(detector, img)
            else:  
                   
                b = _bbox_desde_txt(txt, ancho, alto) if txt.is_file() else None
                if b:
                    bboxes = [b]
                else:
                    if detector is None:
                        detector = _detector_onnx()
                    if detector is None:
                        bboxes = []
                    else:
                        bboxes = _bbox_mayor_area(detector, img)

            placa, conf, intentos = "", 0.0, []
            for bb in bboxes:
                placa, conf, intentos = leer_placa_6(img, bb, args.margen)
                if placa:
                    break
            if not placa:
                sin_lectura += 1

            if real is None:
                sin_etiqueta.append(f"{dia.name}/{img_ruta.name}: "
                                    f"{placa or 'SIN LECTURA'}")
                print(f"  [sin etiqueta] {dia.name}/{img_ruta.name}: "
                      f"{placa or 'SIN LECTURA'}")
                continue

            contador = por_dia.setdefault(dia.name, [0, 0])
            con_etiqueta += 1
            contador[0] += 1
            ok = placa == real
            if ok:
                aciertos += 1
                contador[1] += 1
                print(f"  OK   {dia.name}/{img_ruta.name}: {placa}")
            else:
                print(f"  FALLO {dia.name}/{img_ruta.name}: real={real} "
                      f"motor={placa or 'SIN LECTURA'} conf={conf:.2f}")
                fallos.append(f"{dia.name}/{img_ruta.name}: real={real} "
                              f"motor={placa or '(nada)'}")
        if args.limite and total >= args.limite:
            break

    print("\n=== RESUMEN por dia (solo con etiqueta) ===")
    for d, (n, ok) in sorted(por_dia.items()):
        print(f"  {d}: {ok}/{n} ({100 * ok / n if n else 0:.0f}%)")

    if con_etiqueta:
        pct = 100 * aciertos / con_etiqueta
        print(f"\n  TOTAL con etiqueta: {aciertos}/{con_etiqueta} = {pct:.1f}% "
              f"de acierto (meta >= 90%) -> "
              f"{'PASA' if pct >= 90 else 'NO PASA'}")
    else:
        pct = 0.0
        print("\n  TOTAL con etiqueta: 0 - no hay etiquetas para estas "
              "imagenes (solo lecturas, sin veredicto)")
    print(f"  Sin etiqueta (solo lectura, sin veredicto): {len(sin_etiqueta)}")
    print(f"  Imagenes procesadas: {total} | sin lectura alguna: {sin_lectura}")

    if fallos:
        print(f"\n  Fallos ({len(fallos)}):")
        for f in fallos:
            print(f"    {f}")

    if con_etiqueta == 0:
        return 0
    return 0 if pct >= 90 else 2


if __name__ == "__main__":
    raise SystemExit(main())
