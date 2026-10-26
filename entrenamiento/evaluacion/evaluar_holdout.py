#!/usr/bin/env python3
"""Evaluacion del HOLDOUT (fotos nuevas, fuera de GT-29) en 4 modos.

Modos medidos por separado, nunca mezclados:
  1. CRNN solo       : greedy del CRNN sobre el crop del bbox, sin flota
                       ni snap. Comparacion por canon (sin guiones).
  2. Snap flota      : (1) + corregir(lectura, flota, K=2), la misma
                       receta usada en GT-29.
  3. Config completa : pipeline de produccion (detector ONNX +
                       leer_placa_6), identico a validar_diagnosticos
                       --detector modelo. Comparacion literal placa==real.
  4. Fuera de flota  : de las fotos cuya placa real NO esta en el
                       diccionario: falsos aceptados (salio una placa
                       conocida), SIN LECTURA y lecturas propias fuera
                       de flota (informativas; el tope de flota de
                       produccion las descarta).

REGLAS: SOLO LECTURA -- no importa worker ni SQLite; fuerza
LOG_SIN_LECTURA=0 antes de cualquier cosa. Salida por stdout.

Uso:
    python evaluar_holdout.py --detector modelo
    python evaluar_holdout.py --detector txt --verbose
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# La evaluacion nunca escribe logs de produccion (regla de la casa).
os.environ["LOG_SIN_LECTURA"] = "0"

import cv2

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parents[1] / "app"))

from ocr import leer_placa_6, SNAP_K
from decodificador_flota import flota, corregir, canon, canonicos
import reconocedor
from holdout.diagnosticos import DIAGNOSTICOS, RAIZ


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
                    help="agrega la lectura CRNN/snap de cada foto")
    ap.add_argument("--margen", type=float, default=0.35)
    ap.add_argument("--detector", choices=("txt", "modelo"), default="modelo",
                    help="modelo = ONNX placas.onnx (produccion); "
                         "txt = bbox guardados")
    args = ap.parse_args()

    if not DIAGNOSTICOS:
        print("HOLDOUT VACIO: 0 filas en holdout/diagnosticos.py")
        print("Suma fotos nuevas a holdout/fotos/<dia>/ y sus filas a "
              "holdout/diagnosticos.py (mismo formato que gt_placas.py).")
        return 0

    detector = None
    if args.detector == "modelo":
        from modelo import DetectorPlacas, RUTA_ONNX
        if not RUTA_ONNX.is_file():
            print(f"[ERROR] No existe {RUTA_ONNX}")
            return 1
        detector = DetectorPlacas(RUTA_ONNX)

    F = flota()
    CAN = canonicos()

    contadas = 0
    faltantes = 0
    ok_crnn = ok_snap = ok_cfg = 0
    vistas: dict[str, dict] = {}
    fallos_cfg: list[str] = []
    fuera_n = fa = sl = ep = okf = 0
    det_fuera: list[str] = []

    for dia, nombre, real, nota in DIAGNOSTICOS:
        ruta = RAIZ / dia / nombre
        img = cv2.imread(str(ruta)) if ruta.is_file() else None
        if img is None:
            print(f"[FALTA] {dia}/{nombre}")
            faltantes += 1
            continue
        contadas += 1

        if detector is not None:
            detecciones = detector.detectar(img)
            bboxes = [bb for (bb, _c) in
                      sorted(detecciones,
                             key=lambda par: par[0][2] * par[0][3],
                             reverse=True)]
        else:
            alto, ancho = img.shape[:2]
            bbox = _bbox_desde_txt(ruta.with_suffix(".txt"), ancho, alto)
            bboxes = [bbox] if bbox else []

        # 1. CRNN solo: primer bbox con texto no vacio.
        leido, conf_crnn = "", 0.0
        for bb in bboxes:
            crop = img[bb[1]:bb[1] + bb[3], bb[0]:bb[0] + bb[2]]
            leido, conf_crnn = reconocedor.leer_placa(crop)
            if leido:
                break

        # 2. Snap de flota sobre (1).
        snap = corregir(leido, F, SNAP_K) if leido else ""

        # 3. Config completa: mismo camino que validar_diagnosticos.
        placa, conf, intentos = "", 0.0, []
        for bb in bboxes:
            placa, conf, intentos = leer_placa_6(img, bb, args.margen)
            if placa:
                break

        rc = canon(real)
        e1 = bool(leido) and canon(leido) == rc
        e2 = bool(snap) and canon(snap) == rc
        e3 = placa == real
        ok_crnn += e1
        ok_snap += e2
        ok_cfg += e3

        v = vistas.setdefault(rc, {"real": real, "n": 0,
                                   "c": 0, "s": 0, "p": 0})
        v["n"] += 1
        v["c"] += e1
        v["s"] += e2
        v["p"] += e3

        # 4. Seccion fuera de flota.
        if rc not in CAN:
            fuera_n += 1
            if e3:
                okf += 1
            elif not placa:
                sl += 1
                det_fuera.append(f"{dia}/{nombre}: SIN LECTURA (real {real})")
            elif canon(placa) in CAN:
                fa += 1
                det_fuera.append(f"{dia}/{nombre}: FALSO ACEPTADO "
                                 f"motor={placa} (real {real})")
            else:
                ep += 1
                det_fuera.append(f"{dia}/{nombre}: fuera de flota "
                                 f"motor={placa} (real {real})")

        if e3:
            linea = f"  OK   {dia}/{nombre}: {placa}  ({nota})"
        else:
            linea = (f"  FALLO {dia}/{nombre}: real={real} motor="
                     f"{placa or 'SIN LECTURA'} conf={conf:.2f}  <- {nota}")
            fallos_cfg.append(f"{dia}/{nombre}: real={real} motor="
                              f"{placa or '(nada)'}")
        if args.verbose:
            linea += (f" | crnn={leido or '-'}({conf_crnn:.2f})"
                      f" snap={snap or '-'}")
        print(linea)

    def pct(a: int, b: int) -> float:
        return 100.0 * a / b if b else 0.0

    unic = len(vistas)
    u_c = sum(1 for v in vistas.values() if v["c"] == v["n"])
    u_s = sum(1 for v in vistas.values() if v["s"] == v["n"])
    u_p = sum(1 for v in vistas.values() if v["p"] == v["n"])
    a_c = sum(1 for v in vistas.values() if v["c"] > 0)
    a_s = sum(1 for v in vistas.values() if v["s"] > 0)
    a_p = sum(1 for v in vistas.values() if v["p"] > 0)

    print(f"\n=== HOLDOUT: {contadas} fotos, {unic} placas unicas "
          f"(fuera de flota: {fuera_n}) ===")
    print(f"  1. CRNN solo:                {ok_crnn}/{contadas} = "
          f"{pct(ok_crnn, contadas):.1f}%")
    print(f"  2. Snap flota (K={SNAP_K}):              {ok_snap}/{contadas} = "
          f"{pct(ok_snap, contadas):.1f}%")
    p3 = pct(ok_cfg, contadas)
    print(f"  3. Config completa:          {ok_cfg}/{contadas} = {p3:.1f}% "
          f"(meta >= 90% -> {'PASA' if p3 >= 90 else 'NO PASA'})")
    print("  Unicas: en todas sus fotos / al menos una foto")
    print(f"    CRNN solo:        {u_c}/{unic} / {a_c}/{unic}")
    print(f"    Snap flota:       {u_s}/{unic} / {a_s}/{unic}")
    print(f"    Config completa:  {u_p}/{unic} / {a_p}/{unic}")

    if fuera_n:
        print(f"\n=== Fuera de flota ({fuera_n} fotos) ===")
        print(f"  OK literal: {okf} | falsos aceptados: {fa} | "
              f"SIN LECTURA: {sl} | lectura propia fuera "
              f"(descartada por tope): {ep}")
        for d in det_fuera:
            print(f"    {d}")

    if faltantes:
        print(f"\n  Fotos faltantes en disco: {faltantes} (no cuentan)")
    if fallos_cfg:
        print(f"\n  Fallos config ({len(fallos_cfg)}):")
        for f in fallos_cfg:
            print(f"    {f}")
    return 0 if p3 >= 90 else 2


if __name__ == "__main__":
    raise SystemExit(main())
