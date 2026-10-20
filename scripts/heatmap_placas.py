#!/usr/bin/env python3
"""Heatmap de posiciones de placa (Fase 2): banda espacial tipica.

Agrega las cajas YOLO (.txt, coordenadas normalizadas) de la vista placa
y reporta la banda vertical/horizontal donde caen las placas, por dia y
global. Es la base para calibrar un prior espacial/ROI cuando una camara
se mueve o cambia de canal (medido en produccion: el detector ONNX ya ve
100% de las placas en movimiento; el prior sirve como desempate cuando
hay varios candidatos, no como filtro duro).

Uso:
    python scripts\\heatmap_placas.py                     # resumen en pantalla
    python scripts\\heatmap_placas.py --salida bandas.csv # tambien CSV
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

RUTA_RAIZ = Path(__file__).resolve().parents[1]


def leer_banda(raiz: Path) -> dict[str, list[tuple[float, float]]]:
    """{dia: [(cx%, cy%)]} desde los .txt de la vista placa."""
    por_dia: dict[str, list[tuple[float, float]]] = {}
    for txt in sorted(raiz.rglob("*.txt")):
        try:
            partes = txt.read_text(encoding="utf-8").strip().splitlines()[0]
            _clase, cx, cy, _w, _h = partes.split()[:5]
            por_dia.setdefault(txt.parent.name, []).append(
                (float(cx) * 100.0, float(cy) * 100.0))
        except (OSError, ValueError, IndexError):
            continue
    return por_dia


def percentil(valores: list[float], p: float) -> float:
    ordenados = sorted(valores)
    if not ordenados:
        return 0.0
    i = min(len(ordenados) - 1, int(round(p / 100.0 * (len(ordenados) - 1))))
    return ordenados[i]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raiz", default=str(RUTA_RAIZ
                                          / "imagenes_procesadas" / "placa"),
                    help="carpeta de la vista placa")
    ap.add_argument("--salida", help="ruta CSV opcional")
    args = ap.parse_args()

    raiz = Path(args.raiz)
    if not raiz.is_dir():
        print(f"No existe la carpeta: {raiz}")
        return 1

    por_dia = leer_banda(raiz)
    if not por_dia:
        print(f"Sin cajas .txt bajo {raiz}")
        return 1

    filas_csv: list[tuple] = []
    todos_cx: list[float] = []
    todos_cy: list[float] = []
    print(f"\nBanda espacial de placas (vista placa, {raiz}):\n")
    print(f"  {'DIA':<12} {'N':>4}  {'CX p10..p90':>16}  {'CY p10..p90':>16}")
    print("  " + "-" * 56)
    for dia, pts in sorted(por_dia.items()):
        cx = [p[0] for p in pts]
        cy = [p[1] for p in pts]
        todos_cx += cx
        todos_cy += cy
        rango_cx = f"{percentil(cx, 10):5.1f}..{percentil(cx, 90):5.1f}"
        rango_cy = f"{percentil(cy, 10):5.1f}..{percentil(cy, 90):5.1f}"
        print(f"  {dia:<12} {len(pts):>4}  {rango_cx:>16}  {rango_cy:>16}")
        filas_csv.append((dia, len(pts), percentil(cx, 10), percentil(cx, 50),
                          percentil(cx, 90), percentil(cy, 10),
                          percentil(cy, 50), percentil(cy, 90)))

    print("  " + "-" * 56)
    print(f"  {'GLOBAL':<12} {len(todos_cy):>4}  "
          f"{percentil(todos_cx, 10):5.1f}..{percentil(todos_cx, 90):5.1f}  "
          f"{percentil(todos_cy, 10):5.1f}..{percentil(todos_cy, 90):5.1f}")
    print(f"\n  Banda sugerida (global p10-p90): cx "
          f"{percentil(todos_cx, 10):.1f}%-{percentil(todos_cx, 90):.1f}%, "
          f"cy {percentil(todos_cy, 10):.1f}%-{percentil(todos_cy, 90):.1f}%")
    print("  (el detector ya ve 100% sin prior: use esto para calibrar "
          "desempates o documentar cambios de camara)\n")

    if args.salida:
        destino = Path(args.salida)
        with destino.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["dia", "n", "cx_p10", "cx_p50", "cx_p90",
                        "cy_p10", "cy_p50", "cy_p90"])
            w.writerows(filas_csv)
        print(f"  CSV: {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
