#!/usr/bin/env python3
"""Migra las pasadas historicas guardadas con sufijo EN EL NOMBRE.

Layout viejo (pre-2026-11-10): la pasada vivia en el nombre del archivo
dentro de la carpeta de la placa, todo mezclado:

    procesadas/<vista>/<dia>/<PLACA>/
        C8K-898_api-Roboflow_roi-evaluado.jpg            <- 1a pasada
        C8K-898_api-Roboflow_roi-evaluado_2da-pasada.jpg  <- 2a
        C8K-898_3ra-pasada.jpg, C8K-898_EXCEDIDA.jpg...

Layout nuevo (FASE 2): la pasada vive en su subcarpeta y el nombre del
archivo queda limpio (igual que la 1a pasada):

    procesadas/<vista>/<dia>/<PLACA>/
        C8K-898_api-Roboflow_roi-evaluado.jpg     <- 1a pasada (raiz)
        segunda/...  tercera/...  excedida/...

Este script mueve los archivos viejos a su subcarpeta renombrados sin el
sufijo (acompanantes .txt/_crop/_original van juntos; colisiones de
nombre se resuelven con _1, _2...; `(DUDOSA)` se conserva).

Uso (dry-run por defecto: solo imprime el plan):

    python scripts/migrar_pasadas_carpetas.py D:\\aplicacion-camera\\procesadas
    python scripts/migrar_pasadas_carpetas.py <procesadas> --aplicar

Idempotente: una vez migrados, los archivos no vuelven a matchear y una
segunda corrida no hace nada. NO se corre en prod sin pedido de FELIPE.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

# sufijo de pasada en el nombre + colision _N de _destino_libre (puede
# estar antes o despues de "(DUDOSA)") + cola _crop/_original.
_RE_PASADA = re.compile(
    r"^(?P<base>.+?)"
    r"(?P<suf>_2da-pasada|_3ra-pasada|_EXCEDIDA)"
    r"(?:_(?P<col1>\d+))?"
    r"(?P<dud> \(DUDOSA\))?"
    r"(?:_(?P<col2>\d+))?"
    r"(?P<tail>_crop|_original)?"
    r"\.(?P<ext>jpg|jpeg|png|txt)$")

CARPETAS = {"_2da-pasada": "segunda", "_3ra-pasada": "tercera",
            "_EXCEDIDA": "excedida"}


def _libre(donde: Path, nombre: str) -> Path:
    """Primer nombre disponible en `donde` (nunca pisa lo existente)."""
    destino = donde / nombre
    if not destino.exists():
        return destino
    p = Path(nombre)
    i = 1
    while True:
        alt = donde / f"{p.stem}_{i}{p.suffix}"
        if not alt.exists():
            return alt
        i += 1


def migrar(raiz: Path, aplicar: bool) -> list[tuple[Path, Path]]:
    """Devuelve [(origen, destino)] de archivos de pasadas con sufijo.

    Recorre <raiz>/<vista>/<dia>/<PLACA>/ y solo mira archivos de NIVEL
    RAIZ de la carpeta de placa (los que ya estan en una subcarpeta de
    pasada no se tocan). Dry-run con aplicar=False: no escribe ni crea
    carpetas. Idempotente: al renombrar sin sufijo, en la proxima
    corrida los archivos ya migrados no vuelven a matchear.
    """
    pares: list[tuple[Path, Path]] = []
    if not raiz.is_dir():
        return pares
    for vista_dir in sorted(p for p in raiz.iterdir() if p.is_dir()):
        for dia_dir in sorted(p for p in vista_dir.iterdir() if p.is_dir()):
            for placa_dir in sorted(p for p in dia_dir.iterdir() if p.is_dir()):
                for archivo in sorted(placa_dir.iterdir()):
                    if not archivo.is_file():
                        continue
                    m = _RE_PASADA.match(archivo.name)
                    if not m:
                        continue
                    destino_dir = placa_dir / CARPETAS[m["suf"]]
                    nuevo = (f"{m['base']}{m['dud'] or ''}"
                             f"{m['tail'] or ''}.{m['ext']}")
                    destino = _libre(destino_dir, nuevo)
                    if aplicar:
                        destino_dir.mkdir(parents=True, exist_ok=True)
                        archivo.rename(destino)
                    pares.append((archivo, destino))
    return pares


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migra pasadas historicas con sufijo en el nombre a "
                    "segunda/tercera/excedida (dry-run por defecto).")
    parser.add_argument("raiz", type=Path,
                        help="carpeta procesadas/ (contiene las vistas)")
    parser.add_argument("--aplicar", action="store_true",
                        help="ejecuta los movimientos (sin flag solo imprime)")
    args = parser.parse_args(argv)
    pares = migrar(args.raiz, aplicar=args.aplicar)
    if not pares:
        print("Nada que migrar.")
        return 0
    verbo = "MOVIDO" if args.aplicar else "MOVERIA"
    for origen, destino in pares:
        print(f"{verbo}: {origen} -> {destino}")
    estado = ("migrados." if args.aplicar
              else "pendientes (dry-run: usa --aplicar para ejecutar).")
    print(f"{len(pares)} archivo(s) {estado}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
