#!/usr/bin/env python3
"""Devuelve a ftp_recibidos las fotos retenidas en descartadas\\<dia>\\<motivo>.

El worker las re-procesa solo (dedup, filtros y OCR actualizados): asi una
placa que antes no se leia puede guardarse tras un cambio de modelo o de
flota, sin perder la evidencia.

Uso (en el servidor, con el venv del servicio):
    python scripts\\replay_descartes.py --dry-run            # solo lista
    python scripts\\replay_descartes.py --fecha 2026-10-18   # un dia
    python scripts\\replay_descartes.py --motivo texto_invalido
    python scripts\\replay_descartes.py --fecha 2026-10-18   # mueve de verdad

Mueve (no copia): lo que ya fue reintentado no debe reintentarse otra vez.
El dedup del worker (texto + pHash) evita re-guardar placas
conocidas.
"""
from __future__ import annotations

import argparse
import configparser
import shutil
import sys
from pathlib import Path

RUTA_RAIZ = Path(__file__).resolve().parents[1]
RUTA_CONFIG = RUTA_RAIZ / "config.ini"


def leer_config() -> tuple[Path, Path]:
    """(carpeta_descartes, carpeta_recibidos) desde config.ini."""
    p = configparser.ConfigParser(inline_comment_prefixes=(";",))
    p.read(RUTA_CONFIG, encoding="utf-8-sig")
    recibidos = Path(p.get("rutas", "recibidos",
                           fallback=str(RUTA_RAIZ / "ftp_recibidos")))
    descartes = Path(p.get("filtros", "descartes",
                           fallback=str(recibidos.parent / "descartadas")))
    return descartes, recibidos


def destino_libre(carpeta: Path, nombre: str) -> Path:
    destino = carpeta / nombre
    if not destino.exists():
        return destino
    raiz, ext = Path(nombre).stem, Path(nombre).suffix
    for i in range(1, 1000):
        intento = carpeta / f"{raiz}_{i}{ext}"
        if not intento.exists():
            return intento
    raise RuntimeError(f"no hay nombre libre para {nombre}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fecha", help="YYYY-MM-DD (default: todos los dias)")
    ap.add_argument("--motivo", help="subcarpeta de motivo (default: todas)")
    ap.add_argument("--dry-run", action="store_true",
                    help="solo listar, no mover nada")
    args = ap.parse_args()

    descartes, recibidos = leer_config()
    if not descartes.is_dir():
        print(f"No existe la carpeta de descartes: {descartes}")
        return 1
    recibidos.mkdir(parents=True, exist_ok=True)

    dias = sorted(d for d in descartes.iterdir()
                  if d.is_dir() and (args.fecha is None
                                     or d.name == args.fecha))
    if args.fecha and not dias:
        print(f"Sin carpeta para la fecha {args.fecha} en {descartes}")
        return 1

    movidas, total = 0, 0
    for dia in dias:
        motivos = sorted(m for m in dia.iterdir()
                         if m.is_dir() and (args.motivo is None
                                            or m.name.lower()
                                            == args.motivo.lower()))
        for motivo in motivos:
            for foto in sorted(motivo.iterdir()):
                if foto.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                total += 1
                if args.dry_run:
                    print(f"  [dry-run] {dia.name}\\{motivo.name}\\{foto.name}")
                    continue
                destino = destino_libre(recibidos, foto.name)
                try:
                    shutil.move(str(foto), str(destino))
                    movidas += 1
                    print(f"  {dia.name}\\{motivo.name}\\{foto.name} "
                          f"-> {destino}")
                except OSError as e:
                    print(f"  ERROR {foto.name}: {e}", file=sys.stderr)

    accion = "se moverian" if args.dry_run else "movidas"
    print(f"\nListo: {total} foto(s) {accion} de {descartes} "
          f"a {recibidos} ({movidas if not args.dry_run else total} "
          f"{'ok' if not args.dry_run else 'listadas'}).")
    if not args.dry_run and movidas:
        print("El worker las re-procesara solo; mire el log y 'historial'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
