#!/usr/bin/env python3
"""Exporta 'data dura' de descartadas\\<dia> para el retrain del CRNN.

Copia (NUNCA mueve) los descartes cuya senal sirve para ensenar al
lector: los motivos donde HUBO deteccion pero la lectura fallo. Deja
manifiesto.csv con dia, motivo, archivo y origen. Los originales
quedan intactos: diagnostico, replay y limpieza siguen igual.

Motivos excluidos a proposito:
- sin_placa: sin deteccion no hay anclaje de texto (sirve para entrenar
  el DETECTOR, que es otro modelo); se regenerara aparte si hace falta.
- duplicado_placa / duplicado: ya tienen lectura; no ensenan nada nuevo.

Las descartadas NO traen .txt con el bbox: el paso de retrain
regenera los bboxes con el detector sobre estas copias y un humano
confirma antes de mezclar con entrenamiento/etiquetas_reales.csv.

Uso:
  python exportar_data_dura.py --origen "C:\\...\\descartadas" [--dia 2026-10-19]
  python exportar_data_dura.py --origen "C:\\...\\descartadas" --todos
"""
from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

PROYECTO = Path(__file__).resolve().parent.parent
DESTINO_DEFECTO = PROYECTO / "entrenamiento" / "data_dura"

MOTIVOS_DUROS = (
    "texto_invalido",
    "placa_cortada",
    "sobreexpuesta",
    "ocluida",
    "baja_nitidez",
)

EXTENSIONES = (".jpg", ".jpeg", ".png", ".webp")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--origen", required=True, type=Path,
                    help="Carpeta raiz de descartadas (contiene <dia>/<motivo>/)")
    ap.add_argument("--dia", help="Solo un dia (YYYY-MM-DD); por defecto el ultimo")
    ap.add_argument("--todos", action="store_true",
                    help="Exportar todos los dias que haya en el origen")
    ap.add_argument("--destino", type=Path, default=DESTINO_DEFECTO)
    args = ap.parse_args()

    if not args.origen.is_dir():
        print(f"  No existe el origen: {args.origen}")
        return 1

    
    
    dias = []
    for d in args.origen.iterdir():
        if not d.is_dir():
            continue
        try:
            datetime.strptime(d.name, "%Y-%m-%d")
        except ValueError:
            continue
        dias.append(d)
    dias.sort(key=lambda d: d.name)
    if not dias:
        print(f"  Sin dias dentro de {args.origen}")
        return 1
    if args.todos:
        elegidos = dias
    else:
        nombre = args.dia or dias[-1].name
        elegidos = [d for d in dias if d.name == nombre]
        if not elegidos:
            print(f"  El dia {nombre} no esta en {args.origen}")
            return 1

    total = 0
    for dia_dir in elegidos:
        filas: list[tuple[str, str, str, str]] = []
        for motivo in MOTIVOS_DUROS:
            origen_motivo = dia_dir / motivo
            if not origen_motivo.is_dir():
                continue
            destino_motivo = args.destino / dia_dir.name / motivo
            destino_motivo.mkdir(parents=True, exist_ok=True)
            for archivo in sorted(origen_motivo.iterdir()):
                if not archivo.is_file() or archivo.suffix.lower() not in EXTENSIONES:
                    continue
                copia = destino_motivo / archivo.name
                if not copia.exists():
                    shutil.copy2(archivo, copia)
                filas.append((dia_dir.name, motivo, archivo.name,
                              str(archivo)))
                total += 1
        if filas:
            manifiesto = args.destino / dia_dir.name / "manifiesto.csv"
            existentes = set()
            if manifiesto.is_file():
                with manifiesto.open(encoding="utf-8") as f:
                    existentes = {(r["motivo"], r["archivo"])
                                  for r in csv.DictReader(f)}
            with manifiesto.open("a", newline="", encoding="utf-8") as f:
                escritor = csv.writer(f)
                if manifiesto.stat().st_size == 0:
                    escritor.writerow(("dia", "motivo", "archivo", "origen"))
                for dia, motivo, nombre, origen in filas:
                    if (motivo, nombre) not in existentes:
                        escritor.writerow((dia, motivo, nombre, origen))
            por_motivo: dict[str, int] = {}
            for _d, motivo, _n, _o in filas:
                por_motivo[motivo] = por_motivo.get(motivo, 0) + 1
            detalle = ", ".join(f"{m}={n}" for m, n in sorted(por_motivo.items()))
            print(f"  {dia_dir.name}: {len(filas)} archivo(s) -> {detalle}")
        else:
            print(f"  {dia_dir.name}: sin motivos duros")

    print(f"  Total: {total} archivo(s) en {args.destino}")
    print("  Originales intactos. El bbox se regenera con el detector "
          "en el paso de retrain (las descartadas no traen .txt).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
