# -*- coding: utf-8 -*-
"""Ground truth del HOLDOUT: fotos NUEVAS fuera de GT-29.

Formato identico a app/gt_placas.py: cada fila es
(dia, nombre_archivo, placa_real, nota_verificacion)
y las fotos viven en holdout/fotos/<dia>/<nombre_archivo>
(con <nombre_archivo>.txt YOLO opcional al lado).

REGLAS:
  * Solo fotos que NO se usaron en ninguna decision (ni entrenamiento,
    ni GT-29, ni tuning de margenes/K).
  * NO copiar filas de gt_placas.py ni inventar placas: la placa_real
    sale de verificacion visual de la foto.
  * No alterar GT-29 desde aqui; los dos diagnosticos son independientes.
  * evaluar_holdout.py fuerza LOG_SIN_LECTURA=0: esta evaluacion nunca
    escribe logs ni SQLite de produccion.

Corrida:  python evaluar_holdout.py --detector modelo
"""
from __future__ import annotations

from pathlib import Path

RAIZ = Path(__file__).resolve().parent / "fotos"

# (dia, nombre, placa_real, nota)  -- vacio hasta sumar fotos nuevas.
DIAGNOSTICOS: list[tuple[str, str, str, str]] = [
]
