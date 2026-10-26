#!/usr/bin/env python3
"""Decodificador por diccionario de flota cerrada (edicion distance <= K).

Razon: la flota CYTV es un conjunto conocido y verificado. La salida del
CRNN tiene confusiones sistematicas de glyphs (O<->0, I<->1, E<->F, 5<->9);
si la lectura queda a distancia <= K de UNICA placa de la flota, se
remapea a esa flota. Prioridad: coincidencia exacta > distancia minima
unica > si hay empate se devuelve la lectura original (ambiguo = no tocar).

Fuentes de flota (solo verificacion humana, nada de filenames historicos
que eran ~60% errados):
- gt_placas.DIAGNOSTICOS (29 verificados visualmente)
- entrenamiento/etiquetas_reales.csv (81 transcritos a mano hoy)

Puro python, sin dependencias: corre en el venv de servicio y en el de
entrenamiento.
"""
from __future__ import annotations

import csv
import os
import sys
from functools import lru_cache
from pathlib import Path

APP = Path(__file__).resolve().parent           
PROYECTO = APP.parent                           
sys.path.insert(0, str(APP))

from gt_placas import DIAGNOSTICOS  

CSV_REALES = PROYECTO / "entrenamiento" / "etiquetas_reales.csv"


def canon(texto: str) -> str:
    return texto.upper().replace("-", "").strip()


@lru_cache(maxsize=1)
def flota() -> list[str]:
    """Placas unicas verificadas, en formato canónico con guion.

    CYTV_FLOTA_SIN_GT=1 deja afuera gt_placas.DIAGNOSTICOS: diccionario
    solo de entrenamiento, para medir el holdout sin usar las respuestas.
    """
    placas: dict[str, str] = {}
    if not os.environ.get("CYTV_FLOTA_SIN_GT"):
        for _dia, _arh, placa, _nota in DIAGNOSTICOS:
            c = canon(placa)
            placas.setdefault(c, placa.upper())
    if CSV_REALES.is_file():
        with CSV_REALES.open(encoding="utf-8") as f:
            for fila in csv.DictReader(f):
                c = canon(fila["placa"])
                placas.setdefault(c, fila["placa"].upper())
    return list(placas.values())


@lru_cache(maxsize=1)
def canonicos() -> frozenset[str]:
    """Formas canonicas (sin guion) del diccionario, para puertas rápidas."""
    return frozenset(canon(p) for p in flota())


# Pares confundibles visualmente en placa peruana: sustitucion barata.
_GRUPOS_02 = ("I1", "0OQD", "8B", "5S", "2Z", "7T")
_COSTOS = {}
for _g in _GRUPOS_02:
    for _i, _a in enumerate(_g):
        for _b in _g[_i + 1:]:
            _COSTOS[frozenset((_a, _b))] = 0.2
for _h in "HNMV":
    _COSTOS[frozenset(("W", _h))] = 0.4


def distancia(a: str, b: str) -> float:
    """Distancia de edicion ponderada (Levenshtein con costos).

    Sustituciones entre pares confundibles (I<->1, 0<->O/Q/D, 8<->B,
    5<->S, 2<->Z, 7<->T) cuestan 0.2; W<->H/N/M/V cuestan 0.4; el resto
    de sustituciones y las inserciones/borrados cuestan 1.0.
    """
    if a == b:
        return 0.0
    if not a or not b:
        return float(max(len(a), len(b)))
    fila = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        nueva = [i]
        for j, cb in enumerate(b, 1):
            if ca == cb:
                c = 0.0
            else:
                c = _COSTOS.get(frozenset((ca, cb)), 1.0)
            nueva.append(min(fila[j] + 1.0, nueva[j - 1] + 1.0,
                             fila[j - 1] + c))
        fila = nueva
    return fila[-1]


def corregir(lectura: str, conjunto: list[str], umbral: int) -> str:
    """Remapea lectura al único candidato de flota a distancia <= umbral.

    Devuelve la placa de la flota (con guion) si es exacta o si hay un
    único candidato dentro del umbral; si no, la lectura original.
    """
    if not lectura:
        return lectura
    objetivo = canon(lectura)
    exacta = [p for p in conjunto if canon(p) == objetivo]
    if exacta:
        return exacta[0]
    mejor, candidatos = umbral + 1, []
    for p in conjunto:
        d = distancia(objetivo, canon(p))
        if d < mejor:
            mejor, candidatos = d, [p]
        elif d == mejor:
            candidatos.append(p)
    if mejor <= umbral and len(candidatos) == 1:
        return candidatos[0]
    return lectura
