#!/usr/bin/env python3
"""Experimento: CRNN ONNX + decodificador de flota sobre los 29 GT.

Compara 3 escenarios sin tocar nada de produccion:
  crudo : greedy CTC del modelo (41.4% actual)
  umbral 1: remapeo unico a distancia-1 de flota
  umbral 2: remapeo unico a distancia-2 de flota

Uso: python entrenamiento/eval_flota.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent / "app"))

from decodificador_flota import canon, corregir, flota  # noqa: E402
from eval_rapido import RUTA_ONNX, decodificar  # noqa: E402
from entrenar import cargar_gt, _tensor_desde_crop  # noqa: E402


def main() -> int:
    import onnxruntime as ort

    conjunto = flota()
    print(f"flota: {len(conjunto)} placas verificadas")

    sesion = ort.InferenceSession(str(RUTA_ONNX),
                                  providers=["CPUExecutionProvider"])
    casos = cargar_gt()

    n_crudo = n_d1 = n_d2 = 0
    for img, real in casos:
        t = _tensor_desde_crop(img)
        out = sesion.run(None, {"placa": t[None].numpy()})[0][0]
        leido = decodificar(out)
        d1 = corregir(leido, conjunto, 1)
        d2 = corregir(leido, conjunto, 2)
        ok_c = canon(leido) == canon(real)
        ok_1 = canon(d1) == canon(real)
        ok_2 = canon(d2) == canon(real)
        n_crudo += ok_c
        n_d1 += ok_1
        n_d2 += ok_2
        marca = "OK " if ok_2 else "MAL "
        extra = ""
        if d1 != leido:
            extra += f" d1={d1}"
        if d2 != leido:
            extra += f" d2={d2}"
        print(f"{marca} {real:9s} crudo={leido or '(nada)':12s}{extra}")

    n = len(casos)
    print(f"\ncrudo    : {n_crudo:2d}/{n} = {100 * n_crudo / n:.1f}%")
    print(f"umbral 1 : {n_d1:2d}/{n} = {100 * n_d1 / n:.1f}%")
    print(f"umbral 2 : {n_d2:2d}/{n} = {100 * n_d2 / n:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
