#!/usr/bin/env python3
"""Eval rapida del reconocedor ONNX sobre GT + sanity del preprocesado.

Imprime la lectura de los primeros N crops GT y guarda el tensor de
entrada (160x32) como PNG para ver que ve el modelo exactamente.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent / "app"))

from generador_placas import CHARSET  
from entrenar import cargar_gt, _tensor_desde_crop  

RUTA_ONNX = BASE.parent / "modelo" / "reconocedor.onnx"


def decodificar(logits: np.ndarray) -> str:
    idx = logits.argmax(axis=-1).tolist()
    salida, previo = [], None
    for i in idx:
        if i != len(CHARSET) and i != previo:
            salida.append(CHARSET[i])
        previo = i
    return "".join(salida)


def main() -> int:
    import onnxruntime as ort
    sesion = ort.InferenceSession(str(RUTA_ONNX),
                                  providers=["CPUExecutionProvider"])
    casos = cargar_gt()
    carpeta = BASE / "inputs_gt"
    carpeta.mkdir(exist_ok=True)

    aciertos = 0
    for i, (img, real) in enumerate(casos):
        t = _tensor_desde_crop(img)                       
        out = sesion.run(None, {"placa": t[None].numpy()})[0][0]
        leido = decodificar(out)
        canon = lambda s: s.upper().replace("-", "")
        ok = canon(leido) == canon(real)
        aciertos += ok
        marca = "OK " if ok else "MAL "
        print(f"{marca} {real:9s} -> {leido or '(nada)':12s} "
              f"(img {img.size})")
        if i < 8:
            
            g = Image.fromarray((t[0].numpy() * 255).astype(np.uint8))
            g.resize((640, 128), Image.NEAREST).save(
                carpeta / f"{i:02d}_{real.replace('-', '')}.png")
    print(f"\nTOTAL ONNX/GT: {aciertos}/{len(casos)} = "
          f"{100 * aciertos / len(casos):.1f}%")
    print(f"inputs guardados: {carpeta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
