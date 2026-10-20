#!/usr/bin/env python3
"""Diagnostica el domain gap: stats de crops GT reales vs muestras sinteticas."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent / "app"))

from generador_placas import generar  
from entrenar import cargar_gt  


def stats(img: Image.Image) -> tuple[float, float, float]:
    """(aspecto, media, desvio) de la imagen en grises a 160x32."""
    g = img.convert("L").resize((160, 32), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    return img.width / img.height, float(a.mean()), float(a.std())


def main() -> None:
    rng = random.Random(7)
    casos = cargar_gt()

    reales = [stats(im) for im, _ in casos]
    sintes = [stats(generar(rng)[0]) for _ in range(40)]

    def resumen(nombre, datos):
        arr = np.array(datos)
        print(f"{nombre:10s} aspecto={arr[:,0].mean():.2f} "
              f"(min {arr[:,0].min():.2f} max {arr[:,0].max():.2f})  "
              f"media_gris={arr[:,1].mean():5.1f}  desvio={arr[:,2].mean():5.1f}")

    resumen("REAL GT", reales)
    resumen("SINTETICO", sintes)

    
    raros = [(im.size, im.width / im.height, t)
             for (im, t) in casos if not (3.0 <= im.width / im.height <= 6.0)]
    print(f"\ncrops con aspecto fuera de 3-6 (placa llena el marco?): "
          f"{len(raros)}/{len(casos)}")
    for tam, asp, t in raros:
        print(f"  {t}: {tam} aspecto={asp:.2f}")

    
    tiles = [im.resize((320, 64), Image.BILINEAR) for im, _ in casos[:6]]
    tiles += [generar(rng)[0].resize((320, 64), Image.BILINEAR)
              for _ in range(6)]
    hoja = Image.new("L", (320 * 3, 64 * 4), 128)
    for i, t in enumerate(tiles):
        hoja.paste(t.convert("L"), ((i % 3) * 320, (i // 3) * 64))
    ruta = BASE / "diagnostico_gap.png"
    hoja.save(ruta)
    print(f"\nhoja guardada (arriba 2 filas=REALES, abajo 2=SINTETICOS): "
          f"{ruta}")


if __name__ == "__main__":
    main()
