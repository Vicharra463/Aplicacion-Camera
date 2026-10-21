#!/usr/bin/env python3
"""Genera montajes de crops reales (NO-GT) para etiquetado visual.

9 crops por hoja, cada tile con su nombre de archivo dibujado arriba.
El agente (o una persona) lee la hoja y transcribe la placa visible.
Salida: entrenamiento/montajes/hora_XX.png
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parents[1] / "app"))

from gt_placas import DIAGNOSTICOS, RAIZ  

TILE_W, TILE_H, BANDA = 400, 160, 28
POR_HOJA = 9


def _bbox(ruta_txt: Path, ancho: int, alto: int):
    try:
        linea = ruta_txt.read_text(encoding="utf-8").strip().splitlines()[0]
        _c, cx, cy, w, h = (float(v) for v in linea.split())
    except (OSError, ValueError, IndexError):
        return None
    bw, bh = w * ancho, h * alto
    return int(cx * ancho - bw / 2), int(cy * alto - bh / 2), int(bw), int(bh)


def main() -> int:
    gt = {(d, n) for d, n, _r, _nota in DIAGNOSTICOS}
    casos = []
    for dia_dir in sorted(RAIZ.iterdir()):
        if not dia_dir.is_dir():
            continue
        for jpg in sorted(dia_dir.glob("*.jpg")):
            if (dia_dir.name, jpg.name) in gt:
                continue
            casos.append((dia_dir.name, jpg))

    salida = BASE.parent / "montajes"
    salida.mkdir(exist_ok=True)
    fuente = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 20)

    hoja_n = 0
    tiles = []
    for dia, jpg in casos:
        img = Image.open(jpg)
        ancho, alto = img.size
        bb = _bbox(jpg.with_suffix(".txt"), ancho, alto)
        if not bb:
            print(f"[SIN TXT] {dia}/{jpg.name}")
            continue
        x, y, w, h = bb
        crop = img.crop((x, y, x + w, y + h)).convert("RGB")
        crop = crop.resize((TILE_W, TILE_H), Image.BILINEAR)
        tile = Image.new("RGB", (TILE_W, TILE_H + BANDA), (20, 20, 20))
        tile.paste(crop, (0, BANDA))
        d = ImageDraw.Draw(tile)
        d.text((6, 3), f"{dia}  {jpg.name}", font=fuente, fill=(255, 220, 60))
        tiles.append(tile)
        if len(tiles) == POR_HOJA:
            _guardar(salida, hoja_n, tiles)
            hoja_n += 1
            tiles = []
    if tiles:
        _guardar(salida, hoja_n, tiles)

    total = hoja_n + (1 if tiles else 0)
    print(f"hojas: {total}  crops: {len(casos)}  -> {salida}")
    return 0


def _guardar(salida: Path, n: int, tiles: list[Image.Image]) -> None:
    w, h = tiles[0].size
    hoja = Image.new("RGB", (w * 3, h * 3), (0, 0, 0))
    for i, t in enumerate(tiles):
        hoja.paste(t, ((i % 3) * w, (i // 3) * h))
    hoja.save(salida / f"hoja_{n:02d}.png")


if __name__ == "__main__":
    raise SystemExit(main())
