#!/usr/bin/env python3
"""Fine-tune del CRNN con datos REALES etiquetados + sintetico (mix 50/50).

Arranca de los pesos de la corrida pura-sintetica (crnn_placas.pt) y
ajusta con los 87 crops reales etiquetados a mano (etiquetas_reales.csv,
transcripcion visual directa; los filenames historicos eran ~60% errados).

Datos train: 50% real (aug on-the-fly con jitter de bbox para simular
varianza del detector) + 50% sintetico (para no olvidar generalizacion).
Test honesto: los 29 GT verificados (NUNCA en train).

Uso:
    python entrenamiento/entrenar_real.py                 # 20 epocas
    python entrenamiento/entrenar_real.py --epocas 30
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

BASE = Path(__file__).resolve().parent
PROYECTO = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(PROYECTO))
sys.path.insert(0, str(PROYECTO / "app"))       

from crnn import CRNN, N_CLASSES                      
from generador_placas import CHARSET, a_tensor, generar  
from gt_placas import RAIZ                            
from entrenar import (BLANK, cargar_gt, decodificar,   
                      evaluar_gt, exportar_onnx, _collate)

CSV_REALES = BASE / "etiquetas_reales.csv"
RUTA_INICIO = BASE / "crnn_placas.pt"
RUTA_PESOS = BASE / "crnn_reales.pt"



class Reales(Dataset):
    """Crops reales con augmentation on-the-fly y jitter de bbox."""

    def __init__(self, n_por_epoca: int, semilla: int) -> None:
        self.filas: list[tuple[Path, str]] = []
        with CSV_REALES.open(encoding="utf-8") as f:
            for fila in csv.DictReader(f):
                ruta = RAIZ / fila["dia"] / fila["archivo"]
                if ruta.is_file():
                    self.filas.append((ruta, fila["placa"].upper()))
        if not self.filas:
            raise RuntimeError(f"no hay filas validas en {CSV_REALES}")
        self.n = n_por_epoca
        self.rng = random.Random(semilla)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        rng = random.Random(self.rng.randrange(2**31))
        ruta, texto = self.filas[rng.randrange(len(self.filas))]
        img = Image.open(ruta).convert("RGB")
        ancho, alto = img.size
        x, y, w, h = _bbox_txt(ruta, ancho, alto, rng)
        crop = img.crop((x, y, x + w, y + h))
        crop = _augmentar(crop, rng)
        tensor = torch.from_numpy(a_tensor(crop))
        etiqueta = torch.tensor([CHARSET.index(c) for c in texto],
                                dtype=torch.long)
        return tensor, etiqueta, texto


def _bbox_txt(ruta: Path, ancho: int, alto: int, rng: random.Random):
    """bbox del .txt con jitter aleatorio (simula varianza del detector)."""
    linea = ruta.with_suffix(".txt").read_text(encoding="utf-8")
    _c, cx, cy, w, h = (float(v) for v in linea.strip().splitlines()[0].split())
    bw, bh = w * ancho, h * alto
    
    ex0, ey0 = bw * rng.uniform(0, 0.22), bh * rng.uniform(0, 0.22)
    ex1, ey1 = bw * rng.uniform(0, 0.22), bh * rng.uniform(0, 0.22)
    x = int(cx * ancho - bw / 2 - ex0)
    y = int(cy * alto - bh / 2 - ey0)
    return x, y, int(bw + ex0 + ex1), int(bh + ey0 + ey1)


def _augmentar(img: Image.Image, rng: random.Random) -> Image.Image:
    img = img.rotate(rng.uniform(-4, 4), resample=Image.BILINEAR,
                     expand=False)
    arr = np.asarray(img).astype(np.float32)
    arr *= rng.uniform(0.85, 1.15)
    arr += rng.uniform(-12, 12)
    arr += np.random.default_rng(rng.randrange(2**31)).normal(
        0, rng.uniform(2, 8), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    if rng.random() < 0.6:                      
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(35, 85))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


class Mix(Dataset):
    """50% reales + 50% sinteticos on-the-fly por epoca."""

    def __init__(self, n: int, reales: Reales) -> None:
        self.reales = reales
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        if i % 2 == 0:
            return self.reales[i]
        img, texto = generar()                      
        tensor = torch.from_numpy(a_tensor(img))
        etiqueta = torch.tensor([CHARSET.index(c) for c in texto],
                                dtype=torch.long)
        return tensor, etiqueta, texto


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=20)
    ap.add_argument("--por_epoca", type=int, default=8000)
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--desde", type=str, default=None,
                    help="pesos iniciales (defecto: crnn_placas.pt)")
    args = ap.parse_args()

    dispositivo = torch.device("cuda:0" if torch.cuda.is_available()
                               else "cpu")
    print(f"dispositivo={dispositivo}")

    gt = cargar_gt()
    print(f"GT test: {len(gt)} crops")

    reales = Reales(args.por_epoca, semilla=11)
    print(f"reales etiquetados: {len(reales.filas)}")
    ds = Mix(args.por_epoca, reales)
    dl = DataLoader(ds, batch_size=args.lote, shuffle=True,
                    collate_fn=_collate, num_workers=0)

    modelo = CRNN().to(dispositivo)
    ruta_desde = Path(args.desde) if args.desde else RUTA_INICIO
    if ruta_desde.is_file():
        modelo.load_state_dict(torch.load(ruta_desde,
                                          map_location=dispositivo))
        print(f"arranca de {ruta_desde.name}")
    else:
        print("AVISO: sin pesos previos, desde cero")

    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optim = torch.optim.Adam(modelo.parameters(), lr=args.lr)
    programador = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epocas)

    mejor_gt, mejor_estado = 0.0, None
    for epoca in range(1, args.epocas + 1):
        t0 = time.time()
        modelo.train()
        perdida_total, n_lotes = 0.0, 0
        n_meta = max(1, len(dl))
        n_cada = max(1, n_meta // 10)
        for imgs, etiquetas, longitudes, _textos in dl:
            imgs = imgs.to(dispositivo)
            optim.zero_grad(set_to_none=True)
            logits = modelo(imgs)
            log_probs = logits.log_softmax(-1).permute(1, 0, 2)
            perdida = ctc(log_probs, etiquetas,
                          torch.full_like(longitudes, logits.size(1)),
                          longitudes)
            perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
            optim.step()
            perdida_total += perdida.item()
            n_lotes += 1
            if n_lotes % n_cada == 0:
                prog = n_lotes / n_meta
                barra = "#" * int(prog * 24) + "." * (24 - int(prog * 24))
                print(f"BARRA real epoca {epoca:02d}/{args.epocas} "
                      f"[{barra}] {100 * prog:3.0f}% "
                      f"loss={perdida_total / n_lotes:.3f}", flush=True)
        programador.step()

        gt_pct, fallos = evaluar_gt(modelo, dispositivo, gt)
        dt = time.time() - t0
        print(f"epoca {epoca:02d}/{args.epocas}  "
              f"loss={perdida_total / n_lotes:.3f}  "
              f"GT_TEST={100 * gt_pct:5.1f}%  ({dt:.0f}s)", flush=True)
        if gt_pct >= mejor_gt:
            mejor_gt = gt_pct
            mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
            torch.save(mejor_estado, RUTA_PESOS)
            for f in fallos[:10]:
                print(f"    gt fallo: {f}", flush=True)
            print(f"    (nuevo mejor: {100 * mejor_gt:.1f}%)", flush=True)

    print(f"\nMEJOR GT: {100 * mejor_gt:.1f}%")
    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
        exportar_onnx(modelo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
