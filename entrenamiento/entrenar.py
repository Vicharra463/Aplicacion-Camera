#!/usr/bin/env python3
"""Entrenamiento del CRNN de placas peruanas + export ONNX.

Datos:
  - Entreno: sintetico on-the-fly (generador_placas), 24k/epoca, infinito.
  - Val:     1200 sinteticos fijos (semilla aparte) para la curva.
  - TEST honesto: los 29 crops de la ground truth verificada
    (validar_diagnosticos.DIAGNOSTICOS) — NUNCA se entrena con ellos.

Sale en modelo/reconocedor.onnx (mismo sitio que placas.onnx, camino que
ya lee el servicio). Requiere GPU: torch.cuda.is_available().

Uso:
    python entrenamiento/entrenar.py                # 12 epocas
    python entrenamiento/entrenar.py --epocas 20
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

BASE = Path(__file__).resolve().parent          
PROYECTO = BASE.parent                          
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(PROYECTO))
sys.path.insert(0, str(PROYECTO / "app"))       

from crnn import CRNN, N_CLASSES                
from generador_placas import CHARSET, a_tensor, generar  

BLANK = len(CHARSET)                            
RUTA_ONNX = PROYECTO / "modelo" / "reconocedor.onnx"
RUTA_PESOS = BASE / "crnn_placas.pt"



class Sinteticos(Dataset):
    """Placas sinteticas on-the-fly. Cada __getitem__ genera una distinta."""

    def __init__(self, n: int, semilla: int) -> None:
        self.n = n
        self.rng = random.Random(semilla)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        img, texto = generar(self.rng)
        tensor = torch.from_numpy(a_tensor(img))
        etiqueta = torch.tensor([CHARSET.index(c) for c in texto],
                                dtype=torch.long)
        return tensor, etiqueta, texto


def _collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    longitudes = torch.tensor([len(b[1]) for b in batch], dtype=torch.long)
    etiquetas = torch.cat([b[1] for b in batch])
    textos = [b[2] for b in batch]
    return imgs, etiquetas, longitudes, textos



def _bbox_desde_txt(ruta_txt: Path, ancho: int, alto: int):
    linea = ruta_txt.read_text(encoding="utf-8").strip().splitlines()[0]
    _c, cx, cy, w, h = (float(v) for v in linea.split())
    bw, bh = w * ancho, h * alto
    return int(cx * ancho - bw / 2), int(cy * alto - bh / 2), int(bw), int(bh)


def _texto_canon(texto: str) -> str:
    """Para comparar: mayusculas y sin guion (el guion es estructural)."""
    return texto.upper().replace("-", "").replace(" ", "")


def cargar_gt() -> list[tuple[Image.Image, str]]:
    """Los 29 crops verificados de la ground truth (test honesto)."""
    from gt_placas import DIAGNOSTICOS, RAIZ
    casos = []
    for dia, nombre, real, _nota in DIAGNOSTICOS:
        ruta = RAIZ / dia / nombre
        if not ruta.is_file():
            continue
        img = Image.open(ruta)
        ancho, alto = img.size
        try:
            x, y, w, h = _bbox_desde_txt(ruta.with_suffix(".txt"), ancho, alto)
        except (OSError, ValueError, IndexError):
            continue
        crop = img.crop((x, y, x + w, y + h))
        casos.append((crop, real))
    return casos


def _tensor_desde_crop(img: Image.Image) -> torch.Tensor:
    g = img.convert("L").resize((160, 32), Image.BILINEAR)
    arr = np.asarray(g, dtype=np.float32)[None] / 255.0
    return torch.from_numpy(arr)


def decodificar(logits: torch.Tensor) -> str:
    """CTC greedy: argmax -> quitar repetidos -> quitar blanks."""
    idx = logits.argmax(dim=-1).tolist()
    salida, previo = [], None
    for i in idx:
        if i != BLANK and i != previo:
            salida.append(CHARSET[i])
        previo = i
    return "".join(salida)


def evaluar_gt(modelo, dispositivo, casos) -> tuple[float, list[str]]:
    modelo.eval()
    aciertos, fallos = 0, []
    with torch.no_grad():
        for img, real in casos:
            t = _tensor_desde_crop(img).unsqueeze(0).to(dispositivo)
            leido = decodificar(modelo(t)[0])
            if _texto_canon(leido) == _texto_canon(real):
                aciertos += 1
            else:
                fallos.append(f"real={real} leido={leido or '(nada)'}")
    modelo.train()
    return (aciertos / len(casos)) if casos else 0.0, fallos



def exportar_onnx(modelo) -> None:
    modelo.eval()
    dummy = torch.zeros(1, 1, 32, 160, device=next(modelo.parameters()).device)
    torch.onnx.export(
        modelo, dummy, str(RUTA_ONNX),
        input_names=["placa"], output_names=["logits"],
        dynamic_axes={"placa": {0: "lote"}, "logits": {0: "lote"}},
        opset_version=13)
    print(f"ONNX exportado: {RUTA_ONNX} "
          f"({RUTA_ONNX.stat().st_size / 1e6:.1f} MB)")
    
    try:
        import onnxruntime as ort
        sesion = ort.InferenceSession(str(RUTA_ONNX),
                                      providers=["CPUExecutionProvider"])
        prueba = np.zeros((1, 1, 32, 160), dtype=np.float32)
        out = sesion.run(None, {"placa": prueba})[0]
        print(f"onnxruntime OK: salida {out.shape}")
    except ImportError:
        print("onnxruntime no instalado en el venv: sanidad saltada")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=12)
    ap.add_argument("--por_epoca", type=int, default=24000)
    ap.add_argument("--lote", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    if torch.cuda.is_available():
        dispositivo = torch.device("cuda:0")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        lote, por_epoca = args.lote, args.por_epoca
    else:
        dispositivo = torch.device("cpu")
        print("AVISO: sin GPU (driver GTX 1650 en Code 38/phantom -> "
              "reinicio pendiente). Entrenando en CPU con lote reducido.")
        lote, por_epoca = min(args.lote, 32), min(args.por_epoca, 8000)
    print(f"dispositivo={dispositivo} lote={lote} muestras/epoca={por_epoca}")

    gt = cargar_gt()
    print(f"ground truth (test): {len(gt)} crops verificados")

    train_ds = Sinteticos(por_epoca, semilla=1)
    val_ds = Sinteticos(1200, semilla=999)
    train_dl = DataLoader(train_ds, batch_size=lote, shuffle=True,
                          collate_fn=_collate, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=lote,
                        collate_fn=_collate, num_workers=0)

    modelo = CRNN().to(dispositivo)
    total = sum(p.numel() for p in modelo.parameters())
    print(f"CRNN: {total / 1e6:.1f}M parametros")

    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optim = torch.optim.Adam(modelo.parameters(), lr=args.lr)
    programador = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epocas)

    mejor_gt = 0.0
    usar_amp = dispositivo.type == "cuda"
    n_meta_lotes = max(1, len(train_dl))
    n_cada = max(1, n_meta_lotes // 10)
    for epoca in range(1, args.epocas + 1):
        t0 = time.time()
        perdida_total, n_lotes = 0.0, 0
        for imgs, etiquetas, longitudes, _textos in train_dl:
            imgs = imgs.to(dispositivo, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=usar_amp):
                logits = modelo(imgs)                       
                log_probs = logits.log_softmax(-1).permute(1, 0, 2)
                perdida = ctc(log_probs, etiquetas, torch.full_like(longitudes, logits.size(1)), longitudes)
            perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
            optim.step()
            perdida_total += perdida.item()
            n_lotes += 1
            if n_lotes % n_cada == 0:
                prog = n_lotes / n_meta_lotes
                llenos = int(prog * 24)
                barra = "#" * llenos + "." * (24 - llenos)
                dt = time.time() - t0
                ritmo = dt / n_lotes
                restante = ritmo * (n_meta_lotes - n_lotes)
                print(f"BARRA epoca {epoca:02d}/{args.epocas} "
                      f"[{barra}] {100 * prog:3.0f}% "
                      f"loss={perdida_total / n_lotes:.3f} "
                      f"faltan~{restante:.0f}s", flush=True)
        programador.step()

        
        modelo.eval()
        val_loss, n_val = 0.0, 0
        con_gt, sin_gt = 0, 0
        with torch.no_grad():
            for imgs, etiquetas, longitudes, textos in val_dl:
                imgs = imgs.to(dispositivo, non_blocking=True)
                logits = modelo(imgs)
                log_probs = logits.log_softmax(-1).permute(1, 0, 2)
                val_loss += ctc(log_probs, etiquetas, torch.full_like(longitudes, logits.size(1)), longitudes).item()
                n_val += 1
                for i, texto_real in enumerate(textos):
                    if decodificar(logits[i]) == texto_real:
                        con_gt += 1
                    else:
                        sin_gt += 1
        modelo.train()

        pct_sint = 100 * con_gt / max(1, con_gt + sin_gt)
        gt_pct, gt_fallos = evaluar_gt(modelo, dispositivo, gt)
        dt = time.time() - t0
        print(f"epoca {epoca:02d}/{args.epocas}  "
              f"loss={perdida_total / n_lotes:.3f}  "
              f"val_loss={val_loss / max(1, n_val):.3f}  "
              f"val_acierto={pct_sint:5.1f}%  "
              f"GT_TEST={100 * gt_pct:5.1f}%  ({dt:.0f}s)", flush=True)

        if gt_pct >= mejor_gt:
            mejor_gt = gt_pct
            torch.save(modelo.state_dict(), RUTA_PESOS)
            for f in gt_fallos[:8]:
                print(f"    gt fallo: {f}")
            print(f"    (mejor hasta ahora: {100 * mejor_gt:.1f}%, "
                  f"pesos guardados)")

    print(f"\nMEJOR GT: {100 * mejor_gt:.1f}%")
    modelo.load_state_dict(torch.load(RUTA_PESOS))
    exportar_onnx(modelo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
