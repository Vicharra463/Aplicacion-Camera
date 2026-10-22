#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
ENTRENAMIENTO = BASE.parent
PROYECTO = BASE.parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(PROYECTO))
sys.path.insert(0, str(PROYECTO / "app"))

from crnn import CRNN, N_CLASSES
from generador_placas import CHARSET, a_tensor, generar

BLANK = len(CHARSET)
RUTA_ONNX = PROYECTO / "modelo" / "reconocedor.onnx"
RUTA_PESOS = ENTRENAMIENTO / "pesos" / "crnn_placas.pt"
RUTA_DATASET = PROYECTO / "dataset"
CSV_LABELS = RUTA_DATASET / "labels.csv"

class Sinteticos(Dataset):
    def __init__(self, n: int, semilla: int) -> None:
        self.n = n
        self.rng = random.Random(semilla)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        img, texto = generar(self.rng)
        tensor = torch.from_numpy(a_tensor(img))
        etiqueta = torch.tensor([CHARSET.index(c) for c in texto], dtype=torch.long)
        return tensor, etiqueta, texto

class PlacasRealesDataset(Dataset):
    def __init__(self, ruta_csv: str | Path, ruta_imagenes: str | Path):
        self.ruta_imagenes = Path(ruta_imagenes)
        self.muestras = []
        with open(ruta_csv, encoding='utf-8') as f:
            lector = csv.DictReader(f)
            for fila in lector:
                # Solo agrega la muestra si la imagen existe físicamente en esta carpeta (train o val)
                if (self.ruta_imagenes / fila['archivo']).is_file():
                    self.muestras.append((fila['archivo'], fila['placa']))

    def __len__(self) -> int:
        return len(self.muestras)

    def __getitem__(self, i: int):
        archivo, texto = self.muestras[i]
        ruta = self.ruta_imagenes / archivo
        img = Image.open(ruta).convert("RGB")
        tensor = _tensor_desde_crop(img)[0]
        texto = _texto_canon(texto)
        etiqueta = torch.tensor([CHARSET.index(c) for c in texto if c in CHARSET], dtype=torch.long)
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
    return texto.upper().replace("-", "").replace(" ", "")

def cargar_gt() -> list[tuple[Image.Image, str]]:
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
    idx = logits.argmax(dim=-1).tolist()
    salida, previo = [], None
    for i in idx:
        if i != BLANK and i != previo:
            salida.append(CHARSET[i])
        previo = i
    return "".join(salida)

def _distancia(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                            prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _con_formato(leido: str) -> str:
    a = (leido or "").replace("-", "")
    if len(a) == 7 and a[3:].isdigit():
        return f"{a[:3]}-{a[3:6]}"
    return leido


def evaluar_gt(modelo, dispositivo, casos) -> tuple[float, float, list[str]]:
    if not casos:
        return 0.0, 0.0, []

    modelo.eval()
    aciertos, fallos = 0, []
    errores, total_chars = 0, 0
    with torch.no_grad():
        for img, real in casos:
            t = _tensor_desde_crop(img).unsqueeze(0).to(dispositivo)
            leido = _con_formato(decodificar(modelo(t)[0]))
            rc, lc = _texto_canon(real), _texto_canon(leido)
            errores += _distancia(lc, rc)
            total_chars += len(rc)
            if lc == rc:
                aciertos += 1
            else:
                fallos.append(f"real={real} leido={leido or '(nada)'}")
    modelo.train()
    cer = errores / max(1, total_chars)
    return aciertos / len(casos), cer, fallos

def exportar_onnx(modelo) -> None:
    modelo.eval()
    dummy = torch.zeros(1, 1, 32, 160, device=next(modelo.parameters()).device)
    torch.onnx.export(
        modelo, dummy, str(RUTA_ONNX),
        input_names=["placa"], output_names=["logits"],
        dynamic_axes={"placa": {0: "lote"}, "logits": {0: "lote"}},
        opset_version=13)
    print(f"ONNX exportado: {RUTA_ONNX} ({RUTA_ONNX.stat().st_size / 1e6:.1f} MB)")
    
    try:
        import onnxruntime as ort
        sesion = ort.InferenceSession(str(RUTA_ONNX), providers=["CPUExecutionProvider"])
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
        print("AVISO: sin GPU. Entrenando en CPU con lote reducido.")
        lote, por_epoca = min(args.lote, 32), min(args.por_epoca, 8000)
    print(f"dispositivo={dispositivo} lote={lote} muestras/epoca={por_epoca}")

    gt = cargar_gt()
    print(f"ground truth (test): {len(gt)} crops verificados")

    train_ds = PlacasRealesDataset(ruta_csv=CSV_LABELS, ruta_imagenes=RUTA_DATASET / "images" / "train")
    val_ds = PlacasRealesDataset(ruta_csv=CSV_LABELS, ruta_imagenes=RUTA_DATASET / "images" / "val")
    
    print(f"Dataset real cargado: {len(train_ds)} train, {len(val_ds)} val")

    train_dl = DataLoader(train_ds, batch_size=lote, shuffle=True, collate_fn=_collate, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=lote, collate_fn=_collate, num_workers=0)

    modelo = CRNN().to(dispositivo)
    total = sum(p.numel() for p in modelo.parameters())
    print(f"CRNN: {total / 1e6:.1f}M parametros")

    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optim = torch.optim.Adam(modelo.parameters(), lr=args.lr)
    programador = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epocas)

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
                print(f"BARRA epoca {epoca:02d}/{args.epocas} [{barra}] {100 * prog:3.0f}% loss={perdida_total / n_lotes:.3f} faltan~{restante:.0f}s", flush=True)
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
        gt_pct, gt_cer, gt_fallos = evaluar_gt(modelo, dispositivo, gt)
        dt = time.time() - t0
        print(f"epoca {epoca:02d}/{args.epocas} loss={perdida_total / n_lotes:.3f} val_loss={val_loss / max(1, n_val):.3f} val_acierto={pct_sint:5.1f}% GT_TEST={100 * gt_pct:5.1f}% CER={gt_cer:.3f} ({dt:.0f}s)", flush=True)

        if gt_pct >= mejor_gt:
            mejor_gt = gt_pct
            torch.save(modelo.state_dict(), RUTA_PESOS)
            for f in gt_fallos[:8]:
                print(f"    gt fallo: {f}")
            print(f"    (mejor hasta ahora: {100 * mejor_gt:.1f}%, pesos guardados)")

    print(f"\nMEJOR GT: {100 * mejor_gt:.1f}%")
    modelo.load_state_dict(torch.load(RUTA_PESOS))
    exportar_onnx(modelo)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())