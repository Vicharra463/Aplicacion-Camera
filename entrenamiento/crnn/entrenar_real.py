#!/usr/bin/env python3
r"""Fine-tune del CRNN con datos REALES etiquetados + sintetico (mix 50/50).

Un solo archivo con los dos modos de entrenamiento con datos reales:

  one-shot (defecto)   Arranca de los pesos de la corrida pura-sintetica
                       (crnn_placas.pt) y ajusta con los crops reales
                       etiquetados a mano (etiquetas_reales.csv). Datos
                       train: 50% real (aug on-the-fly con jitter de bbox)
                       + 50% sintetico. Test honesto: los 29 GT verificados
                       (NUNCA en train).

  --continuo           Daemon que consume cola_entrenamiento\ (la alimenta
                       el worker): solo entrena en el reposo de la cola,
                       con compuerta GT (no publica si el examen baja mas
                       de 0.5pp) y lock por PID. Lo usa `cytv entrenador`.

Uso:
    python entrenamiento/crnn/entrenar_real.py                 # 20 epocas
    python entrenamiento/crnn/entrenar_real.py --epocas 30
    python entrenamiento/crnn/entrenar_real.py --continuo      # daemon
    python entrenamiento/crnn/entrenar_real.py --continuo --once
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import io
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
ENTRENAMIENTO = BASE.parent
PROYECTO = BASE.parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(PROYECTO))
sys.path.insert(0, str(PROYECTO / "app"))

CSV_REALES = ENTRENAMIENTO / "etiquetas_reales.csv"
RUTA_INICIO = ENTRENAMIENTO / "pesos" / "crnn_placas.pt"
RUTA_PESOS = ENTRENAMIENTO / "pesos" / "crnn_reales.pt"
COLA = PROYECTO / "cola_entrenamiento"
LOG = COLA / "entrenador.log"
LOCK = COLA / "entrenador.lock"
PROCESADAS = COLA / "procesadas.csv"
TOLERANCIA_GT = 0.005

_ERROR_DEPS = None
try:
    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image, ImageFilter
    from torch.utils.data import DataLoader, Dataset

    from crnn import CRNN
    from generador_placas import CHARSET, a_tensor, generar
    from gt_placas import RAIZ
    from entrenar import (BLANK, cargar_gt, decodificar, evaluar_gt,
                          exportar_onnx, _collate)
except ImportError as e:                       # pragma: sin cover
    _ERROR_DEPS = e


# ---------------------------------------------------------------- utilidades
# (no requieren torch: se importan siempre)

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


def _log(mensaje: str) -> None:
    linea = f"{datetime.now():%Y-%m-%d %H:%M:%S} {mensaje}"
    print(linea, flush=True)
    try:
        COLA.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(linea + "\n")
    except OSError:
        pass


def _pid_vivo(pid: int) -> bool:
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int,
                                    ctypes.c_uint32]
        k32.GetExitCodeProcess.argtypes = [ctypes.c_void_p,
                                           ctypes.POINTER(ctypes.c_ulong)]
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        manejador = k32.OpenProcess(0x1000, False, pid)
        if not manejador:
            return False
        codigo = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(manejador, ctypes.byref(codigo))
        k32.CloseHandle(manejador)
        return bool(ok) and codigo.value == 259
    except Exception:
        return False


def _tomar_lock() -> bool:
    COLA.mkdir(parents=True, exist_ok=True)
    if LOCK.is_file():
        try:
            pid = int(LOCK.read_text(encoding="utf-8").split()[0])
        except (OSError, ValueError, IndexError):
            pid = None
        if pid and _pid_vivo(pid):
            return False
        _log(f"lock de instancia anterior (PID {pid}) obsoleto: lo tomo")
    LOCK.write_text(str(os.getpid()), encoding="utf-8")
    return True


def _firmar_cola() -> str:
    partes = []
    for ruta in sorted(COLA.glob("*/*")):
        if not ruta.is_file():
            continue
        if ruta.name in ("entrenador.log", "entrenador.lock",
                         "procesadas.csv"):
            continue
        if ruta.suffix not in (".csv", ".jpg", ".txt"):
            continue
        try:
            st = ruta.stat()
            partes.append(f"{ruta.name}:{st.st_mtime_ns}:{st.st_size}")
        except OSError:
            continue
    return hashlib.md5("|".join(partes).encode()).hexdigest()


def _filas_procesadas() -> set[tuple[str, str]]:
    ya: set[tuple[str, str]] = set()
    if PROCESADAS.is_file() and PROCESADAS.stat().st_size:
        with PROCESADAS.open(encoding="utf-8") as f:
            for fila in csv.DictReader(f):
                ya.add((fila["dia"], fila["archivo"]))
    return ya


def _pendientes() -> list[tuple[Path, str]]:
    ya = _filas_procesadas()
    filas: list[tuple[Path, str]] = []
    for csv_dia in sorted(COLA.glob("*/etiquetas.csv")):
        try:
            with csv_dia.open(encoding="utf-8") as f:
                for fila in csv.DictReader(f):
                    clave = (fila["dia"], fila["archivo"])
                    if clave in ya:
                        continue
                    ruta = COLA / fila["dia"] / fila["archivo"]
                    if ruta.is_file() and fila["placa"].strip():
                        filas.append((ruta, fila["placa"].upper()))
        except OSError:
            continue
    return filas


def _consumir(filas: list[tuple[Path, str]]) -> None:
    ya = _filas_procesadas()
    nuevas = [(p.parent.name, p.name) for p, _t in filas
              if (p.parent.name, p.name) not in ya]
    if not nuevas:
        return
    COLA.mkdir(parents=True, exist_ok=True)
    vacia = (not PROCESADAS.is_file()) or PROCESADAS.stat().st_size == 0
    with PROCESADAS.open("a", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        if vacia:
            escritor.writerow(("dia", "archivo"))
        escritor.writerows(nuevas)


# ---------------------------------------------------------------- one-shot

def main() -> int:
    if "--continuo" in sys.argv[1:]:
        return main_continuo()
    if _ERROR_DEPS is not None:
        print(f"[ERROR] faltan deps de entrenamiento "
              f"({sys.executable}): {_ERROR_DEPS}")
        return 2

    class Reales(Dataset):
        """Crops reales con augmentation on-the-fly y jitter de bbox."""

        def __init__(self, n_por_epoca: int, semilla: int,
                     ruta_csv: Path | None = None) -> None:
            self.filas: list[tuple[Path, str]] = []
            csv_usado = ruta_csv or CSV_REALES
            with csv_usado.open(encoding="utf-8") as f:
                for fila in csv.DictReader(f):
                    ruta = RAIZ / fila["dia"] / fila["archivo"]
                    if ruta.is_file():
                        self.filas.append((ruta, fila["placa"].upper()))
            if not self.filas:
                raise RuntimeError(f"no hay filas validas en {csv_usado}")
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

    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=20)
    ap.add_argument("--por_epoca", type=int, default=8000)
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--csv", default=None,
                    help="CSV de reales (defecto: etiquetas_reales.csv)")
    ap.add_argument("--salida", default=None,
                    help="ruta .pt de salida; si se pone NO exporta ONNX")
    args = ap.parse_args()
    csv_ruta = Path(args.csv) if args.csv else CSV_REALES
    ruta_pesos = Path(args.salida) if args.salida else RUTA_PESOS

    dispositivo = torch.device("cuda:0" if torch.cuda.is_available()
                               else "cpu")
    print(f"dispositivo={dispositivo}")

    gt = cargar_gt()
    print(f"GT test: {len(gt)} crops")

    reales = Reales(args.por_epoca, semilla=11, ruta_csv=csv_ruta)
    print(f"reales etiquetados: {len(reales.filas)} (csv {csv_ruta.name})")
    ds = Mix(args.por_epoca, reales)
    dl = DataLoader(ds, batch_size=args.lote, shuffle=True,
                    collate_fn=_collate, num_workers=0)

    modelo = CRNN().to(dispositivo)
    if RUTA_INICIO.is_file():
        modelo.load_state_dict(torch.load(RUTA_INICIO,
                                          map_location=dispositivo))
        print(f"arranca de {RUTA_INICIO.name} (corrida sintetica)")
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

        gt_pct, gt_cer, fallos = evaluar_gt(modelo, dispositivo, gt)
        dt = time.time() - t0
        print(f"epoca {epoca:02d}/{args.epocas}  "
              f"loss={perdida_total / n_lotes:.3f}  "
              f"GT_TEST={100 * gt_pct:5.1f}%  CER={gt_cer:.3f}  ({dt:.0f}s)", flush=True)
        if gt_pct >= mejor_gt:
            mejor_gt = gt_pct
            mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
            torch.save(mejor_estado, ruta_pesos)
            for f in fallos[:10]:
                print(f"    gt fallo: {f}", flush=True)
            print(f"    (nuevo mejor: {100 * mejor_gt:.1f}%)", flush=True)

    print(f"\nMEJOR GT: {100 * mejor_gt:.1f}%")
    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
        if args.salida:
            print(f"pesos guardados en {ruta_pesos} (sin export ONNX)")
        else:
            exportar_onnx(modelo)
    return 0


# ---------------------------------------------------------------- continuo

def _corrida_continua(pendientes: list[tuple[Path, str]], gt, gt_base: float,
                      args) -> float:
    class Cola(Dataset):
        def __init__(self) -> None:
            self.rng = random.Random(time.time_ns() & 0x7FFFFFFF)

        def __len__(self) -> int:
            return args.por_epoca // 2

        def __getitem__(self, i):
            rng = random.Random(self.rng.randrange(2**31))
            ruta, texto = pendientes[rng.randrange(len(pendientes))]
            img = Image.open(ruta).convert("RGB")
            ancho, alto = img.size
            x, y, w, h = _bbox_txt(ruta, ancho, alto, rng)
            crop = img.crop((x, y, x + w, y + h))
            crop = _augmentar(crop, rng)
            tensor = torch.from_numpy(a_tensor(crop))
            etiqueta = torch.tensor([CHARSET.index(c) for c in texto],
                                    dtype=torch.long)
            return tensor, etiqueta, texto

    class Sinteticos(Dataset):
        def __init__(self) -> None:
            self.rng = random.Random((time.time_ns() >> 8) & 0x7FFFFFFF)

        def __len__(self) -> int:
            return args.por_epoca - args.por_epoca // 2

        def __getitem__(self, i):
            img, texto = generar(self.rng)
            tensor = torch.from_numpy(a_tensor(img))
            etiqueta = torch.tensor([CHARSET.index(c) for c in texto],
                                    dtype=torch.long)
            return tensor, etiqueta, texto

    class Mix(Dataset):
        def __init__(self) -> None:
            self.cola = Cola()
            self.sint = Sinteticos()

        def __len__(self) -> int:
            return args.por_epoca

        def __getitem__(self, i):
            if i % 2 == 0:
                return self.cola[i]
            return self.sint[i]

    dispositivo = torch.device("cuda:0" if torch.cuda.is_available()
                               else "cpu")
    modelo = CRNN().to(dispositivo)
    origen = RUTA_PESOS if RUTA_PESOS.is_file() else RUTA_INICIO
    if origen.is_file():
        modelo.load_state_dict(torch.load(origen, map_location=dispositivo))
        _log(f"arranca de {origen.name}")
    else:
        _log("AVISO: sin pesos base, se entrena desde cero")

    dl = DataLoader(Mix(), batch_size=args.lote, shuffle=True,
                    collate_fn=_collate, num_workers=0)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optim = torch.optim.Adam(modelo.parameters(), lr=args.lr)

    modelo.train()
    perdida_total, n_lotes = 0.0, 0
    for epoca in range(1, args.epocas + 1):
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
        _log(f"corrida epoca {epoca}/{args.epocas} "
             f"loss={perdida_total / max(1, n_lotes):.4f}")

    gt_nuevo, cer_nuevo, fallos = evaluar_gt(modelo, dispositivo, gt)
    _log(f"GT: {100 * gt_base:.1f}% -> {100 * gt_nuevo:.1f}% "
         f"CER={cer_nuevo:.3f} (tolerancia {100 * TOLERANCIA_GT:.1f}pp)")
    for f in fallos[:5]:
        _log(f"  gt fallo: {f}")

    if gt_nuevo >= gt_base - TOLERANCIA_GT:
        copia = RUTA_PESOS.with_suffix(".pt.bak")
        if RUTA_PESOS.is_file():
            shutil.copy2(RUTA_PESOS, copia)
        torch.save(modelo.state_dict(), RUTA_PESOS)
        try:
            exportar_onnx(modelo)
            _log("PUBLICADO: pesos + reconocedor.onnx actualizados")
        except Exception as e:
            _log(f"export ONNX fallo ({e}); restauro pesos anteriores")
            if copia.is_file():
                shutil.copy2(copia, RUTA_PESOS)
            return gt_base
        if copia.is_file():
            copia.unlink(missing_ok=True)
        return gt_nuevo
    _log("GT bajaria mas de la tolerancia: corrida DESCARTADA "
         "(pesos y ONNX sin tocar)")
    return gt_base


def main_continuo() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idle", type=float, default=600,
                    help="segundos sin cambios en la cola para entrenar")
    ap.add_argument("--min_muestras", type=int, default=5)
    ap.add_argument("--epocas", type=int, default=3)
    ap.add_argument("--por_epoca", type=int, default=3000)
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--poll", type=float, default=60)
    ap.add_argument("--once", action="store_true",
                    help="una corrida (omite la espera idle) y sale")
    args = ap.parse_args()

    if _ERROR_DEPS is not None:
        _log(f"torch NO disponible en este interpreter "
             f"({sys.executable}): {_ERROR_DEPS}")
        _log("configura entrenador_python en config.ini con un python que "
             "tenga torch, o corre este script a mano con el tuyo")
        return 2

    if not _tomar_lock():
        _log("otra instancia ya esta corriendo: salgo")
        return 0

    try:
        gt = cargar_gt()
        _log(f"GT verificado: {len(gt)} crops")
        dispositivo = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")
        if RUTA_PESOS.is_file():
            ruta_base_eval = RUTA_PESOS
        elif RUTA_INICIO.is_file():
            ruta_base_eval = RUTA_INICIO
        else:
            ruta_base_eval = None
        gt_base = 0.0
        if ruta_base_eval is not None:
            modelo_base = CRNN().to(dispositivo)
            modelo_base.load_state_dict(
                torch.load(ruta_base_eval, map_location=dispositivo))
            gt_base, _cer, _ = evaluar_gt(modelo_base, dispositivo, gt)
            del modelo_base
            _log(f"baseline GT de {ruta_base_eval.name}: "
                 f"{100 * gt_base:.1f}%")
        _log(f"dispositivo={dispositivo} idle={args.idle}s "
             f"min_muestras={args.min_muestras}")

        firma_prev = _firmar_cola()
        quiete_desde = time.time()
        while True:
            if LOCK.is_file():
                try:
                    LOCK.write_text(str(os.getpid()), encoding="utf-8")
                except OSError:
                    pass
            pend = _pendientes()
            firma = _firmar_cola()
            if firma != firma_prev:
                firma_prev = firma
                quiete_desde = time.time()
                _log("la cola cambio: reinicio la espera idle")
            if args.once:
                if pend and len(pend) >= args.min_muestras:
                    gt_base = _corrida_continua(pend, gt, gt_base, args)
                    _consumir(pend)
                else:
                    _log(f"pendientes {len(pend)} < min {args.min_muestras}")
                return 0
            en_reposo = (time.time() - quiete_desde) >= args.idle
            if not pend or len(pend) < args.min_muestras:
                _log(f"esperando: {len(pend)} pendientes "
                     f"(min {args.min_muestras})")
            elif not en_reposo:
                _log(f"esperando reposo: cola activa hace "
                     f"{time.time() - quiete_desde:.0f}s "
                     f"(idle {args.idle:.0f}s)")
            else:
                _log(f"reposo + {len(pend)} muestras: entreno")
                gt_base = _corrida_continua(pend, gt, gt_base, args)
                _consumir(pend)
                firma_prev = _firmar_cola()
                quiete_desde = time.time()
            time.sleep(args.poll)
    finally:
        try:
            if LOCK.is_file() and LOCK.read_text(encoding="utf-8").strip() \
                    == str(os.getpid()):
                LOCK.unlink(missing_ok=True)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
