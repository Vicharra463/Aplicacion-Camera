#!/usr/bin/env python3
r"""Entrenamiento continuo del CRNN: solo en el reposo, con compuerta GT.

Reglas:
  - espera a que cola_entrenamiento\ este SIN CAMBIOS durante --idle segundos
    (no ha entrado nada) y haya >= --min_muestras pendientes;
  - fine-tune corto desde crnn_reales.pt (o crnn_placas.pt) con mix
    50% cola real (aug) / 50% sintetico;
  - solo publica (pesos + ONNX) si el examen GT (29 crops verificados)
    no baja mas de 0.005 respecto al ultimo modelo publicado;
  - lo que entreno o descarta se marca en cola_entrenamiento\procesadas.csv;
  - lock por PID: nunca corren dos instancias a la vez.

Uso:
    python entrenamiento/entrenador_continuo.py
    python entrenamiento/entrenador_continuo.py --once
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

BASE = Path(__file__).resolve().parent
PROYECTO = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(PROYECTO))
sys.path.insert(0, str(PROYECTO / "app"))

COLA = PROYECTO / "cola_entrenamiento"
LOG = COLA / "entrenador.log"
LOCK = COLA / "entrenador.lock"
PROCESADAS = COLA / "procesadas.csv"
RUTA_PESOS = BASE / "crnn_reales.pt"
RUTA_INICIO = BASE / "crnn_placas.pt"
TOLERANCIA_GT = 0.005


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


def _importar_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as e:
        _log(f"torch NO disponible en este interpreter "
             f"({sys.executable}): {e}")
        _log("configura entrenador_python en config.ini con un python que "
             "tenga torch, o corre este script a mano con el tuyo")
        return None
    return torch, nn, DataLoader, Dataset


def _corrida(pendientes: list[tuple[Path, str]], gt, gt_base: float,
             args, modulo) -> float:
    torch, nn, DataLoader, Dataset = modulo
    from crnn import CRNN
    from generador_placas import CHARSET, a_tensor, generar
    from entrenar import (BLANK, _collate, evaluar_gt,
                          exportar_onnx)
    from entrenar_real import _augmentar, _bbox_txt

    dispositivo = torch.device("cuda:0" if torch.cuda.is_available()
                               else "cpu")
    modelo = CRNN().to(dispositivo)
    origen = RUTA_PESOS if RUTA_PESOS.is_file() else RUTA_INICIO
    if origen.is_file():
        modelo.load_state_dict(torch.load(origen, map_location=dispositivo))
        _log(f"arranca de {origen.name}")
    else:
        _log("AVISO: sin pesos base, se entrena desde cero")

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

    gt_nuevo, fallos = evaluar_gt(modelo, dispositivo, gt)
    _log(f"GT: {100 * gt_base:.1f}% -> {100 * gt_nuevo:.1f}% "
         f"(tolerancia {100 * TOLERANCIA_GT:.1f}pp)")
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


def main() -> int:
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

    modulo = _importar_torch()
    if modulo is None:
        return 2
    torch = modulo[0]

    if not _tomar_lock():
        _log("otra instancia ya esta corriendo: salgo")
        return 0

    try:
        from entrenar import cargar_gt, evaluar_gt
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
            from crnn import CRNN
            modelo_base = CRNN().to(dispositivo)
            modelo_base.load_state_dict(
                torch.load(ruta_base_eval, map_location=dispositivo))
            gt_base, _ = evaluar_gt(modelo_base, dispositivo, gt)
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
                    gt_base = _corrida(pend, gt, gt_base, args, modulo)
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
                gt_base = _corrida(pend, gt, gt_base, args, modulo)
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
