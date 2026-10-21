from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger("cytv.reconocedor")

BASE = Path(__file__).resolve().parents[1]
RUTA_ONNX = BASE / "modelo" / "reconocedor.onnx"

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
_BLANK = len(CHARSET)
_IDX_GUION = CHARSET.index("-")
_IDX_DIG = [i for i, c in enumerate(CHARSET) if c.isdigit()]

_sesion = None
_sesion_mtime: float | None = None


def _obtener_sesion():
    global _sesion, _sesion_mtime
    if not RUTA_ONNX.is_file():
        raise FileNotFoundError(f"modelo no encontrado: {RUTA_ONNX}")
    try:
        mtime = RUTA_ONNX.stat().st_mtime
    except OSError:
        if _sesion is not None:
            return _sesion
        raise
    if _sesion is not None and mtime == _sesion_mtime:
        return _sesion
    import onnxruntime as ort
    sesion_op = ort.SessionOptions()
    sesion_op.intra_op_num_threads = 1
    sesion_op.inter_op_num_threads = 1
    sesion_op.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sesion_op.enable_cpu_mem_arena = False
    sesion_op.log_severity_level = 4
    try:
        nueva = ort.InferenceSession(str(RUTA_ONNX),
                                     sess_options=sesion_op,
                                     providers=["CPUExecutionProvider"])
    except Exception:
        if _sesion is not None:
            logger.warning("El ONNX nuevo no se pudo abrir; sigo con los "
                           "pesos anteriores", exc_info=True)
            return _sesion
        raise
    if _sesion is None:
        logger.info("CRNN cargado (%s)", RUTA_ONNX.name)
    else:
        logger.info("CRNN recargado con pesos nuevos (%s)", RUTA_ONNX.name)
    _sesion = nueva
    _sesion_mtime = mtime
    return _sesion


def _tensor_desde_bgr(crop: np.ndarray) -> np.ndarray:
    if crop.ndim == 3 and crop.shape[2] == 3:
        rgb = np.ascontiguousarray(crop[:, :, ::-1])
    else:
        rgb = np.ascontiguousarray(crop)
    g = Image.fromarray(rgb).convert("L").resize((160, 32), Image.BILINEAR)
    arr = np.asarray(g, dtype=np.float32)[None] / 255.0
    return arr[None]


def _decodificar(logits: np.ndarray) -> str:
    idx = logits.argmax(axis=-1).tolist()
    salida, previo = [], None
    for i in idx:
        if i != _BLANK and i != previo:
            salida.append(CHARSET[i])
        previo = i
    return "".join(salida)


def leer_placa(crop: np.ndarray) -> tuple[str, float]:
    if crop is None or getattr(crop, "size", 0) == 0:
        return "", 0.0
    try:
        sesion = _obtener_sesion()
    except Exception:
        logger.debug("CRNN no disponible (fallo de carga)", exc_info=True)
        return "", 0.0
    try:
        logits = sesion.run(None, {"placa": _tensor_desde_bgr(crop)})[0][0]
        leido = _decodificar(logits)
        if not leido:
            return "", 0.0
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        conf = float((probs / probs.sum(axis=-1, keepdims=True))
                     .max(axis=-1).mean())
        return leido, conf
    except Exception:
        logger.exception("Fallo el reconocedor CRNN")
        return "", 0.0


def _probs(crop: np.ndarray) -> np.ndarray:
    sesion = _obtener_sesion()
    logits = sesion.run(None, {"placa": _tensor_desde_bgr(crop)})[0][0]
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _por_caracter(probs: np.ndarray) -> list:
    idx = probs.argmax(axis=-1)
    salida, previo, tramo = [], None, []
    for t, i in enumerate(idx):
        if i != previo:
            if previo is not None and previo != _BLANK:
                salida.append((previo, np.mean(tramo, axis=0)))
            tramo = []
        tramo.append(probs[t])
        previo = i
    if previo is not None and previo != _BLANK:
        salida.append((previo, np.mean(tramo, axis=0)))
    return [(CHARSET[i], v) for i, v in salida if CHARSET[i] != "-"]


def leer_consenso(crops: list) -> dict | None:
    lecturas = []
    for c in crops:
        if c is None or getattr(c, "size", 0) == 0:
            continue
        try:
            lectura = _por_caracter(_probs(c))
        except Exception:
            logger.debug("CRNN: una variante fallo", exc_info=True)
            continue
        if 5 <= len(lectura) <= 7:
            lecturas.append(lectura)
    if not lecturas:
        return None
    por_largo: dict = {}
    for lectura in lecturas:
        por_largo.setdefault(len(lectura), []).append(lectura)
    n, grupo = max(por_largo.items(), key=lambda kv: (len(kv[1]), kv[0]))
    medio = np.mean([[v for _c, v in lectura] for lectura in grupo],
                    axis=0)[:, :len(CHARSET)]
    medio[:, _IDX_GUION] = 0.0
    if n in (6, 7):
        for p in range(3, n):
            fila = np.zeros(len(CHARSET))
            fila[_IDX_DIG] = medio[p, _IDX_DIG]
            medio[p] = fila
    medio = medio + 1e-9
    medio = medio / medio.sum(axis=1, keepdims=True)
    conf = medio.max(axis=1)
    return {"texto": "".join(CHARSET[i] for i in medio.argmax(axis=1)),
            "conf": float(conf.mean()),
            "conf_min": float(conf.min()),
            "acuerdo": len(grupo) / len(lecturas),
            "probs": medio}