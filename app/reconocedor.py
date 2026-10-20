#!/usr/bin/env python3
"""Reconocedor de placas CYTV: CRNN propio (ONNX) + diccionario de flota.

Lector principal de la Fase 0 v2. Closed-set: solo responde cuando la
lectura (o su correccion unica a distancia <= 2) aterriza en una placa
de la flota verificada; si se abstiene, el llamador cae al motor
heredado de 6 intentos (ocr.leer_placa_6), que queda intacto.

Preproceso IDENTICO al entrenamiento (si se rompe, el accuracy cae en
silencio): PIL convert("L").resize((160, 32), BILINEAR) / 255.0.
Sin torch: corre en el python de servicio (3.13) con numpy+PIL+ort.
"""
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

_sesion = None
_sesion_mtime: float | None = None
_flota: list[str] | None = None
_conjunto: set[str] | None = None


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
    try:
        nueva = ort.InferenceSession(str(RUTA_ONNX),
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


def _obtener_flota():
    global _flota, _conjunto
    if _flota is None:
        from decodificador_flota import canon, flota
        _flota = flota()
        _conjunto = {canon(p) for p in _flota}
        logger.info("flota cargada: %d placas verificadas", len(_flota))
    return _flota, _conjunto


def _tensor_desde_bgr(crop: np.ndarray) -> np.ndarray:
    """BGR uint8 (cv2) -> tensor (1,1,32,160) float [0,1]."""
    if crop.ndim == 3 and crop.shape[2] == 3:
        rgb = np.ascontiguousarray(crop[:, :, ::-1])
    else:
        rgb = np.ascontiguousarray(crop)
    g = Image.fromarray(rgb).convert("L").resize((160, 32), Image.BILINEAR)
    arr = np.asarray(g, dtype=np.float32)[None] / 255.0
    return arr[None]


def _decodificar(logits: np.ndarray) -> str:
    """CTC greedy: argmax -> quitar repetidos -> quitar blanks."""
    idx = logits.argmax(axis=-1).tolist()
    salida, previo = [], None
    for i in idx:
        if i != _BLANK and i != previo:
            salida.append(CHARSET[i])
        previo = i
    return "".join(salida)


def leer_placa(crop: np.ndarray) -> tuple[str, float]:
    """(placa canonica de flota, confianza cruda) o ("", 0.0) si abstuvo.

    Confianza = media del maximo softmax por timestep (0..1, no
    calibrada: sirve como senal relativa y para el log).
    """
    from decodificador_flota import canon, corregir

    if crop is None or getattr(crop, "size", 0) == 0:
        return "", 0.0
    try:
        sesion = _obtener_sesion()
        conjunto, canonicos = _obtener_flota()
    except Exception:
        logger.debug("CRNN no disponible (fallo de carga)", exc_info=True)
        return "", 0.0
    try:
        logits = sesion.run(None, {"placa": _tensor_desde_bgr(crop)})[0][0]
        leido = _decodificar(logits)
        if not leido:
            return "", 0.0
        corregido = corregir(leido, conjunto, 2)
        
        
        if canon(corregido) not in canonicos:
            return "", 0.0
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        conf = float((probs / probs.sum(axis=-1, keepdims=True))
                     .max(axis=-1).mean())
        return corregido, conf
    except Exception:
        logger.exception("Fallo el reconocedor CRNN")
        return "", 0.0
