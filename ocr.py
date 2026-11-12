#!/usr/bin/env python3
"""OCR de placas peruanas sobre el recorte de la placa (RapidOCR, ONNX, CPU).

Este modulo resuelve dos cosas distintas:

1. LEER: el texto de la placa. Se llama SOLO con el recorte, nunca con la foto
   completa, porque la camara quema el timestamp OSD arriba a la izquierda y el
   OCR lo leeria como si fuera una placa ("10-09-2026 Fri 15:16:33").

2. LIMPIAR: el texto sucio que sale del motor se deja en mayusculas y sin
   signos. NO se valida formato de placa: el texto limpio se guarda tal cual
   (camara, web o cualquier fuente) y la revision la hace el operador.

Uso:
    from ocr import leer_placa
    texto = leer_placa(recorte)      # "CBN-449" o "" si no se pudo leer
"""
from __future__ import annotations

import logging
import re

import numpy as np

logger = logging.getLogger("cytv.ocr")

# Motor cargado una sola vez: RapidOCR tarda ~0.45 s en el primer llamado
# (carga de los pesos ONNX) y ~0.05 s despues.
_motor = None


def _obtener_motor():
    global _motor
    if _motor is None:
        from rapidocr_onnxruntime import RapidOCR
        _motor = RapidOCR()
        logger.info("OCR cargado (RapidOCR ONNX, CPU)")
    return _motor


def _limpiar(texto: str) -> str:
    """Mayusculas + solo alfanumerico y guion. NO valida formato de placa:
    el texto limpio se devuelve tal cual y la revision es del operador."""
    bruto = texto or ""
    # El OSD de la camara quema la hora con dos puntos ("15:16:33"): si el
    # OCR trae esos signos, no es texto de placa.
    if any(c in bruto for c in ":.,/\\"):
        return ""
    limpio = re.sub(r"[^A-Za-z0-9-]", "", bruto).upper().strip("-")
    # Piso minimo de longitud: la placa mas corta (AB-1234) tiene 5
    # caracteres; "PERU" (4, va impresa en toda placa) queda fuera solo
    # por largo, no por formato.
    return limpio if len(limpio) >= 5 else ""


def leer_placa(img: np.ndarray) -> str:
    """Devuelve el texto limpio del OCR, o '' si no se pudo leer."""
    if img is None or getattr(img, "size", 0) == 0:
        return ""

    try:
        resultado, _ = _obtener_motor()(img)
    except Exception:
        logger.exception("Fallo el OCR")
        return ""
    if not resultado:
        return ""

    items = []
    for caja, texto, score in resultado:
        try:
            valor = float(score)
        except (TypeError, ValueError):
            valor = 0.0
        items.append((caja, str(texto), valor))

    # 1) Cada caja por separado, de la mas confiable a la menos.
    for _, texto, _ in sorted(items, key=lambda x: -x[2]):
        limpio = _limpiar(texto)
        if limpio:
            return limpio

    # 2) La placa llega partida en varias cajas: pruebo vecinas seguidas.
    items.sort(key=lambda x: min(p[0] for p in x[0]))
    for inicio in range(len(items) - 1):
        for fin in range(inicio + 1, min(inicio + 3, len(items))):
            unido = "".join(t for _, t, _ in items[inicio:fin + 1])
            limpio = _limpiar(unido)
            if limpio:
                return limpio
    return ""
