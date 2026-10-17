#!/usr/bin/env python3
"""OCR de placas peruanas sobre el recorte de la placa (RapidOCR, ONNX, CPU).

Este modulo resuelve dos cosas distintas:

1. LEER: el texto de la placa. Se llama SOLO con el recorte, nunca con la foto
   completa, porque la camara quema el timestamp OSD arriba a la izquierda y el
   OCR lo leeria como si fuera una placa ("10-09-2026 Fri 15:16:33").

2. NORMALIZAR: el dedup y el nombre del archivo necesitan una identidad
   canonica ("CBN-449"), no el texto sucio que sale del motor. La
   normalizacion valida contra los formatos de placa peruana y corrige las
   confusiones tipicas del OCR solo dentro de cada zona (letras: 0->O,
   1->I, 5->S | numeros: O->0, I->1, S->5, B->8, T->7...). El texto que no
   encaja en ningun formato valido se descarta: no es una placa.

Formatos de placa peruana aceptados:
    livianos/pesados   3 letras + 3 numeros   ABC-123
    motos / viejas     2 letras + 4 numeros   AB-1234
    motos nuevas       3 letras + 4 numeros   ABC-1234
    serie nueva mixta  3 alfanum. + 3 numeros B71-923, F8E-876, E7X-165
                       (la zona "de letras" de la serie nueva trae digitos:
                        no se puede cortar en el primer digito; se enumeran
                        todos los cortes validos)

Uso:
    from ocr import leer_placa, leer_placa_detallado
    texto = leer_placa(recorte)                  # "CBN-449" o ""
    texto, conf = leer_placa_detallado(recorte)  # ademas la confianza
"""
from __future__ import annotations

import logging
import re

import cv2
import numpy as np

logger = logging.getLogger("cytv.ocr")

# Motor cargado una sola vez: RapidOCR tarda ~0.45 s en el primer llamado
# (carga de los pesos ONNX) y ~0.05 s despues.
_motor = None

_FORMATOS = (
    re.compile(r"^([A-Z]{3})-(\d{3})$"),       # ABC-123 (liviano/pesado)
    re.compile(r"^([A-Z]{2})-(\d{4})$"),       # AB-1234 (moto/vieja)
    re.compile(r"^([A-Z]{3})-(\d{4})$"),       # ABC-1234 (moto nueva)
    re.compile(r"^([A-Z0-9]{3})-(\d{3})$"),    # serie nueva mixta: B71-923,
                                               # F8E-876, E7X-165 (la zona
                                               # "de letras" trae digitos)
)

# Confusiones de OCR, corregidas SOLO dentro de la zona que corresponde.
# En letras nunca se "corrige" un 7 a T ni un 4 a A: solo digitos que el
# motor confunde con letras, y al reves en la zona de numeros.
_NUM_POR_LETRA = str.maketrans({"O": "0", "I": "1", "S": "5", "Z": "2",
                                "G": "6", "B": "8", "T": "7"})
_LETRA_POR_NUM = str.maketrans({"0": "O", "1": "I", "5": "S", "2": "Z",
                                "8": "B", "6": "G"})
# El 0 leido en zona de letras casi nunca es cero de verdad: en la serie
# nueva la confusion tipica del OCR es Q->0 (C2Q-618 leido C20-618). Se
# prueba ANTES que 0->O y solo cuando la zona no es letra pura.
_LETRA_POR_CERO_Q = str.maketrans({"0": "Q"})


def _obtener_motor():
    global _motor
    if _motor is None:
        from rapidocr_onnxruntime import RapidOCR
        _motor = RapidOCR()
        logger.info("OCR cargado (RapidOCR ONNX, CPU)")
    return _motor


def _normalizar(texto: str) -> str:
    """Texto del OCR -> placa canonica ("CBN-449") o '' si no encaja."""
    bruto = texto or ""
    # El OCR confunde el guion con puntos medios y variantes de bullet
    # ("F8E·876"); se pasan a guion ANTES del gate de puntuacion.
    for sep in ("·", "•", "°", "`"):
        bruto = bruto.replace(sep, "-")
    # Las placas no llevan puntos, dos puntos ni barras: si el OCR trae uno de
    # esos signos, lo que leyo es otra cosa (p. ej. la hora del OSD
    # "Fri 15:16:33", que si no se colaria como FRI-1633 y pasaria el gate).
    if any(c in bruto for c in ":.,/\\"):
        return ""
    bruto = re.sub(r"[^A-Za-z0-9-]", "", bruto).upper()
    bruto = re.sub(r"-{2,}", "-", bruto).strip("-")
    alnum = bruto.replace("-", "")
    if not (5 <= len(alnum) <= 7):
        return ""

    # La serie nueva mezcla digitos dentro de la zona de letras (F8E-876):
    # no hay un unico "corte en la primera cifra" correcto. Se enumeran los
    # cortes validos (primero 3+3, que es la mayoria; luego 2+4 de motos).
    for corte in (3, 2):
        if corte >= len(alnum):
            continue
        letras, numeros = alnum[:corte], alnum[corte:]
        if not numeros.isdigit() or not any(ch.isalpha() for ch in letras):
            continue
        # Orden de la "traduccion" de la zona de letras: tal cual, luego el 0
        # como Q (C2Q-618 leido C20-618), luego las confusiones tipicas
        # (1->I, 5->S, ...). Zona de numeros: tal cual y luego O->0, I->1...
        variantes_L = (letras, letras.translate(_LETRA_POR_CERO_Q),
                       letras.translate(_LETRA_POR_NUM))
        variantes_N = (numeros, numeros.translate(_NUM_POR_LETRA))
        for L in variantes_L:
            for N in variantes_N:
                for patron in _FORMATOS:
                    m = patron.match(f"{L}-{N}")
                    if m:
                        return f"{m.group(1)}-{m.group(2)}"
    return ""


# Equivalencia de confusones para agrupar votos: Z1S y ZIS son la MISMA
# placa votando en el motor de 6 intentos. La identidad de guardado sigue
# siendo el texto del voto ganador; esto solo decide que dos lecturas
# distintas cuentan como el mismo acuerdo.
#   I/1, O/0, Z/7  confusiones de la serie mixta.
#   Q/0            el 0 leido en la zona mixta suele ser la Q de verdad
#                  (C2Q-618 leido C20-618 por el motor).
#   Y/V            la V del motor sobre la Y real (C6Y-488 leido C6V-488
#                  en 3 de 4 intentos; el umbral invertido si ve la Y).
_EQUIV_VOTOS = str.maketrans({"I": "1", "O": "0", "Z": "7", "Q": "0",
                              "Y": "V", "V": "V"})


def _forma_voto(texto: str) -> str:
    return texto.translate(_EQUIV_VOTOS)


# ---------------------------------------------------------------------------
# Motor de 6 intentos (Fase 0): una detección se procesa con hasta 6
# variantes del recorte y se vota. Consenso >= 2 intentos iguales = lectura
# ganadora; sin consenso => la detección no se puede leer (sin_placa).
# ---------------------------------------------------------------------------

# Grosor/padding heredados del worker; el recorte aqui es local al modulo.
def _recorte(img: np.ndarray, bbox: tuple[int, int, int, int],
             margen: float) -> np.ndarray | None:
    """Recorte del bbox con aire alrededor (margen fraccional del bbox)."""
    x, y, w, h = bbox
    dx, dy = int(w * margen), int(h * margen)
    alto, ancho = img.shape[:2]
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(ancho, x + w + dx), min(alto, y + h + dy)
    recorte = img[y0:y1, x0:x1]
    return recorte if recorte.size else None


def _clahe_gamma(crop: np.ndarray) -> np.ndarray:
    """CLAHE (contraste local) + gamma 1.25: placas con sombra o bajo contraste."""
    gris = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ecu = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gris)
    tabla = np.array([((i / 255.0) ** (1 / 1.25)) * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.cvtColor(cv2.LUT(ecu, tabla), cv2.COLOR_GRAY2BGR)


def _umbral_invertido(crop: np.ndarray) -> np.ndarray:
    """Binarizacion Otsu invertida (texto blanco sobre negro): a veces es lo
    unico que hace legibles las placas sucias o con reflejo."""
    gris = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binarizada = cv2.threshold(gris, 0, 255,
                                  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return cv2.cvtColor(binarizada, cv2.COLOR_GRAY2BGR)


def variantes_intento(img: np.ndarray, bbox: tuple[int, int, int, int],
                      margen_base: float = 0.35) -> list[tuple[str, np.ndarray]]:
    """Las 6 variantes del recorte para la votacion (Fase 0)."""
    base = _recorte(img, bbox, margen_base)
    if base is None:
        return []
    grande = cv2.resize(base, None, fx=2.0, fy=2.0,
                        interpolation=cv2.INTER_CUBIC)
    p5 = _recorte(img, bbox, margen_base + 0.05)
    p10 = _recorte(img, bbox, margen_base + 0.10)
    variantes: list[tuple[str, np.ndarray]] = [("recorte", base)]
    if p5 is not None:
        variantes.append(("margen+5%", p5))
    if p10 is not None:
        variantes.append(("margen+10%", p10))
    variantes += [("upscale2x", grande),
                  ("clahe_gamma", _clahe_gamma(base)),
                  ("umbral_inv", _umbral_invertido(base))]
    return variantes


def leer_placa_6(img: np.ndarray, bbox: tuple[int, int, int, int],
                 margen_base: float = 0.35
                 ) -> tuple[str, float, list[tuple[str, str, float]]]:
    """Motor de 6 intentos con votacion ponderada por confianza OCR.

    Devuelve (placa_canonica, confianza_media, registro) donde registro es
    [(intento, texto_o_vacio, confianza)] para dejar constancia en el log.

    Reglas de decision (todas por PUNTUACION = conf * peso_variante):
    0. PRIMERO el reconocedor propio (CRNN + flota cerrada, ver
       reconocedor.py): closed-set, ~82.8% sobre los 29 GT diagnosticos
       y ~4x mas rapido que el motor de 6 intentos. Si se abstiene
       (lectura fuera de flota, modelo ausente, fallo de carga), cae al
       motor de 6 intentos de abajo SIN CAMBIOS: cero regresion.
    1. Los votos se agrupan por equivalencia (I/1, O/0, Z/7, Q/0, Y/V):
       Z1S y ZIS son el MISMO acuerdo. umbral_inv pesa x1.2: en los 6
       conflictos posicionales medidos (B1Y/E7X/C6Y/F5A/ZIS/ARI) la
       lectura correcta era su minoria, con confianza cruda menor que la
       mayoria correlacionada que se equivocaba.
    2. Un grupo (>= 2 equivalentes) puntua su MEDIA ponderada; un
       singleton puntua su confianza ponderada y solo entra en la
       contienda si esa puntuacion es >= 0.70 (5 intentos vacios no
       votan en contra de 1 lectura limpia: T2K-790 y B71-923 se leen
       solo en umbral_inv; E7X-165 solo en umbral_inv a 0.71; F5A-947
       solo en umbral_inv a 0.67 pese al consenso errado E5A x3).
    3. Gana la mayor puntuacion; el texto es el miembro de mayor
       conf*peso. Sin ningun candidato => ("", 0.0).
    """
    # 0) Reconocedor propio CRNN+flota (closed-set): si aterriza en una
    #    placa de la flota verificada, esa es la lectura. Si se abstiene,
    #    sigue el motor heredado de 6 intentos de abajo, intacto.
    base = _recorte(img, bbox, margen_base)
    if base is not None and base.size:
        import reconocedor
        texto_crnn, conf_crnn = reconocedor.leer_placa(base)
        if texto_crnn:
            registro = [("crnn_flota", texto_crnn, conf_crnn)]
            logger.debug("CRNN: %s (conf=%.2f)", texto_crnn, conf_crnn)
            return texto_crnn, conf_crnn, registro

    UMBRAL_PUNTUACION = 0.70
    PESO_UMBRAL_INV = 1.2
    registro: list[tuple[str, str, float]] = []
    grupos: dict[str, list[tuple[str, float, float]]] = {}
    for nombre, crop in variantes_intento(img, bbox, margen_base):
        texto, conf = leer_placa_detallado(crop)
        registro.append((nombre, texto, conf))
        if texto:
            peso = PESO_UMBRAL_INV if nombre == "umbral_inv" else 1.0
            clave = _forma_voto(texto)
            grupos.setdefault(clave, []).append((texto, conf, peso))

    candidatos: list[tuple[float, list[tuple[str, float, float]]]] = []
    for miembros in grupos.values():
        if len(miembros) >= 2:
            puntaje = sum(c * p for _t, c, p in miembros) / len(miembros)
            candidatos.append((puntaje, miembros))
        else:
            _t, c, p = miembros[0]
            if c * p >= UMBRAL_PUNTUACION:
                candidatos.append((c * p, miembros))
    if not candidatos:
        return "", 0.0, registro
    _puntaje, miembros = max(candidatos, key=lambda par: par[0])
    conf = sum(c for _t, c, _p in miembros) / len(miembros)
    ganador = max(miembros, key=lambda tcp: tcp[1] * tcp[2])[0]
    return ganador, conf, registro


def leer_placa_detallado(img: np.ndarray) -> tuple[str, float]:
    """(placa canonica, confianza del motor). ("", 0.0) si no se pudo leer."""
    if img is None or getattr(img, "size", 0) == 0:
        return "", 0.0

    try:
        resultado, _ = _obtener_motor()(img)
    except Exception:
        logger.exception("Fallo el OCR")
        return "", 0.0
    if not resultado:
        return "", 0.0

    items = []
    for caja, texto, score in resultado:
        try:
            valor = float(score)
        except (TypeError, ValueError):
            valor = 0.0
        items.append((caja, str(texto), valor))

    # 1) Cada caja por separado, de la mas confiable a la menos.
    for _, texto, score in sorted(items, key=lambda x: -x[2]):
        normal = _normalizar(texto)
        if normal:
            return normal, score

    # 2) La placa llega partida en varias cajas: pruebo vecinas seguidas.
    items.sort(key=lambda x: min(p[0] for p in x[0]))
    for inicio in range(len(items) - 1):
        for fin in range(inicio + 1, min(inicio + 3, len(items))):
            unido = "".join(t for _, t, _ in items[inicio:fin + 1])
            normal = _normalizar(unido)
            if normal:
                scores = [s for _, _, s in items[inicio:fin + 1]]
                return normal, sum(scores) / len(scores)
    return "", 0.0


def leer_placa(img: np.ndarray) -> str:
    """Devuelve la placa en texto canonico, o '' si no se pudo leer."""
    return leer_placa_detallado(img)[0]
