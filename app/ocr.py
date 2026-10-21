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



_motor = None



_detector_onnx = None

_FORMATOS = (
    re.compile(r"^([A-Z]{3})-(\d{3})$"),       
    re.compile(r"^([A-Z]{2})-(\d{4})$"),       
    re.compile(r"^([A-Z]{3})-(\d{4})$"),       
    re.compile(r"^([A-Z0-9]{3})-(\d{3})$"),    
                                               
                                               
)




_NUM_POR_LETRA = str.maketrans({"O": "0", "I": "1", "S": "5", "Z": "2",
                                "G": "6", "B": "8", "T": "7"})
_LETRA_POR_NUM = str.maketrans({"0": "O", "1": "I", "5": "S", "2": "Z",
                                "8": "B", "6": "G"})



_LETRA_POR_CERO_Q = str.maketrans({"0": "Q"})


def _limitar_hilos_rapidocr() -> None:
    """1 hilo por sesion interna de RapidOCR (det, cls y rec).

    rapidocr_onnxruntime construye sus InferenceSession con los defaults
    de onnxruntime (intra_op = todos los nucleos del equipo). En el
    servidor eso enciende todos los cores en CADA foto. Se envuelve la
    clase ya importada en utils antes de construir el motor; idempotente.
    """
    try:
        import onnxruntime as ort
        import rapidocr_onnxruntime.utils as ru
        original = ru.InferenceSession
        if getattr(original, "_cytv_hilos", False):
            return

        def _limitada(*args, **kwargs):
            so = kwargs.get("sess_options")
            if so is not None:
                so.intra_op_num_threads = 1
                so.inter_op_num_threads = 1
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            return original(*args, **kwargs)

        _limitada._cytv_hilos = True
        ru.InferenceSession = _limitada
    except Exception:
        logger.debug("No se pudieron limitar los hilos de RapidOCR",
                     exc_info=True)


def _obtener_motor():
    global _motor
    if _motor is None:
        _limitar_hilos_rapidocr()
        from rapidocr_onnxruntime import RapidOCR
        _motor = RapidOCR()
        logger.info("OCR cargado (RapidOCR ONNX, CPU, 1 hilo)")
    return _motor


def _normalizar(texto: str) -> str:
    """Texto del OCR -> placa canonica ("CBN-449") o '' si no encaja."""
    bruto = texto or ""
    
    
    for sep in ("·", "•", "°", "`"):
        bruto = bruto.replace(sep, "-")
    
    
    
    if any(c in bruto for c in ":.,/\\"):
        return ""
    bruto = re.sub(r"[^A-Za-z0-9-]", "", bruto).upper()
    bruto = re.sub(r"-{2,}", "-", bruto).strip("-")
    alnum = bruto.replace("-", "")
    if not (5 <= len(alnum) <= 7):
        return ""

    
    
    
    for corte in (3, 2):
        if corte >= len(alnum):
            continue
        letras, numeros = alnum[:corte], alnum[corte:]
        if not numeros.isdigit() or not any(ch.isalpha() for ch in letras):
            continue
        
        
        
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











_EQUIV_VOTOS = str.maketrans({"I": "1", "O": "0", "Z": "7", "Q": "0",
                              "Y": "V", "V": "V"})


def _forma_voto(texto: str) -> str:
    return texto.translate(_EQUIV_VOTOS)










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


def _noche(crop: np.ndarray) -> np.ndarray:
    """CLAHE agresivo + bilateral + gamma 1.4: placas de noche.

    El ruido IR de la camara sube el piso de grises y el CLAHE suave
    (2.0/8x8) no alcanza; bilateral limpia el grano sin matar los bordes
    de las letras.
    """
    gris = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ecu = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6)).apply(gris)
    suave = cv2.bilateralFilter(ecu, 9, 60, 60)
    tabla = np.array([((i / 255.0) ** (1 / 1.4)) * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.cvtColor(cv2.LUT(suave, tabla), cv2.COLOR_GRAY2BGR)


def variantes_intento(img: np.ndarray, bbox: tuple[int, int, int, int],
                      margen_base: float = 0.35) -> list[tuple[str, np.ndarray]]:
    """Las variantes del recorte para la votacion (Fase 0 + noche Fase 2)."""
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
                  ("noche", _noche(base)),
                  ("umbral_inv", _umbral_invertido(base))]
    return variantes


UMBRAL_CONF_RESUELTA = 0.95    
UMBRAL_PUNTUACION = 0.70
PESO_UMBRAL_INV = 1.0










def _correr_6(img: np.ndarray, bbox: tuple[int, int, int, int],
              margen_base: float
              ) -> tuple[list[tuple[str, str, float]],
                         dict[str, list[tuple[str, float, float]]]]:
    """Los intentos -> (registro, votos agrupados por equivalencia)."""
    registro: list[tuple[str, str, float]] = []
    grupos: dict[str, list[tuple[str, float, float]]] = {}
    for nombre, crop in variantes_intento(img, bbox, margen_base):
        texto, conf = leer_placa_detallado(crop)
        registro.append((nombre, texto, conf))
        if texto:
            peso = PESO_UMBRAL_INV if nombre == "umbral_inv" else 1.0
            grupos.setdefault(_forma_voto(texto),
                              []).append((texto, conf, peso))
    return registro, grupos


def _elegir_voto(grupos: dict[str, list[tuple[str, float, float]]]
                 ) -> tuple[str, float, list[tuple[str, float, float]]]:
    """(texto_ganador, conf_media, miembros); [] si no hay voto.

    Reglas 2-4: si existe algun grupo (>= 2) los singletons quedan
    fuera; gana la mayor puntuacion; el texto es el miembro de mayor
    conf*peso. Sin snap ni flota: el texto sale tal cual lo vio la
    mayoria.
    """
    hay_grupo = any(len(miembros) >= 2 for miembros in grupos.values())
    candidatos: list[tuple[float, list[tuple[str, float, float]]]] = []
    for miembros in grupos.values():
        if len(miembros) >= 2:
            puntaje = sum(c * p for _t, c, p in miembros) / len(miembros)
            candidatos.append((puntaje, miembros))
        elif not hay_grupo:
            _t, c, p = miembros[0]
            if c * p >= UMBRAL_PUNTUACION:
                candidatos.append((c * p, miembros))
    if not candidatos:
        return "", 0.0, []
    _puntaje, miembros = max(candidatos, key=lambda par: par[0])
    conf = sum(c for _t, c, _p in miembros) / len(miembros)
    ganador = max(miembros, key=lambda tcp: tcp[1] * tcp[2])[0]
    return ganador, conf, miembros


def _reintento_onnx(img: np.ndarray, bbox: tuple[int, int, int, int],
                    margen_base: float
                    ) -> tuple[str, float, list[tuple[str, str, float]]]:
    """(texto, conf, registro): segunda oportunidad con el bbox del ONNX.

    Se invoca SOLO cuando CRNN y todos los intentos quedaron vacios sobre el
    bbox anotado: el .txt puede cortar la placa o dar aire distinto al
    que ve produccion. Umbral de aceptacion (medido 10-19): un voto que
    viene de un GRUPO (>= 2 miembros, intentos +/- CRNN) cuyo mejor
    miembro alcanza UMBRAL_PUNTUACION. Sin ese filtro el reintento
    inventaba 3 errores (CU4625 en NO_PLACA con grupo CU4-625 a 0.65,
    A7-935 con singleton umbral_inv A07-935 y NTG-774 con CRNN 'AXF-748'
    a 0.935); con el filtro conserva los 2 aciertos exclusivos de aqui:
    OF868 -> FOF-868 y placa_15-28-03 -> F2P-703.
    """
    global _detector_onnx
    registro: list[tuple[str, str, float]] = []
    try:
        if _detector_onnx is None:
            from modelo import DetectorPlacas, RUTA_ONNX
            if not RUTA_ONNX.is_file():
                return "", 0.0, registro
            _detector_onnx = DetectorPlacas(RUTA_ONNX)
        detecciones = _detector_onnx.detectar(img)
        if not detecciones:
            return "", 0.0, registro
        b2 = max(detecciones, key=lambda par: par[0][2] * par[0][3])[0]
        b2 = tuple(int(v) for v in b2)
        if b2 == tuple(bbox):
            return "", 0.0, registro
    except Exception:
        logger.debug("reintento ONNX no disponible", exc_info=True)
        return "", 0.0, registro
    registro.append(("retry_onnx", f"bbox={b2}", 0.0))
    base = _recorte(img, b2, margen_base)
    texto_crnn, conf_crnn = "", 0.0
    if base is not None and base.size:
        import reconocedor
        texto_crnn, conf_crnn = reconocedor.leer_placa(base)
        if texto_crnn:
            registro.append(("crnn", texto_crnn, conf_crnn))
    voto_reg, grupos = _correr_6(img, b2, margen_base)
    registro += voto_reg
    if texto_crnn and any(t for _n, t, _c in voto_reg):
        grupos.setdefault(_forma_voto(texto_crnn),
                          []).append((texto_crnn, conf_crnn, 1.0))
    ganador, conf, miembros = _elegir_voto(grupos)
    if (len(miembros) >= 2
            and max(c for _t, c, _p in miembros) >= UMBRAL_PUNTUACION):
        return ganador, conf, registro
    return "", 0.0, registro


def leer_placa_6(img: np.ndarray, bbox: tuple[int, int, int, int],
                 margen_base: float = 0.35
                 ) -> tuple[str, float, list[tuple[str, str, float]]]:
    """Motor de votacion ponderada por confianza OCR (7 intentos: 6 del
    Fase 0 + variante noche del Fase 2).

    Devuelve (placa_canonica, confianza_media, registro) donde registro es
    [(intento, texto_o_vacio, confianza)] para dejar constancia en el log.

    Reglas de decision (todas por PUNTUACION = conf * peso_variante):
    0. El CRNN propio (ver reconocedor.py) es UN VOTO MAS, nunca un rey:
       su confianza NO esta calibrada (decia 1.00 en lecturas
       inventadas: C5R-622, AZK-44, PK-8760). Se suma al grupo de
       equivalencia SOLO si los 6 intentos tambien vieron texto (dos
       motores independientes). Si decide solo, no decide: ("", 0.0) --
       asi CRW-025 (foto SIN placa donde el CRNN inventa 'C4W-25' a
       0.934 y RapidOCR esta vacio en los 6 cortes) sigue sin falso
       positivo.
    1. Los votos se agrupan por equivalencia (I/1, O/0, Z/7, Q/0, Y/V):
       Z1S y ZIS son el MISMO acuerdo. umbral_inv pesa x1.2: en los 6
       conflictos posicionales medidos (B1Y/E7X/C6Y/F5A/ZIS/ARI) la
       lectura correcta era su minoria, con confianza cruda menor que la
       mayoria correlacionada que se equivocaba.
    2. Un grupo (>= 2 equivalentes) puntua su MEDIA ponderada. Si existe
       al menos un grupo, los singletons quedan FUERA de la contienda:
       una mayoria correlacionada de 4-5 intentos no puede perder contra
       1 sola lectura. (9 de los 16 errores del CRNN venian de un
       singleton de umbral_inv robando el voto: HTC-798, COV-876,
       CZC-910, HTG-954, V2M-407...). Los singletons solo compiten
       cuando NO hay ningun grupo y su puntuacion es >= 0.70 (T2K-790,
       B71-923 y E7X-165 se leen unicamente en umbral_inv).
    3. Gana la mayor puntuacion; el texto es el miembro de mayor
       conf*peso. Sin ningun candidato => ("", 0.0).
    4. Si CRNN y todos los intentos quedaron vacios, _reintento_onnx prueba
       con la deteccion del propio modelo ONNX (el bbox del .txt corta
       la placa o da aire distinto). El reintento exige evidencia
       fuerte -- ver _reintento_onnx: grupo con mejor miembro >= 0.70.
    Sin snap ni flota: el texto que decide el modelo es el texto final.
    """
    
    
    base = _recorte(img, bbox, margen_base)
    texto_crnn, conf_crnn = "", 0.0
    if base is not None and base.size:
        import reconocedor
        texto_crnn, conf_crnn = reconocedor.leer_placa(base)

    registro: list[tuple[str, str, float]] = []
    if texto_crnn:
        registro.append(("crnn", texto_crnn, conf_crnn))
        logger.debug("CRNN: %s (conf=%.2f)", texto_crnn, conf_crnn)
    voto_reg, grupos = _correr_6(img, bbox, margen_base)
    registro += voto_reg
    if texto_crnn and any(t for _n, t, _c in voto_reg):
        grupos.setdefault(_forma_voto(texto_crnn),
                          []).append((texto_crnn, conf_crnn, 1.0))
    ganador, conf, miembros = _elegir_voto(grupos)

    if miembros:
        return ganador, conf, registro
    if texto_crnn:
        # CRNN solo: no decide (no fabrica).
        return "", 0.0, registro
    t2, c2, reg2 = _reintento_onnx(img, bbox, margen_base)
    return t2, c2, registro + reg2


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

    
    for _, texto, score in sorted(items, key=lambda x: -x[2]):
        normal = _normalizar(texto)
        if normal:
            return normal, score

    
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
















UMBRAL_VERDE = 90
UMBRAL_AMARILLO = 50


def evidencia_y_color(texto: str, conf: float,
                      registro: list[tuple[str, str, float]] | None = None
                      ) -> tuple[int, str]:
    """(puntaje 0-100, 'verde'|'amarillo'|'rojo') para una lectura.

    ``registro`` es la lista de intentos de leer_placa_6:
    [(intento, texto_o_vacio, conf)] -- se cuenta cuantos intentos ven al
    ganador (por canonic o por la forma equivalente I/1, O/0, ...).
    """
    if not texto:
        return 0, "rojo"

    objetivo = (texto.upper().replace("-", "").strip()
                if texto else "")
    forma_ganador = _forma_voto(texto)
    a_favor = 1  
    for _nombre, t, _c in registro or []:
        if not t:
            continue
        forma = t.upper().replace("-", "").strip()
        if forma == objetivo or _forma_voto(t) == forma_ganador:
            a_favor += 1
    a_favor = min(a_favor, 6)

    
    
    es_crnn_directa = (conf >= UMBRAL_CONF_RESUELTA
                       and any(n == "crnn" for n, _t, _c
                               in registro or []))
    if es_crnn_directa:
        base = 70.0 + 30.0 * max(0.0, min(conf, 1.0))
    else:
        base = (25.0 + 11.0 * min(a_favor, 5)
                + 40.0 * max(0.0, min(conf, 1.0)))

    puntaje = int(round(min(base, 100.0)))
    if puntaje >= UMBRAL_VERDE:
        color = "verde"
    elif puntaje >= UMBRAL_AMARILLO:
        color = "amarillo"
    else:
        color = "rojo"
    return puntaje, color
