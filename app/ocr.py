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
    re.compile(r"^([A-Z0-9]{3})-(\d{3})$"),
)

_NUM_POR_LETRA = str.maketrans({"O": "0", "I": "1", "S": "5", "Z": "2",
                                "G": "6", "B": "8", "T": "7",
                                "Q": "0", "D": "0"})
_LETRA_POR_NUM = str.maketrans({"0": "O", "1": "I", "5": "S", "2": "Z",
                                "8": "B", "6": "G"})
_LETRA_POR_CERO_Q = str.maketrans({"0": "Q"})

_EQUIV_VOTOS = str.maketrans({"I": "1", "O": "0", "Z": "7", "Q": "0",
                              "Y": "V", "V": "V"})

UMBRAL_CONF_RESUELTA = 0.95
UMBRAL_PUNTUACION = 0.70
PESO_UMBRAL_INV = 1.0

PARES_CONFUSOS = {frozenset(p) for p in
                  ("WH", "WV", "QO", "QD", "OD", "MN", "UV")}
UMBRAL_ARBITRAJE = 0.60
UMBRAL_CRNN_FUERTE = 0.95
RAPIDO = {"recorte", "clahe_gamma"}
MARGENES_CRNN = (0.05, 0.11, 0.18)
SNAP_K = 2
SNAP_K_SINGLETON = 1

UMBRAL_VERDE = 90
UMBRAL_AMARILLO = 50


def _limitar_hilos_rapidocr() -> None:
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
    bruto = texto or ""
    for sep in ("·", "•", "°", "`"):
        bruto = bruto.replace(sep, "-")
    if any(c in bruto for c in ":.,/\\"):
        return ""
    bruto = re.sub(r"[^A-Za-z0-9-]", "", bruto).upper()
    bruto = re.sub(r"-{2,}", "-", bruto).strip("-")
    alnum = bruto.replace("-", "")
    if len(alnum) == 7 and alnum[3:].isdigit():
        alnum = alnum[:6]
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


def _forma_voto(texto: str) -> str:
    return texto.translate(_EQUIV_VOTOS)


def _snap_flota(texto: str, k: int) -> str:
    """corregir() del decodificador: exacta o vecina UNICA a distancia <= k."""
    if not texto:
        return texto
    try:
        from decodificador_flota import corregir, flota
        return corregir(texto, flota(), k) or texto
    except Exception:
        logger.debug("flota no disponible para el snap", exc_info=True)
        return texto


def _en_flota(texto: str) -> bool:
    try:
        from decodificador_flota import canonicos, canon
        return canon(texto) in canonicos()
    except Exception:
        logger.debug("flota no disponible para el gate", exc_info=True)
        return True


def _snap_sin_consenso(registro: list[tuple[str, str, float]]
                       ) -> tuple[str, float] | None:
    """Rescate (Fase 2a): mejor singleton que cae a distancia <= SNAP_K de
    una placa UNICA de flota y se MOVIÓ hacia ella. Solo sin CRNN (el CRNN
    gateado manda antes) y solo si el remapeo cambia el texto."""
    candidatos = [(c, t) for n, t, c in registro
                  if t and n not in ("decision", "arbitraje",
                                     "arbitraje_descartado", "snap_flota",
                                     "crnn", "crnn_flota", "retry_onnx")]
    if not candidatos:
        return None
    mejor_c, mejor_t = max(candidatos)
    destino = _snap_flota(mejor_t, SNAP_K)
    if destino == mejor_t:
        return None
    return destino, mejor_c


def _recorte(img: np.ndarray, bbox: tuple[int, int, int, int],
             margen: float) -> np.ndarray | None:
    x, y, w, h = bbox
    dx, dy = int(w * margen), int(h * margen)
    alto, ancho = img.shape[:2]
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(ancho, x + w + dx), min(alto, y + h + dy)
    recorte = img[y0:y1, x0:x1]
    return recorte if recorte.size else None


def _clahe_gamma(crop: np.ndarray) -> np.ndarray:
    gris = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ecu = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gris)
    tabla = np.array([((i / 255.0) ** (1 / 1.25)) * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.cvtColor(cv2.LUT(ecu, tabla), cv2.COLOR_GRAY2BGR)


def _umbral_invertido(crop: np.ndarray) -> np.ndarray:
    gris = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binarizada = cv2.threshold(gris, 0, 255,
                                  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return cv2.cvtColor(binarizada, cv2.COLOR_GRAY2BGR)


def _noche(crop: np.ndarray) -> np.ndarray:
    gris = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ecu = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6)).apply(gris)
    suave = cv2.bilateralFilter(ecu, 9, 60, 60)
    tabla = np.array([((i / 255.0) ** (1 / 1.4)) * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.cvtColor(cv2.LUT(suave, tabla), cv2.COLOR_GRAY2BGR)


def variantes_intento(img: np.ndarray, bbox: tuple[int, int, int, int],
                      margen_base: float = 0.35) -> list[tuple[str, np.ndarray]]:
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


def _correr_6(img: np.ndarray, bbox: tuple[int, int, int, int],
              margen_base: float, solo=None, excluir=()
              ) -> tuple[list[tuple[str, str, float]],
                         dict[str, list[tuple[str, float, float]]]]:
    registro: list[tuple[str, str, float]] = []
    grupos: dict[str, list[tuple[str, float, float]]] = {}
    for nombre, crop in variantes_intento(img, bbox, margen_base):
        if (solo is not None and nombre not in solo) or nombre in excluir:
            continue
        texto, conf = leer_placa_detallado(crop)
        registro.append((nombre, texto, conf))
        if texto:
            peso = PESO_UMBRAL_INV if nombre == "umbral_inv" else 1.0
            grupos.setdefault(_forma_voto(texto),
                              []).append((texto, conf, peso))
    return registro, grupos


def _elegir_voto(grupos: dict[str, list[tuple[str, float, float]]]
                 ) -> tuple[str, float, list[tuple[str, float, float]]]:
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
    k = SNAP_K if len(miembros) >= 2 else SNAP_K_SINGLETON
    nuevo = _snap_flota(ganador, k)
    if nuevo != ganador:
        logger.debug("snap post-voto: %s -> %s", ganador, nuevo)
    return nuevo, conf, miembros


def _crnn_consenso(img: np.ndarray, bbox: tuple[int, int, int, int],
                   margen_base: float) -> dict | None:
    try:
        import reconocedor
    except Exception:
        return None
    crops = []
    for margen in MARGENES_CRNN:
        r = _recorte(img, bbox, margen)
        if r is not None and r.size:
            crops.append(r)
    if not crops:
        return None
    medio = _recorte(img, bbox, MARGENES_CRNN[1])
    if medio is not None and medio.size:
        crops += [_clahe_gamma(medio), _noche(medio)]
    resultado = reconocedor.leer_consenso(crops)
    if not resultado:
        return None
    normal = _normalizar(resultado["texto"])
    if not normal:
        return None
    destino = _snap_flota(normal, SNAP_K)
    if not _en_flota(destino):
        return None
    resultado["texto"] = destino
    resultado["normal"] = destino
    return resultado


def _arbitrar(ganador: str, crnn: dict) -> str:
    import reconocedor
    g, t = ganador.replace("-", ""), crnn["texto"]
    if len(g) != len(t):
        return ganador
    salida = list(g)
    for i, (a, b) in enumerate(zip(g, t)):
        if a == b or frozenset((a, b)) not in PARES_CONFUSOS:
            continue
        pa = crnn["probs"][i][reconocedor.CHARSET.index(a)]
        pb = crnn["probs"][i][reconocedor.CHARSET.index(b)]
        if pb >= UMBRAL_ARBITRAJE and pb > pa:
            salida[i] = b
    nuevo = "".join(salida)
    if nuevo == g:
        return ganador
    return _normalizar(nuevo) or ganador


def _votar(grupos, t_crnn, c_crnn, hubo_rapid):
    copia = {k: list(v) for k, v in grupos.items()}
    if t_crnn and hubo_rapid:
        copia.setdefault(_forma_voto(t_crnn), []).append((t_crnn, c_crnn, 1.0))
    return _elegir_voto(copia)


def _leer_en_bbox(img: np.ndarray, bbox: tuple[int, int, int, int],
                  margen_base: float):
    crnn = _crnn_consenso(img, bbox, margen_base)
    t_crnn = crnn["normal"] if crnn else ""
    c_crnn = crnn["conf"] if crnn else 0.0
    registro: list[tuple[str, str, float]] = []
    if t_crnn:
        registro.append(("crnn_flota", t_crnn, c_crnn))

    reg, grupos = _correr_6(img, bbox, margen_base, solo=RAPIDO)
    registro += reg
    hubo = any(t for _n, t, _c in reg)
    ganador, conf, miembros = _votar(grupos, t_crnn, c_crnn, hubo)

    dos_motores = (bool(t_crnn) and len(miembros) >= 2
                   and _forma_voto(ganador) == _forma_voto(t_crnn))
    if not dos_motores:
        reg2, grupos2 = _correr_6(img, bbox, margen_base, excluir=RAPIDO)
        registro += reg2
        for clave, valores in grupos2.items():
            grupos.setdefault(clave, []).extend(valores)
        hubo = hubo or any(t for _n, t, _c in reg2)
        ganador, conf, miembros = _votar(grupos, t_crnn, c_crnn, hubo)

    if t_crnn and c_crnn >= UMBRAL_CRNN_FUERTE:
        registro.append(("decision", f"crnn_fuerte ({c_crnn:.3f})", 0.0))
        return t_crnn, c_crnn, registro, miembros, t_crnn
    if not miembros:
        if t_crnn:
            registro.append(("decision",
                             "crnn_unico (rapidocr sin lectura)", 0.0))
            return t_crnn, c_crnn, registro, miembros, t_crnn
        rescate = _snap_sin_consenso(registro)
        if rescate:
            texto_r, conf_r = rescate
            registro.append(("snap_flota", texto_r, conf_r))
            return texto_r, conf_r, registro, [], t_crnn
        return "", 0.0, registro, [], t_crnn
    if t_crnn and len(miembros) < 2:
        registro.append(("decision", "crnn_por_falta_de_apoyo", 0.0))
        return t_crnn, c_crnn, registro, miembros, t_crnn
    if crnn:
        corregido = _arbitrar(ganador, crnn)
        if corregido != ganador:
            if corregido == t_crnn:
                registro.append(("arbitraje", f"{ganador}->{corregido}", 0.0))
                ganador = corregido
            else:
                registro.append(("arbitraje_descartado",
                                 f"{ganador}->{corregido}", 0.0))
    return ganador, conf, registro, miembros, t_crnn


def _reintento_onnx(img: np.ndarray, bbox: tuple[int, int, int, int],
                    margen_base: float
                    ) -> tuple[str, float, list[tuple[str, str, float]]]:
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
    ganador, conf, reg, miembros, _t = _leer_en_bbox(img, b2, margen_base)
    registro += reg
    if (len(miembros) >= 2
            and max(c for _x, c, _p in miembros) >= UMBRAL_PUNTUACION):
        return ganador, conf, registro
    return "", 0.0, registro


def leer_placa_6(img: np.ndarray, bbox: tuple[int, int, int, int],
                 margen_base: float = 0.35, reintento: bool = True
                 ) -> tuple[str, float, list[tuple[str, str, float]]]:
    ganador, conf, registro, miembros, t_crnn = _leer_en_bbox(
        img, bbox, margen_base)
    if miembros:
        return ganador, conf, registro
    if ganador:
        return ganador, conf, registro
    if t_crnn:
        return t_crnn, c_crnn, registro
    if not reintento:
        return "", 0.0, registro
    t2, c2, reg2 = _reintento_onnx(img, bbox, margen_base)
    return t2, c2, registro + reg2


def leer_placa_detallado(img: np.ndarray) -> tuple[str, float]:
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
    return leer_placa_detallado(img)[0]


def evidencia_y_color(texto: str, conf: float,
                      registro: list[tuple[str, str, float]] | None = None
                      ) -> tuple[int, str]:
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
                       and any(n in ("crnn", "crnn_flota")
                               for n, _t, _c in registro or []))
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