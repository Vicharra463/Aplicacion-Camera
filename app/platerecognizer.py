import io
import time
import logging

import cv2
import requests

logger = logging.getLogger("cytv.platerecognizer")
URL = "https://api.platerecognizer.com/v1/plate-reader/"
TIMEOUT = 10
CUOTA_SEG = 3600     # 402/403: la clave queda 1 h fuera
RATE_SEG = 60        # 429 persistente: esa clave 1 min fuera
SIN_RED_SEG = 120    # sin conexion: toda la AOI 2 min fuera

# Claves en orden de prueba. Cada fuente es una cadena 'a, b, c'
# (config.ini [aoi] tokens y/o PLATERECOGNIZER_TOKEN).
# NO se commitean claves aca: se cargan por config o variable de entorno.
_tokens: list[str] = []
# Claves apartadas por cuota/rate: clave -> (hasta_ts, motivo)
_cuota: dict[str, tuple[float, str]] = {}
_apagada_hasta = 0.0   # apagado GLOBAL (solo errores de red)
_motivo = ""

# Sin caches ni ventanas de tiempo: "una llamada a la API por vehiculo por
# dia" lo garantiza el worker con el pre-chequeo mismo-carro (si el carro ya
# tiene confirmacion de la API hoy, la toma siguiente reutiliza la placa y
# no se vuelve a llamar).

_EST_OK, _EST_VACIO, _EST_AGOTADA, _EST_RATE, _EST_RED, _EST_ERROR = range(6)


def configurar(*fuentes) -> list[str]:
    """Concatena claves de cada fuente, limpia espacios y deduplica.

    Conserva el orden: es el orden de prueba contra la API.
    """
    global _tokens
    vistos, salida = set(), []
    for fuente in fuentes:
        for tok in str(fuente or "").split(","):
            tok = tok.strip()
            if tok and tok not in vistos:
                vistos.add(tok)
                salida.append(tok)
    _tokens = salida
    return salida


def hay_claves() -> bool:
    return bool(_tokens)


def _vigente(tok: str) -> bool:
    return time.time() >= _cuota.get(tok, (0.0, ""))[0]


def api_disponible() -> bool:
    """True si hay al menos una clave sin cuota y la AOI no esta apagada."""
    if not _tokens or time.time() < _apagada_hasta:
        return False
    return any(_vigente(t) for t in _tokens)


def _apagar(segundos, motivo):
    global _apagada_hasta, _motivo
    _apagada_hasta, _motivo = time.time() + segundos, motivo
    logger.warning("API apagada %d min: %s", max(1, int(segundos // 60)), motivo)


def motivo_sin_api() -> str:
    if not _tokens:
        return "sin_token"
    if time.time() < _apagada_hasta:
        return _motivo
    if not any(_vigente(t) for t in _tokens):
        inactivos = sorted({_cuota[t][1] for t in _tokens if not _vigente(t)})
        return ",".join(inactivos) or "cuota_agotada"
    return "sin_lectura_api"   # la API respondio sin placa, o fallo puntualmente


def _recorte(img, bbox, margen=0.35):
    """Recorte MEJORADO para la API: margen 0.35 (el mismo que siempre
    recibio el OCR local; antes a la API le mandabamos solo 0.15) y
    ampliado si queda chico, porque la API lee mejor placas grandes."""
    x, y, w, h = bbox
    mx, my = int(w * margen), int(h * margen)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(img.shape[1], x + w + mx), min(img.shape[0], y + h + my)
    recorte = img[y1:y2, x1:x2]
    if recorte.size and recorte.shape[1] < 320:
        escala = min(4.0, 320.0 / recorte.shape[1])
        recorte = cv2.resize(recorte, None, fx=escala, fy=escala,
                             interpolation=cv2.INTER_CUBIC)
    return recorte


def _con_guion(p: str) -> str:
    return f"{p[:3]}-{p[3:]}" if len(p) == 6 else p


def _post_clave(token: str, buf) -> tuple[int, object]:
    """POST del recorte con UNA clave. Devuelve (estado, dato):
    _EST_OK/data, _EST_VACIO, _EST_AGOTADA/'http_402|403',
    _EST_RATE/'http_429', _EST_RED/'sin_conexion', _EST_ERROR/'fallo'.
    """
    rate = False
    for intento in range(3):
        try:
            r = requests.post(
                URL, headers={"Authorization": f"Token {token}"},
                files={"upload": ("placa.jpg", io.BytesIO(buf.tobytes()),
                                  "image/jpeg")},
                data={"regions": "pe"}, timeout=TIMEOUT)
            if r.status_code in (402, 403):   # cuota agotada o clave invalida
                return _EST_AGOTADA, f"http_{r.status_code}"
            if r.status_code == 429:          # limite de velocidad
                rate = True
                time.sleep(1 + intento)
                continue
            rate = False
            r.raise_for_status()
            data = r.json()
            if data.get("results"):
                return _EST_OK, data
            return _EST_VACIO, None
        except (requests.ConnectionError, requests.Timeout) as e:
            logger.warning("PlateRecognizer sin conexion (%s)", e)
            return _EST_RED, "sin_conexion"
        except (requests.RequestException, ValueError) as e:
            logger.warning("PlateRecognizer fallo (%s), intento %d", e, intento + 1)
            time.sleep(1)
    return (_EST_RATE, "http_429") if rate else (_EST_ERROR, "fallo")


def leer_api(img, bbox):
    """Devuelve el JSON completo de la API, probando cada clave vigente en
    orden hasta que una responda; None si ninguna (cuota de todas, sin red
    o sin lectura)."""
    if not api_disponible():
        return None
    recorte = _recorte(img, bbox)
    if recorte is None or recorte.size == 0:
        return None
    ok, buf = cv2.imencode(".jpg", recorte, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return None
    for token in list(_tokens):
        if not _vigente(token):
            continue
        est, dato = _post_clave(token, buf)
        if est == _EST_OK:
            return dato
        if est == _EST_VACIO:
            return None      # la API hablo y no vio placa: no se rota
        if est == _EST_RED:
            _apagar(SIN_RED_SEG, "sin_conexion")
            return None      # sin red no se rotra: la siguiente clave fallaria igual
        if est == _EST_AGOTADA:
            _cuota[token] = (time.time() + CUOTA_SEG, str(dato))
            logger.warning("Clave AOI %s... sin cuota (%s); probando la siguiente",
                           token[:4] + "***", dato)
            continue
        if est == _EST_RATE:
            _cuota[token] = (time.time() + RATE_SEG, str(dato))
            continue
        return None          # _EST_ERROR: 3 reintentos fallidos
    return None              # todas las claves agotadas


def _elegir(data, es_flota):
    """Primera placa detectada; la flota arbitra entre SUS candidatos."""
    m = data["results"][0]
    placa = str(m.get("plate", "")).upper()
    for c in m.get("candidates", []):
        cand = str(c.get("plate", "")).upper()
        if es_flota(cand):
            placa = cand
            break
    return placa, float(m.get("score", 0.0))


def leer_placa_6_api(img, bbox, **kw):
    """Flujo MODELO-CROPS + API-LECTURA (el que pidio el usuario).

    1. El modelo solo detecta y recorta: manda el recorte MEJORADO (mas
       margen y ampliado si queda chico) a PlateRecognizer.
    2. La API es la lectorA: si responde, su placa manda y queda anotada
       como fuente 'api' (se guarda la imagen con sufijo -Api).
    3. Si la API no responde (sin cuota / sin red / sin placa), se usa el
       motor local (CRNN + RapidOCR) como lectura provisional (-Modelo).
    4. Sin caches ni ventanas de tiempo: "una llamada a la API por
       vehiculo por dia" lo garantiza el worker con el pre-chequeo
       mismo-carro (si el carro ya tiene confirmacion de la API hoy, la
       toma siguiente reutiliza la placa y no se vuelve a llamar).

    El primer elemento de `intentos` indica la fuente: ("api", ...)
    si respondio la API, o los del motor local si quedo provisional.
    """
    import ocr

    # 1)+2) Recorte mejorado a la API: ella lee y decide.
    data = leer_api(img, bbox)
    if data:
        placa, conf = _elegir(data, ocr._en_flota)
        if placa:
            bruta = str(data["results"][0].get("plate", "")).upper()
            final = _con_guion(placa)
            return final, conf, [("api", _con_guion(bruta), conf)]

    # 3) API no respondio: motor local como lectura provisional.
    return ocr.leer_placa_6(img, bbox, **kw)
