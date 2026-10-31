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
CONFIRMAR_SEG = 30.0  # misma placa: no volver a llamar a la API en este lapso

# Claves en orden de prueba. Cada fuente es una cadena 'a, b, c'
# (config.ini [aoi] tokens y/o PLATERECOGNIZER_TOKEN).
# NO se commitean claves aca: se cargan por config o variable de entorno.
_tokens: list[str] = []
# Claves apartadas por cuota/rate: clave -> (hasta_ts, motivo)
_cuota: dict[str, tuple[float, str]] = {}
_apagada_hasta = 0.0   # apagado GLOBAL (solo errores de red)
_motivo = ""

# Caché de confirmaciones: texto local normalizado -> (ts, placa_api, conf_api).
# Si la API ya confirmo esta misma placa hace menos de CONFIRMAR_SEG, no se
# vuelve a llamar (evita quemar cuota con rafagas del mismo carro).
_confirmadas: dict[str, tuple[float, str, float]] = {}

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


def _recorte(img, bbox, margen=0.15):
    x, y, w, h = bbox
    mx, my = int(w * margen), int(h * margen)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(img.shape[1], x + w + mx), min(img.shape[0], y + h + my)
    return img[y1:y2, x1:x2]


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
    """Flujo MODELO-PRIMERO (ahorra cuota de la API).

    1. El motor local (CRNN + RapidOCR) lee el recorte. Su _normalizar solo
       devuelve texto si es una placa VALIDA de 6 digitos (3 letras + 3
       digitos); si no, devuelve vacio.
    2. Solo si el local ya leyo una placa limpia se consulta la API para
       confirmar/limpiarla. Si el local no leyo, NO se gasta cuota.
    3. Si la API ya confirmo ESTA MISMA placa hace menos de CONFIRMAR_SEG,
       se reutiliza esa confirmacion sin volver a llamar (evita quemar
       cuota con rafagas del mismo carro en la ventana de dedup).
    4. Si la API responde, manda su lectura (es el verificador). Si no
       responde, se queda la del local como provisional.

    El primer elemento de `intentos` indica la fuente: ("api", ...)
    si confirmo la API, o los del motor local si quedo provisional.
    """
    import ocr

    # 1) Motor local primero: detecta y lee la placa.
    texto_local, conf_local, intentos_local = ocr.leer_placa_6(img, bbox, **kw)

    # 2) Sin lectura local limpia => no se toca la API (ahorro de cuota).
    if not texto_local:
        return texto_local, conf_local, intentos_local

    # 3) Caché: si la API ya confirmo esta misma placa hace poco, no llamar.
    clave = texto_local.replace("-", "").upper()
    hit = _confirmadas.get(clave)
    if hit is not None:
        ts, placa_api, conf_api = hit
        if time.time() - ts < CONFIRMAR_SEG:
            return placa_api, conf_api, [("api", placa_api, conf_api)]
        _confirmadas.pop(clave, None)

    # 4) El local ya tiene placa de 6 digitos: la API confirma/limpia.
    data = leer_api(img, bbox)
    if data:
        placa, conf = _elegir(data, ocr._en_flota)
        if placa:
            bruta = str(data["results"][0].get("plate", "")).upper()
            final = _con_guion(placa)
            _confirmadas[clave] = (time.time(), final, conf)
            return final, conf, [("api", _con_guion(bruta), conf)]

    # API no confirmo (sin cuota / sin red / sin placa): local provisional.
    return texto_local, conf_local, intentos_local
