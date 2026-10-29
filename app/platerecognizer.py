import io
import os
import time
import logging

import cv2
import requests

logger = logging.getLogger("cytv.platerecognizer")
URL = "https://api.platerecognizer.com/v1/plate-reader/"
TOKEN = os.environ.get("PLATERECOGNIZER_TOKEN", "")   # nunca un valor por defecto
TIMEOUT = 10

_apagada_hasta = 0.0
_motivo = ""


def api_disponible() -> bool:
    return bool(TOKEN) and time.time() >= _apagada_hasta


def _apagar(segundos, motivo):
    global _apagada_hasta, _motivo
    _apagada_hasta, _motivo = time.time() + segundos, motivo
    logger.warning("API apagada %d min: %s", max(1, int(segundos // 60)), motivo)


def motivo_sin_api() -> str:
    if not TOKEN:
        return "sin_token"
    if time.time() < _apagada_hasta:
        return _motivo
    return "sin_lectura_api"      # la API respondio sin placa, o fallo puntualmente


def _recorte(img, bbox, margen=0.15):
    x, y, w, h = bbox
    mx, my = int(w * margen), int(h * margen)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(img.shape[1], x + w + mx), min(img.shape[0], y + h + my)
    return img[y1:y2, x1:x2]


def _con_guion(p: str) -> str:
    return f"{p[:3]}-{p[3:]}" if len(p) == 6 else p


def leer_api(img, bbox):
    """Devuelve la respuesta JSON completa de la API, o None si falla / no hay lectura."""
    if not api_disponible():
        return None
    recorte = _recorte(img, bbox)
    if recorte is None or recorte.size == 0:
        return None
    ok, buf = cv2.imencode(".jpg", recorte, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return None
    for intento in range(3):
        try:
            r = requests.post(
                URL, headers={"Authorization": f"Token {TOKEN}"},
                files={"upload": ("placa.jpg", io.BytesIO(buf.tobytes()),
                                  "image/jpeg")},
                data={"regions": "pe"}, timeout=TIMEOUT)
            if r.status_code in (402, 403):      # cuota agotada o token invalido (verificalo)
                _apagar(3600, f"http_{r.status_code}")
                return None
            if r.status_code == 429:             # limite de velocidad: temporal
                time.sleep(1 + intento)
                continue
            r.raise_for_status()
            data = r.json()
            return data if data.get("results") else None
        except (requests.ConnectionError, requests.Timeout) as e:
            logger.warning("PlateRecognizer sin conexion (%s)", e)
            _apagar(120, "sin_conexion")         # el motor local trabaja 2 min sin reintentar
            return None
        except (requests.RequestException, ValueError) as e:
            logger.warning("PlateRecognizer fallo (%s), intento %d", e, intento + 1)
            time.sleep(1)
    return None


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
    """Misma firma que ocr.leer_placa_6. El primer elemento de `intentos`
    indica la fuente: ("api", lectura_cruda, conf) o los del motor local."""
    import ocr
    data = leer_api(img, bbox)
    if data:
        placa, conf = _elegir(data, ocr._en_flota)
        if placa:
            bruta = str(data["results"][0].get("plate", "")).upper()
            return _con_guion(placa), conf, [("api", _con_guion(bruta), conf)]
    return ocr.leer_placa_6(img, bbox, **kw)     # provisional: motor local