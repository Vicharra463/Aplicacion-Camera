import time
import base64
import logging

import cv2
import requests

logger = logging.getLogger("cytv.roboflow")
API_URL = "https://serverless.roboflow.com"
SIN_RED_SEG = 120    # sin conexion: la cadena Roboflow 2 min fuera
ERROR_SEG = 300       # error de la API: 5 min fuera
TIMEOUT_SEG = 30      # timeout de la llamada HTTP

_sesion = None
_workspace = ""
_workflow = ""
_apagada_hasta = 0.0   # apagado GLOBAL
_motivo = ""

_CLAVES_RED = ("connect", "timeout", "network", "unreachable",
               "sin red", "dns", "proxy", "temporary failure",
               "read timed out", "max retries")


def configurar(api_key, workspace, workflow_id) -> bool:
    """Arma la sesion HTTPS del workflow. False si falta cualquier dato."""
    global _sesion, _workspace, _workflow
    api_key = str(api_key or "").strip()
    workspace = str(workspace or "").strip()
    workflow_id = str(workflow_id or "").strip()
    if not api_key or not workspace or not workflow_id:
        return False
    sesion = requests.Session()
    sesion.headers.update({
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
    })
    _sesion, _workspace, _workflow = sesion, workspace, workflow_id
    return True


def api_disponible() -> bool:
    return _sesion is not None and time.time() >= _apagada_hasta


def _apagar(segundos, motivo):
    global _apagada_hasta, _motivo
    _apagada_hasta, _motivo = time.time() + segundos, motivo
    logger.warning("Roboflow apagado %d min: %s",
                   max(1, int(segundos // 60)), motivo)


def motivo_sin_api() -> str:
    if _sesion is None:
        return "sin_clave"
    if time.time() < _apagada_hasta:
        return _motivo
    return "sin_lectura_api"    # respondio sin placa, o fallo puntualmente


def _es_error_red(exc) -> bool:
    if isinstance(exc, OSError):
        return True
    texto = str(exc).lower()
    return any(clave in texto for clave in _CLAVES_RED)


def _entero(valor):
    try:
        return int(round(float(valor)))
    except (TypeError, ValueError):
        return 0


def _acotar(v, tope):
    if tope <= 0:
        return max(0, v)
    return max(0, min(v, tope - 1))


def parsear(result) -> list[tuple[tuple[int, int, int, int], str, float]]:
    """Convierte la salida del workflow en [(bbox_esquina, texto, conf)].

    'predictions' trae el centro (x, y) y el tamano (width, height);
    'plate_text' viene en lista paralela a las detecciones. Si el
    desalineo (o falta), el texto queda vacio y la lectura cae en
    PlateRecognizer / motor local.
    """
    salidas: list[tuple[tuple[int, int, int, int], str, float]] = []
    items = result if isinstance(result, list) else [result]
    for item in items:
        if not isinstance(item, dict):
            continue
        pred = item.get("predictions")
        if not isinstance(pred, dict):
            pred = {}
        dims = pred.get("image") if isinstance(pred.get("image"), dict) else {}
        ancho, alto = _entero(dims.get("width")), _entero(dims.get("height"))
        detecciones = pred.get("predictions") or []
        if not isinstance(detecciones, list):
            continue
        textos = item.get("plate_text")
        if isinstance(textos, str):
            textos = [textos]
        if not isinstance(textos, list):
            textos = []
        paralelo = len(textos) == len(detecciones)
        for i, det in enumerate(detecciones):
            if not isinstance(det, dict):
                continue
            w, h = _entero(det.get("width")), _entero(det.get("height"))
            if w <= 0 or h <= 0:
                continue
            x0 = _acotar(_entero(float(det.get("x", 0)) - w / 2), ancho)
            y0 = _acotar(_entero(float(det.get("y", 0)) - h / 2), alto)
            ancho_b, alto_b = w, h
            if ancho > 0:
                ancho_b = min(w, ancho - x0)
            if alto > 0:
                alto_b = min(h, alto - y0)
            if ancho_b <= 0 or alto_b <= 0:
                continue
            texto = ""
            if paralelo and textos[i] is not None:
                texto = str(textos[i]).strip().upper()
            try:
                conf = float(det.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            salidas.append(((x0, y0, ancho_b, alto_b), texto, conf))
    return salidas


def detectar_leer(img) -> list[tuple[tuple[int, int, int, int], str, float]]:
    """Full image -> detecciones + texto en un solo POST HTTPS.

    Nunca lanza: red/API fallida devuelve [] y apaga la cadena un rato.
    """
    if not api_disponible():
        return []
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return []
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    url = f"{API_URL}/{_workspace}/workflows/{_workflow}"
    payload = {"inputs": {"image": {"type": "base64", "value": b64}}}
    try:
        resp = _sesion.post(url, json=payload, timeout=TIMEOUT_SEG)
    except Exception as exc:
        if _es_error_red(exc):
            _apagar(SIN_RED_SEG, "sin_conexion")
        else:
            _apagar(ERROR_SEG, "error: " + str(exc)[:60])
        logger.warning("Roboflow fallo: %s", str(exc)[:120])
        return []
    if resp.status_code != 200:
        _apagar(ERROR_SEG, f"error HTTP {resp.status_code}")
        logger.warning("Roboflow HTTP %s: %s",
                       resp.status_code, resp.text[:120])
        return []
    try:
        data = resp.json()
    except ValueError:
        _apagar(ERROR_SEG, "respuesta no JSON")
        logger.warning("Roboflow devolvio algo que no es JSON")
        return []
    salidas = data.get("outputs") if isinstance(data, dict) else data
    if not isinstance(salidas, list):
        salidas = []
    return parsear(salidas)
