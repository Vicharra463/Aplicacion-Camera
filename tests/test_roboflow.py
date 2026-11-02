import json
from pathlib import Path

import numpy as np
import pytest

import roboflow

FIXTURE = Path(__file__).parent / "fixtures" / "roboflow_result.json"


class _Respuesta:
    def __init__(self, status_code=200, data=None, text=""):
        self.status_code = status_code
        self._data = data
        self.text = text

    def json(self):
        if self._data is None:
            raise ValueError("no es JSON")
        return self._data


class _SesionFalsa:
    """Fake de requests.Session: registra el POST y devuelve/lanza."""

    def __init__(self, respuesta=None, excepcion=None):
        self._respuesta = respuesta
        self._excepcion = excepcion
        self.llamadas = []

    def post(self, url, json=None, timeout=None):
        self.llamadas.append({"url": url, "json": json, "timeout": timeout})
        if self._excepcion is not None:
            raise self._excepcion
        return self._respuesta


@pytest.fixture(autouse=True)
def _limpiar_estado():
    roboflow._sesion = None
    roboflow._workspace = ""
    roboflow._workflow = ""
    roboflow._apagada_hasta = 0.0
    roboflow._motivo = ""
    yield
    roboflow._sesion = None
    roboflow._workspace = ""
    roboflow._workflow = ""
    roboflow._apagada_hasta = 0.0
    roboflow._motivo = ""


def _configurar_con(sesion, workspace="mi_ws", workflow="mi_wf"):
    assert roboflow.configurar("sk_x", workspace, workflow) is True
    roboflow._sesion = sesion
    return sesion


def test_parsea_resultado_real():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    res = roboflow.parsear(data)

    assert len(res) == 1
    bbox, texto, conf = res[0]
    # centro (998.5, 1157) con 225x232 -> esquina (886, 1041)
    assert bbox == (886, 1041, 225, 232)
    assert texto == "AWI-935"
    assert conf == pytest.approx(0.87, abs=0.01)


def test_parsea_vacio_devuelve_lista_vacia():
    assert roboflow.parsear([]) == []
    assert roboflow.parsear([{"predictions": {"predictions": []},
                              "plate_text": []}]) == []


def test_parsea_multi_placa():
    result = [{
        "predictions": {
            "image": {"width": 2560, "height": 1440},
            "predictions": [
                {"x": 500.0, "y": 400.0, "width": 200.0, "height": 200.0,
                 "confidence": 0.9, "class": "License_Plate"},
                {"x": 1500.0, "y": 400.0, "width": 220.0, "height": 210.0,
                 "confidence": 0.8, "class": "License_Plate"},
            ],
        },
        "plate_text": ["ABC-123", "DEF-456"],
    }]
    res = roboflow.parsear(result)

    assert len(res) == 2
    assert res[0][0] == (400, 300, 200, 200)
    assert res[0][1] == "ABC-123"
    assert res[1][0] == (1390, 295, 220, 210)
    assert res[1][1] == "DEF-456"


def test_textos_desalineados_no_atribuye_texto():
    result = [{
        "predictions": {
            "image": {"width": 2560, "height": 1440},
            "predictions": [
                {"x": 500.0, "y": 400.0, "width": 200.0, "height": 200.0,
                 "confidence": 0.9},
                {"x": 1500.0, "y": 400.0, "width": 220.0, "height": 210.0,
                 "confidence": 0.8},
            ],
        },
        "plate_text": ["ABC-123"],
    }]
    res = roboflow.parsear(result)

    assert len(res) == 2
    assert all(texto == "" for _bb, texto, _c in res)


def test_bbox_se_acota_a_la_imagen():
    result = [{
        "predictions": {
            "image": {"width": 100, "height": 100},
            "predictions": [
                {"x": 95.0, "y": 95.0, "width": 40.0, "height": 40.0,
                 "confidence": 0.7},
            ],
        },
        "plate_text": ["XYZ-999"],
    }]
    res = roboflow.parsear(result)

    x, y, w, h = res[0][0]
    assert x >= 0 and y >= 0
    assert x + w <= 100 and y + h <= 100


def test_configurar_requiere_todo():
    assert roboflow.configurar("", "ws", "wf") is False
    assert roboflow.configurar("sk_x", "", "wf") is False
    assert roboflow.configurar("sk_x", "ws", "") is False
    assert roboflow.api_disponible() is False


def test_configurar_ok_arma_sesion_con_bearer():
    assert roboflow.configurar("sk_x", "ws", "wf") is True
    assert roboflow.api_disponible() is True
    assert roboflow._sesion.headers["Authorization"] == "Bearer sk_x"
    # API viva sin lectura -> mismo contrato que platerecognizer
    assert roboflow.motivo_sin_api() == "sin_lectura_api"


def test_sin_sesion_no_disponible():
    assert roboflow.api_disponible() is False
    assert roboflow.detectar_leer(np.zeros((4, 4, 3), np.uint8)) == []


def test_falla_red_devuelve_vacia_y_apaga_global():
    sesion = _configurar_con(_SesionFalsa(
        excepcion=ConnectionError("sin red")))
    img = np.zeros((4, 4, 3), np.uint8)

    assert roboflow.detectar_leer(img) == []
    assert roboflow.api_disponible() is False
    assert "sin_conexion" in roboflow.motivo_sin_api()
    assert len(sesion.llamadas) == 1


def test_falla_api_devuelve_vacia_y_se_recupera():
    _configurar_con(_SesionFalsa(excepcion=RuntimeError("boom interno")))
    img = np.zeros((4, 4, 3), np.uint8)

    assert roboflow.detectar_leer(img) == []
    assert roboflow.api_disponible() is False
    assert "error" in roboflow.motivo_sin_api()


def test_http_500_apaga_sin_lanzar():
    _configurar_con(_SesionFalsa(
        respuesta=_Respuesta(status_code=500, text="oops")))

    assert roboflow.detectar_leer(np.zeros((4, 4, 3), np.uint8)) == []
    assert roboflow.api_disponible() is False
    assert "error HTTP 500" in roboflow.motivo_sin_api()


def test_respuesta_no_json_apaga():
    _configurar_con(_SesionFalsa(
        respuesta=_Respuesta(status_code=200, data=None, text="<html>")))

    assert roboflow.detectar_leer(np.zeros((4, 4, 3), np.uint8)) == []
    assert "respuesta no JSON" in roboflow.motivo_sin_api()


def test_post_usa_url_bearer_y_payload_base64():
    sesion = _configurar_con(
        _SesionFalsa(respuesta=_Respuesta(data={"outputs": []})))
    roboflow.detectar_leer(np.zeros((4, 4, 3), np.uint8))

    assert len(sesion.llamadas) == 1
    llamada = sesion.llamadas[0]
    assert llamada["url"] == ("https://serverless.roboflow.com/"
                              "mi_ws/workflows/mi_wf")
    assert llamada["timeout"] == roboflow.TIMEOUT_SEG
    imagen = llamada["json"]["inputs"]["image"]
    assert imagen["type"] == "base64"
    # base64 CRUDO: sin prefijo data:image (así lo acepta el workflow)
    assert not imagen["value"].startswith("data:")


def test_outputs_se_parsean_como_detecciones():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    outputs = fixture if isinstance(fixture, list) else [fixture]
    _configurar_con(_SesionFalsa(
        respuesta=_Respuesta(data={"outputs": outputs})))

    res = roboflow.detectar_leer(np.zeros((4, 4, 3), np.uint8))

    assert len(res) == 1
    bbox, texto, conf = res[0]
    assert bbox == (886, 1041, 225, 232)
    assert texto == "AWI-935"
    assert conf == pytest.approx(0.87, abs=0.01)
