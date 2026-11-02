import json
from pathlib import Path

import numpy as np
import pytest

import roboflow

FIXTURE = Path(__file__).parent / "fixtures" / "roboflow_result.json"


@pytest.fixture(autouse=True)
def _limpiar_estado():
    roboflow._cliente = None
    roboflow._workspace = ""
    roboflow._workflow = ""
    roboflow._apagada_hasta = 0.0
    roboflow._motivo = ""
    yield
    roboflow._cliente = None
    roboflow._workspace = ""
    roboflow._workflow = ""
    roboflow._apagada_hasta = 0.0
    roboflow._motivo = ""


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


def test_configurar_ok():
    assert roboflow.configurar("sk_x", "ws", "wf") is True
    assert roboflow.api_disponible() is True
    # API viva sin lectura -> mismo contrato que platerecognizer
    assert roboflow.motivo_sin_api() == "sin_lectura_api"


def test_sin_cliente_no_disponible():
    assert roboflow.api_disponible() is False
    assert roboflow.detectar_leer(np.zeros((4, 4, 3), np.uint8)) == []


def test_falla_red_devuelve_vacia_y_apaga_global(monkeypatch):
    assert roboflow.configurar("sk_x", "ws", "wf") is True

    class Boom:
        def run_workflow(self, **_kw):
            raise ConnectionError("sin red")

    monkeypatch.setattr(roboflow, "_cliente", Boom())
    img = np.zeros((4, 4, 3), np.uint8)

    assert roboflow.detectar_leer(img) == []
    assert roboflow.api_disponible() is False
    assert "sin_conexion" in roboflow.motivo_sin_api()


def test_falla_api_devuelve_vacia_y_se_recupera(monkeypatch):
    assert roboflow.configurar("sk_x", "ws", "wf") is True

    class Boom:
        def run_workflow(self, **_kw):
            raise RuntimeError("HTTP 500")

    monkeypatch.setattr(roboflow, "_cliente", Boom())
    img = np.zeros((4, 4, 3), np.uint8)

    assert roboflow.detectar_leer(img) == []
    assert roboflow.api_disponible() is False
    assert "error" in roboflow.motivo_sin_api()


def test_usa_workspace_y_workflow_configurados(monkeypatch):
    assert roboflow.configurar("sk_x", "mi_ws", "mi_wf") is True
    llamada = {}

    class Espia:
        def run_workflow(self, **kw):
            llamada.update(kw)
            return []

    monkeypatch.setattr(roboflow, "_cliente", Espia())
    roboflow.detectar_leer(np.zeros((4, 4, 3), np.uint8))

    assert llamada["workspace_name"] == "mi_ws"
    assert llamada["workflow_id"] == "mi_wf"
    assert llamada["images"]["image"].startswith("data:image/jpeg;base64,")
