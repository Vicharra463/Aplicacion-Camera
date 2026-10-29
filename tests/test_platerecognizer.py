"""Tests del adaptador AOI (PlateRecognizer): parseo y fallback local."""
import numpy as np
import pytest

import platerecognizer


class _Respuesta:
    def __init__(self, status=200, data=None):
        self.status_code = status
        self._data = data if data is not None else {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return self._data


@pytest.fixture
def img_bbox():
    return np.zeros((80, 240, 3), dtype=np.uint8), (10, 10, 120, 40)


def test_sin_token_devuelve_none(monkeypatch, img_bbox):
    monkeypatch.setattr(platerecognizer, "TOKEN", "")
    img, bbox = img_bbox
    assert platerecognizer.leer_api(img, bbox) is None


def test_parsea_resultado_ok(monkeypatch, img_bbox):
    monkeypatch.setattr(platerecognizer, "TOKEN", "tok")
    data = {"results": [{"plate": "ABC123", "score": 0.91,
                         "candidates": [{"plate": "ABC123", "score": 0.91},
                                        {"plate": "AB0123", "score": 0.40}]}],
            "processing_time": 0.5}
    monkeypatch.setattr(platerecognizer.requests, "post",
                        lambda *a, **k: _Respuesta(200, data))
    img, bbox = img_bbox
    r = platerecognizer.leer_api(img, bbox)
    assert r is not None
    assert r["results"][0]["plate"] == "ABC123"
    assert r["results"][0]["score"] == pytest.approx(0.91)


def test_sin_resultados_devuelve_none(monkeypatch, img_bbox):
    monkeypatch.setattr(platerecognizer, "TOKEN", "tok")
    monkeypatch.setattr(platerecognizer.requests, "post",
                        lambda *a, **k: _Respuesta(200, {"results": []}))
    img, bbox = img_bbox
    assert platerecognizer.leer_api(img, bbox) is None


def test_reintenta_en_429_y_devuelve(monkeypatch, img_bbox):
    monkeypatch.setattr(platerecognizer, "TOKEN", "tok")
    monkeypatch.setattr(platerecognizer.time, "sleep", lambda s: None)
    data = {"results": [{"plate": "XYZ789", "score": 0.80}]}
    llamadas = []

    def _post(*a, **k):
        llamadas.append(1)
        if len(llamadas) < 2:
            return _Respuesta(429)
        return _Respuesta(200, data)

    monkeypatch.setattr(platerecognizer.requests, "post", _post)
    img, bbox = img_bbox
    texto, conf, intentos = platerecognizer.leer_placa_6_api(img, bbox)
    assert texto == "XYZ-789"
    assert intentos[0][0] == "api"
    assert len(llamadas) == 2


def test_falla_red_devuelve_none(monkeypatch, img_bbox):
    monkeypatch.setattr(platerecognizer, "TOKEN", "tok")
    monkeypatch.setattr(platerecognizer.time, "sleep", lambda s: None)

    def _post(*a, **k):
        raise platerecognizer.requests.RequestException("sin red")

    monkeypatch.setattr(platerecognizer.requests, "post", _post)
    img, bbox = img_bbox
    assert platerecognizer.leer_api(img, bbox) is None


def test_leer_placa_6_api_fallback_local(monkeypatch, img_bbox):
    llamado = {}

    def _local(img, bbox, **kw):
        llamado["si"] = True
        return "AAA-111", 0.5, [("crnn", "AAA-111", 0.5)]

    import ocr
    monkeypatch.setattr(ocr, "leer_placa_6", _local)
    monkeypatch.setattr(platerecognizer, "TOKEN", "")
    img, bbox = img_bbox
    texto, conf, cands = platerecognizer.leer_placa_6_api(img, bbox)
    assert llamado.get("si") is True
    assert texto == "AAA-111"


def test_leer_placa_6_api_prefiere_api(monkeypatch, img_bbox):
    import ocr
    monkeypatch.setattr(platerecognizer, "TOKEN", "tok")
    monkeypatch.setattr(ocr, "_en_flota", lambda t: t.upper().replace("-", "")
                        == "ABC123")
    data = {"results": [{"plate": "ABO123", "score": 0.90,
                         "candidates": [{"plate": "ABO123", "score": 0.90},
                                        {"plate": "ABC123", "score": 0.70}]}]}
    monkeypatch.setattr(platerecognizer.requests, "post",
                        lambda *a, **k: _Respuesta(200, data))
    img, bbox = img_bbox
    texto, conf, cands = platerecognizer.leer_placa_6_api(img, bbox)
    assert texto == "ABC-123"
    assert cands[0][0] == "api"
