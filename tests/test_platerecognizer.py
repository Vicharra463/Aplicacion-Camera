"""Tests del adaptador AOI (PlateRecognizer): parseo, rotacion de claves
y fallback local."""
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


@pytest.fixture(autouse=True)
def estado_limpio(monkeypatch):
    """Snapshot+restore de los globales de estado por test."""
    monkeypatch.setattr(platerecognizer, "_tokens", [])
    monkeypatch.setattr(platerecognizer, "_cuota", {})
    monkeypatch.setattr(platerecognizer, "_apagada_hasta", 0.0)
    monkeypatch.setattr(platerecognizer, "_motivo", "")


@pytest.fixture
def img_bbox():
    return np.zeros((80, 240, 3), dtype=np.uint8), (10, 10, 120, 40)


def test_sin_token_devuelve_none(img_bbox):
    platerecognizer.configurar("")
    img, bbox = img_bbox
    assert platerecognizer.leer_api(img, bbox) is None


def test_configurar_limpia_y_dedupe():
    tokens = platerecognizer.configurar(" a, b ,a,", "b,c")
    assert tokens == ["a", "b", "c"]
    assert platerecognizer._tokens == ["a", "b", "c"]


def test_parsea_resultado_ok(monkeypatch, img_bbox):
    platerecognizer.configurar("tok")
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
    platerecognizer.configurar("tok")
    monkeypatch.setattr(platerecognizer.requests, "post",
                        lambda *a, **k: _Respuesta(200, {"results": []}))
    img, bbox = img_bbox
    assert platerecognizer.leer_api(img, bbox) is None


def test_reintenta_en_429_y_devuelve(monkeypatch, img_bbox):
    platerecognizer.configurar("tok")
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


def test_falla_red_devuelve_none_y_apaga_global(monkeypatch, img_bbox):
    platerecognizer.configurar("tok1,tok2")
    monkeypatch.setattr(platerecognizer.time, "sleep", lambda s: None)
    llamadas = []

    def _post(*a, **k):
        llamadas.append(1)
        raise platerecognizer.requests.ConnectionError("sin red")

    monkeypatch.setattr(platerecognizer.requests, "post", _post)
    img, bbox = img_bbox
    assert platerecognizer.leer_api(img, bbox) is None
    assert len(llamadas) == 1          # sin red no rota: el 2do token no se prueba
    assert platerecognizer.api_disponible() is False


def test_rotacion_402_usa_siguiente_clave(monkeypatch, img_bbox):
    platerecognizer.configurar("tok1,tok2")
    data = {"results": [{"plate": "DEF456", "score": 0.88}]}
    autorizaciones = []

    def _post(url, headers=None, **k):
        autorizaciones.append((headers or {}).get("Authorization"))
        if len(autorizaciones) == 1:
            return _Respuesta(402)
        return _Respuesta(200, data)

    monkeypatch.setattr(platerecognizer.requests, "post", _post)
    img, bbox = img_bbox
    r = platerecognizer.leer_api(img, bbox)
    assert r is not None
    assert r["results"][0]["plate"] == "DEF456"
    assert len(autorizaciones) == 2
    assert autorizaciones[0].endswith("tok1")
    assert autorizaciones[1].endswith("tok2")


def test_clave_agotada_queda_fuera_y_no_repite(monkeypatch, img_bbox):
    platerecognizer.configurar("tok1,tok2")
    llamadas = []

    def _post(url, headers=None, **k):
        autoriza = (headers or {}).get("Authorization", "")
        llamadas.append(autoriza)
        return _Respuesta(402)

    monkeypatch.setattr(platerecognizer.requests, "post", _post)
    img, bbox = img_bbox
    assert platerecognizer.leer_api(img, bbox) is None
    assert len(llamadas) == 2           # una por clave
    # las dos quedaron en cuota: ya no hay claves vigentes
    assert platerecognizer.api_disponible() is False
    # siguiente foto: cero llamadas HTTP, directo al motor local
    assert platerecognizer.leer_api(img, bbox) is None
    assert len(llamadas) == 2


def test_403_tambien_agota_la_clave(monkeypatch, img_bbox):
    platerecognizer.configurar("mala")
    llamadas = []
    data = {"results": [{"plate": "GHI789", "score": 0.7}]}

    def _post(*a, **k):
        llamadas.append(1)
        return _Respuesta(403 if len(llamadas) == 1 else 200, data)

    monkeypatch.setattr(platerecognizer.requests, "post", _post)
    img, bbox = img_bbox
    assert platerecognizer.leer_api(img, bbox) is None
    assert platerecognizer.motivo_sin_api() == "http_403"


def test_todas_agotadas_motivo_cuota(monkeypatch, img_bbox):
    platerecognizer.configurar("a,b")
    monkeypatch.setattr(platerecognizer.requests, "post",
                        lambda *a, **k: _Respuesta(402))
    img, bbox = img_bbox
    assert platerecognizer.leer_api(img, bbox) is None
    assert platerecognizer.motivo_sin_api() == "http_402"


def test_leer_placa_6_api_fallback_local(monkeypatch, img_bbox):
    llamado = {}

    def _local(img, bbox, **kw):
        llamado["si"] = True
        return "AAA-111", 0.5, [("crnn", "AAA-111", 0.5)]

    import ocr
    monkeypatch.setattr(ocr, "leer_placa_6", _local)
    platerecognizer.configurar("")
    img, bbox = img_bbox
    texto, conf, cands = platerecognizer.leer_placa_6_api(img, bbox)
    assert llamado.get("si") is True
    assert texto == "AAA-111"


def test_leer_placa_6_api_prefiere_api(monkeypatch, img_bbox):
    import ocr
    platerecognizer.configurar("tok")
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


def test_modelo_interviene_solo_tras_agotar_todas(monkeypatch, img_bbox):
    """La rotacion completa de claves ocurre ANTES del motor local."""
    import ocr
    llamadas = {"http": 0, "local": 0}

    def _post(*a, **k):
        llamadas["http"] += 1
        return _Respuesta(402)

    def _local(img, bbox, **kw):
        llamadas["local"] += 1
        return "ZZZ-999", 0.4, [("crnn", "ZZZ-999", 0.4)]

    monkeypatch.setattr(platerecognizer.requests, "post", _post)
    monkeypatch.setattr(ocr, "leer_placa_6", _local)
    platerecognizer.configurar("k1,k2,k3")
    img, bbox = img_bbox
    texto, _conf, _cands = platerecognizer.leer_placa_6_api(img, bbox)
    assert llamadas["http"] == 3
    assert llamadas["local"] == 1
    assert texto == "ZZZ-999"
