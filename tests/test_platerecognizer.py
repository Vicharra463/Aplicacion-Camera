import ocr
import platerecognizer as pr
import pytest


@pytest.fixture(autouse=True)
def _cache_limpio():
    pr._confirmadas.clear()
    yield
    pr._confirmadas.clear()


def _data(placa: str, score: float = 0.9) -> dict:
    return {"results": [{"plate": placa, "score": score, "candidates": []}]}


def _local_falso(placa: str = "LOC-123"):
    def fake(img, bbox, **kw):
        return (placa, 0.5, [("crnn", placa, 0.5)])
    return fake


def test_local_lee_y_api_confirma(monkeypatch):
    # MODELO-PRIMERO: el local lee la placa y la API la confirma/limpia;
    # la lectura de la API es la que manda.
    monkeypatch.setattr(ocr, "leer_placa_6", _local_falso("ABC-123"))
    llamadas = []

    def espia(img, bbox):
        llamadas.append(1)
        return _data("ABC123")
    monkeypatch.setattr(pr, "leer_api", espia)

    placa, conf, intentos = pr.leer_placa_6_api(None, None)
    assert placa == "ABC-123"
    assert conf == 0.9
    assert intentos[0][0] == "api"
    assert llamadas == [1]


def test_sin_lectura_local_no_llama_la_api(monkeypatch):
    # AHORRO DE CUOTA: si el motor local no leyo nada, la API no se toca.
    def no_llamar(img, bbox):
        raise AssertionError("sin lectura local no debe gastar cuota")
    monkeypatch.setattr(pr, "leer_api", no_llamar)
    monkeypatch.setattr(ocr, "leer_placa_6", _local_falso(""))

    placa, _conf, intentos = pr.leer_placa_6_api(None, None)
    assert placa == ""
    assert intentos[0][0] == "crnn"


def test_cache_de_confirmacion_evita_segunda_llamada(monkeypatch):
    # Rafaga del mismo carro: la confirmacion se reutiliza CONFIRMAR_SEG
    # segundos y no se vuelve a gastar cuota.
    monkeypatch.setattr(ocr, "leer_placa_6", _local_falso("ABC-123"))
    llamadas = []

    def espia(img, bbox):
        llamadas.append(1)
        return _data("ABC123")
    monkeypatch.setattr(pr, "leer_api", espia)

    primera, _c1, i1 = pr.leer_placa_6_api(None, None)
    segunda, _c2, i2 = pr.leer_placa_6_api(None, None)

    assert primera == segunda == "ABC-123"
    assert i1[0][0] == "api" and i2[0][0] == "api"
    assert llamadas == [1]      # solo UNA llamada para las dos fotos


def test_api_rechaza_placa_distinta_de_6_y_se_queda_con_la_local(monkeypatch):
    # La regla del sistema es EXACTAMENTE 6 alfanumericos: si la API
    # devuelve 7 (o 5), su lectura no vale y se queda la del local.
    monkeypatch.setattr(ocr, "leer_placa_6", _local_falso("LOC-123"))
    monkeypatch.setattr(pr, "leer_api", lambda img, bbox: _data("ABC1234"))

    placa, _conf, intentos = pr.leer_placa_6_api(None, None)
    assert placa == "LOC-123"
    assert not any(nombre == "api" for nombre, _t, _c in intentos)


def test_api_rechaza_placa_de_5_y_se_queda_con_la_local(monkeypatch):
    monkeypatch.setattr(ocr, "leer_placa_6", _local_falso("LOC-123"))
    monkeypatch.setattr(pr, "leer_api", lambda img, bbox: _data("ABC12"))

    placa, _conf, intentos = pr.leer_placa_6_api(None, None)
    assert placa == "LOC-123"
    assert not any(nombre == "api" for nombre, _t, _c in intentos)


def test_api_rechaza_placa_todo_digitos_o_todo_letras(monkeypatch):
    # Misma regla que ocr._normalizar: pide letras Y digitos mezclados.
    monkeypatch.setattr(ocr, "leer_placa_6", _local_falso("AAA-111"))
    monkeypatch.setattr(pr, "leer_api", lambda img, bbox: _data("123456"))
    placa, _c, intentos = pr.leer_placa_6_api(None, None)
    assert placa == "AAA-111"
    assert not any(nombre == "api" for nombre, _t, _c in intentos)

    monkeypatch.setattr(ocr, "leer_placa_6", _local_falso("BBB-222"))
    monkeypatch.setattr(pr, "leer_api", lambda img, bbox: _data("ABCDEF"))
    placa, _c, intentos = pr.leer_placa_6_api(None, None)
    assert placa == "BBB-222"
    assert not any(nombre == "api" for nombre, _t, _c in intentos)


def test_api_no_responde_se_queda_con_la_local(monkeypatch):
    # Sin cuota / sin red: la lectura local queda como provisional.
    monkeypatch.setattr(ocr, "leer_placa_6", _local_falso("LOC-123"))
    monkeypatch.setattr(pr, "leer_api", lambda img, bbox: None)

    placa, _conf, intentos = pr.leer_placa_6_api(None, None)
    assert placa == "LOC-123"
    assert intentos[0][0] == "crnn"
