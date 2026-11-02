import ocr
import platerecognizer as pr


def _data(placa: str, score: float = 0.9) -> dict:
    return {"results": [{"plate": placa, "score": score, "candidates": []}]}


def _local_falso(placa: str = "LOC-123"):
    def fake(img, bbox, **kw):
        return (placa, 0.5, [("crnn", placa, 0.5)])
    return fake


def test_api_acepta_placa_de_6_alfanumericos(monkeypatch):
    monkeypatch.setattr(pr, "leer_api", lambda img, bbox: _data("ABC123"))

    def no_deberia_llamar(*a, **k):
        raise AssertionError("placa valida no debe caer al motor local")
    monkeypatch.setattr(ocr, "leer_placa_6", no_deberia_llamar)

    placa, conf, intentos = pr.leer_placa_6_api(None, None)
    assert placa == "ABC-123"
    assert conf == 0.9
    assert intentos[0][0] == "api"
    assert intentos[0][1] == "ABC-123"


def test_api_rechaza_placa_distinta_de_6_y_cae_al_motor_local(monkeypatch):
    # La regla del sistema es EXACTAMENTE 6 alfanumericos: si la API
    # devuelve 7 (o 5), la lectura no es forma de placa y hay que
    # delegar en el motor local en vez de guardar basura.
    monkeypatch.setattr(pr, "leer_api", lambda img, bbox: _data("ABC1234"))
    monkeypatch.setattr(ocr, "leer_placa_6", _local_falso())

    placa, _conf, intentos = pr.leer_placa_6_api(None, None)
    assert placa == "LOC-123"
    assert not any(nombre == "api" for nombre, _t, _c in intentos)


def test_api_rechaza_placa_de_5_y_cae_al_motor_local(monkeypatch):
    monkeypatch.setattr(pr, "leer_api", lambda img, bbox: _data("ABC12"))
    monkeypatch.setattr(ocr, "leer_placa_6", _local_falso())

    placa, _conf, intentos = pr.leer_placa_6_api(None, None)
    assert placa == "LOC-123"
    assert not any(nombre == "api" for nombre, _t, _c in intentos)


def test_api_rechaza_placa_todo_digitos_o_todo_letras(monkeypatch):
    # Misma regla que ocr._normalizar: pide letras Y digitos mezclados.
    monkeypatch.setattr(pr, "leer_api", lambda img, bbox: _data("123456"))
    monkeypatch.setattr(ocr, "leer_placa_6", _local_falso("AAA-111"))
    placa, _c, intentos = pr.leer_placa_6_api(None, None)
    assert placa == "AAA-111"
    assert not any(nombre == "api" for nombre, _t, _c in intentos)

    monkeypatch.setattr(pr, "leer_api", lambda img, bbox: _data("ABCDEF"))
    monkeypatch.setattr(ocr, "leer_placa_6", _local_falso("BBB-222"))
    placa, _c, intentos = pr.leer_placa_6_api(None, None)
    assert placa == "BBB-222"
    assert not any(nombre == "api" for nombre, _t, _c in intentos)


def test_sin_lectura_api_delega_en_el_motor_local(monkeypatch):
    monkeypatch.setattr(pr, "leer_api", lambda img, bbox: None)
    monkeypatch.setattr(ocr, "leer_placa_6", _local_falso())

    placa, _conf, intentos = pr.leer_placa_6_api(None, None)
    assert placa == "LOC-123"
    assert intentos[0][0] == "crnn"
