import ocr
import platerecognizer as pr
import pytest


def _data(placa: str, score: float = 0.9) -> dict:
    return {"results": [{"plate": placa, "score": score, "candidates": []}]}


def _local_falso(placa: str = "LOC-123"):
    def fake(img, bbox, **kw):
        return (placa, 0.5, [("crnn", placa, 0.5)])
    return fake


def _local_prohibido(img, bbox, **kw):
    raise AssertionError("con la API respondiendo no debe usarse el local")


def test_api_primero_responde_y_manda(monkeypatch):
    # MODELO-CROPS + API-LECTURA: la API es la lectora; si responde con una
    # placa valida, su lectura manda (fuente 'api') y el local ni se toca.
    llamadas = []

    def espia(img, bbox):
        llamadas.append(1)
        return _data("ABC123")
    monkeypatch.setattr(pr, "leer_api", espia)
    monkeypatch.setattr(ocr, "leer_placa_6", _local_prohibido)

    placa, conf, intentos = pr.leer_placa_6_api(None, None)
    assert placa == "ABC-123"
    assert conf == 0.9
    assert intentos[0][0] == "api"
    assert llamadas == [1]


def test_api_sin_respuesta_usa_el_local(monkeypatch):
    # Sin cuota / sin red / sin placa: la API no respondio y el motor
    # local (CRNN + RapidOCR) queda como lectura provisional (-Modelo).
    monkeypatch.setattr(pr, "leer_api", lambda img, bbox: None)
    monkeypatch.setattr(ocr, "leer_placa_6", _local_falso("LOC-123"))

    placa, _conf, intentos = pr.leer_placa_6_api(None, None)
    assert placa == "LOC-123"
    assert intentos[0][0] == "crnn"


def test_sin_cache_cada_toma_llama_a_la_api(monkeypatch):
    # Sin caches ni ventanas de tiempo en este nivel: desde 2026-11-10
    # TODA foto que se guarda paso por la cadena (el worker no reutiliza
    # lecturas: pre-chequeo por hash y _api_confirmada_hoy eliminados),
    # asi que aca tampoco hay cache.
    llamadas = []

    def espia(img, bbox):
        llamadas.append(1)
        return _data("ABC123")
    monkeypatch.setattr(pr, "leer_api", espia)
    monkeypatch.setattr(ocr, "leer_placa_6", _local_prohibido)

    primera, _c1, i1 = pr.leer_placa_6_api(None, None)
    segunda, _c2, i2 = pr.leer_placa_6_api(None, None)

    assert primera == segunda == "ABC-123"
    assert i1[0][0] == "api" and i2[0][0] == "api"
    assert llamadas == [1, 1]     # dos tomas, dos llamadas: sin cache


def test_api_rechaza_placa_distinta_de_6_y_cae_al_local(monkeypatch):
    # La regla del sistema es EXACTAMENTE 6 alfanumericos: si la API
    # devuelve 7 (o 5), su lectura no vale y se cae al motor local.
    monkeypatch.setattr(pr, "leer_api", lambda img, bbox: _data("ABC1234"))
    monkeypatch.setattr(ocr, "leer_placa_6", _local_falso("LOC-123"))

    placa, _conf, intentos = pr.leer_placa_6_api(None, None)
    assert placa == "LOC-123"
    assert not any(nombre == "api" for nombre, _t, _c in intentos)


def test_api_rechazada_se_marca_en_intentos(monkeypatch):
    # La traza de rechazadas_api necesita SABER que la API fue consultada
    # y no confirmo (el worker la copia a rechazadas_api\ para diagnostico).
    # El marcador va AL FINAL: no cambia la fuente ni el orden de lectura.
    monkeypatch.setattr(pr, "leer_api", lambda img, bbox: None)
    monkeypatch.setattr(ocr, "leer_placa_6", _local_falso("LOC-123"))

    placa, _conf, intentos = pr.leer_placa_6_api(None, None)

    assert placa == "LOC-123"
    assert intentos[-1][0] == "api_rechazada"
    assert not any(nombre == "api" for nombre, _t, _c in intentos)


def test_api_rechazada_placa_invalida_tambien_se_marca(monkeypatch):
    monkeypatch.setattr(pr, "leer_api",
                        lambda img, bbox: _data("ABC1234"))
    monkeypatch.setattr(ocr, "leer_placa_6", _local_falso("LOC-123"))

    _placa, _conf, intentos = pr.leer_placa_6_api(None, None)

    assert intentos[-1][0] == "api_rechazada"


def test_sin_barrido_no_se_marca_rechazo(monkeypatch):
    # Si ni el barrido gratis ve texto, la API NUNCA se llamo: no hay
    # rechazo que registrar (no se gasto cuota).
    monkeypatch.setattr(ocr, "hay_texto", lambda img, bbox, **k: False)
    monkeypatch.setattr(pr, "leer_api",
                        lambda img, bbox: pytest.fail("no debe llamar"))
    monkeypatch.setattr(ocr, "leer_placa_6", _local_falso("LOC-123"))

    _placa, _conf, intentos = pr.leer_placa_6_api(None, None)

    assert not any(n == "api_rechazada" for n, _t, _c in intentos)


def test_api_rechaza_placa_de_5_y_cae_al_local(monkeypatch):
    monkeypatch.setattr(pr, "leer_api", lambda img, bbox: _data("ABC12"))
    monkeypatch.setattr(ocr, "leer_placa_6", _local_falso("LOC-123"))

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


def test_api_con_placa_invalida_y_local_tambien_sin_lectura(monkeypatch):
    # API invalida + local sin lectura: no hay placa (el worker la
    # descartara como texto_invalido si ocr_requerido esta en 'si').
    monkeypatch.setattr(pr, "leer_api", lambda img, bbox: _data("ABC1234"))
    monkeypatch.setattr(ocr, "leer_placa_6", _local_falso(""))

    placa, _conf, intentos = pr.leer_placa_6_api(None, None)
    assert placa == ""
    assert not any(nombre == "api" for nombre, _t, _c in intentos)

