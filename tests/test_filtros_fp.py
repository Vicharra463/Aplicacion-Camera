"""Falsos positivos: gate de color + barrido OCR gratis antes de la API."""
import numpy as np

import ocr
import platerecognizer as pr
import worker


# ---------------------------------------------------------------- color ---

def test_fraccion_rojo_detecta_cinta_reflectora():
    # Rojo puro tipo cinta: mucha fraccion (> umbral 15%)
    cinta = np.full((55, 46, 3), (0, 0, 255), np.uint8)
    assert worker._fraccion_rojo(cinta) > 15.0


def test_fraccion_rojo_placa_blanca_es_casi_cero():
    # Fondo blanco de placa con "texto" oscuro: casi nada de rojo
    placa = np.full((90, 140, 3), (240, 240, 240), np.uint8)
    placa[20:70, 15:125] = (30, 30, 30)
    assert worker._fraccion_rojo(placa) < 15.0


def test_fraccion_rojo_entre_los_dos_casos():
    # Media imagen roja -> mitad: tambien cae del lado de "no placa"
    mixta = np.full((40, 40, 3), (240, 240, 240), np.uint8)
    mixta[:, :20] = (0, 0, 255)
    fraccion = worker._fraccion_rojo(mixta)
    assert 40.0 < fraccion < 60.0
    assert fraccion > 15.0


# -------------------------------------------------------------- barrido ---

class _MotorFalso:
    def __init__(self, resultado=None, falla=False):
        self._resultado = resultado or []
        self._falla = falla

    def __call__(self, _crop):
        if self._falla:
            raise RuntimeError("motor roto")
        return self._resultado, 0.05


def test_hay_texto_true_con_lectura(monkeypatch):
    monkeypatch.setattr(ocr, "_obtener_motor",
                        lambda: _MotorFalso([(("0", "0", "1", "1"),
                                              "A8R-635", 0.9)]))
    img = np.zeros((60, 80, 3), np.uint8)
    assert ocr.hay_texto(img, (5, 5, 40, 30)) is True


def test_hay_texto_false_sin_lectura(monkeypatch):
    monkeypatch.setattr(ocr, "_obtener_motor", lambda: _MotorFalso([]))
    img = np.zeros((60, 80, 3), np.uint8)
    assert ocr.hay_texto(img, (5, 5, 40, 30)) is False


def test_hay_texto_fail_open_si_el_motor_falla(monkeypatch):
    monkeypatch.setattr(ocr, "_obtener_motor",
                        lambda: _MotorFalso(falla=True))
    img = np.zeros((60, 80, 3), np.uint8)
    assert ocr.hay_texto(img, (5, 5, 40, 30)) is True


def test_hay_texto_fail_open_sin_imagen():
    assert ocr.hay_texto(None, (1, 1, 10, 10)) is True
    assert ocr.hay_texto(np.zeros((0, 0, 3), np.uint8), (1, 1, 10, 10)) is True


# ------------------------------------- pre-barrido antes de PlateRecognizer

def test_api_no_se_gasta_si_no_hay_texto(monkeypatch):
    llamadas = {"api": 0}

    def _leer_api(_img, _bbox):
        llamadas["api"] += 1
        return {"results": [{"plate": "A8R-635"}]}

    monkeypatch.setattr(ocr, "hay_texto",
                        lambda img, bbox, margen=0.15: False)
    monkeypatch.setattr(pr, "leer_api", _leer_api)
    monkeypatch.setattr(ocr, "leer_placa_6",
                        lambda img, bbox, **kw: ("", 0.0, []))

    res = pr.leer_placa_6_api(None, None)

    assert llamadas["api"] == 0          # la cinta no gasto cuota
    assert res == ("", 0.0, [])


def test_api_se_llama_cuando_si_hay_texto(monkeypatch):
    llamadas = {"api": 0}

    def _leer_api(_img, _bbox):
        llamadas["api"] += 1
        return None                       # API sin placa -> local

    monkeypatch.setattr(ocr, "hay_texto",
                        lambda img, bbox, margen=0.15: True)
    monkeypatch.setattr(pr, "leer_api", _leer_api)
    monkeypatch.setattr(ocr, "leer_placa_6",
                        lambda img, bbox, **kw: ("ABC-123", 0.7,
                                                 [("local", "ABC-123", 0.7)]))

    res = pr.leer_placa_6_api(None, None)

    assert llamadas["api"] == 1
    assert res[0] == "ABC-123"


# ------------------------------------------------- gate ancho lateral ---

def test_gate_ancho_placa_real_pasa():
    # GT-29: min 0.037, max 0.082 del ancho de imagen
    assert worker._pasa_gate_ancho(80, 1920, 0.20) is True   # 0.042
    assert worker._pasa_gate_ancho(157, 1920, 0.20) is True  # 0.082 (max GT)


def test_gate_ancho_texto_lateral_falla():
    # Texto lateral omnibus/camion medido: 0.24-0.46
    assert worker._pasa_gate_ancho(460, 1920, 0.20) is False  # 0.24
    assert worker._pasa_gate_ancho(883, 1920, 0.20) is False  # 0.46


def test_gate_ancho_en_el_limite():
    # Umbral estricto: 0.20 exacto NO pasa (>= umbral falla)
    assert worker._pasa_gate_ancho(384, 1920, 0.20) is False  # 0.20
    assert worker._pasa_gate_ancho(383, 1920, 0.20) is True   # 0.1995


def test_gate_ancho_ancho_invalido():
    # Division por cero / imagen sin ancho -> no pasa
    assert worker._pasa_gate_ancho(100, 0, 0.20) is False


# ------------------------------------------- lectura lateral (dedup) ---

def test_lectura_lateral_texto_cercano_hash_lejano_es_lateral():
    # CSV-990 vs C8V-990: dist texto 1, hash lejano -> lateral
    assert worker._es_lectura_lateral(
        0xFFFF0000FFFF0000, "C8V-990",
        0x0000FFFF0000FFFF, "CSV-990", umbral=18) is True


def test_lectura_lateral_texto_igual_no_es_lateral():
    # Misma placa, mismo texto (hash cercano o no): MEJOR FOTO legitimo
    assert worker._es_lectura_lateral(
        0xFFFF0000FFFF0000, "BDY-412",
        0x0000FFFF0000FFFF, "BDY-412", umbral=18) is False


def test_lectura_lateral_hash_cercano_no_es_lateral():
    # Texto distinto pero hash cercano = misma foto, otra lectura OCR
    # (ej. BLN-7U5 -> BLN-703): permite MEJOR FOTO
    h = 0x0000000000000000
    h2 = h ^ (1 << 5)  # 5 bits de distancia <= 18
    assert worker._es_lectura_lateral(
        h2, "BLN-703", h, "BLN-7U5", umbral=18) is False


def test_lectura_lateral_texto_vacio_no_es_lateral():
    assert worker._es_lectura_lateral(
        0xFFFF, "", 0x0000, "CSV-990", umbral=18) is False
    assert worker._es_lectura_lateral(
        0xFFFF, "C8V-990", 0x0000, "", umbral=18) is False


def test_lectura_lateral_hash_en_el_limite():
    # 18 bits exactos NO es lateral (umbral estricto >)
    h = 0
    h2 = sum(1 << i for i in range(18))  # exactamente 18 bits
    assert worker._es_lectura_lateral(
        h2, "ABC-123", h, "ABC-124", umbral=18) is False
    h3 = sum(1 << i for i in range(19))  # 19 bits -> lateral
    assert worker._es_lectura_lateral(
        h3, "ABC-123", h, "ABC-124", umbral=18) is True
