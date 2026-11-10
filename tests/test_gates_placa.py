"""Gates anti-fantasma (2026-11-06): formato peruano de placa,
confianza minima de la deteccion y cooldown POR EVENTO (rafaga).

El problema: un mismo carro generaba varias "placas fantasmas" (cartel
lateral de micro 5TC-000, placard de explosivos UN1-910, sticker de
camioneta WO3-330) y el cooldown existente (cooldown_placa_seg) era
solo una pausa de sueno, no un filtro.
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

import worker

RAIZ = Path(__file__).resolve().parents[1]
SUP, INF = 0.35, 0.85
ALTO, ANCHO = 160, 120
ROJO = (0, 0, 255)


class _VeRojo:
    """Detector fake: ve cualquier zona no-blanca, con conf configurable."""

    def __init__(self, conf=0.9):
        self.conf = conf
        self.llamadas = []

    def detectar(self, img, enmascarar=True):
        self.llamadas.append(enmascarar)
        b = img[..., 0].astype(int)
        g = img[..., 1].astype(int)
        r = img[..., 2].astype(int)
        negro = (r == 0) & (g == 0) & (b == 0)
        mascara = ((r < 150) | (g < 150) | (b < 150)) & ~negro
        ys, xs = np.where(mascara)
        if len(ys) == 0:
            return []
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        return [((x0, y0, x1 - x0 + 1, y1 - y0 + 1), self.conf)]


@pytest.fixture
def proc(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = {
            "umbral": 8.0,
            "roi_habilitado": True,
            "roi_linea_verde_sup": SUP,
            "roi_linea_verde_inf": INF,
            "gates_fp": False,
            "modo_descarte": "retener",
            "procesadas": tmp / "procesadas",
            "descartes": tmp / "descartadas",
            "rechazadas_api": tmp / "rechazadas_api",
            "ocr_requerido": "si",
            "umbral_texto_min": 5.0,
            "umbral_texto_max": 55.0,
            "umbral_nitidez": 120.0,
            "umbral_sobreexpuesto": 2.0,
            "umbral_bordes": 2.0,
            "umbral_rojo_max": 15.0,
            "umbral_ancho_bbox": 0.20,
            "umbral_dedup": 12,
            "umbral_modelo": 0.40,
            "ventana_reinspeccion_min": 60.0,
            "ventana_reinspeccion_dias": 30,
            "max_pasadas": 3,
            "cooldown_placa_seg": 0.0,
            "cooldown_sin_placa_seg": 0.0,
            "retencion": 30,
        }

        descartes: list[tuple[str, str]] = []
        state = {"texto": "ABC-123", "conf": 0.9}

        def _leer(img, bbox, **kw):
            t = state["texto"]
            return t, state["conf"], [("crnn", t, state["conf"])]

        def _descartar_espia(ruta, motivo, detalle="", auditar=True):
            descartes.append((motivo, detalle))
            ruta.unlink(missing_ok=True)

        detector = _VeRojo()
        monkeypatch.setattr(worker, "_cfg", cfg)
        monkeypatch.setattr(worker, "_ruta_registro", None)
        monkeypatch.setattr(worker, "_detector", detector)
        monkeypatch.setattr(worker, "_ROBOFLOW_ACTIVO", False)
        monkeypatch.setattr(worker, "_AOI_ACTIVA", False)
        monkeypatch.setattr(worker, "_ultima_imagen", {})
        monkeypatch.setattr(worker, "_hashes", {})
        monkeypatch.setattr(worker, "_placas", {})
        monkeypatch.setattr(worker, "_ultima_placa_foto_ts", None)
        monkeypatch.setattr(worker, "_actualizar_dia", lambda: None)
        monkeypatch.setattr(worker, "_descartar", _descartar_espia)
        monkeypatch.setattr(worker, "leer_placa_6", _leer)
        monkeypatch.setattr(worker, "_pasada_de_placa",
                            lambda *a, **k: (1, float("inf")))
        monkeypatch.setattr(worker, "_registrar_pasada", lambda *a: None)
        monkeypatch.setattr(worker, "_encolar_entrenamiento",
                            lambda *a: False)
        monkeypatch.setattr(worker, "evidencia_y_color",
                            lambda *a: (1, "verde"))
        monkeypatch.setattr(worker, "_cooldown_placa", lambda: None)
        monkeypatch.setattr(worker, "_cooldown_sin_placa", lambda: None)
        monkeypatch.setattr(worker, "_trazar_aoi", lambda *a, **k: None)
        monkeypatch.setattr(worker, "_marcar_json", lambda: None)
        import log_sin_lectura
        monkeypatch.setattr(log_sin_lectura, "registrar_worker",
                            lambda **k: None)

        class _H:
            pass

        h = _H()
        h.cfg = cfg
        h.tmp = tmp
        h.descartes = descartes
        h.leer = state
        h.detector = detector

        def _foto(nombre, y0, y1, color=ROJO, x0=30, x1=90):
            img = np.full((ALTO, ANCHO, 3), 255, np.uint8)
            if y1 > y0:
                img[y0:y1, x0:x1] = color
            ruta = tmp / nombre
            assert cv2.imwrite(str(ruta), img)
            return ruta

        h.foto = _foto
        yield h


def _guardadas(proc):
    raiz = proc.cfg["procesadas"]
    if not raiz.is_dir():
        return []
    return sorted(p for p in raiz.rglob("*.jpg")
                  if not p.stem.endswith(("_crop", "_original")))


# ------------------------------------------------------- formato PE ---

@pytest.mark.parametrize("texto", [
    "ABC-123",    # ABC-123
    "A4L-438",    # A1B-234
    "F2P-703",    # A1B-234
    "UN1-910",    # AB1-234 (placard: lo mata la confianza, no el formato)
    "WO3-330",    # AB1-234 (sticker: idem)
    "AB1-234",    # AB1-234
    "AB-1234",    # vieja 2 letras + 4 numeros
    "abc-123",    # minusculas del OCR
    "ABC123",     # sin guion
])
def test_formato_valido_peruano(texto):
    assert worker._formato_placa_valido(texto)


@pytest.mark.parametrize("texto", [
    "5TC-000",    # empieza con digito: cartel lateral de micro
    "00F-245",    # digito primero
    "VST-0NY",    # letra en el bloque de numeros
    "SES-05A",    # letra en el bloque de numeros
    "A68-778",    # digito en posiciones de letras (no es A1B/AB1)
    "AB-123",     # numeros de más cortos
    "ABC-12",     # numeros insuficientes
    "",
])
def test_formato_invalido_peruano(texto):
    assert not worker._formato_placa_valido(texto)


def test_formato_invalido_no_se_guarda(proc):
    # El caso real: 5TC-000 = cartel lateral de un micro.
    # 2026-11-09: se archiva como texto_invalido (uno de los 3 motivos).
    proc.leer["texto"] = "5TC-000"
    ruta = proc.foto("fcarton.jpg", 90, 110)
    worker.procesar_imagen(ruta)

    assert proc.descartes and proc.descartes[0][0] == "texto_invalido"
    assert _guardadas(proc) == []


def test_formato_valido_si_se_guarda(proc):
    proc.leer["texto"] = "A4L-438"
    ruta = proc.foto("freal.jpg", 90, 110)
    worker.procesar_imagen(ruta)

    assert proc.descartes == []
    assert len(_guardadas(proc)) == 1


def test_formato_apagado_guarda_basura(proc):
    proc.cfg["validar_formato"] = False
    proc.leer["texto"] = "5TC-000"
    ruta = proc.foto("fapagado.jpg", 90, 110)
    worker.procesar_imagen(ruta)

    assert proc.descartes == []
    assert len(_guardadas(proc)) == 1


# ----------------------------------------------------- confianza ---

def test_confianza_baja_no_se_guarda(proc):
    # UN1-910 (placard) 0.64 y WO3-330 (sticker) 0.65 < 0.66.
    # 2026-11-09: se archiva como texto_invalido (uno de los 3 motivos).
    proc.detector.conf = 0.5
    ruta = proc.foto("fbaja.jpg", 90, 110)
    worker.procesar_imagen(ruta)

    assert proc.descartes and proc.descartes[0][0] == "texto_invalido"
    assert _guardadas(proc) == []


def test_confianza_minima_configurable(proc):
    proc.cfg["confianza_minima"] = 0.40
    proc.detector.conf = 0.5
    ruta = proc.foto("fmina.jpg", 90, 110)
    worker.procesar_imagen(ruta)

    assert proc.descartes == []
    assert len(_guardadas(proc)) == 1


def test_confianza_alta_si_se_guarda(proc):
    proc.detector.conf = 0.88
    ruta = proc.foto("falta.jpg", 90, 110)
    worker.procesar_imagen(ruta)

    assert proc.descartes == []
    assert len(_guardadas(proc)) == 1


# ------------------------------------------- cooldown por evento ---

TS1 = "192.168.100.41_01_20261106140422407_MD_WITH_TARGET.jpg"
TS2 = "192.168.100.41_01_20261106140426409_MD_WITH_TARGET.jpg"   # +4 s
TS3 = "192.168.100.41_01_20261106140530000_MD_WITH_TARGET.jpg"   # +68 s


def test_ts_de_foto_parsea_nombre_de_camara():
    ts = worker._ts_de_foto(TS1)
    assert ts is not None
    assert (ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second) \
        == (2026, 11, 6, 14, 4, 22)


def test_ts_de_foto_sin_timestamp_devuelve_none():
    assert worker._ts_de_foto("ABC-123_api-Roboflow_roi-evaluado.jpg") \
        is None


def test_cooldown_evento_rechaza_misma_rafaga(proc):
    # Foto 1 se guardo (placa ABC-123 a las 14:04:22).
    ruta1 = proc.foto(TS1, 90, 110)
    worker.procesar_imagen(ruta1)
    assert len(_guardadas(proc)) == 1

    # Foto 2: +4 s de camara, OTRO texto inventado por el OCR (la misma
    # rafaga del mismo carro): se rechaza aunque el texto no se parezca.
    # 2026-11-09: cooldown_evento es filtro INTERNO — no crea carpeta
    # en descartadas; solo borra la foto.
    proc.leer["texto"] = "XYZ-999"
    ruta2 = proc.foto(TS2, 100, 120)
    worker.procesar_imagen(ruta2)

    assert not ruta2.exists()          # borrada, sin carpeta
    assert len(_guardadas(proc)) == 1  # sigue siendo solo la 1
    assert not any("cooldown" in m for m, _ in proc.descartes)


def test_cooldown_evento_pasa_con_tiempo(proc):
    ruta1 = proc.foto(TS1, 90, 110)
    worker.procesar_imagen(ruta1)
    assert len(_guardadas(proc)) == 1

    # Otro carro +68 s despues: pasa (ventana 15 s).
    proc.leer["texto"] = "XYZ-999"
    ruta2 = proc.foto(TS3, 100, 120)
    worker.procesar_imagen(ruta2)

    assert len(_guardadas(proc)) == 2


def test_cooldown_evento_sin_timestamp_no_aplica(proc):
    ruta1 = proc.foto(TS1, 90, 110)
    worker.procesar_imagen(ruta1)
    assert len(_guardadas(proc)) == 1

    # Replay: archivo ya renombrado, sin timestamp de camara -> no
    # aplica el cooldown (conservador: mejor dejar pasar que perder).
    proc.leer["texto"] = "XYZ-999"
    ruta2 = proc.foto("fotareplay.jpg", 100, 120)
    worker.procesar_imagen(ruta2)

    assert len(_guardadas(proc)) == 2


def test_cooldown_evento_apagado(proc):
    proc.cfg["cooldown_evento_seg"] = 0.0
    ruta1 = proc.foto(TS1, 90, 110)
    worker.procesar_imagen(ruta1)

    proc.leer["texto"] = "XYZ-999"
    ruta2 = proc.foto(TS2, 100, 120)
    worker.procesar_imagen(ruta2)

    assert len(_guardadas(proc)) == 2


# ------------------------------------------------- config expuesta ---

def test_config_ini_expone_gates_nuevos():
    import configparser
    p = configparser.ConfigParser()
    p.read(RAIZ / "config.ini", encoding="utf-8-sig")
    assert p.get("filtros", "validar_formato").strip().lower() in ("si", "no")
    assert p.getfloat("filtros", "confianza_minima") > 0
    assert p.getfloat("camara", "cooldown_evento_seg") >= 0


def test_cargar_config_carga_gates_nuevos(monkeypatch):
    texto = (RAIZ / "config.ini").read_text(encoding="utf-8-sig")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "config.ini").write_text(texto, encoding="utf-8")
        import ftp_server
        monkeypatch.setattr(ftp_server, "BASE", Path(tmp))
        cfg = ftp_server.cargar_config()
    assert cfg["validar_formato"] is True
    assert cfg["confianza_minima"] == pytest.approx(0.66)
    assert cfg["cooldown_evento_seg"] == pytest.approx(15.0)
