"""Ventana de evaluacion y gate de posicion (2026-11-10, pedido FELIPE):

1. Gate de posicion: con linea_prioridad calibrada, solo llega a
   evaluar la foto cuya placa cae dentro de linea_prioridad +/- tolerancia_ventana
   (config 2026-11-11: 0.20; el fixture prueba el gate con 0.04).
   Fuera de la ventana se descarta como
   fuera_de_area (naranja si quedo pegada, rojo si quedo lejos) SIN
   llamar a la API.
2. Ventana best-of: de la RAFAGA se evalua UNA foto por
   eval_ventana_seg (default 15 s): la mas cerca de la linea (mas
   grande si empate). La elegida vuelve a procesarse al cerrar la
   ventana y ahi SI pasa por la cadena OCR/API; las peores se
   descartan internas (sin carpeta, sin cuota).
3. Atajos de API eliminados: el pre-chequeo por hash y
   _api_confirmada_hoy ya no reutilizan lecturas: TODA foto que se
   guarda paso por la cadena y queda etiquetada _api-*.
"""
from __future__ import annotations

import configparser
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "app"))
sys.path.insert(0, str(RAIZ))

import worker  # noqa: E402

SUP, INF = 0.35, 0.85
ALTO, ANCHO = 160, 120
ROJO = (0, 0, 255)
LIN = 0.60
TOL_VENT = 0.04
SEG_VENT = 15.0


class _VeRojo:
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
def vent(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = {
            "umbral": 8.0,
            "roi_habilitado": True,
            "roi_linea_verde_sup": SUP,
            "roi_linea_verde_inf": INF,
            "roi_linea_prioridad": LIN,
            "roi_tolerancia_ventana": TOL_VENT,
            "roi_eval_ventana_seg": SEG_VENT,
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
            "ventana_reinspeccion_min": 0.0,
            "ventana_reinspeccion_dias": 30,
            "max_pasadas": 3,
            "cooldown_placa_seg": 0.0,
            "cooldown_sin_placa_seg": 0.0,
            "cooldown_evento_seg": 0.0,
            "retencion": 30,
            "umbral_leven_pasada": 1,
            "revisar_carpetas_dia": True,
            "validar_formato": True,
            "confianza_minima": 0.0,
        }

        descartes: list[tuple[str, str]] = []
        rf_llamadas: list[tuple] = []
        leer_llamadas: list[tuple] = []
        timers: dict[str, list[float]] = {}
        encolados: list[str] = []
        state = {"texto": "ABC-123", "conf": 0.9}

        def _rf(img, bbox):
            rf_llamadas.append(bbox)
            return state["texto"], 0.95

        def _leer(img, bbox, **kw):
            leer_llamadas.append(bbox)
            return (state["texto"], state["conf"],
                    [("crnn", state["texto"], state["conf"])])

        def _descartar_espia(ruta, motivo, detalle="", auditar=True):
            descartes.append((motivo, detalle))
            ruta.unlink(missing_ok=True)

        def _encolar_espia(ruta):
            encolados.append(ruta)
            return True

        def _programar_espia(vista, seg):
            timers.setdefault(vista, []).append(seg)

        monkeypatch.setattr(worker, "_cfg", cfg)
        monkeypatch.setattr(worker, "_ruta_registro", None)
        monkeypatch.setattr(worker, "_detector", _VeRojo())
        monkeypatch.setattr(worker, "_ROBOFLOW_ACTIVO", False)
        monkeypatch.setattr(worker, "_AOI_ACTIVA", False)
        monkeypatch.setattr(worker, "_ultima_imagen", {})
        monkeypatch.setattr(worker, "_hashes", {})
        monkeypatch.setattr(worker, "_placas", {})
        monkeypatch.setattr(worker, "_ultima_placa_foto_ts", None)
        monkeypatch.setattr(worker, "_eval_ventanas", {})
        monkeypatch.setattr(worker, "_actualizar_dia", lambda: None)
        monkeypatch.setattr(worker, "_descartar", _descartar_espia)
        monkeypatch.setattr(worker, "_rf_leer_recorte", _rf)
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
        monkeypatch.setattr(worker, "_reemplazar_si_mejor",
                            lambda *a, **k: False)
        monkeypatch.setattr(worker, "encolar", _encolar_espia)
        monkeypatch.setattr(worker, "_programar_cierre_ventana",
                            _programar_espia)

        import log_sin_lectura
        monkeypatch.setattr(log_sin_lectura, "registrar_worker",
                            lambda **k: None)

        class _H:
            pass

        h = _H()
        h.cfg = cfg
        h.tmp = tmp
        h.descartes = descartes
        h.state = state
        h.rf_llamadas = rf_llamadas
        h.leer_llamadas = leer_llamadas
        h.timers = timers
        h.encolados = encolados
        h.vista = worker._carpeta_de("f1.jpg")

        def _foto(nombre, y0, y1, color=ROJO, x0=30, x1=90):
            img = np.full((ALTO, ANCHO, 3), 255, np.uint8)
            if y1 > y0:
                img[y0:y1, x0:x1] = color
            ruta = tmp / nombre
            assert cv2.imwrite(str(ruta), img)
            return ruta

        h.foto = _foto
        yield h


def _guardadas(vent):
    raiz = vent.cfg["procesadas"]
    if not raiz.is_dir():
        return []
    return sorted(p for p in raiz.rglob("*.jpg")
                  if not p.stem.endswith(("_crop", "_original")))


# ------------------------------------------------- gate de posicion ---

def test_fuera_de_ventana_linea_es_naranja_sin_api(vent):
    # cy = 85/160 = 0.531 -> dist 0.069 de la linea 0.60: pegada pero
    # afuera de +/-0.04 -> naranja, sin gastar API.
    ruta = vent.foto("lejos_1.jpg", 70, 100)
    worker.procesar_imagen(ruta)
    assert vent.descartes
    motivo, detalle = vent.descartes[0]
    assert motivo == "fuera_de_area/naranja"
    assert "ventana de linea" in detalle
    assert not vent.rf_llamadas
    assert not vent.leer_llamadas
    assert not _guardadas(vent)


def test_fuera_de_ventana_linea_lejos_es_rojo(vent):
    # cy = 72.5/160 = 0.453 -> dist 0.147: lejos de la ventana -> rojo
    ruta = vent.foto("lejos_2.jpg", 65, 80)
    worker.procesar_imagen(ruta)
    assert vent.descartes
    motivo, detalle = vent.descartes[0]
    assert motivo == "fuera_de_area/rojo"
    assert "ventana de linea" in detalle
    assert not vent.rf_llamadas
    assert not _guardadas(vent)


def test_dentro_de_ventana_abre_ventana_sin_evaluar(vent):
    # cy = 95/160 = 0.594 -> dentro de +/-0.04: pasa el gate pero NO
    # se evalua ya: queda como candidata con timer de 15 s.
    ruta = vent.foto("cerca.jpg", 85, 105)
    worker.procesar_imagen(ruta)
    assert not vent.descartes
    assert not _guardadas(vent)
    assert not vent.rf_llamadas
    assert ruta.is_file()
    est = worker._eval_ventanas.get(vent.vista)
    assert est is not None
    assert est["fase"] == "abierta"
    assert est["ruta"] == ruta
    assert vent.timers.get(vent.vista) == [SEG_VENT]


def test_sin_linea_calibrada_no_hay_gate(vent):
    vent.cfg["roi_linea_prioridad"] = 0.0
    vent.cfg["roi_eval_ventana_seg"] = 0.0
    ruta = vent.foto("sin_linea.jpg", 70, 100)
    worker.procesar_imagen(ruta)
    assert not vent.descartes
    assert len(_guardadas(vent)) == 1


def test_foto_real_cy_041_pasa_con_tolerancia_del_config(vent):
    # 2026-11-11 (pedido FELIPE): 7/9 fotos reales de un carro cayeron
    # en fuera_de_area (cy 0.41-0.70 vs linea 0.60 +/- 0.04). Con la
    # tolerancia DEL CONFIG (0.20 en verde) la foto de cy 0.410 tiene
    # que pasar el gate de posicion.
    import ftp_server
    tol = ftp_server.cargar_config()["roi_tolerancia_ventana"]
    vent.cfg["roi_tolerancia_ventana"] = tol
    # y0=58 y1=74 en ALTO=160 -> cy = 66/160 = 0.4125 (dist 0.1875)
    ruta = vent.foto("real_cy041.jpg", 58, 74)
    worker.procesar_imagen(ruta)
    motivos = [m for m, _ in vent.descartes]
    assert not any(m.startswith("fuera_de_area") for m in motivos), motivos
    assert not vent.rf_llamadas      # sin ventana abierta no hay API
    assert ruta.is_file()            # quedo como candidata, no descartada


# ------------------------------------------------- ventana best-of ---

def test_rafaga_foto_peor_se_descarta_interna(vent):
    buena = vent.foto("a_cerca.jpg", 85, 105)      # dist 0.006
    worker.procesar_imagen(buena)
    mala = vent.foto("b_lejos.jpg", 92, 112)       # dist 0.038: peor
    worker.procesar_imagen(mala)
    assert mala.exists() is False
    assert not vent.descartes        # interno: sin carpeta de motivos
    assert worker._eval_ventanas[vent.vista]["ruta"] == buena
    assert not _guardadas(vent)
    assert not vent.rf_llamadas


def test_rafaga_foto_mejor_reemplaza_candidata(vent):
    vieja = vent.foto("a_lejos.jpg", 92, 112)      # dist 0.038
    worker.procesar_imagen(vieja)
    mejor = vent.foto("b_cerca.jpg", 85, 105)      # dist 0.006
    worker.procesar_imagen(mejor)
    assert vieja.exists() is False
    assert mejor.is_file()
    assert worker._eval_ventanas[vent.vista]["ruta"] == mejor
    assert not vent.descartes
    assert not _guardadas(vent)


def test_cierre_encola_la_mejor_y_guarda_con_api(vent):
    vieja = vent.foto("a_lejos.jpg", 92, 112)
    worker.procesar_imagen(vieja)
    mejor = vent.foto("b_cerca.jpg", 85, 105)
    worker.procesar_imagen(mejor)

    # cierra la ventana (lo hace el timer en produccion)
    worker._eval_ventana_cerrar(vent.vista)
    assert vent.encolados == [str(mejor)]
    est = worker._eval_ventanas[vent.vista]
    assert est["fase"] == "lista"
    assert est["ruta"] == mejor

    # la elegida vuelve a procesarse y AHI pasa por la API
    worker.procesar_imagen(mejor)
    assert len(vent.rf_llamadas) == 1
    guardadas = _guardadas(vent)
    assert len(guardadas) == 1
    assert "_api-Roboflow" in guardadas[0].name
    assert mejor.exists() is False
    assert not worker._eval_ventanas


def test_misma_candidata_re_llegada_no_se_borra(vent):
    ruta = vent.foto("cand.jpg", 85, 105)
    worker.procesar_imagen(ruta)
    worker.procesar_imagen(ruta)   # barrido/pendiente la re-encola
    assert ruta.is_file()
    est = worker._eval_ventanas[vent.vista]
    assert est["fase"] == "abierta"
    assert est["ruta"] == ruta
    assert len(vent.timers.get(vent.vista, [])) == 1   # sin reprogramar
    assert not vent.descartes


def test_fase_lista_otra_foto_abre_ventana_nueva(vent):
    elegida = vent.foto("elegida.jpg", 92, 112)
    worker.procesar_imagen(elegida)
    worker._eval_ventana_cerrar(vent.vista)
    otra = vent.foto("otra.jpg", 80, 100)         # cy 0.5625: pasa gate
    worker.procesar_imagen(otra)
    est = worker._eval_ventanas[vent.vista]
    assert est["fase"] == "abierta"
    assert est["ruta"] == otra
    assert otra.is_file()
    assert len(vent.timers[vent.vista]) == 2


def test_red_de_seguridad_solo_con_ventana_abierta(vent):
    ruta = vent.foto("cand.jpg", 85, 105)
    worker.procesar_imagen(ruta)
    assert worker._eval_ventana_salta_rf(vent.vista) is True
    worker._eval_ventana_cerrar(vent.vista)
    # fase "lista": la elegida si pasa por la red de seguridad
    assert worker._eval_ventana_salta_rf(vent.vista) is False


# ------------------------------------------------ atajos API muertos ---

def test_atajo_hash_y_confirmada_hoy_no_evitan_la_api(vent, monkeypatch):
    vent.cfg["roi_eval_ventana_seg"] = 0.0   # directo, sin ventana
    # con los atajos viejos estos valores reutilizarian la lectura SIN
    # llamar a la API (intentos "hash"/"api" y sin etiqueta _api-*)
    monkeypatch.setattr(worker, "_buscar_misma",
                        lambda *a, **k: (999, "ABC-123"))
    # raising=False: estos atajos existen en RED y se eliminan en GREEN
    monkeypatch.setattr(worker, "_api_confirmada_hoy",
                        lambda *a: True, raising=False)
    monkeypatch.setattr(worker, "_es_mejor_toma",
                        lambda *a, **k: False, raising=False)

    ruta = vent.foto("directo.jpg", 85, 105)
    worker.procesar_imagen(ruta)

    assert len(vent.rf_llamadas) == 1        # la API se evaluo igual
    assert not vent.leer_llamadas            # el local no hizo falta
    guardadas = _guardadas(vent)
    assert len(guardadas) == 1
    assert "_api-Roboflow" in guardadas[0].name


# ---------------------------------------------------------- config ---

def test_config_ini_expone_claves_de_ventana():
    p = configparser.ConfigParser()
    p.read(RAIZ / "config.ini", encoding="utf-8-sig")
    tol = p.getfloat("roi", "tolerancia_ventana")
    seg = p.getfloat("roi", "eval_ventana_seg")
    assert 0.0 < tol < 0.25
    assert seg > 0.0


def test_cargar_config_pasa_las_claves_de_ventana_al_worker():
    # mismo gotcha conocido: sin la clave en cargar_config() el ini no
    # hace nada (default silencioso)
    import ftp_server
    cfg = ftp_server.cargar_config()
    assert "roi_tolerancia_ventana" in cfg
    assert "roi_eval_ventana_seg" in cfg
    assert 0.0 < cfg["roi_tolerancia_ventana"] < 0.25
    assert cfg["roi_eval_ventana_seg"] > 0.0
