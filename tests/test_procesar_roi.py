"""procesar_imagen con ROI: la validacion es SOLO la franja verde
(gates FP apagables), lo que esta en rojo no se detecta, el motivo de
placa fuera de la franja es propio (fuera_franja, sin reevaluar) y las
fotos que la API rechazo quedan copiadas en rechazadas_api\\ con traza.
"""
import json
import sys
import tempfile
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

import worker

RAIZ = Path(__file__).resolve().parents[1]
SUP, INF = 0.45, 0.75
ALTO, ANCHO = 160, 120

ROJO = (0, 0, 255)


class _VeRojo:
    """Detector fake que 've' cualquier zona no-blanca (la placa).

    Ignora el negro puro (0,0,0): es como se pinta la mascara OSD y el
    ONNX real no detecta rectangulos negros lisos. Registra el kwarg
    enmascarar que le pasa _candidatos_placa.
    """

    def __init__(self):
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
        return [((x0, y0, x1 - x0 + 1, y1 - y0 + 1), 0.9)]


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
            # 2026-11-10: directo a la cadena, sin ventana best-of
            "roi_eval_ventana_seg": 0.0,
            "retencion": 30,
        }

        descartes: list[tuple[str, str]] = []
        leer_llamadas: list[tuple] = []
        leer_kwargs: list[dict] = []

        def _leer(img, bbox, **kw):
            leer_llamadas.append(bbox)
            leer_kwargs.append(dict(kw))
            return "ABC-123", 0.9, [("crnn", "ABC-123", 0.9)]

        def _descartar_espia(ruta, motivo, detalle="", auditar=True):
            descartes.append((motivo, detalle))
            ruta.unlink(missing_ok=True)

        monkeypatch.setattr(worker, "_cfg", cfg)
        monkeypatch.setattr(worker, "_ruta_registro", None)
        monkeypatch.setattr(worker, "_detector", _VeRojo())
        monkeypatch.setattr(worker, "_ROBOFLOW_ACTIVO", False)
        monkeypatch.setattr(worker, "_AOI_ACTIVA", False)
        monkeypatch.setattr(worker, "_ultima_imagen", {})
        monkeypatch.setattr(worker, "_hashes", {})
        monkeypatch.setattr(worker, "_placas", {})
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
        h.leer_llamadas = leer_llamadas
        h.leer_kwargs = leer_kwargs
        h.detector = worker._detector
        h.rechazadas = cfg["rechazadas_api"]

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
    # _guardar escribe la foto completa, el _crop y el _original: la
    # entrada de la placa es la completa.
    return sorted(p for p in raiz.rglob("*.jpg")
                  if not p.stem.endswith(("_crop", "_original")))


# --------------------------------------------------------------- gates ---

def test_gates_fp_apagados_no_filtran(proc):
    # placa NEGRO (nitidez ~0): con los gates FP activos seria
    # baja_nitidez; con gates_fp = no la validacion es solo la franja.
    ruta = proc.foto("fotoscuro.jpg", 90, 110, color=(30, 30, 30))
    worker.procesar_imagen(ruta)

    assert proc.descartes == []
    assert len(proc.leer_llamadas) == 1          # llego al OCR
    assert len(_guardadas(proc)) == 1            # y se guardo
    assert proc.detector.llamadas == [False]     # enmascarar=False (ROI)


def test_gates_fp_encendidos_sigue_filtrando(proc):
    proc.cfg["gates_fp"] = True
    ruta = proc.foto("fotoscuro.jpg", 90, 110, color=(30, 30, 30))
    worker.procesar_imagen(ruta)

    assert proc.descartes and proc.descartes[0][0] == "baja_nitidez"
    assert proc.leer_llamadas == []              # no gasto OCR ni API
    assert _guardadas(proc) == []


def test_cargar_config_carga_gates_fp(monkeypatch):
    texto = (RAIZ / "config.ini").read_text(encoding="utf-8-sig")
    assert "gates_fp" in texto
    texto = texto.replace("gates_fp = no", "gates_fp = si")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "config.ini").write_text(texto, encoding="utf-8")
        import ftp_server
        monkeypatch.setattr(ftp_server, "BASE", Path(tmp))
        cfg = ftp_server.cargar_config()
    assert cfg["gates_fp"] is True


def test_config_ini_expone_gates_fp():
    import configparser
    p = configparser.ConfigParser()
    p.read(RAIZ / "config.ini", encoding="utf-8-sig")
    assert p.get("filtros", "gates_fp").strip().lower() in ("si", "no")


# ------------------------------------------------------- fuera_de_area ---

def test_placa_fuera_de_la_franja_es_motivo_propio(proc):
    # placa POR DEBAJO de la linea verde de inf (auto demasiado cerca)
    # -> fuera_de_area/naranja (cruza la linea de abajo)
    ruta = proc.foto("fotofuera.jpg", 130, 150)
    worker.procesar_imagen(ruta)

    assert proc.descartes and proc.descartes[0][0] == "fuera_de_area/naranja"
    assert proc.leer_llamadas == []   # sin reeval: la placa no se mueve
    assert _guardadas(proc) == []


def test_fuera_de_area_no_se_reevalua(proc):
    # fuera_de_area esta en FALLOS_SIN_REEVAL: ni siquiera se intenta
    # leer de nuevo (el REEVALUACION llamaria a leer_placa_6)
    assert "fuera_de_area" in worker.FALLOS_SIN_REEVAL


def test_placa_contra_el_borde_izquierdo_es_fuera_de_area(proc):
    # corte vertical: placa x 0..41 de 120 con izq = 0.15 (limite 18 + 4
    # de tolerancia = 22); la y pasa la franja, la que manda es la x.
    # Vertical = rojo.
    proc.cfg["roi_linea_verde_izq"] = 0.15
    proc.cfg["roi_linea_verde_der"] = 0.95
    ruta = proc.foto("fverticalizq.jpg", 90, 110, x0=0, x1=40)
    worker.procesar_imagen(ruta)

    assert proc.descartes and proc.descartes[0][0] == "fuera_de_area/rojo"
    assert "vertical" in proc.descartes[0][1]
    assert proc.leer_llamadas == []   # se corta antes del OCR
    assert _guardadas(proc) == []


def test_placa_contra_el_borde_derecho_es_fuera_de_area(proc):
    # x 95..120 de 120 con der = 0.95 (limite 114 + 4 = 118): 120 lo cruza.
    proc.cfg["roi_linea_verde_izq"] = 0.15
    proc.cfg["roi_linea_verde_der"] = 0.95
    ruta = proc.foto("fverticalder.jpg", 90, 110, x0=95, x1=119)
    worker.procesar_imagen(ruta)

    assert proc.descartes and proc.descartes[0][0] == "fuera_de_area/rojo"
    assert proc.leer_llamadas == []
    assert _guardadas(proc) == []


def test_sin_claves_verticales_no_corta_nada(proc):
    # defaults 0.0/1.0: sin las claves en cfg el corte no existe y la
    # placa contra el borde izquierdo pasa como siempre (rollback).
    ruta = proc.foto("fsinvertical.jpg", 90, 110, x0=0, x1=40)
    worker.procesar_imagen(ruta)

    assert proc.descartes == []
    assert len(_guardadas(proc)) == 1


def test_foto_guardada_lleva_las_lineas_verticales(proc):
    proc.cfg["roi_linea_verde_izq"] = 0.15
    proc.cfg["roi_linea_verde_der"] = 0.95
    ruta = proc.foto("fverticalbandas.jpg", 90, 110)
    worker.procesar_imagen(ruta)

    guardadas = _guardadas(proc)
    assert len(guardadas) == 1
    img = cv2.imread(str(guardadas[0]))
    assert img is not None

    # linea vertical blanca en x = 0.15*120 = 18 (sobre la franja verde)
    assert int(img[100, 18, 0]) > 230
    assert int(img[100, 18, 1]) > 230
    assert int(img[100, 18, 2]) > 230

    # tinte "se corta" a la izquierda de la linea (fila verde, x < 18)
    assert int(img[100, 8, 2]) > int(img[100, 8, 1]) + 40


def test_guarda_tambien_la_imagen_original(proc):
    # El pedido de FELIPE: el set de una placa es placa (bandas+caja),
    # recorte, txt y TAMBIEN la imagen original sin anotar.
    ruta = proc.foto("foriginal.jpg", 90, 110)
    worker.procesar_imagen(ruta)

    guardadas = _guardadas(proc)
    assert len(guardadas) == 1
    destino = guardadas[0]
    original = destino.with_name(destino.stem + "_original.jpg")
    assert original.is_file()
    # la etiqueta YOLO es de la foto principal; la original no la lleva
    assert not original.with_suffix(".txt").exists()

    img_o = cv2.imread(str(original))
    img_a = cv2.imread(str(destino))
    assert img_o is not None
    # sin bandas ROI: la fila 10 (tintada de rojo en la anotada) queda
    # blanco puro en la original
    assert int(img_o[10, 5, 1]) > 200
    assert int(img_a[10, 5, 2]) > int(img_a[10, 5, 1])
    # foto COMPLETA, no el recorte
    assert img_o.shape[0] == ALTO
    assert img_o.shape[1] == ANCHO


# ------------------------------------------- lo rojo no se detecta E2E ---

def test_placa_en_zona_roja_no_se_detecta(proc):
    # la placa esta ENTERA por encima de la linea verde de arriba:
    # el recorte se la lleva y el detector (que solo ve el recorte)
    # no regresa ningun candidato -> sin_placa, sin OCR, sin API.
    ruta = proc.foto("fotorojo.jpg", 10, 30)
    worker.procesar_imagen(ruta)

    assert proc.descartes and proc.descartes[0][0] == "sin_placa"
    assert proc.leer_llamadas == []
    assert _guardadas(proc) == []


def test_placa_dentro_de_la_franja_si_se_detecta(proc):
    ruta = proc.foto("fotoverde.jpg", 90, 110)
    worker.procesar_imagen(ruta)

    assert proc.descartes == []
    assert len(_guardadas(proc)) == 1


# ----------------------------------------------------- rechazadas_api ---

def test_api_rechazada_copia_la_foto_con_traza(proc, monkeypatch):
    def _leer_rechaza(img, bbox, **kw):
        proc.leer_llamadas.append(bbox)
        proc.leer_kwargs.append(dict(kw))
        return ("ABC-123", 0.4, [("crnn", "ABC-123", 0.4),
                                 ("api_rechazada", "api: sin respuesta",
                                  0.0)])
    monkeypatch.setattr(worker, "leer_placa_6", _leer_rechaza)

    ruta = proc.foto("fotoapi.jpg", 90, 110)
    worker.procesar_imagen(ruta)

    assert len(_guardadas(proc)) == 1            # la lectura local salvo
    hoy = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
    carpeta = proc.rechazadas / hoy / "placa"
    copias = list(carpeta.glob("*.jpg")) if carpeta.is_dir() else []
    assert len(copias) == 1                      # foto ORIGINAL copiada
    trazas = list(carpeta.glob("rechazadas.jsonl"))
    assert len(trazas) == 1
    linea = json.loads(trazas[0].read_text(encoding="utf-8").splitlines()[0])
    assert linea["resultado"] == "guardado"
    assert linea["foto"] == "fotoapi.jpg"
    nombres = [n for n, _t, _c in linea["intentos"]]
    assert "api_rechazada" in nombres


def test_sin_rechazo_de_api_no_copia_nada(proc):
    ruta = proc.foto("fotalocal.jpg", 90, 110)
    worker.procesar_imagen(ruta)

    assert len(_guardadas(proc)) == 1
    assert not proc.rechazadas.exists()


def test_roboflow_sin_deteccion_no_es_rechazo(proc, monkeypatch):
    # calle vacia: RF no confirmo pero tampoco vio nada -> no copiar
    rf = types.ModuleType("roboflow")
    rf.detectar_leer = lambda img: []
    rf.motivo_sin_api = lambda: "sin_deteccion"
    monkeypatch.setitem(sys.modules, "roboflow", rf)
    monkeypatch.setattr(worker, "_ROBOFLOW_ACTIVO", True)

    ruta = proc.foto("fotovacia.jpg", 0, 0)
    worker.procesar_imagen(ruta)

    assert proc.descartes and proc.descartes[0][0] == "sin_placa"
    assert not proc.rechazadas.exists()


def test_roboflow_que_ve_no_confirma_es_rechazo(proc, monkeypatch):
    # RF detecto algo (gasto cuota) pero el texto no es placa -> rechazo.
    # El local tampoco lee la calle vacia -> la foto se descarta y queda
    # copiada en rechazadas_api con rf_rechazado = True.
    rf = types.ModuleType("roboflow")
    rf.detectar_leer = lambda img: [((30, 18, 60, 20), "PERU", 0.9)]
    rf.motivo_sin_api = lambda: "texto_invalido"
    monkeypatch.setitem(sys.modules, "roboflow", rf)
    monkeypatch.setattr(worker, "_ROBOFLOW_ACTIVO", True)
    monkeypatch.setattr(
        worker, "leer_placa_6",
        lambda *a, **kw: ("", 0.0, [("crnn", "", 0.0)]))

    ruta = proc.foto("fotorf.jpg", 0, 0)
    worker.procesar_imagen(ruta)

    assert proc.descartes and proc.descartes[0][0] == "texto_invalido"
    hoy = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
    carpeta = proc.rechazadas / hoy / "placa"
    assert list(carpeta.glob("*.jpg"))
    traza = json.loads(
        (carpeta / "rechazadas.jsonl").read_text(
            encoding="utf-8").splitlines()[0])
    assert traza["resultado"] == "texto_invalido"
    assert traza["rf_rechazado"] is True


# -------------------------------------------------- foto guardada ROI ---

def test_guarda_foto_completa_con_bandas_y_etiqueta_roi(proc):
    # El pedido de FELIPE: la foto guardada lleva el nombre de la placa,
    # las bandas de colores del ROI y la etiqueta "ROI", para poder VER
    # en la foto donde queda la placa respecto de las lineas.
    ruta = proc.foto("fotoroi.jpg", 90, 110)
    worker.procesar_imagen(ruta)

    guardadas = _guardadas(proc)
    assert len(guardadas) == 1
    assert "ABC-123" in guardadas[0].name       # nombre = la placa

    img = cv2.imread(str(guardadas[0]))
    assert img is not None
    # Se guarda la foto ORIGINAL completa (160), no el recorte (88):
    # el corte es solo para el detector.
    assert img.shape[0] == ALTO

    # Bandas de colores (BGR): roja arriba, verde en la franja,
    # naranja abajo. Mismo estilo que scripts\vista_roi.py.
    assert int(img[10, 5, 2]) > int(img[10, 5, 1])            # R > G: rojo
    assert int(img[100, 5, 1]) > int(img[100, 5, 2])          # G > R: verde
    assert (int(img[150, 5, 2]) > int(img[150, 5, 1])
            > int(img[150, 5, 0]))                             # naranja

    # Etiqueta "ROI": texto blanco sobre la banda roja (arriba-izq).
    region = img[8:60, 8:ANCHO]
    assert bool(np.all(region > 230, axis=2).any()), \
        "falta la etiqueta ROI en la foto guardada"

    # El .txt queda normalizado a la foto COMPLETA (coherente con la
    # imagen guardada): cy = ((90+110)/2) / 160 = 0.625.
    partes = guardadas[0].with_suffix(".txt").read_text(
        encoding="utf-8").split()
    assert abs(float(partes[2]) - ((90 + 110) / 2 / ALTO)) < 0.01


def test_roi_apagado_guarda_sin_bandas(proc):
    proc.cfg["roi_habilitado"] = False
    ruta = proc.foto("fotocolor.jpg", 90, 110)
    worker.procesar_imagen(ruta)

    guardadas = _guardadas(proc)
    assert len(guardadas) == 1
    img = cv2.imread(str(guardadas[0]))
    assert img.shape[0] == ALTO
    # Sin ROI no hay tintas: el fondo blanco queda intacto.
    assert int(img[10, 5, 1]) > 200


# ------------------------------------------------------------- purga ---

def test_purga_rechazadas_por_retencion(proc):
    vieja = proc.rechazadas / "2020-01-01" / "placa"
    vieja.mkdir(parents=True)
    (vieja / "vieja.jpg").write_bytes(b"x")
    hoy = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
    actual = proc.rechazadas / hoy / "placa"
    actual.mkdir(parents=True)
    (actual / "nueva.jpg").write_bytes(b"x")

    worker._purgar_rechazadas_api()

    assert not vieja.exists()
    assert (actual / "nueva.jpg").exists()


def test_cargar_config_carga_rechazadas_api(monkeypatch):
    texto = (RAIZ / "config.ini").read_text(encoding="utf-8-sig")
    assert "rechazadas_api" in texto
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "config.ini").write_text(texto, encoding="utf-8")
        import ftp_server
        monkeypatch.setattr(ftp_server, "BASE", Path(tmp))
        cfg = ftp_server.cargar_config()
    assert "rechazadas_api" in str(cfg["rechazadas_api"])
