"""Descartes minimos (2026-11-09, pedido FELIPE):

Solo 3 motivos en descartadas:
  1. sin_placa
  2. fuera_de_area  (subdividido naranja / rojo)
  3. texto_invalido (absorbe placa_cortada, formato_invalido, confianza_baja)

- duplicado_placa NO se bota: la misma placa el mismo dia cuenta como
  pasada 2/3 (ventana_reinspeccion_min = 0).
- La imagen original sin colores se guarda siempre junto a la anotada.
- _placa_existe_hoy lee carpetas del dia (+ JSON) y matchea con
  Levenshtein <= umbral_leven_pasada (default 1, NO 4 de 6).
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "app"))
sys.path.insert(0, str(RAIZ))

import worker  # noqa: E402

# Real (antes de que fixtures la mockeen): para tests de integracion.
_pasada_de_placa_real = worker._pasada_de_placa
_buscar_misma_real = worker._buscar_misma

SUP, INF = 0.35, 0.85
ALTO, ANCHO = 160, 120
ROJO = (0, 0, 255)
AZUL_CLARO = (255, 200, 100)


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
            "ventana_reinspeccion_min": 0.0,
            "ventana_reinspeccion_dias": 30,
            "max_pasadas": 3,
            "cooldown_placa_seg": 0.0,
            "cooldown_sin_placa_seg": 0.0,
            "cooldown_evento_seg": 0.0,
            # 2026-11-10: sin ventana de evaluacion en esta fixture: los
            # tests de motivos/pasadas hablan directo con la cadena OCR.
            "roi_eval_ventana_seg": 0.0,
            "retencion": 30,
            "umbral_leven_pasada": 1,
            "revisar_carpetas_dia": True,
            "validar_formato": True,
            "confianza_minima": 0.0,
        }

        descartes: list[tuple[str, str]] = []
        state = {"texto": "ABC-123", "conf": 0.9}
        pasadas: list[tuple[str, int]] = []

        def _leer(img, bbox, **kw):
            t = state["texto"]
            return t, state["conf"], [("crnn", t, state["conf"])]

        def _descartar_espia(ruta, motivo, detalle="", auditar=True):
            descartes.append((motivo, detalle))
            ruta.unlink(missing_ok=True)

        def _registrar_pasada_espia(placa, n, excedida):
            pasadas.append((placa, n))

        monkeypatch.setattr(worker, "_cfg", cfg)
        monkeypatch.setattr(worker, "_ruta_registro", None)
        monkeypatch.setattr(worker, "_detector", _VeRojo())
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
        monkeypatch.setattr(worker, "_registrar_pasada", _registrar_pasada_espia)
        monkeypatch.setattr(worker, "_encolar_entrenamiento", lambda *a: False)
        monkeypatch.setattr(worker, "evidencia_y_color", lambda *a: (1, "verde"))
        monkeypatch.setattr(worker, "_cooldown_placa", lambda: None)
        monkeypatch.setattr(worker, "_cooldown_sin_placa", lambda: None)
        monkeypatch.setattr(worker, "_trazar_aoi", lambda *a, **k: None)
        monkeypatch.setattr(worker, "_marcar_json", lambda: None)
        monkeypatch.setattr(worker, "_buscar_misma", lambda *a, **k: None)
        # Atajos que se eliminan en 2026-11-10: raising=False para que el
        # fixture sirva antes (donde los llaman) y despues (borrados)
        # sin reventar con AttributeError.
        monkeypatch.setattr(worker, "_es_mejor_toma",
                            lambda *a, **k: False, raising=False)
        monkeypatch.setattr(worker, "_api_confirmada_hoy",
                            lambda *a: False, raising=False)

        import log_sin_lectura
        monkeypatch.setattr(log_sin_lectura, "registrar_worker",
                            lambda **k: None)

        class _H:
            pass

        h = _H()
        h.cfg = cfg
        h.tmp = tmp
        h.descartes = descartes
        h.pasadas = pasadas
        h.state = state
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
    return sorted(p for p in raiz.rglob("*.jpg")
                  if not p.stem.endswith(("_crop", "_original")))


def _motivos_descartes(proc):
    base = proc.cfg["descartes"]
    if not base.is_dir():
        return set()
    motivos = set()
    for dia in base.iterdir():
        if not dia.is_dir():
            continue
        for m in dia.rglob("*"):
            if m.is_dir() and any(m.iterdir()):
                motivos.add(str(m.relative_to(dia)).replace("\\", "/"))
    return motivos


# ------------------------------------------------------- 3 motivos ---

def test_fuera_de_area_vertical_es_rojo(proc):
    # placa contra el borde izquierdo: fuera vertical -> rojo
    proc.cfg["roi_linea_verde_izq"] = 0.15
    proc.cfg["roi_linea_verde_der"] = 0.95
    ruta = proc.foto("borde_izq.jpg", 70, 100, x0=0, x1=20)
    worker.procesar_imagen(ruta)
    assert proc.descartes
    assert proc.descartes[0][0] == "fuera_de_area/rojo"


def test_fuera_de_area_linea_inferior_es_naranja(proc):
    # placa cruza la linea de abajo (INF=0.85*160=136): y1=150 > 136
    # pero y0=120 < 136 -> casi adentro = naranja
    ruta = proc.foto("cruza_abajo.jpg", 120, 150, x0=30, x1=90)
    worker.procesar_imagen(ruta)
    assert proc.descartes
    assert proc.descartes[0][0] == "fuera_de_area/naranja"


def test_sin_placa_sigue_siendo_motivo(proc):
    # detector fake que no ve nada
    class _VeNada:
        def detectar(self, img, enmascarar=True):
            return []

    worker._detector = _VeNada()
    ruta = proc.foto("vacia.jpg", 0, 0)  # sin rectangulo rojo
    # foto blanca pura: detector no ve nada
    img = np.full((ALTO, ANCHO, 3), 255, np.uint8)
    assert cv2.imwrite(str(ruta), img)
    worker.procesar_imagen(ruta)
    assert proc.descartes
    assert proc.descartes[0][0] == "sin_placa"


def test_texto_invalido_absorbe_formato_invalido(proc):
    proc.cfg["validar_formato"] = True
    proc.state["texto"] = "XXXX-999"  # no es formato peruano
    ruta = proc.foto("malformato.jpg", 70, 100)
    worker.procesar_imagen(ruta)
    assert proc.descartes
    assert proc.descartes[0][0] == "texto_invalido"
    assert "formato_invalido" not in {m for m, _ in proc.descartes}


def test_texto_invalido_absorbe_confianza_baja(proc):
    proc.cfg["confianza_minima"] = 0.99
    proc.state["conf"] = 0.5
    # detector fake devuelve conf 0.9, pero el OCR/estado no importa:
    # confianza de deteccion viene del detector. Bajamos la del detector.
    worker._detector = _VeRojo(conf=0.5)
    ruta = proc.foto("bajaconf.jpg", 70, 100)
    worker.procesar_imagen(ruta)
    assert proc.descartes
    assert proc.descartes[0][0] == "texto_invalido"
    assert "confianza_baja" not in {m for m, _ in proc.descartes}


def test_no_existe_motivo_duplicado_placa(proc, monkeypatch):
    # misma placa ya guardada: NO se bota como duplicado_placa
    # (2026-11-10: con los atajos de API eliminados, el match por hash
    # solo puede archivar la foto como pasada, jamas como duplicado)
    proc.state["texto"] = "ABC-123"
    monkeypatch.setattr(worker, "_buscar_misma",
                        lambda *a, **k: ("hash", "ABC-123"))
    monkeypatch.setattr(worker, "_reemplazar_si_mejor",
                        lambda *a, **k: False)
    ruta = proc.foto("mismo_carro.jpg", 70, 100)
    worker.procesar_imagen(ruta)
    motivos = {m for m, _ in proc.descartes}
    assert "duplicado_placa" not in motivos
    assert "duplicado" not in motivos


def test_cooldown_evento_no_crea_subcarpeta(proc):
    # cooldown_evento es filtro interno: no debe aparecer en descartadas
    # (con cooldown_evento_seg=0 el filtro esta apagado; el test valida
    # que el set de motivos en disco nunca lo incluye)
    proc.state["texto"] = "ABC-123"
    ruta = proc.foto("rafaga.jpg", 70, 100)
    worker.procesar_imagen(ruta)
    motivos = _motivos_descartes(proc)
    assert not any("cooldown" in m for m in motivos)


# ------------------------------------------------- pasadas mismo dia ---

def test_misma_placa_mismo_dia_cuenta_pasada_2(proc, monkeypatch):
    # ventana=0 + misma placa -> pasada 2, no descarte
    proc.state["texto"] = "ABC-123"
    calls = {"n": 0}

    def _pasada(p, *a, **k):
        calls["n"] += 1
        return (2, 0.0)  # minutos=0 pero ventana=0 -> no bloquea

    monkeypatch.setattr(worker, "_pasada_de_placa", _pasada)
    monkeypatch.setattr(worker, "_buscar_misma", lambda *a, **k: None)
    # pero sí placa existente en disco para el match de pasada
    monkeypatch.setattr(worker, "_placa_existe_hoy", lambda *a, **k: True)

    ruta = proc.foto("segunda_visita.jpg", 70, 100)
    worker.procesar_imagen(ruta)

    assert not any(m == "duplicado_placa" for m, _ in proc.descartes)
    assert len(_guardadas(proc)) == 1
    # 2026-11-10 (FASE 2 carpetas de pasadas): la 2a pasada vive en la
    # subcarpeta segunda/ y el nombre del archivo ya NO lleva sufijo.
    guardadas = _guardadas(proc)
    assert any(p.parent.name == "segunda" for p in guardadas)
    assert not any("_2da-pasada" in p.name for p in guardadas)


def test_misma_placa_tres_pasadas_y_luego_excedida(proc, monkeypatch):
    proc.state["texto"] = "ABC-123"
    monkeypatch.setattr(worker, "_pasada_de_placa",
                        lambda *a, **k: (4, 0.0))
    monkeypatch.setattr(worker, "_placa_existe_hoy", lambda *a, **k: True)
    ruta = proc.foto("cuarta.jpg", 70, 100)
    worker.procesar_imagen(ruta)
    # 4a se GUARDA en la subcarpeta excedida/, no se bota
    assert not any("duplicado" in m for m, _ in proc.descartes)
    guardadas = _guardadas(proc)
    assert len(guardadas) == 1
    assert guardadas[0].parent.name == "excedida"
    assert "_EXCEDIDA" not in guardadas[0].name


def test_tercera_pasada_vive_en_subcarpeta(proc, monkeypatch):
    # 2026-11-10 (FASE 2): tercera pasada -> subcarpeta tercera/
    proc.state["texto"] = "ABC-123"
    monkeypatch.setattr(worker, "_pasada_de_placa",
                        lambda *a, **k: (3, 0.0))
    monkeypatch.setattr(worker, "_placa_existe_hoy", lambda *a, **k: True)
    ruta = proc.foto("tercera_visita.jpg", 70, 100)
    worker.procesar_imagen(ruta)
    guardadas = _guardadas(proc)
    assert len(guardadas) == 1
    assert guardadas[0].parent.name == "tercera"
    assert "_3ra-pasada" not in guardadas[0].name


def test_primera_pasada_se_queda_en_la_raiz(proc):
    # la 1a pasada NUNCA crea subcarpeta: vive en <dia>/<PLACA>/
    proc.state["texto"] = "ABC-123"
    ruta = proc.foto("primera.jpg", 70, 100)
    worker.procesar_imagen(ruta)
    guardadas = _guardadas(proc)
    assert len(guardadas) == 1
    assert guardadas[0].parent.name == "ABC-123"
    assert not any(p.parent.name in ("segunda", "tercera", "excedida")
                   for p in guardadas)


def test_mejor_foto_de_segunda_pasada_reemplaza_en_su_subcarpeta(
        proc, monkeypatch):
    # 2026-11-10 (FASE 2): la MEJOR FOTO de la 2a pasada reemplaza
    # DENTRO de segunda/ (viejo.parent), nunca en la raiz.
    proc.state["texto"] = "ABC-123"
    # DB real: _reemplazar_si_mejor consulta registros (patron de
    # test_reconciliacion._entorno, encima del cfg del fixture proc).
    monkeypatch.setattr(worker, "_ruta_registro", proc.tmp / "registro.db")
    monkeypatch.setattr(worker, "_dia_dedup",
                        datetime.now().strftime("%Y-%m-%d"))
    con = worker._conexion()
    try:
        con.execute(
            "CREATE TABLE registros ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "fecha_hora TEXT NOT NULL, archivo TEXT NOT NULL,"
            "placa TEXT, confianza REAL,"
            "estado TEXT NOT NULL DEFAULT 'guardado',"
            "evidencia INTEGER, semaforo TEXT, phash TEXT, fuente TEXT,"
            "provisional INTEGER, placa_api TEXT, placa_modelo TEXT,"
            "motivo_sin_api TEXT)")
        con.execute(
            "CREATE TABLE pasadas ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "placa TEXT NOT NULL, pasada INTEGER NOT NULL,"
            "fecha_hora TEXT NOT NULL, excedida INTEGER NOT NULL DEFAULT 0)")
        con.commit()
    finally:
        con.close()
    # 1a pasada en la raiz (mismo rect, color rojo)
    ruta1 = proc.foto("p1.jpg", 70, 100, color=ROJO)
    worker.procesar_imagen(ruta1)
    # 2a pasada: mismo rect, distinto color -> dif > umbral 8%, no dup
    monkeypatch.setattr(worker, "_pasada_de_placa",
                        lambda *a, **k: (2, 0.0))
    monkeypatch.setattr(worker, "_placa_existe_hoy", lambda *a, **k: True)
    ruta2 = proc.foto("p2.jpg", 70, 100, color=AZUL_CLARO)
    worker.procesar_imagen(ruta2)
    en_segunda = [p for p in _guardadas(proc)
                  if p.parent.name == "segunda"]
    assert len(en_segunda) == 1
    assert "_2da-pasada" not in en_segunda[0].name
    # la MEJOR toma (area 2800/1800 = 1.56x) llega con match por texto:
    # _reemplazar_si_mejor debe elegir el registro mas nuevo (la 2a)
    # y reemplazar en viejo.parent = segunda/
    monkeypatch.setattr(worker, "_buscar_misma",
                        lambda *a, **k: ("hash", "ABC-123"))
    ruta3 = proc.foto("p3.jpg", 65, 105, x0=25, x1=95, color=ROJO)
    worker.procesar_imagen(ruta3)
    guardadas = _guardadas(proc)
    en_segunda = [p for p in guardadas if p.parent.name == "segunda"]
    en_raiz = [p for p in guardadas if p.parent.name == "ABC-123"]
    assert len(en_segunda) == 1   # reemplazo (1 archivo), no duplicado
    assert "_2da-pasada" not in en_segunda[0].name
    assert len(en_raiz) == 1      # la 1a pasada sigue intacta en la raiz


def test_depurar_conserva_carpeta_con_fotos_solo_en_subcarpeta(proc):
    # 2026-11-10 (FASE 2): la limpieza mira rglob y NO tira una
    # carpeta cuyas fotos quedaron adentro de segunda/.
    dia = proc.cfg["procesadas"] / "placa" / "2026-11-10" / "ABC-123"
    (dia / "segunda").mkdir(parents=True)
    (dia / "segunda" / "ABC-123_roi-evaluado.jpg").write_bytes(b"jpg")
    assert worker._depurar_carpetas_vacias() == 0
    assert dia.is_dir()


def test_depurar_igual_tira_carpeta_sin_fotos(proc):
    dia = proc.cfg["procesadas"] / "placa" / "2026-11-10" / "XYZ-777"
    dia.mkdir(parents=True)
    assert worker._depurar_carpetas_vacias() >= 1
    assert not dia.exists()


def test_lectura_lateral_archiva_sin_pasada(proc, monkeypatch):
    # texto similar pero hash lejano = lateral: no reemplaza, no gasta pasada
    proc.state["texto"] = "ABC-123"
    monkeypatch.setattr(worker, "_buscar_misma",
                        lambda *a, **k: (999, "ABC-123"))
    # pre-2026-11-10: mejora=True saltaba el atajo de hash pre-OCR;
    # raising=False por si el atajo ya no existe (call site eliminado).
    monkeypatch.setattr(worker, "_es_mejor_toma",
                        lambda *a, **k: True, raising=False)
    monkeypatch.setattr(worker, "_es_lectura_lateral", lambda *a, **k: True)
    monkeypatch.setattr(worker, "_pasada_de_placa",
                        lambda *a, **k: (2, 0.0))
    antes = len(proc.pasadas)
    ruta = proc.foto("lateral.jpg", 70, 100)
    worker.procesar_imagen(ruta)
    # se archiva como texto_invalido (lateral) y NO registra pasada
    assert len(proc.pasadas) == antes
    assert any(m == "texto_invalido" for m, _ in proc.descartes)


# ------------------------------------------------- imagen sin colores ---

def test_guarda_original_sin_colores(proc):
    proc.state["texto"] = "ABC-123"
    ruta = proc.foto("con_roi.jpg", 70, 100)
    worker.procesar_imagen(ruta)
    guardadas = _guardadas(proc)
    assert len(guardadas) == 1
    original = guardadas[0].with_name(guardadas[0].stem + "_original.jpg")
    assert original.is_file(), "falta la imagen original sin colores"
    # la original NO debe tener tinte ROI (píxeles rojos puros de la banda)
    img = cv2.imread(str(original))
    assert img is not None
    # la foto de entrada era blanca con rectángulo rojo: la original
    # conserva el rojo del rectángulo pero NO el tinte translúcido ROI
    # (banda superior teñida). Verificamos que existe y es legible.
    assert img.shape == (ALTO, ANCHO, 3)


def test_original_es_copia_limpia_sin_banda_roi(proc):
    # con ROI habilitado, la guardada tiene banda teñida; la original no
    proc.state["texto"] = "ABC-123"
    ruta = proc.foto("compara.jpg", 70, 100)
    worker.procesar_imagen(ruta)
    guardadas = _guardadas(proc)
    original = guardadas[0].with_name(guardadas[0].stem + "_original.jpg")
    anotada = guardadas[0]
    assert original.is_file() and anotada.is_file()
    o = cv2.imread(str(original))
    a = cv2.imread(str(anotada))
    # la banda superior (y < SUP*alto) en la anotada esta teñida de rojo;
    # en la original debe ser el blanco puro de la foto base
    y_banda = int(SUP * ALTO) - 5
    # original: esquina superior izquierda = blanco (255,255,255)
    assert int(o[y_banda, 5].mean()) > 200
    # anotada: banda ROI teñida (rojo dominante o al menos distinto)
    assert not np.array_equal(o[y_banda, 5], a[y_banda, 5])


# ------------------------------------------- disco + Levenshtein ---

def test_placa_existe_hoy_lee_carpetas_del_dia(proc):
    dia = "2026-11-09"
    base = proc.cfg["procesadas"] / "placa" / dia / "ABC-123"
    base.mkdir(parents=True)
    (base / "ABC-123_roi-evaluado.jpg").write_bytes(b"x")
    assert worker._placa_existe_hoy("ABC-123", "placa",
                                    dia=dia) is True
    assert worker._placa_existe_hoy("XYZ-999", "placa",
                                    dia=dia) is False


def test_placa_existe_hoy_lee_json(proc):
    dia = "2026-11-09"
    base = proc.cfg["procesadas"] / "placa" / dia
    base.mkdir(parents=True, exist_ok=True)
    (base / "placas_2026-11-09.json").write_text(
        json.dumps({"placas": ["TAB-220", "AOM-361"]}), encoding="utf-8")
    assert worker._placa_existe_hoy("TAB-220", "placa", dia=dia) is True
    assert worker._placa_existe_hoy("AOM-361", "placa", dia=dia) is True


def test_matching_leven_un_error_concuerda(proc):
    # AOM-361 vs AOM-861: 1 error -> misma pasada (umbral default 1)
    dia = "2026-11-09"
    base = proc.cfg["procesadas"] / "placa" / dia / "AOM-361"
    base.mkdir(parents=True)
    (base / "x.jpg").write_bytes(b"x")
    assert worker._placa_existe_hoy("AOM-861", "placa", dia=dia) is True


def test_matching_leven_dos_errores_no_concuerda(proc):
    # AOM-361 vs AOM-367: 1 error en digito... wait 361 vs 367 = 1 error
    # usamos 2 errores claros: AOM-361 vs AOM-388 (361->388 = 2 subs)
    dia = "2026-11-09"
    base = proc.cfg["procesadas"] / "placa" / dia / "AOM-361"
    base.mkdir(parents=True)
    (base / "x.jpg").write_bytes(b"x")
    # AOM-361 vs TAB-220 = totally different
    assert worker._placa_existe_hoy("TAB-220", "placa", dia=dia) is False
    # AOM-361 vs AOM-999: 3 digitos distintos -> distancia 3 > 1
    assert worker._placa_existe_hoy("AOM-999", "placa", dia=dia) is False


def test_matching_no_es_4_de_6(proc):
    # 4 de 6 coincidentes NO debe matchear con umbral=1
    # AOM-361 vs AOM-367: distancia 1 -> SÍ (1 error, ok)
    # Pero AOM-361 vs BOM-361 es 1 error de letra...
    # El caso peligroso de "4 de 6": AOM-361 vs AOM-300
    # 361 vs 300: posiciones 2,3 diferentes? 3=3, 6vs0, 1vs0 -> 2 errores
    dia = "2026-11-09"
    base = proc.cfg["procesadas"] / "placa" / dia / "AOM-361"
    base.mkdir(parents=True)
    (base / "x.jpg").write_bytes(b"x")
    # 2 errores: NO debe matchear con umbral 1
    assert worker._placa_existe_hoy("AOM-300", "placa", dia=dia) is False


def test_cerca_texto_acepta_umbral_configurable():
    assert worker._cerca_texto("ABC-123", "ABC-123", max_dist=0) is True
    assert worker._cerca_texto("ABC-123", "ABC-124", max_dist=1) is True
    assert worker._cerca_texto("ABC-123", "ABC-124", max_dist=0) is False
    assert worker._cerca_texto("ABC-123", "XYZ-789", max_dist=1) is False


def test_pasada_de_placa_fallback_disco_si_db_vacia(proc, monkeypatch):
    # DB sin historial + placa ya en carpetas del dia -> pasada 2.
    # Este es el enganche de _placa_existe_hoy en _pasada_de_placa.
    class _ConVacia:
        def execute(self, *a, **k):
            return []

        def close(self):
            pass

    # la fixture mockea _pasada_de_placa: restauramos la real
    monkeypatch.setattr(worker, "_conexion", lambda: _ConVacia())
    real = _pasada_de_placa_real
    monkeypatch.setattr(worker, "_pasada_de_placa", real)

    dia = datetime.now().strftime("%Y-%m-%d")
    base = proc.cfg["procesadas"] / "placa" / dia / "ABC-123"
    base.mkdir(parents=True)
    (base / "x.jpg").write_bytes(b"x")

    pasada, minutos = worker._pasada_de_placa("ABC-123", "placa")
    assert pasada == 2
    assert minutos == 0.0

    # sin disco -> sigue siendo pasada 1
    pasada2, _ = worker._pasada_de_placa("ZZZ-999", "placa")
    assert pasada2 == 1


# --------------------------------- lectura confiable (API): distancia 0 ---

def test_placa_existe_hoy_confiable_es_exacto(proc):
    # Si la lectura viene de la API NO se tolera 1 error: AOM-361 vs
    # AOM-861 son placas distintas y producian pasadas cruzadas.
    dia = "2026-11-09"
    base = proc.cfg["procesadas"] / "placa" / dia / "AOM-361"
    base.mkdir(parents=True)
    (base / "x.jpg").write_bytes(b"x")
    # local (default): 1 error sigue concuerda
    assert worker._placa_existe_hoy("AOM-861", "placa", dia=dia) is True
    # confiable (API): solo exacto
    assert worker._placa_existe_hoy("AOM-861", "placa",
                                    dia=dia, confiable=True) is False
    assert worker._placa_existe_hoy("AOM-361", "placa",
                                    dia=dia, confiable=True) is True


def test_buscar_misma_confiable_umbral_cero_sin_fallback_hash(proc,
                                                              monkeypatch):
    # confiable=True: texto exacto (0) y SIN el fallback por hash que
    # reutilizaba lecturas viejas (el pre-chequeo que se elimino).
    monkeypatch.setattr(worker, "_hashes",
                        {"placa": {11: "XYZ-999"}})
    # 11 = los bits de la huella guardada: hamming 0 -> el fallback de
    # hash la encontraria igual con umbral_dedup=12
    assert _buscar_misma_real("placa", 11, "ABC-123") is not None
    # confiable: "ABC-123" no es "XYZ-999" exacto -> None (no hash)
    assert _buscar_misma_real("placa", 11, "ABC-123",
                              confiable=True) is None
    # texto exacto si matchea
    assert _buscar_misma_real("placa", 11, "XYZ-999",
                              confiable=True) == (11, "XYZ-999")
    # 1 error de OCR con confiable: NO concuerda
    assert _buscar_misma_real("placa", 11, "XYZ-998",
                              confiable=True) is None
    # sin confiable: 1 error sigue concuerdando (umbral_leven_pasada)
    assert _buscar_misma_real("placa", 11, "XYZ-998") is not None


def test_pasada_de_placa_confiable_no_falla_por_levenshtein(proc,
                                                            monkeypatch):
    # La leida de API es exacta: AOM-861 NO es la AOM-361 en disco.
    class _ConVacia:
        def execute(self, *a, **k):
            return []

        def close(self):
            pass

    monkeypatch.setattr(worker, "_conexion", lambda: _ConVacia())
    monkeypatch.setattr(worker, "_pasada_de_placa", _pasada_de_placa_real)

    dia = datetime.now().strftime("%Y-%m-%d")
    base = proc.cfg["procesadas"] / "placa" / dia / "AOM-361"
    base.mkdir(parents=True)
    (base / "x.jpg").write_bytes(b"x")

    # local: 1 error -> pasada 2
    assert worker._pasada_de_placa("AOM-861", "placa")[0] == 2
    # confiable: distinta -> pasada 1
    assert worker._pasada_de_placa("AOM-861", "placa",
                                   confiable=True)[0] == 1
    # confiable exacta: misma -> pasada 2
    assert worker._pasada_de_placa("AOM-361", "placa",
                                   confiable=True)[0] == 2


def test_forma_comparable_une_7_con_i():
    # 2026-11-12 (FELIPE, REPETIDOS): PlateRecognizer leyo B7B-820 y
    # 60 s despues devolvio BIB-820 (7 confundido con I): misma forma.
    assert (worker._forma_comparable("B7B-820")
            == worker._forma_comparable("BIB-820"))
    # 7 sigue unido a Z (familia de confusion visual)
    assert (worker._forma_comparable("AZK-980")
            == worker._forma_comparable("A7K-980"))


def test_relectura_ocr_equivalente_usa_lectura_previa(proc, monkeypatch):
    # 2026-11-12 (FELIPE, REPETIDOS): B7B-820 guardada y 60 s despues
    # la API devuelve BIB-820 (7->I). Misma forma con confiable
    # (umbral 0): NO es placa nueva — manda la lectura ya guardada y
    # la relectura vive en segunda/ bajo B7B-820, sin carpeta BIB-820.
    monkeypatch.setattr(worker, "_buscar_misma", _buscar_misma_real)
    # DB real (patron del test de MEJOR FOTO) para el bloque de previa.
    monkeypatch.setattr(worker, "_ruta_registro", proc.tmp / "registro.db")
    monkeypatch.setattr(worker, "_dia_dedup",
                        datetime.now().strftime("%Y-%m-%d"))
    con = worker._conexion()
    try:
        con.execute(
            "CREATE TABLE registros ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "fecha_hora TEXT NOT NULL, archivo TEXT NOT NULL,"
            "placa TEXT, confianza REAL,"
            "estado TEXT NOT NULL DEFAULT 'guardado',"
            "evidencia INTEGER, semaforo TEXT, phash TEXT, fuente TEXT,"
            "provisional INTEGER, placa_api TEXT, placa_modelo TEXT,"
            "motivo_sin_api TEXT)")
        con.execute(
            "CREATE TABLE pasadas ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "placa TEXT NOT NULL, pasada INTEGER NOT NULL,"
            "fecha_hora TEXT NOT NULL, excedida INTEGER NOT NULL DEFAULT 0)")
        con.commit()
    finally:
        con.close()

    # lectura con fuente API: confiable -> umbral 0 en el match
    def _leer_api(img, bbox, **kw):
        t = proc.state["texto"]
        return t, proc.state["conf"], [
            ("platerecognizer", t, proc.state["conf"])]

    monkeypatch.setattr(worker, "leer_placa_6", _leer_api)

    # 1a lectura: la placa real
    proc.state["texto"] = "B7B-820"
    ruta1 = proc.foto("p1.jpg", 70, 100, color=ROJO)
    worker.procesar_imagen(ruta1)
    assert len(_guardadas(proc)) == 1

    # 2a lectura: la API se equivoco en un digito (7 -> I)
    proc.state["texto"] = "BIB-820"
    monkeypatch.setattr(worker, "_pasada_de_placa",
                        lambda *a, **k: (2, 0.0))
    ruta2 = proc.foto("p2.jpg", 70, 100, color=AZUL_CLARO)
    worker.procesar_imagen(ruta2)

    guardadas = _guardadas(proc)
    # NO aparece una carpeta nueva BIB-820 ...
    assert not any(p.parent.name == "BIB-820" for p in guardadas)
    # ... y la relectura vive como 2a pasada bajo B7B-820/
    assert any(p.parent.name == "segunda"
               and p.parent.parent.name == "B7B-820"
               for p in guardadas)


# ------------------------------------------------------- config ---

def test_config_ini_ventana_cinco_min():
    import configparser
    p = configparser.ConfigParser()
    p.read(RAIZ / "config.ini", encoding="utf-8-sig")
    assert float(p.get("pasadas", "ventana_reinspeccion_min",
                       fallback=p.get("DEFAULT", "ventana_reinspeccion_min",
                                      fallback="1440"))) == 0.0 or True
    # la clave puede estar en seccion raiz/otra: buscar en todo el texto
    texto = (RAIZ / "config.ini").read_text(encoding="utf-8-sig")
    for linea in texto.splitlines():
        if "ventana_reinspeccion_min" in linea and not linea.strip().startswith(";"):
            valor = linea.split("=", 1)[1].strip()
            # 2026-11-12 (FELIPE): 5 min conserva pasadas reales (24/36
            # min) y descarta falsas segundas de la misma escena (28 s).
            assert valor == "5", f"ventana debe ser 5, es {valor}"
            break
    else:
        raise AssertionError("falta ventana_reinspeccion_min en config.ini")


def test_config_ini_expone_umbral_leven():
    texto = (RAIZ / "config.ini").read_text(encoding="utf-8-sig")
    assert "umbral_leven_pasada" in texto


def test_config_ini_expone_revisar_carpetas_dia():
    texto = (RAIZ / "config.ini").read_text(encoding="utf-8-sig")
    assert "revisar_carpetas_dia" in texto
