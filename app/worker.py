from __future__ import annotations

import json
import csv
import logging
import os
import queue
import re
import shutil
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2
import numpy as np

cv2.setNumThreads(1)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

try:
    from modelo import DetectorPlacas, RUTA_ONNX, enmascarar_osd
except Exception:
    DetectorPlacas = None
    RUTA_ONNX = None
    enmascarar_osd = None

try:
    from ocr import (leer_placa, leer_placa_6, evidencia_y_color,
                     _normalizar, es_placa_6)
except Exception:
    leer_placa = None
    leer_placa_6 = None
    evidencia_y_color = None
    _normalizar = None
    es_placa_6 = None

logger = logging.getLogger("cytv.worker")

_AOI_ACTIVA = False
_ROBOFLOW_ACTIVO = False
_leer_local = leer_placa_6

COLA: "queue.Queue[str]" = queue.Queue(maxsize=1000)
_encolados: set[str] = set()
_encolados_lock = threading.Lock()

_MOTIVOS_VOLUMEN = {"sin_texto", "color_no_placa"}

FOLDER_POR_CANAL = {"01": "placa", "02": "placa"}
CARPETA_POR_DEFECTO = "placa"
_RE_CANAL = re.compile(r"_(\d{2})_")

FALLOS_SIN_REEVAL = {"baja_nitidez", "sobreexpuesta", "ocluida",
                     "fuera_franja", "fuera_de_area"}
LETRAS_REVISAR = "WQH"

_ultima_imagen: dict[str, np.ndarray] = {}

_hashes: dict[str, dict[int, str]] = {}
_placas: dict[str, set[str]] = {}

_dia_dedup: str = ""

# Timestamp de camara (YYYYMMDDHHMMSS...) de la ULTIMA placa guardada:
# base del cooldown por evento/rafaga (un carro = una placa por pasada).
_ultima_placa_foto_ts: datetime | None = None

_cfg: dict = {}
_ruta_registro: Path | None = None

_detector = None
_reposo_seg: float = 600.0
_t_ultima_actividad: float = 0.0
_en_reposo = False
_json_pendiente = False

# Ventana de evaluacion por vista (2026-11-10, pedido FELIPE): de una
# RAFAGA se evalua UNA sola foto cada roi_eval_ventana_seg (la mas
# cerca de la linea de prioridad); las demas se descartan internas.
# {vista: {"ruta": Path, "score": (dist_linea, -area), "fase": ...}}
# fases: "abierta" (candidata vigente) | "lista" (ventana cerrada,
# elegida encolada esperando su reproceso -> ahi SI pasa por API).
_eval_ventanas: dict[str, dict] = {}
_eval_ventanas_lock = threading.Lock()


def _conexion() -> sqlite3.Connection:
    con = sqlite3.connect(_ruta_registro)
    con.execute("PRAGMA busy_timeout=5000")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def _prioridad(normal: bool) -> None:
    try:
        import ctypes
        import ctypes.wintypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [ctypes.wintypes.HANDLE,
                                         ctypes.wintypes.DWORD]
        k32.SetPriorityClass.restype = ctypes.wintypes.BOOL
        # activo = BELOW_NORMAL (0x80): la camara cede CPU a las demas apps
        # del server; reposo = IDLE (0x40). Antes activo era NORMAL (0x20)
        # y con trafico continuo el servicio no soltaba el procesador.
        ok = k32.SetPriorityClass(k32.GetCurrentProcess(),
                                  0x80 if normal else 0x40)
        if not ok:
            raise OSError("SetPriorityClass fallo")
    except Exception:
        logger.debug("No se pudo cambiar la prioridad del proceso",
                     exc_info=True)


def _asegurar_detector():
    global _detector
    if (_detector is None and DetectorPlacas is not None
            and RUTA_ONNX is not None and RUTA_ONNX.is_file()):
        try:
            _detector = DetectorPlacas(
                RUTA_ONNX, umbral=float(_cfg.get("umbral_modelo", 0.25)))
        except Exception:
            logger.exception("No se pudo recargar el detector ONNX")
            _detector = None
    return _detector


def _entrar_en_reposo() -> None:
    global _detector, _en_reposo
    if _en_reposo:
        return
    _en_reposo = True
    _detector = None
    try:
        import ocr
        ocr._detector_onnx = None
        ocr._motor = None
    except Exception:
        pass
    try:
        import reconocedor
        reconocedor._sesion = None
        reconocedor._sesion_mtime = None
    except Exception:
        pass
    _prioridad(False)
    logger.info("REPOSO: modelos liberados, prioridad idle")


def _despertar() -> None:
    global _en_reposo, _t_ultima_actividad
    _t_ultima_actividad = time.time()
    if not _en_reposo:
        return
    _en_reposo = False
    _asegurar_detector()
    _prioridad(True)
    logger.info("DESPERTAR: detector recargado al recibir foto")


def _marcar_json() -> None:
    global _json_pendiente
    _json_pendiente = True


def _vaciar_json_pendiente() -> None:
    global _json_pendiente
    if not _json_pendiente:
        return
    _json_pendiente = False
    try:
        generar_json_dia(_dia_dedup)
    except Exception:
        logger.exception("No se pudo actualizar el reporte JSON del dia")


def _carpeta_de(nombre_archivo: str) -> str:
    encontrado = _RE_CANAL.search(nombre_archivo)
    canal = encontrado.group(1) if encontrado else ""
    return FOLDER_POR_CANAL.get(canal, CARPETA_POR_DEFECTO)


def etiqueta_fuente(intentos: list | None) -> str:
    """Sufijo de nombre que documenta que la API leo la placa.

    Prioridad roboflow > PlateRecognizer; el motor local no etiqueta.
    """
    nombres = [n for n, _t, _c in (intentos or [])]
    if "roboflow" in nombres:
        return "_api-Roboflow"
    if "api" in nombres:
        return "_api-PlateRecognizer"
    return ""


def fuente_de_intentos(intentos: list | None) -> str:
    """Origen de la lectura para la columna registros.fuente.

    Unica fuente de verdad de la semantica: 'roboflow' / 'platerecognizer'
    / 'modelo'. La usan registrar(), el reemplazo de MEJOR FOTO y el
    matching confiable (Levenshtein 0) de las lecturas de API.
    """
    nombres = [n for n, _t, _c in (intentos or [])]
    if "roboflow" in nombres:
        return "roboflow"
    if "api" in nombres:
        return "platerecognizer"
    return "modelo"


def texto_de_api(tx: str | None) -> str:
    """Texto devuelto por una API validado con la regla del motor local.

    REGLA: una placa son EXACTAMENTE 6 alfanumericos (3+3). Se aplica
    es_placa_6() a cada linea normalizada.

    plate_text de Roboflow viene multi-linea con rótulos al lado
    ('PERU\\nT2K-790', 'D5W-782\\nTOYOTA'): se valida CADA linea y solo
    se acepta si hay EXACTAMENTE UNA candidata con forma de placa. Si
    hay varias (ambigua) o ninguna (rotulo puro: PERU, VISTONY) queda
    vacio y la lectura cae a PlateRecognizer/motor local sobre la misma
    bbox.
    """
    if not tx or _normalizar is None:
        return ""
    candidatas: list[str] = []
    for linea in str(tx).splitlines():
        normal = _normalizar(linea.strip())
        if not normal or not es_placa_6(normal):
            continue          # no tiene 6 alfanumericos: no es placa
        if normal not in candidatas:
            candidatas.append(normal)
    return candidatas[0] if len(candidatas) == 1 else ""


def _carpeta_de_ruta(relativo: str) -> str:
    parte = Path(relativo).parts[0] if relativo else ""
    return parte if parte in FOLDER_POR_CANAL.values() else CARPETA_POR_DEFECTO


def _cargar_dedup_dia(con, limite_dia: str) -> None:
    """Carga la memoria del dedup del dia (_hashes/_placas) desde registros.

    La DB es la memoria; el archivo en disco es solo evidencia: NO se
    chequea is_file(). Con ese filtro, una foto movida o borrada (p. ej.
    llevada a FOTOS CAMARA) borraba la placa de la memoria y la segunda
    toma del mismo carro entraba como si fuera nueva: repeticiones.
    """
    global _hashes, _placas
    _hashes = {}
    _placas = {}
    for valor, archivo, placa in con.execute(
            "SELECT phash, archivo, placa FROM registros "
            "WHERE phash IS NOT NULL AND fecha_hora >= ?",
            (limite_dia,)):
        if not archivo:
            continue
        _hashes.setdefault(_carpeta_de_ruta(archivo), {})[
            int(valor, 16)] = placa or ""
    for texto, archivo in con.execute(
            "SELECT placa, archivo FROM registros "
            "WHERE placa IS NOT NULL AND fecha_hora >= ?",
            (limite_dia,)):
        if not texto or not archivo:
            continue
        _placas.setdefault(_carpeta_de_ruta(archivo), set()).add(texto)


def iniciar(cfg: dict) -> None:
    global _cfg, _ruta_registro, _hashes, _placas, _dia_dedup, _detector, _reposo_seg
    global leer_placa_6, _AOI_ACTIVA, _ROBOFLOW_ACTIVO
    _cfg = cfg
    _ruta_registro = cfg["registro"]
    _reposo_seg = max(0.0, float(cfg.get("reposo_minutos", 10.0))) * 60.0

    fuentes = (cfg.get("aoi_tokens", ""),
               os.environ.get("PLATERECOGNIZER_TOKEN", ""))
    if _leer_local is not None and any(str(f).strip() for f in fuentes):
        try:
            import platerecognizer
            tokens = platerecognizer.configurar(*fuentes)
            if tokens:
                leer_placa_6 = platerecognizer.leer_placa_6_api
                _AOI_ACTIVA = True
                logger.info("AOI activa: %d clave(s) de PlateRecognizer "
                            "(orden de prueba: %s)", len(tokens),
                            ", ".join(t[:4] + "***" for t in tokens))
        except Exception:
            logger.exception("AOI (PlateRecognizer) no disponible; "
                             "sigo con el motor local")

    fuentes_rf = (cfg.get("roboflow_key", ""),
                  os.environ.get("ROBOFLOW_API_KEY", ""))
    rf_key = next((str(f).strip() for f in fuentes_rf if str(f).strip()), "")
    if rf_key:
        try:
            import roboflow
            if roboflow.configurar(rf_key, cfg.get("roboflow_workspace", ""),
                                   cfg.get("roboflow_workflow", "")):
                _ROBOFLOW_ACTIVO = True
                logger.info("Roboflow activo: workflow %s de %s",
                            cfg.get("roboflow_workflow", ""),
                            cfg.get("roboflow_workspace", ""))
        except Exception:
            logger.exception("Roboflow no disponible; sigo con la cadena "
                             "local")
    else:
        logger.info("Roboflow sin clave (config.ini [roboflow] api_key o "
                    "ROBOFLOW_API_KEY); encadena con el detector local")

    cfg["procesadas"].mkdir(parents=True, exist_ok=True)
    if str(cfg.get("modo_descarte", "eliminar")).strip().lower() == "retener":
        _carpeta_descartes().mkdir(parents=True, exist_ok=True)

    con = _conexion()
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS registros (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha_hora TEXT    NOT NULL,
                archivo    TEXT    NOT NULL,
                placa      TEXT,
                confianza  REAL,
                estado     TEXT    NOT NULL DEFAULT 'guardado',
                evidencia  INTEGER,
                semaforo   TEXT
            )
            """
        )
        columnas = {fila[1] for fila in con.execute("PRAGMA table_info(registros)")}
        if "phash" not in columnas:
            con.execute("ALTER TABLE registros ADD COLUMN phash TEXT")
        if "evidencia" not in columnas:
            con.execute("ALTER TABLE registros ADD COLUMN evidencia INTEGER")
        if "semaforo" not in columnas:
            con.execute("ALTER TABLE registros ADD COLUMN semaforo TEXT")
        for col, tipo in (("fuente", "TEXT"), ("provisional", "INTEGER"),
                          ("placa_api", "TEXT"), ("placa_modelo", "TEXT"),
                          ("motivo_sin_api", "TEXT")):
            if col not in columnas:
                con.execute(f"ALTER TABLE registros ADD COLUMN {col} {tipo}")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS pasadas (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                placa      TEXT    NOT NULL,
                pasada     INTEGER NOT NULL,
                fecha_hora TEXT    NOT NULL,
                excedida   INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_pasadas_placa "
            "ON pasadas (placa, fecha_hora)"
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS candidatas (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha_hora TEXT    NOT NULL,
                placa      TEXT    NOT NULL,
                archivo    TEXT    NOT NULL,
                evidencia  INTEGER,
                votos      INTEGER,
                estado     TEXT    NOT NULL DEFAULT 'pendiente'
            )
            """
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_candidatas_estado "
            "ON candidatas (estado, placa)"
        )
        con.commit()
        _dia_dedup = datetime.now().strftime("%Y-%m-%d")
        limite_dia = f"{_dia_dedup}T00:00:00"
        # REGLA primero: ninguna carpeta de placa sin su foto. Luego
        # DISCO -> reporte ANTES de cargar el dedup y el JSON: carpetas
        # <vista>\<dia>\<PLACA>\ cuyas filas no llegaron a 'guardado' (o
        # quedaron huerfanas tras un error) se anotan aqui.
        try:
            _depurar_carpetas_vacias()
        except Exception:
            logger.exception("Depuracion de carpetas vacias fallo "
                             "(el servicio sigue)")
        try:
            _reconciliar_carpetas_dia(con, _dia_dedup)
        except Exception:
            logger.exception("Reconciliacion de carpetas del dia fallo "
                             "(el servicio sigue)")
        _cargar_dedup_dia(con, limite_dia)
    finally:
        con.close()
    total_huellas = sum(len(v) for v in _hashes.values())
    total_placas = sum(len(v) for v in _placas.values())
    logger.info("Worker listo: registro en %s, %d huellas y %d placas cargadas "
                "(dedup del dia %s, vistas: %s)",
                _ruta_registro.name, total_huellas, total_placas, _dia_dedup,
                ", ".join(f"{k}={len(v)} placas"
                          for k, v in sorted(_placas.items())) or "ninguna")

    if DetectorPlacas is not None and RUTA_ONNX is not None and RUTA_ONNX.is_file():
        try:
            _detector = DetectorPlacas(RUTA_ONNX,
                                       umbral=float(cfg.get("umbral_modelo", 0.25)))
            logger.info("Deteccion: MODELO ONNX activo (%s)", RUTA_ONNX.name)
            try:
                import ocr
                ocr._detector_onnx = _detector
            except Exception:
                pass
        except Exception:
            logger.exception("No se pudo cargar el modelo ONNX; "
                             "sigo con el detector geometrico")
            _detector = None
    else:
        logger.info("Deteccion: sin modelo ONNX, detector geometrico")

    if leer_placa is not None:
        try:
            leer_placa(np.zeros((48, 160, 3), dtype=np.uint8))
            logger.info("OCR: RapidOCR listo (dedup exacto por texto de placa)")
        except Exception:
            logger.exception("No se pudo iniciar el OCR; sigo con el dedup "
                             "por hash perceptual")
    else:
        logger.info("OCR: no disponible, dedup solo por hash perceptual")

    try:
        ruta_json = generar_json_dia(_dia_dedup)
        logger.info("Reporte JSON del dia %s actualizado: %s",
                    _dia_dedup, ruta_json)
    except Exception:
        logger.exception("No se pudo generar el reporte JSON del dia %s",
                         _dia_dedup)
    try:
        con = _conexion()
        try:
            dias = [fila[0] for fila in con.execute(
                "SELECT DISTINCT substr(fecha_hora, 1, 10) FROM registros "
                "WHERE estado = 'guardado' AND placa IS NOT NULL "
                "AND placa != '' ORDER BY 1")]
        finally:
            con.close()
        anteriores = [d for d in dias if d != _dia_dedup]
        for fecha in anteriores:
            generar_json_dia(fecha)
        if anteriores:
            logger.info("Reportes JSON de dias anteriores regenerados: %d (%s)",
                        len(anteriores), ", ".join(anteriores))
    except Exception:
        logger.exception("No se pudieron regenerar los reportes JSON anteriores")

    _replay_descartes_inicio()


def _actualizar_dia() -> None:
    global _dia_dedup, _hashes, _placas, _ultima_imagen
    hoy = datetime.now().strftime("%Y-%m-%d")
    if hoy == _dia_dedup:
        return
    if _dia_dedup:
        try:
            ruta_json = generar_json_dia(_dia_dedup)
            logger.info("Reporte JSON del dia %s cerrado: %s",
                        _dia_dedup, ruta_json)
        except Exception:
            logger.exception("No se pudo cerrar el reporte JSON del dia %s",
                             _dia_dedup)
        logger.info("Nuevo dia (%s): dedup reiniciado, las placas de ayer "
                    "pueden registrarse otra vez", hoy)
    _dia_dedup = hoy
    _hashes = {}
    _placas = {}
    _ultima_imagen = {}
    try:
        _purgar_rechazadas_api()
    except Exception:
        logger.debug("No se pudo purgar rechazadas_api", exc_info=True)


def registrar(archivo: str, placa: str | None = None,
              confianza: float | None = None, estado: str = "guardado",
              phash: int | None = None, evidencia: int | None = None,
              semaforo: str | None = None, fuente: str | None = None,
              provisional: bool = False, placa_api: str | None = None,
              placa_modelo: str | None = None,
              motivo_sin_api: str | None = None) -> None:
    if _ruta_registro is None:
        return
    con = _conexion()
    try:
        con.execute(
            "INSERT INTO registros (fecha_hora, archivo, placa, confianza, "
            "estado, phash, evidencia, semaforo, fuente, provisional, "
            "placa_api, placa_modelo, motivo_sin_api) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), archivo, placa,
             confianza, estado, None if phash is None else f"{phash:016x}",
             evidencia, semaforo, fuente, 1 if provisional else 0,
             placa_api, placa_modelo, motivo_sin_api),
        )
        con.commit()
    finally:
        con.close()


UMBRAL_ETIQUETA_AUTO = 0.55


def _etiqueta_auto(texto: str, intentos: list) -> str:
    if not texto:
        return ""

    def _c(s: str) -> str:
        return (s or "").upper().replace("-", "").strip()

    texto_c = _c(texto)
    for nombre, t, _conf in intentos:
        if nombre == "crnn" and t and _c(t) == texto_c:
            return texto.upper()
    return ""


def _encolar_entrenamiento(ruta: Path, img: np.ndarray,
                           bbox: tuple[int, int, int, int] | None,
                           texto: str, intentos: list) -> bool:
    etiqueta = _etiqueta_auto(texto, intentos)
    if not etiqueta:
        logger.info("Sin etiqueta auto (no se encola): %s | %s",
                    texto or "(sin texto)", ruta.name)
        return False
    try:
        dia_dir = _carpeta_descartes().parent / "cola_entrenamiento" \
            / datetime.now().strftime("%Y-%m-%d")
        dia_dir.mkdir(parents=True, exist_ok=True)
        revisar = any(ch in etiqueta for ch in LETRAS_REVISAR)
        csv_dia = dia_dir / ("etiquetas_REVISAR.csv" if revisar
                             else "etiquetas.csv")
        if csv_dia.is_file():
            with csv_dia.open(encoding="utf-8") as f:
                for fila in csv.DictReader(f):
                    if fila["archivo"] == ruta.name:
                        return False
        destino = _destino_libre(dia_dir, ruta.name)
        shutil.copy2(ruta, destino)
        if bbox is not None:
            x, y, w, h = bbox
            alto, ancho = img.shape[:2]
            etiqueta_txt = (f"0 {(x + w / 2) / ancho:.6f} "
                            f"{(y + h / 2) / alto:.6f} {w / ancho:.6f} "
                            f"{h / alto:.6f}\n")
            destino.with_suffix(".txt").write_text(etiqueta_txt,
                                                   encoding="utf-8")
        with csv_dia.open("a", newline="", encoding="utf-8") as f:
            escritor = csv.writer(f)
            if csv_dia.stat().st_size == 0:
                escritor.writerow(("dia", "archivo", "placa"))
            escritor.writerow((dia_dir.name, destino.name, etiqueta))
        logger.info("COLA ENTRENAMIENTO%s: %s <- %s",
                    " (REVISAR)" if revisar else "", etiqueta, ruta.name)
        return True
    except Exception:
        logger.exception("No se pudo encolar la foto %s", ruta.name)
        return False


def _carpeta_reportes() -> Path:
    propio = _cfg.get("reportes")
    if propio:
        return Path(propio)
    return _cfg["procesadas"].parent / "reportes"


def generar_json_dia(fecha: str) -> Path:
    if _ruta_registro is None:
        raise RuntimeError("worker no inicializado")
    con = _conexion()
    try:
        filas = con.execute(
            "SELECT fecha_hora, archivo, placa, confianza, evidencia, "
            "semaforo, fuente, provisional FROM registros "
            "WHERE fecha_hora LIKE ? AND estado = 'guardado' "
            "AND placa IS NOT NULL AND placa != '' ORDER BY fecha_hora",
            (f"{fecha}%",),
        ).fetchall()
        pasadas_dia: dict[str, tuple[int, bool]] = {}
        for placa, pasada, excedida in con.execute(
                "SELECT placa, pasada, excedida FROM pasadas "
                "WHERE fecha_hora LIKE ? ORDER BY fecha_hora",
                (f"{fecha}%",)):
            pasadas_dia[placa] = (pasada, bool(excedida))
    finally:
        con.close()

    por_placa: dict[str, dict] = {}
    for (fecha_hora, archivo, placa, confianza, evidencia, semaforo,
         fuente, provisional) in filas:
        entrada = por_placa.get(placa)
        if entrada is None:
            pasada, excedida = pasadas_dia.get(placa, (1, False))
            entrada = por_placa[placa] = {
                "placa": placa,
                "pasada": pasada,
                "excedida": excedida,
                "primera_vez": fecha_hora,
                "ultima_vez": fecha_hora,
                "detecciones": 0,
                "confianza_max": 0.0,
                "evidencia_max": 0,
                "semaforo": "sin_dato",
                "fuente": fuente,
                "provisional": bool(provisional),
                "imagenes": {"placa": [], "recorte": []},
            }
        entrada["ultima_vez"] = fecha_hora
        entrada["detecciones"] += 1
        entrada["fuente"] = fuente
        entrada["provisional"] = bool(provisional)
        if confianza is not None:
            entrada["confianza_max"] = max(entrada["confianza_max"], confianza)
        if evidencia is not None and evidencia > entrada["evidencia_max"]:
            entrada["evidencia_max"] = evidencia
            if semaforo:
                entrada["semaforo"] = semaforo
        ruta_foto = _cfg["procesadas"] / archivo
        entrada["imagenes"]["placa"].append(str(ruta_foto))
        crop = ruta_foto.with_name(ruta_foto.stem + "_crop.jpg")
        if crop.is_file():
            entrada["imagenes"]["recorte"].append(str(crop))

    datos = {
        "fecha": fecha,
        "generado": datetime.now().isoformat(timespec="seconds"),
        "total_placas": len(por_placa),
        "total_detecciones": sum(e["detecciones"] for e in por_placa.values()),
        "placas": sorted(por_placa.values(), key=lambda e: e["primera_vez"]),
    }

    carpeta = _carpeta_reportes()
    carpeta.mkdir(parents=True, exist_ok=True)
    destino = carpeta / f"placas-{fecha}.json"
    temporal = carpeta / f"placas-{fecha}.json.tmp"
    temporal.write_text(json.dumps(datos, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    temporal.replace(destino)
    return destino


def _depurar_carpetas_vacias() -> int:
    r"""REGLA: no existen carpetas de placa sin sus archivos.

    Una carpeta <vista>/<fecha>/<PLACA>/ sin foto principal (*.jpg que no
    sea _crop ni _original) es el residuo de un guardado interrumpido: se
    elimina completa (con sus .txt/.crop huerfanos). Tambien se quitan las
    carpetas de fecha que quedaron sin nada. Las carpetas de vista NO se
    tocan: el FTP las verifica al arrancar.
    """
    procesadas = _cfg["procesadas"]
    if not procesadas.is_dir():
        return 0
    eliminadas = 0
    for vista_dir in sorted(p for p in procesadas.iterdir() if p.is_dir()):
        for dia_dir in sorted(p for p in vista_dir.iterdir() if p.is_dir()):
            for placa_dir in sorted(p for p in dia_dir.iterdir() if p.is_dir()):
                # rglob (2026-11-10, FASE 2): las fotos de las pasadas
                # 2+ viven en subcarpetas (segunda/tercera/excedida).
                principales = [f for f in placa_dir.rglob("*.jpg")
                               if not f.stem.endswith(("_crop",
                                                       "_original"))]
                if principales:
                    continue
                try:
                    shutil.rmtree(placa_dir)
                    eliminadas += 1
                    logger.warning("CARPETA VACIA eliminada: %s "
                                   "(sin foto principal)", placa_dir)
                except OSError:
                    logger.exception("No se pudo eliminar la carpeta "
                                     "vacia %s", placa_dir)
            # La carpeta de fecha quedo sin nada (los layout viejos con
            # fotos sueltas tienen archivos: ahi NO se toca nada).
            try:
                if dia_dir.exists() and not any(dia_dir.iterdir()):
                    dia_dir.rmdir()
            except OSError:
                pass
    if eliminadas:
        logger.info("DEPURACION: %d carpeta(s) de placa sin foto "
                    "principal eliminadas", eliminadas)
    return eliminadas


def _reconciliar_carpetas_dia(con: sqlite3.Connection, hoy: str) -> int:
    r"""DISCO -> reporte: carpetas <vista>/<dia>/<PLACA>/ del dia se
    anotan en registros si falta la fila 'guardado'.

    El JSON del reporte sale SOLO de registros con estado 'guardado' y
    placa no vacia; nunca mira el disco. Si una foto se guardo en su
    carpeta de placa pero la fila quedo 'pendiente' (detector no cargado)
    o el registro.db se perdio/recreo, la placa no llegaba al reporte.
    Reglas: fila ausente -> se inserta 'guardado'; solo 'pendiente' ->
    se promueve; ya 'guardado' -> NO se toca, salvo que su archivo ya no
    exista en disco (se repunta a la foto viva de la carpeta).
    """
    vistas = sorted(set(FOLDER_POR_CANAL.values()) | {CARPETA_POR_DEFECTO})
    procesadas = _cfg["procesadas"]
    anotadas = 0
    for vista in vistas:
        raiz = procesadas / vista / hoy
        if not raiz.is_dir():
            continue
        for carpeta in sorted(p for p in raiz.iterdir() if p.is_dir()):
            placa = carpeta.name
            # _guardar creo la carpeta con _placa_base: solo se
            # reconcilian carpetas con forma de placa (6 alnum, letras y
            # digitos, con guion); el resto ('recortes', etc.) se ignora.
            base = placa.replace("-", "")
            if not (len(base) == 6 and any(c.isalpha() for c in base)
                    and any(c.isdigit() for c in base)):
                continue
            fotos = [f for f in carpeta.glob("*.jpg")
                     if not f.stem.endswith(("_crop", "_original"))]
            if not fotos:
                continue
            foto = max(fotos, key=lambda f: f.stat().st_mtime)
            relativo = str(foto.relative_to(procesadas))
            # La fecha es la de la CARPETA (dia del reporte); solo se toma
            # la hora del archivo para conservar el orden cronologico.
            hora = datetime.fromtimestamp(foto.stat().st_mtime)
            fecha_hora = f"{hoy}T{hora.strftime('%H:%M:%S')}"
            filas = con.execute(
                "SELECT id, estado, archivo FROM registros "
                "WHERE placa = ? AND fecha_hora LIKE ?",
                (placa, f"{hoy}%")).fetchall()
            if not filas:
                con.execute(
                    "INSERT INTO registros (fecha_hora, archivo, placa, "
                    "estado) VALUES (?, ?, ?, 'guardado')",
                    (fecha_hora, relativo, placa))
                if not con.execute(
                        "SELECT 1 FROM pasadas WHERE placa = ? "
                        "AND fecha_hora LIKE ? LIMIT 1",
                        (placa, f"{hoy}%")).fetchone():
                    con.execute(
                        "INSERT INTO pasadas (placa, pasada, fecha_hora, "
                        "excedida) VALUES (?, 1, ?, 0)",
                        (placa, fecha_hora))
                anotadas += 1
            elif all(e != "guardado" for _i, e, _a in filas):
                # La carpeta existe = la foto se guardo: se anota. Solo
                # se actualizan las filas 'pendiente' de ESTA placa/dia.
                con.execute(
                    "UPDATE registros SET estado = 'guardado', archivo = ? "
                    "WHERE id IN (%s)" % ",".join("?" * len(filas)),
                    (relativo, *[i for i, _e, _a in filas]))
                anotadas += len(filas)
            else:
                # Ya esta en el reporte: no se toca, salvo repuntar filas
                # cuyo archivo ya no existe (divergencia disco/DB).
                for id_fila, _estado, archivo in filas:
                    if archivo and not (procesadas / archivo).is_file():
                        con.execute(
                            "UPDATE registros SET archivo = ? WHERE id = ?",
                            (relativo, id_fila))
                        anotadas += 1
    if anotadas:
        con.commit()
        logger.info("RECONCILIACION: %d fila(s) del %s anotadas o "
                    "repuntadas desde las carpetas de disco",
                    anotadas, hoy)
    return anotadas


def calcular_diferencia(img1: np.ndarray, img2: np.ndarray) -> float:
    img1_peq = cv2.resize(img1, (640, 480))
    img2_peq = cv2.resize(img2, (640, 480))
    gris1 = cv2.cvtColor(img1_peq, cv2.COLOR_BGR2GRAY)
    gris2 = cv2.cvtColor(img2_peq, cv2.COLOR_BGR2GRAY)
    diferencia = cv2.absdiff(gris1, gris2)
    _, umbral = cv2.threshold(diferencia, 35, 255, cv2.THRESH_BINARY)
    return (np.count_nonzero(umbral) / (umbral.shape[0] * umbral.shape[1])) * 100


def encolar(ruta: str) -> bool:
    with _encolados_lock:
        if ruta in _encolados:
            return False
        try:
            COLA.put_nowait(ruta)
            _encolados.add(ruta)
            return True
        except queue.Full:
            logger.warning("Cola llena: %s queda para el barrido de huerfanos", Path(ruta).name)
            return False


_MOTIVOS_SIN_REPLAY = {"duplicado", "duplicado_placa"}
_EXT_FOTO = (".jpg", ".jpeg", ".png")


def _limpiar_carpetas_vacias(raiz: Path) -> int:
    # Bottom-up: primero hojas (fuera_de_area/naranja), despues padres
    # (fuera_de_area), despues la fecha. rglob cubre la profundidad.
    borradas = 0
    for fecha in sorted((d for d in raiz.iterdir() if d.is_dir()),
                        reverse=True):
        dirs = sorted((p for p in fecha.rglob("*") if p.is_dir()),
                      key=lambda p: len(p.parts), reverse=True)
        for d in dirs:
            try:
                next(d.iterdir())
            except StopIteration:
                try:
                    d.rmdir()
                    borradas += 1
                except OSError:
                    continue
            except OSError:
                continue
        try:
            next(fecha.iterdir())
        except StopIteration:
            try:
                fecha.rmdir()
                borradas += 1
            except OSError:
                continue
        except OSError:
            continue
    if borradas:
        logger.info("Replay: %d carpeta(s) vacia(s) eliminadas de "
                    "descartadas", borradas)
    return borradas


def _fotos_descartadas_para_replay(raiz: Path) -> list[Path]:
    pendientes: list[Path] = []
    for dia in sorted(d for d in raiz.iterdir() if d.is_dir()):
        for f in sorted(dia.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in _EXT_FOTO:
                continue
            # Excluye si CUALQUIER directorio padre (relativo al dia)
            # esta en _MOTIVOS_SIN_REPLAY (ej: duplicado/, y el
            # anidado fuera_de_area/naranja no esta en la lista).
            partes = f.relative_to(dia).parts[:-1]
            if any(p.lower() in _MOTIVOS_SIN_REPLAY for p in partes):
                continue
            pendientes.append(f)
    return pendientes


def _replay_descartes_inicio() -> None:
    activo = str(_cfg.get("replay_inicio", "si")).strip().lower()
    if activo in ("no", "false", "0", "off"):
        logger.info("Replay al arrancar desactivado (replay_descartes = no)")
        return
    raiz = _carpeta_descartes()
    if not raiz.is_dir():
        return
    _limpiar_carpetas_vacias(raiz)
    pendientes = _fotos_descartadas_para_replay(raiz)
    if not pendientes:
        logger.info("Replay al arrancar: sin fotos pendientes en descartadas")
        return
    pausa = max(0.0, float(_cfg.get("replay_pausa_seg", 1.0)))
    logger.info("Replay al arrancar: %d foto(s) de descartadas se re-encolan "
                "(%.1fs de pausa; duplicados excluidos; prioridad de lo en "
                "vivo)", len(pendientes), pausa)

    def _alimentar() -> None:
        reencoladas = 0
        for foto in pendientes:
            while COLA.qsize() > 1:
                time.sleep(1.0)
            if encolar(str(foto)):
                reencoladas += 1
            if pausa:
                time.sleep(pausa)
        logger.info("Replay al arrancar terminado: %d/%d foto(s) "
                    "re-encoladas", reencoladas, len(pendientes))
        try:
            COLA.join()
            _limpiar_carpetas_vacias(raiz)
        except Exception:
            logger.exception("No se pudieron limpiar las carpetas vacias "
                             "de descartadas")

    threading.Thread(target=_alimentar, daemon=True,
                     name="replay-inicio").start()


def _candidatos_placa(img: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
    if _detector is None:
        _asegurar_detector()
    if _detector is not None:
        # ROI activo: img ya viene enmascarada (mask antes del corte);
        # enmascarar=False evita que ZONAS_OSD (normalizadas al encuadre
        # original) tapen la placa del recorte.
        return _detector.detectar(
            img, enmascarar=not _cfg.get("roi_habilitado", False))

    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris = cv2.bilateralFilter(gris, 9, 75, 75)
    gradiente = cv2.morphologyEx(gris, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    _, binarizada = cv2.threshold(gradiente, 0, 255,
                                  cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binarizada = cv2.morphologyEx(binarizada, cv2.MORPH_CLOSE,
                                  np.ones((9, 2), np.uint8))
    contornos, _ = cv2.findContours(binarizada, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    alto, ancho = img.shape[:2]
    area_img = alto * ancho
    ordenados: list[tuple[float, tuple[int, int, int, int]]] = []

    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0 or w == 0:
            continue
        relacion = w / h
        if not 1.6 <= relacion <= 6.0:
            continue
        if not 0.0015 * area_img <= w * h <= 0.30 * area_img:
            continue
        if x < 0 or y < 0 or x + w > ancho or y + h > alto:
            continue
        ordenados.append((1.0 - abs(relacion - 4.6) / 4.6, (x, y, w, h)))

    ordenados.sort(key=lambda par: par[0], reverse=True)

    elegidos: list[tuple[tuple[int, int, int, int], float]] = []
    for score, (x, y, w, h) in ordenados:
        if len(elegidos) >= 8:
            break
        solape = False
        for (ox, oy, ow, oh), _ in elegidos:
            ix = max(0, min(x + w, ox + ow) - max(x, ox))
            iy = max(0, min(y + h, oy + oh) - max(y, oy))
            inter = ix * iy
            if inter and inter / min(w * h, ow * oh) > 0.4:
                solape = True
                break
        if not solape:
            elegidos.append(((x, y, w, h), round(min(max(score, 0.0), 1.0), 3)))

    return elegidos


def _a_gris(recorte: np.ndarray) -> np.ndarray:
    return recorte if recorte.ndim == 2 else cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)


def _fraccion_oscuros(recorte: np.ndarray) -> float:
    gris = _a_gris(recorte)
    _, texto = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return float(np.count_nonzero(texto)) / texto.size * 100.0


def _nitidez(recorte: np.ndarray) -> float:
    return float(cv2.Laplacian(_a_gris(recorte), cv2.CV_64F).var())


def _fraccion_sobreexpuesta(recorte: np.ndarray) -> float:
    gris = _a_gris(recorte)
    return float(np.count_nonzero(gris > 250)) / gris.size * 100.0


def _fraccion_bordes(recorte: np.ndarray) -> float:
    gris = _a_gris(recorte)
    bordes = cv2.Canny(gris, 80, 200)
    return float(np.count_nonzero(bordes)) / bordes.size * 100.0


def _fraccion_rojo(recorte: np.ndarray) -> float:
    """% de pixeles rojos saturados (cintas reflectoras, luces, stickers).

    Las placas reales miden <4% (promedio ~1%); la cinta reflectora del
    falso positivo conocido mide ~32%.
    """
    hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    rojo = (((h <= 10) | (h >= 170)) & (s >= 70) & (v >= 50))
    return float(np.count_nonzero(rojo)) / rojo.size * 100.0


def _pasa_gate_ancho(w: int, ancho: int, umbral: float = 0.20) -> bool:
    """True si el bbox es estrecho enough para ser placa (no texto lateral).

    Texto lateral de omnibus/camion: 0.24-0.46 del ancho de imagen.
    Placa real (GT-29): 0.037-0.082. Umbral 0.20 deja ~2.4x de margen.
    """
    if ancho <= 0:
        return False
    return (w / ancho) < umbral


# ------------------------------------------------- ROI franja verde ---
# La camara es fija: la linea verde de arriba es el limite del corte
# (todo lo de atras --fondo, timestamp, pared-- se descarta) y la franja
# entre las dos lineas verdes es la referencia de donde debe quedar el
# carro con la placa TOTALMENTE visible. Coordenadas normalizadas (0-1)
# sobre la imagen ORIGINAL, calibradas con scripts\calibrar_roi.py.

def _franja_verde() -> tuple[float, float]:
    """(y_sup, y_inf) normalizadas sobre la imagen original.

    ROI apagado o config rota -> (0.0, 1.0): se degrada a sin ROI en
    vez de fallar (rollback total con habilitado = no).
    """
    if not _cfg.get("roi_habilitado"):
        return 0.0, 1.0
    sup = float(_cfg.get("roi_linea_verde_sup", 0.0))
    inf = float(_cfg.get("roi_linea_verde_inf", 1.0))
    if not (0.0 <= sup < inf <= 1.0) or (inf - sup) < 0.05:
        return 0.0, 1.0
    return sup, inf


def _recortar_roi(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Recorta la foto por encima de la linea verde de arriba.

    Devuelve (imagen_recortada, y_sup_en_px). ROI apagado -> (img, 0.0).
    """
    sup, _ = _franja_verde()
    if sup <= 0.0:
        return img, 0.0
    alto = img.shape[0]
    y_sup = int(round(alto * sup))
    if y_sup <= 0 or y_sup >= alto:
        return img, 0.0
    return img[y_sup:, :], float(y_sup)


def _preparar_imagen(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Mascara OSD (solo con ROI activo) y recorte de la franja verde.

    ORDEN CRITICO: la mascara ZONAS_OSD esta calibrada sobre el encuadre
    ORIGINAL (timestamp arriba-izq, 'Camera 01' abajo-dcha). Si se aplica
    despues del corte, la zona de arriba-izquierda cae sobre la placa del
    recorte y la tapa: medido con GT-29, AFH-559 cumple la franja pero
    con_osd=0 / sin_osd=1 (el detector no la veia). Con ROI activo se
    enmascara ANTES de recortar y el detector entra con enmascarar=False.
    Con ROI apagado no se toca nada: la mascara la sigue aplicando
    detectar por su cuenta, como siempre (rollback total).
    """
    if _cfg.get("roi_habilitado") and enmascarar_osd is not None:
        img = enmascarar_osd(img)
    return _recortar_roi(img)


def _franja_inf_px(alto: int) -> int | None:
    """Borde inferior de la franja en pixeles de la imagen en proceso.

    `alto` es de la imagen YA recortada (arranca en linea_verde_sup).
    None = sin franja que chequear (ROI apagado o solo corte superior).
    """
    if not _cfg.get("roi_habilitado"):
        return None
    sup, inf = _franja_verde()
    if inf >= 1.0:
        return None
    return int(round((inf - sup) / (1.0 - sup) * alto))


def _placa_en_franja(y: int, h: int, alto: int) -> bool:
    """True si el bbox queda TOTALMENTE dentro de la franja verde.

    La placa que cruza la linea verde de abajo (auto que paso de mas) o
    la de arriba (toca_borde sobre el corte) no es "totalmente visible".
    Tolerancia 4px, la misma que toca_borde. ROI apagado -> siempre True.
    """
    limite = _franja_inf_px(alto)
    if limite is None:
        return True
    return (y + h) <= limite + 4


def _placa_en_franja_x(x: int, w: int, ancho: int) -> bool:
    """True si el bbox queda TOTALMENTE entre linea_verde_izq y der.

    Corte vertical: la placa contra el borde izquierdo o derecho del
    encuadre queda mitad afuera y "depende de la suerte" leerla; no es
    la toma que queremos. Misma tolerancia de 4px que la franja y.
    Sin claves verticales (defaults 0.0/1.0) o ROI apagado -> True.
    """
    if not _cfg.get("roi_habilitado"):
        return True
    izq = float(_cfg.get("roi_linea_verde_izq", 0.0))
    der = float(_cfg.get("roi_linea_verde_der", 1.0))
    return (x >= izq * ancho - 4) and (x + w <= der * ancho + 4)


def _linea_prioridad() -> float | None:
    """y normalizada (0-1) de la linea de PRIORIDAD o None si esta apagada.

    Es la linea pintada en el piso donde debe parar el carro (dibujada
    en azul con etiqueta PRIORIDAD sobre la foto guardada). Config:
    [roi] linea_prioridad. 0 o fuera de rango -> desactivada.
    """
    try:
        lin = float(_cfg.get("roi_linea_prioridad", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    return lin if 0.0 < lin < 1.0 else None


def _tolerancia_prioridad() -> float:
    """Mejora minima (0-1) para que una toma reemplace a la guardada.

    Chica a proposito: "un poquito" mas cerca de la linea ya gana, pero
    >0 para que dos tomas casi identicas no se reemplacen entre si.
    Config: [roi] tolerancia_prioridad. Default 0.02.
    """
    try:
        tol = float(_cfg.get("roi_tolerancia_prioridad", 0.02) or 0.0)
    except (TypeError, ValueError):
        return 0.02
    return max(0.0, tol)


def _distancia_franja(cy: float, sup: float, inf: float) -> float:
    """Distancia (0-1) del centro de la placa al punto de referencia.

    Referencia = linea de prioridad si esta calibrada; si no, el centro
    de la franja verde (criterio viejo). `cy` esta normalizado sobre la
    IMAGEN ORIGINAL (coords de la foto guardada y del .txt: con ROI
    activo la foto se guarda COMPLETA con las bandas, el recorte es
    solo para el detector).
    """
    ref = _linea_prioridad()
    if ref is None:
        ref = (sup + inf) / 2.0
    return abs(cy - ref)


def _mejora_por_posicion(cy_nuevo: float, cy_viejo: float | None) -> bool:
    """True si la toma nueva esta mas cerca de la linea de prioridad
    (o del centro de la franja si no hay linea) que la ya guardada,
    mejorando en al menos tolerancia_prioridad.

    La mas CERCA gana: basta con venir "un poquito" mas cerca para
    reemplazar; la tolerancia evita que tomas casi identicas se
    reemplacen entre si. Solo con ROI activo; si no, manda solo el
    area (criterio viejo de MEJOR FOTO).
    """
    if not _cfg.get("roi_habilitado") or cy_viejo is None:
        return False
    sup, inf = _franja_verde()
    nueva = _distancia_franja(cy_nuevo, sup, inf)
    vieja = _distancia_franja(cy_viejo, sup, inf)
    return nueva < vieja - _tolerancia_prioridad()


def _tolerancia_ventana() -> float:
    """Semiancho (0-1) de la VENTANA de posicion sobre la linea de
    prioridad (2026-11-10, pedido FELIPE).

    Solo llega a evaluarse la foto cuya placa cae dentro de
    linea_prioridad +/- esto ("un poquito lejos, no tan lejos");
    afuera se descarta como fuera_de_area sin gastar API. Naranja si
    quedo pegada (<= 2x), rojo si quedo lejos. Config:
    [roi] tolerancia_ventana. 2026-11-11: default = config (0.20;
    con 0.04 se descartaban placas reales de cy 0.41-0.70).
    """
    try:
        tol = float(_cfg.get("roi_tolerancia_ventana", 0.20) or 0.0)
    except (TypeError, ValueError):
        return 0.20
    return max(0.0, tol)


def _eval_ventana_seg() -> float:
    """Segundos de la ventana de evaluacion por vista (default 15).

    0 = apagado: cada foto se evalua directo (comportamiento sin
    ventana, usado por fixtures de tests). Config:
    [roi] eval_ventana_seg.
    """
    try:
        seg = float(_cfg.get("roi_eval_ventana_seg", 15.0) or 0.0)
    except (TypeError, ValueError):
        return 15.0
    return max(0.0, seg)


def _programar_cierre_ventana(vista: str, seg: float) -> None:
    """Timer daemon que cierra la ventana de evaluacion de la vista."""
    def _cerrar() -> None:
        try:
            _eval_ventana_cerrar(vista)
        except Exception:
            logger.exception("Fallo el cierre de ventana de %s", vista)

    t = threading.Timer(seg, _cerrar)
    t.daemon = True
    t.start()


def _eval_ventana_cerrar(vista: str) -> None:
    """Cierra la ventana ABIERTA: encola a la elegida para que se
    reprocese (y ahi SI pase por OCR/API) y deja la ventana en fase
    'lista'. Sin estado o ya cerrada -> no hace nada (timer viejo).
    """
    with _eval_ventanas_lock:
        est = _eval_ventanas.get(vista)
        if est is None or est["fase"] != "abierta":
            return
        est["fase"] = "lista"
        elegida = est["ruta"]
    if not encolar(str(elegida)):
        # Cola llena / ya encolada: la elegida queda en fase 'lista' y
        # la reprocesa el proximo barrido o una re-llegada de la foto.
        logger.warning("No se pudo encolar la elegida de la ventana: %s",
                       elegida.name)


def _eval_ventana_salta_rf(vista: str) -> bool:
    """True si hay una ventana ABIERTA: Roboflow (red de seguridad) se
    salta en esa vista para no gastar cuota en fotos que la ventana va
    a descartar. Fase 'lista' -> False: la elegida SI pasa por la red.
    """
    if _eval_ventana_seg() <= 0:
        return False
    with _eval_ventanas_lock:
        est = _eval_ventanas.get(vista)
        return bool(est and est["fase"] == "abierta")


def _eval_ventana_probar(vista: str, ruta: Path, cy: float,
                         area: float) -> str:
    """Ventana best-of por vista: 1 foto evaluada por ventana.

    `cy` = centro de la placa sobre la foto completa; `area` = w*h del
    bbox (desempate: mas grande gana). Devuelve:
      "procesar"  -> sigue por la cadena OCR/API (seg=0 directo, es la
                     elegida re-llegada, o su reproceso post-cierre);
      "espera"    -> es/queda como candidata de la ventana: NO se
                     evalua aun y NO se borra;
      "descartar" -> candidata peor: se borra el archivo aca mismo
                     (interno, sin carpeta de motivos, sin API).
    El reemplazo de una candidata por otra mejor borra la vieja.
    """
    seg = _eval_ventana_seg()
    if seg <= 0:
        return "procesar"
    puntaje = (_distancia_franja(cy, *_franja_verde()), -float(area))
    programar = False
    with _eval_ventanas_lock:
        est = _eval_ventanas.get(vista)
        if est is None:
            _eval_ventanas[vista] = {"ruta": ruta, "score": puntaje,
                                     "fase": "abierta"}
            programar = True
        elif est["fase"] == "lista":
            if est["ruta"] == ruta:
                # La elegida vuelve a llegar (la re-encolo el cierre):
                # se evalua AHORA, con toda la cadena OCR/API.
                del _eval_ventanas[vista]
                return "procesar"
            # Llego otra foto con la ventana ya cerrada: ventana nueva.
            _eval_ventanas[vista] = {"ruta": ruta, "score": puntaje,
                                     "fase": "abierta"}
            programar = True
        elif est["ruta"] == ruta:
            # Misma candidata re-llegada (barrido/pendiente): sigue
            # siendo la mejor y no se reprograma el timer.
            return "espera"
        elif puntaje < est["score"]:
            vieja = est["ruta"]
            est["ruta"], est["score"] = ruta, puntaje
            vieja.unlink(missing_ok=True)
            return "espera"
        else:
            ruta.unlink(missing_ok=True)
            return "descartar"
    # Fuera del lock: nunca armar el timer con el lock tomado.
    if programar:
        _programar_cierre_ventana(vista, seg)
    return "espera"


def _etiqueta_centro_y(jpg: Path) -> float | None:
    """cy normalizado (0-1) del bbox guardado en el .txt YOLO vecino."""
    try:
        partes = jpg.with_suffix(".txt").read_text(encoding="utf-8").split()
        return float(partes[2])
    except Exception:
        return None


def _dibujar_roi(img: np.ndarray, sup: float, inf: float,
                 izq: float = 0.0, der: float = 1.0) -> np.ndarray:
    """Bandas de colores del ROI + etiqueta "ROI" sobre la foto COMPLETA.

    La foto guardada conserva el encuadre original (el corte es solo
    para el detector) y pinta, igual que scripts\\vista_roi.py:
      * banda ROJA  [0, sup)          -> se corta, no se detecta
      * banda VERDE [sup, inf)        -> aqui debe estar el vehiculo
      * banda NARANJA [inf, 1)        -> lo que cruza la linea no cuenta
      * franjas IZQ/DER (si izq>0 o der<1) con el mismo tinte rojo y
        linea blanca -> lo que esta de ese lado no se detecta
    Mas la etiqueta "ROI" arriba-izq para poder VER en la foto donde
    queda la placa respecto de las lineas.
    """
    alto, ancho = img.shape[:2]
    y_sup = int(round(sup * alto))
    y_inf = int(round(inf * alto))
    vista = img.copy()
    rojo = vista[0:y_sup]
    if rojo.size:
        vista[0:y_sup] = (rojo * 0.45 + np.array([0, 0, 180]) * 0.55
                          ).astype(np.uint8)
    verde = vista[y_sup:y_inf]
    if verde.size:
        vista[y_sup:y_inf] = (verde * 0.55 + np.array([0, 150, 0]) * 0.45
                              ).astype(np.uint8)
    bajo = vista[y_inf:]
    if bajo.size:
        vista[y_inf:] = (bajo * 0.6 + np.array([0, 140, 255]) * 0.4
                         ).astype(np.uint8)
    x_izq = int(round(izq * ancho))
    x_der = int(round(der * ancho))
    if izq > 0.0 and x_izq > 0:
        tinta = vista[:, 0:x_izq]
        if tinta.size:
            vista[:, 0:x_izq] = (tinta * 0.45
                                 + np.array([0, 0, 180]) * 0.55
                                 ).astype(np.uint8)
    if der < 1.0 and x_der < ancho:
        tinta = vista[:, x_der:]
        if tinta.size:
            vista[:, x_der:] = (tinta * 0.45
                                + np.array([0, 0, 180]) * 0.55
                                ).astype(np.uint8)
    cv2.line(vista, (0, y_sup), (ancho - 1, y_sup), (0, 255, 0), 2)
    cv2.line(vista, (0, y_inf), (ancho - 1, y_inf), (0, 200, 255), 2)
    if izq > 0.0 and x_izq > 0:
        cv2.line(vista, (x_izq, 0), (x_izq, alto - 1), (255, 255, 255), 2)
    if der < 1.0 and x_der < ancho:
        cv2.line(vista, (x_der, 0), (x_der, alto - 1), (255, 255, 255), 2)
    # Linea de PRIORIDAD (azul): donde debe parar el carro. En MEJOR FOTO
    # gana la toma cuya placa queda mas cerca de esta linea.
    lin = _linea_prioridad()
    if lin is not None:
        y_lin = int(round(lin * alto))
        if 0 <= y_lin < alto:
            cv2.line(vista, (0, y_lin), (ancho - 1, y_lin),
                     (255, 0, 0), 3)
            # Sombra blanca + texto azul para que se lea sobre el verde.
            cv2.putText(vista, "PRIORIDAD", (12, y_lin - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 5,
                        cv2.LINE_AA)
            cv2.putText(vista, "PRIORIDAD", (12, y_lin - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2,
                        cv2.LINE_AA)
    cv2.putText(vista, "ROI", (10, max(40, min(y_sup - 8, 56))),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3,
                cv2.LINE_AA)
    return vista


def _hash_perceptual(img: np.ndarray) -> int:
    gris = _a_gris(img)
    pequena = np.float32(cv2.resize(gris, (32, 32), interpolation=cv2.INTER_AREA))
    coef = np.float32(cv2.dct(pequena))[:8, :8].reshape(-1)
    mediana = float(np.median(coef))
    valor = 0
    for bit in coef > mediana:
        valor = (valor << 1) | int(bit)
    return valor


_COLOR_ROJO = (0, 0, 255)
_COLOR_AMARILLO = (0, 255, 255)
_COLOR_VERDE = (0, 255, 0)
_COLOR_AZUL = (255, 0, 0)


def color_caja(sufijo: str) -> tuple[int, int, int]:
    """Color del recuadro dibujado en la imagen completa segun la fuente.

    Verde = Roboflow, ambar = PlateRecognizer, azul = motor local.
    El sufijo ya documenta la fuente ('_api-Roboflow', '_api-PlateRecognizer');
    desde 2026-11-10 el nombre del archivo no lleva sufijo de pasada
    (la pasada vive en su subcarpeta), asi que solo decide la fuente.
    """
    if "_api-Roboflow" in sufijo:
        return _COLOR_VERDE
    if "_api-PlateRecognizer" in sufijo:
        return _COLOR_AMARILLO
    return _COLOR_AZUL


def _dibujar_caja(img: np.ndarray, bbox: tuple[int, int, int, int],
                  color: tuple[int, int, int], padding: float = 0.08,
                  grosor: int | None = None) -> np.ndarray:
    salida = img.copy()
    x, y, w, h = bbox
    dx, dy = int(w * padding), int(h * padding)
    alto, ancho = salida.shape[:2]
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(ancho, x + w + dx), min(alto, y + h + dy)
    if grosor is None:
        grosor = max(2, round(min(alto, ancho) / 300))
    cv2.rectangle(salida, (x0, y0), (x1, y1), color, grosor)
    return salida


def _pasada_de_placa(placa: str, vista: str | None = None,
                     confiable: bool = False) -> tuple[int, float]:
    dias = int(_cfg.get("ventana_reinspeccion_dias", 30))
    limite = (datetime.now() - timedelta(days=dias)).isoformat(timespec="seconds")
    con = _conexion()
    try:
        filas = [fila[0] for fila in con.execute(
            "SELECT fecha_hora FROM pasadas WHERE placa = ? AND fecha_hora >= ? "
            "ORDER BY fecha_hora", (placa, limite))]
    finally:
        con.close()
    if not filas:
        # Sin historial en DB: el disco del dia manda (carpetas + JSON
        # con Levenshtein). Si ya esta guardada hoy => pasada 2.
        if vista and _placa_existe_hoy(placa, vista, confiable=confiable):
            return 2, 0.0
        return 1, float("inf")
    ultima = datetime.fromisoformat(filas[-1])
    minutos = (datetime.now() - ultima).total_seconds() / 60.0
    return len(filas) + 1, minutos


def _registrar_pasada(placa: str, pasada: int, excedida: bool) -> None:
    con = _conexion()
    try:
        con.execute(
            "INSERT INTO pasadas (placa, pasada, fecha_hora, excedida) "
            "VALUES (?, ?, ?, ?)",
            (placa, pasada, datetime.now().isoformat(timespec="seconds"),
             1 if excedida else 0),
        )
        con.commit()
    finally:
        con.close()


def _sufijo_pasada(pasada: int) -> str:
    # Solo para el log de cierre (desde 2026-11-10 el nombre del archivo
    # ya NO lleva el sufijo: lo dice la subcarpeta, ver _carpeta_pasada).
    return {1: "", 2: "_2da-pasada", 3: "_3ra-pasada"}.get(pasada, "_EXCEDIDA")


def _carpeta_pasada(pasada: int) -> str:
    """Subcarpeta de la pasada dentro de <dia>/<PLACA>/ ("" = raiz)."""
    return {1: "", 2: "segunda", 3: "tercera"}.get(pasada, "excedida")


# Formatos de placa peruana validos (MTC): ABC-123, A1B-234, AB1-234
# (desde 2010) y la vieja AB-1234. Mata lo que NO es placa aunque el
# OCR "lea" algo: 5TC-000 (cartel lateral de micro), 00F-245,
# VST-0NY, SES-05A, A68-778.
_RE_PLACA_PE = re.compile(
    r"^(?:[A-Z]{3}|[A-Z]\d[A-Z]|[A-Z]{2}\d)[-\s]?\d{3}$"
    r"|^[A-Z]{2}[-\s]?\d{4}$")


def _formato_placa_valido(texto: str) -> bool:
    """True si el texto tiene formato de placa peruana."""
    t = (texto or "").strip().upper()
    return bool(_RE_PLACA_PE.match(t))


def _ts_de_foto(nombre: str) -> datetime | None:
    """Timestamp de camara embebido en el nombre (20261106140422407).

    Devuelve None en archivos ya renombrados (replay): sin timestamp no
    se aplica el cooldown por evento (mejor dejar pasar que perder).
    """
    m = re.search(r"(20\d{15})", nombre)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1)[:14], "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _cooldown_placa() -> None:
    """Pausa tras guardar una placa (cooldown_placa_seg, por defecto 90 s).

    Solo duerme si la cola quedo vacia: con fotos pendientes el worker
    drena primero y el sleep no bloquea el drenaje.
    """
    seg = float(_cfg.get("cooldown_placa_seg", 90.0))
    if seg > 0 and COLA.qsize() == 0:
        logger.debug("Cooldown %.1f s tras guardar placa", seg)
        time.sleep(seg)


def _cooldown_sin_placa() -> None:
    """Pausa tras descartar una foto SIN placa detectada
    (cooldown_sin_placa_seg, por defecto 5 s).

    Evita quemar CPU en la calle vacia: mientras no aparezca una placa,
    el worker espera antes de procesar la proxima foto. Si la cola aun
    tiene fotos (backlog), no duerme: drenar es prioritario.
    """
    seg = float(_cfg.get("cooldown_sin_placa_seg", 5.0))
    if seg > 0 and COLA.qsize() == 0:
        logger.debug("Cooldown %.1f s tras descartar foto sin placa", seg)
        time.sleep(seg)


def _trazar_aoi(ruta_foto: Path, img: np.ndarray,
                bbox: tuple[int, int, int, int], texto: str, conf: float,
                intentos: list) -> None:
    """Graba el crop enviado a la AOI y una linea JSONL con el desempeno."""
    if not (_AOI_ACTIVA or _ROBOFLOW_ACTIVO):
        return
    try:
        dia = datetime.now().strftime("%Y-%m-%d")
        carpeta = _carpeta_reportes() / "aoi" / dia
        carpeta.mkdir(parents=True, exist_ok=True)
        recorte = _recorte_con_margen(img, bbox, 0.15)
        marca = datetime.now().strftime("%H-%M-%S-%f")[:-3]
        nombre_crop = f"{marca}_{ruta_foto.stem}.jpg"
        destino = carpeta / nombre_crop
        if recorte.size:
            cv2.imwrite(str(destino), recorte,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        evento = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "foto": ruta_foto.name,
            "bbox": [int(v) for v in bbox],
            "texto": texto,
            "conf": round(float(conf), 4),
            "via_api": any(n == "api" for n, _t, _c in intentos or []),
            "via_roboflow": any(n == "roboflow" for n, _t, _c in intentos
                                or []),
            "intentos": [{"nombre": n, "texto": t or "", "conf": round(c, 4)}
                         for n, t, c in (intentos or [])],
            "crop": str(destino) if recorte.size else "",
        }
        jsonl = carpeta / f"aoi-{dia}.jsonl"
        with jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evento, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("No se pudo trazar el envio a la AOI", exc_info=True)


def _recorte_con_margen(img: np.ndarray, bbox: tuple[int, int, int, int],
                        margen: float = 0.35) -> np.ndarray:
    x, y, w, h = bbox
    dx, dy = int(w * margen), int(h * margen)
    alto, ancho = img.shape[:2]
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(ancho, x + w + dx), min(alto, y + h + dy)
    return img[y0:y1, x0:x1]


def _cerca_texto(a: str, b: str, max_dist: int = 2) -> bool:
    if a == b:
        return True
    if not a or not b:
        return False
    if abs(len(a) - len(b)) > max_dist:
        return False
    anterior = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        actual = [i]
        for j, cb in enumerate(b, 1):
            actual.append(min(anterior[j] + 1, actual[j - 1] + 1,
                              anterior[j - 1] + (ca != cb)))
        anterior = actual
    return anterior[-1] <= max_dist


def _sin_guion(texto: str) -> str:
    return texto.replace("-", "")


_EQUIV_COMPARAR = str.maketrans({"I": "1", "O": "0", "Z": "7"})


def _forma_comparable(texto: str) -> str:
    return _sin_guion(texto).upper().translate(_EQUIV_COMPARAR)


def _buscar_misma(vista: str, huella: int, texto: str = "",
                  confiable: bool = False):
    """Match de misma placa en el historial de la vista.

    confiable = la lectura viene de la API (roboflow/PlateRecognizer):
    texto EXACTO (Levenshtein 0, un error de la API es otra placa) y
    SIN el fallback por hash (2026-11-10: los atajos que reutilizaban
    lecturas viejas sin pasar por la API se eliminaron).
    """
    guardados = _hashes.get(vista, {})
    if texto:
        forma = _forma_comparable(texto)
        umbral_leven = (0 if confiable
                        else int(_cfg.get("umbral_leven_pasada", 1)))
        for h, t in guardados.items():
            if t and _cerca_texto(forma, _forma_comparable(t),
                                  max_dist=umbral_leven):
                return h, t
    if confiable:
        return None
    umbral = _cfg.get("umbral_dedup", 18)
    for h, t in guardados.items():
        if bin(huella ^ h).count("1") <= umbral:
            return h, t
    return None


def _placas_del_dia(vista: str, dia: str | None = None) -> list[str]:
    """Nombres de placa del dia en disco: carpetas + JSON indice."""
    if not _cfg.get("revisar_carpetas_dia", True):
        return []
    dia = dia or datetime.now().strftime("%Y-%m-%d")
    base = Path(_cfg.get("procesadas", Path("."))) / vista / dia
    placas: list[str] = []
    json_path = base / f"placas_{dia}.json"
    if json_path.is_file():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            items = data.get("placas", data) if isinstance(data, dict) \
                else data
            if isinstance(items, list):
                placas.extend(str(p) for p in items if p)
        except (OSError, ValueError, TypeError):
            logger.debug("No se pudo leer indice JSON %s", json_path)
    if base.is_dir():
        for sub in base.iterdir():
            if sub.is_dir() and sub.name:
                placas.append(sub.name)
    return placas


def _placa_existe_hoy(texto: str, vista: str,
                      dia: str | None = None,
                      confiable: bool = False) -> bool:
    """True si la placa ya existe hoy en carpetas del dia o el JSON.

    Matching con Levenshtein <= umbral_leven_pasada (default 1): un error
    de OCR cuenta como la misma placa; 2+ errores NO (evita el falso
    positivo de '4 de 6' entre placas distintas). confiable = lectura de
    API -> distancia 0 (exacta): AOM-861 NO es la AOM-361 guardada.
    """
    if not texto:
        return False
    umbral = (0 if confiable
              else int(_cfg.get("umbral_leven_pasada", 1)))
    forma = _forma_comparable(texto)
    for p in _placas_del_dia(vista, dia=dia):
        if _cerca_texto(forma, _forma_comparable(p), max_dist=umbral):
            return True
    return False


def _es_misma_placa(vista: str, huella: int, texto: str = "",
                    confiable: bool = False) -> bool:
    return _buscar_misma(vista, huella, texto,
                         confiable=confiable) is not None


def _es_lectura_lateral(huella: int, texto: str, previa_huella: int,
                        texto_previo: str, umbral: int = 18) -> bool:
    """True si es la misma placa leida distinto desde el costado.

    Texto similar (dist <= 2 via _cerca_texto) pero hash lejano =
    sticker/angulo lateral, no la placa del frente: no reemplazar.
    """
    if not texto or not texto_previo:
        return False
    if _forma_comparable(texto) == _forma_comparable(texto_previo):
        return False
    return bin(huella ^ previa_huella).count("1") > umbral


def _carpeta_descartes() -> Path:
    propio = _cfg.get("descartes")
    if propio:
        return Path(propio)
    return _cfg["procesadas"].parent / "descartadas"


def _carpeta_rechazadas_api() -> Path:
    propio = _cfg.get("rechazadas_api")
    if propio:
        return Path(propio)
    return _cfg["procesadas"].parent / "rechazadas_api"


def _guardar_rechazada_api(ruta: Path, intentos: list, resultado: str,
                           detalle: str = "",
                           rf_rechazado: bool = False) -> None:
    """Copia la foto ORIGINAL (sin recortar) + traza JSONL.

    Documenta por que la API no confirmo la lectura: rechazo de
    Roboflow (rf_rechazado) y/o de PlateRecognizer (intento
    'api_rechazada'). Solo se llama cuando hubo rechazo real (gasto de
    cuota sin confirmacion), nunca ante calle vacia.
    """
    dia = datetime.now().strftime("%Y-%m-%d")
    vista = _carpeta_de(ruta.name)
    carpeta = _carpeta_rechazadas_api() / dia / vista
    carpeta.mkdir(parents=True, exist_ok=True)
    destino = _destino_libre(carpeta, ruta.name)
    try:
        shutil.copy2(str(ruta), str(destino))
    except OSError:
        logger.warning("No se pudo copiar a rechazadas_api: %s", ruta.name)
        return
    traza = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "foto": ruta.name,
        "copia": str(destino),
        "resultado": resultado,
        "detalle": detalle,
        "rf_rechazado": bool(rf_rechazado),
        "intentos": [[n, t, round(float(c), 3)] for n, t, c in intentos],
        "roi": {"habilitado": bool(_cfg.get("roi_habilitado")),
                "sup": _cfg.get("roi_linea_verde_sup"),
                "inf": _cfg.get("roi_linea_verde_inf"),
                "izq": _cfg.get("roi_linea_verde_izq", 0.0),
                "der": _cfg.get("roi_linea_verde_der", 1.0)},
    }
    try:
        with (carpeta / "rechazadas.jsonl").open("a",
                                                 encoding="utf-8") as fh:
            fh.write(json.dumps(traza, ensure_ascii=False) + "\n")
    except OSError:
        logger.warning("No se pudo escribir la traza de rechazadas_api")
    logger.info("Rechazo de API copiado: %s -> %s (resultado=%s%s)",
                ruta.name, destino.name, resultado,
                f", {detalle}" if detalle else "")


def _purgar_rechazadas_api() -> None:
    """Borra rechazadas_api\\<dia> mas viejas que [log] retencion_dias."""
    raiz = _carpeta_rechazadas_api()
    if not raiz.is_dir():
        return
    dias = int(_cfg.get("retencion", 30))
    limite = (datetime.now() - timedelta(days=dias)).date()
    for carpeta in raiz.iterdir():
        if not carpeta.is_dir():
            continue
        try:
            fecha = datetime.strptime(carpeta.name, "%Y-%m-%d").date()
        except ValueError:
            continue
        if fecha < limite:
            shutil.rmtree(carpeta, ignore_errors=True)
            logger.info("rechazadas_api: purgada %s (>%d dias)",
                        carpeta.name, dias)


def _destino_libre(carpeta: Path, nombre: str) -> Path:
    destino = carpeta / nombre
    indice = 1
    while destino.exists():
        destino = carpeta / f"{Path(nombre).stem}_{indice}{Path(nombre).suffix}"
        indice += 1
    return destino


_RE_NOMBRE_PLACA = re.compile(r"[^A-Za-z0-9-]+")


def _placa_base(texto_placa: str) -> str:
    """Placa sanitizada en mayusculas: base del nombre Y de la subcarpeta."""
    base = _RE_NOMBRE_PLACA.sub("-", texto_placa or "").strip("-")
    return re.sub(r"-{2,}", "-", base).upper()


def _nombre_con_placa(vista: str, texto_placa: str, ahora: datetime,
                      sufijo: str = "") -> str:
    base = _placa_base(texto_placa)
    if not base:
        return f"{vista}_{ahora.strftime('%H-%M-%S')}_{ahora.microsecond:03d}"
    return base + sufijo


def _descartar(ruta: Path, motivo: str, detalle: str = "",
               auditar: bool = True) -> None:
    nombre = ruta.name
    sufijo = f" {detalle}" if detalle else ""
    retener = str(_cfg.get("modo_descarte", "eliminar")
                  ).strip().lower() == "retener"

    if not retener:
        ruta.unlink(missing_ok=True)
        accion = "Imagen eliminada"
    else:
        carpeta = _carpeta_descartes() / datetime.now().strftime("%Y-%m-%d") \
            / motivo
        carpeta.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(ruta), str(_destino_libre(carpeta, nombre)))
            accion = "Imagen movida a descartadas (modo retener)"
        except OSError:
            logger.warning("No se pudo mover el descarte %s", nombre)
            ruta.unlink(missing_ok=True)
            accion = "Imagen eliminada"

    if auditar:
        registrar(nombre, estado=motivo, evidencia=0, semaforo="rojo")

    nivel = logging.DEBUG if motivo in _MOTIVOS_VOLUMEN else logging.INFO
    if motivo == "sin_placa":
        logger.info("No se detecta placa: %s", nombre)
    logger.log(nivel, "%s: %s | motivo: %s%s", accion, nombre, motivo, sufijo)


def _guardar(img: np.ndarray, bbox: tuple[int, int, int, int] | None,
             vista: str, texto_placa: str = "", sufijo: str = "",
             dudosa: bool = False, pasada: int = 1,
             carpeta: Path | None = None) -> Path:
    ahora = datetime.now()
    dia = _cfg["procesadas"] / vista / ahora.strftime("%Y-%m-%d")
    base = _placa_base(texto_placa)
    raiz = dia / base if base else dia
    if carpeta is None:
        # 2026-11-10 (FASE 2): cada pasada vive en su subcarpeta
        # (segunda/tercera/excedida) y el nombre del archivo ya NO
        # lleva sufijo de pasada.
        sub = _carpeta_pasada(pasada)
        carpeta = raiz / sub if sub else raiz
    carpeta.mkdir(parents=True, exist_ok=True)

    nombre = _nombre_con_placa(vista, texto_placa, ahora, sufijo)
    if dudosa:
        nombre = f"{nombre} (DUDOSA)"
    destino = _destino_libre(carpeta, f"{nombre}.jpg")
    salida = img
    if _cfg.get("roi_habilitado"):
        # Foto guardada COMPLETA con las bandas de colores y la etiqueta
        # ROI: sirve para ver donde queda la placa respecto de las
        # lineas (el corte es solo para el detector).
        sup, inf = _franja_verde()
        izq = float(_cfg.get("roi_linea_verde_izq", 0.0))
        der = float(_cfg.get("roi_linea_verde_der", 1.0))
        salida = _dibujar_roi(img, sup, inf, izq=izq, der=der)
    if vista == "placa" and bbox is not None:
        color = color_caja(sufijo)
        if dudosa:
            fino = max(1, round(min(img.shape[:2]) / 600))
            salida = _dibujar_caja(salida, bbox, color, padding=0.18,
                                   grosor=fino)
        else:
            salida = _dibujar_caja(salida, bbox, color)
    if not cv2.imwrite(str(destino), salida,
                       [int(cv2.IMWRITE_JPEG_QUALITY), 100]):
        # REGLA: una carpeta de placa SIEMPRE tiene sus archivos. Si la
        # foto no se pudo escribir (disco lleno, permisos), no dejo la
        # carpeta vacia atras y fallo ruidoso para no registrar un
        # archivo que no existe en el reporte.
        try:
            carpeta.rmdir()          # solo si quedo recien creada y vacia
        except OSError:
            pass                     # tenia archivos previos: se conservan
        raise OSError(f"No se pudo escribir la foto {destino}")

    # La IMAGEN ORIGINAL sin anotar (sin bandas ROI ni caja) es parte
    # del set de la placa: sirve para re-anotar o comparar sin perder
    # los pixeles crudos de la camara.
    original = destino.with_name(destino.stem + "_original.jpg")
    if not cv2.imwrite(str(original), img,
                       [int(cv2.IMWRITE_JPEG_QUALITY), 100]):
        raise OSError(f"No se pudo escribir la imagen original {original}")

    if bbox is not None:
        x, y, w, h = bbox
        alto, ancho = img.shape[:2]
        etiqueta = (f"0 {(x + w / 2) / ancho:.6f} {(y + h / 2) / alto:.6f} "
                    f"{w / ancho:.6f} {h / alto:.6f}\n")
        (destino.with_suffix(".txt")).write_text(etiqueta, encoding="utf-8")
        if vista == "placa":
            recorte = _recorte_con_margen(img, bbox, 0.0)
            if recorte.size:
                cv2.imwrite(str(destino.with_name(destino.stem + "_crop.jpg")),
                            recorte, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return destino


def _area_etiqueta(jpg: Path) -> float:
    try:
        partes = jpg.with_suffix(".txt").read_text(encoding="utf-8").split()
        return float(partes[3]) * float(partes[4])
    except Exception:
        return 0.0


def _reemplazar_si_mejor(vista: str, h_vieja: int, t_viejo: str,
                         img: np.ndarray,
                         bbox: tuple[int, int, int, int], texto: str,
                         evidencia: int, color: str, confianza: float,
                         huella: int, etiqueta: str = "",
                         fuente: str | None = None,
                         origen: str = "") -> bool:
    if not t_viejo:
        return False
    desde = f"{_dia_dedup}T00:00:00"
    con = _conexion()
    try:
        fila = con.execute(
            "SELECT id, archivo, evidencia FROM registros WHERE placa = ? "
            "AND estado = 'guardado' AND fecha_hora >= ? AND archivo LIKE ? "
            "ORDER BY id DESC LIMIT 1", (t_viejo, desde, f"{vista}%")
        ).fetchone()
    finally:
        con.close()
    if not fila:
        return False
    id_registro, relativo_viejo, ev_vieja = fila
    viejo = _cfg["procesadas"] / relativo_viejo
    alto, ancho = img.shape[:2]
    area_nueva = (bbox[2] * bbox[3]) / (ancho * alto)
    area_vieja = _area_etiqueta(viejo)
    # Reemplaza si la nueva toma crece 1.2x (criterio original) O si esta
    # claramente mejor posicionada en la franja verde (>=25% mas cerca
    # del centro): el carro mas asentado en la linea verde gana aunque
    # el area no crezca. La evidencia sigue siendo obligatoria: nunca se
    # degrada la calidad de la lectura guardada.
    mejora_area = area_nueva >= 1.2 * area_vieja
    mejora_pos = _mejora_por_posicion(
        (bbox[1] + bbox[3] / 2) / alto if alto else 0.0,
        _etiqueta_centro_y(viejo))
    if not ((mejora_area or mejora_pos)
            and evidencia >= (ev_vieja or 0)):
        return False
    if not etiqueta:
        etiqueta = next((e for e in ("_api-Roboflow", "_api-PlateRecognizer")
                         if e in viejo.stem), "")
    viejo.unlink(missing_ok=True)
    viejo.with_suffix(".txt").unlink(missing_ok=True)
    viejo.with_name(viejo.stem + "_crop.jpg").unlink(missing_ok=True)
    viejo.with_name(viejo.stem + "_original.jpg").unlink(missing_ok=True)
    # Si el texto cambio, _guardar creara OTRA carpeta: esta quedo
    # sin archivos y viviria hasta el proximo arranque. Sin este
    # rmdir, MEJOR FOTO deja carpetas fantasma todo el dia.
    try:
        viejo.parent.rmdir()
    except OSError:
        pass  # no vacia (otros archivos) o ya no existe
    
    # NUEVO: Sufijo condicional si el ROI está habilitado
    sufijo_roi = "_roi-evaluado" if _cfg.get("roi_habilitado") else ""

    # 2026-11-10 (FASE 2): el reemplazo vuelve a la MISMA carpeta de
    # viejo (raiz, segunda/, tercera/ o excedida/) cuando la placa no
    # cambio; si cambio, _guardar arma la carpeta de la placa nueva.
    nuevo = _guardar(img, bbox, vista, texto,
                     sufijo=etiqueta + sufijo_roi,
                     carpeta=viejo.parent if texto == t_viejo else None)
    relativo = str(nuevo.relative_to(_cfg["procesadas"]))
    con = _conexion()
    try:
        con.execute("UPDATE registros SET archivo = ?, placa = ?, "
                    "confianza = ?, evidencia = ?, semaforo = ?, phash = ? "
                    "WHERE id = ?",
                    (relativo, texto, confianza, evidencia, color,
                     f"{huella:016x}", id_registro))
        # Mantener fuente al dia: si esta toma llego por la API, el
        # registro queda como 'roboflow'/'platerecognizer'. Nunca se
        # degrada a 'modelo': la confirmacion de la API no se pierde si
        # una toma posterior se lee local.
        # (Antes aca leia una variable 'fuente' inexistente: NameError en
        # cada MEJOR FOTO, que hacia rollback del UPDATE mientras el
        # archivo nuevo ya estaba en disco -> carpeta sin fila en el
        # reporte.)
        if fuente in ("roboflow", "platerecognizer"):
            con.execute("UPDATE registros SET fuente = ? WHERE id = ?",
                        (fuente, id_registro))
        if texto != t_viejo:
            con.execute("UPDATE pasadas SET placa = ? WHERE placa = ? "
                        "AND fecha_hora >= ?", (texto, t_viejo, desde))
        con.commit()
    finally:
        con.close()
    _hashes.setdefault(vista, {}).pop(h_vieja, None)
    _hashes[vista][huella] = texto
    _placas.setdefault(vista, set()).discard(t_viejo)
    _placas[vista].add(texto)
    _marcar_json()
    logger.info("MEJOR FOTO: %s -> %s (area %.4f vs %.4f): %s <- %s",
                t_viejo, texto, area_nueva, area_vieja, relativo,
                origen or "(origen desconocido)")
    return True


def _rf_leer_recorte(img: np.ndarray,
                     bbox: tuple[int, int, int, int]) -> tuple[str, float]:
    """API 1 de la cadena dual: Roboflow lee el recorte del detector.

    Primera respuesta valida gana: devuelve ('', 0.0) si esta apagado,
    falla o no trae placa de 6, y asi entra la siguiente API
    (PlateRecognizer) sin gastar de mas. El recorte va tal cual lo
    detecto el modelo.
    """
    if not _ROBOFLOW_ACTIVO:
        return "", 0.0
    try:
        import roboflow
        x, y, w, h = bbox
        recorte = img[y:y + h, x:x + w]
        if recorte.size == 0:
            return "", 0.0
        umbral_rf = float(_cfg.get("umbral_modelo", 0.4))
        mejor_texto, mejor_conf = "", 0.0
        for _bb, tx, cf in roboflow.detectar_leer(recorte):
            if cf < umbral_rf:
                continue
            texto = texto_de_api(tx)
            if texto and cf > mejor_conf:
                mejor_texto, mejor_conf = texto, cf
        return mejor_texto, mejor_conf
    except Exception:
        logger.exception("Roboflow fallo leyendo el recorte; "
                         "sigo con PlateRecognizer")
        return "", 0.0


def procesar_imagen(ruta: Path) -> None:
    global _ultima_imagen

    if not ruta.exists():
        logger.info("Archivo ya no existe (omitido): %s", ruta.name)
        return

    _actualizar_dia()

    # Cooldown POR EVENTO/rafaga (2026-11-10, pedido FELIPE): SIEMPRE
    # primero, ANTES de decodificar, detectar y consultar APIs. Si la
    # foto viene de la misma ventana de tiempo de la camara
    # (+/-cooldown_evento_seg = 90 s, estimado de permanencia del
    # vehiculo en la escena) que la ultima placa guardada, se RECHAZA
    # aunque el texto sea DISTINTO (un carro = una placa por pasada):
    # lo que entre en la ventana se borra SIN gastar cuota ni CPU.
    # 2026-11-09: filtro INTERNO — no crea subcarpeta en descartadas.
    # Sin timestamp en el nombre (replay renombrado) no aplica.
    global _ultima_placa_foto_ts
    ventana_ev = float(_cfg.get("cooldown_evento_seg", 90.0))
    if ventana_ev > 0 and _ultima_placa_foto_ts is not None:
        ts_foto = _ts_de_foto(ruta.name)
        if ts_foto is not None:
            delta = abs((ts_foto - _ultima_placa_foto_ts).total_seconds())
            if delta <= ventana_ev:
                detalle_ev = (f"(misma rafaga que la placa guardada hace "
                              f"{delta:.0f}s)")
                _nombre_cool = ruta.name
                ruta.unlink(missing_ok=True)
                logger.debug("cooldown_evento: %s | %s", _nombre_cool,
                             detalle_ev)
                return

    img = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("Imagen ilegible, se elimina: %s", ruta.name)
        ruta.unlink(missing_ok=True)
        return

    # ROI franja verde: mascara OSD (si activa) + corta lo de atras de
    # la linea verde de arriba ANTES de todo (diff, detector, Roboflow).
    # Todo lo demas corre sobre el recorte, con coordenadas consistentes.
    # La foto que se GUARDA es la original COMPLETA (con bandas ROI):
    # el corte es solo para el detector; y_roi es el desplazamiento en
    # px para llevar el bbox del recorte a coords de la foto completa.
    img_full = img.copy() if _cfg.get("roi_habilitado") else None
    img, y_roi = _preparar_imagen(img)
    dy_roi = int(round(y_roi))
    foto_save = img_full if img_full is not None else img
    alto_save, ancho_save = foto_save.shape[:2]

    def _bbox_full(b):
        if b is None:
            return None
        return (b[0], b[1] + dy_roi, b[2], b[3])

    # Traza de rechazo de API: si alguna API fue consultada y no confirmo
    # (Roboflow sin lectura valida o PlateRecognizer 'api_rechazada'), la
    # foto ORIGINAL se copia a rechazadas_api\ con un JSONL para poder
    # diagnosticar cuota/validacion con la foto en la mano.
    api_rechazado = False
    ya_enmascarada = bool(_cfg.get("roi_habilitado", False))
    conf_ocr = 0.0
    intentos_ocr: list = []

    def _anotar_rechazo(resultado: str, detalle: str = "") -> None:
        if not (api_rechazado
                or any(n == "api_rechazada" for n, _t, _c in intentos_ocr)):
            return
        try:
            _guardar_rechazada_api(ruta, intentos_ocr, resultado,
                                   detalle, rf_rechazado=api_rechazado)
        except Exception:
            logger.exception("No se pudo copiar a rechazadas_api %s",
                               ruta.name)

    vista = _carpeta_de(ruta.name)
    logger.debug("Imagen %s -> vista %s (%dx%d)", ruta.name, vista,
                 img.shape[1], img.shape[0])

    porcentaje = 0.0
    ultima = _ultima_imagen.get(vista)
    if ultima is not None:
        porcentaje = calcular_diferencia(ultima, img)
        if porcentaje < _cfg["umbral"]:
            # Filtro interno de rafaga (mismo frame): NO crea carpeta
            # en descartadas (2026-11-09, solo 3 motivos).
            _nombre_dup = ruta.name
            ruta.unlink(missing_ok=True)
            logger.debug("duplicado interno: %s | %.2f%% de cambio",
                         _nombre_dup, porcentaje)
            return

    # Cadena dual-API: el detector LOCAL primero (gratis y siempre);
    # Roboflow sobre la foto completa es SOLO red de seguridad cuando el
    # modelo no ve nada. La lectura (RF recorte -> PlateRecognizer ->
    # motor local) pasa despues de los filtros, para no gastar cuota en
    # fotos que igual se descartan.
    lecturas_rf: dict[tuple[int, int, int, int], tuple[str, float]] = {}
    candidatos_rf: list[tuple[tuple[int, int, int, int], float]] = []
    candidatos = _candidatos_placa(img)
    # Ventana abierta en esta vista -> sin red de seguridad: la foto o
    # es la candidata de la ventana o se descarta; Roboflow gastaria
    # cuota para nada. (La elegida, fase 'lista', SI pasa por la red.)
    if (not candidatos and _ROBOFLOW_ACTIVO
            and not _eval_ventana_salta_rf(vista)):
        try:
            import roboflow
            detectados = roboflow.detectar_leer(img)
        except Exception:
            logger.exception("Roboflow fallo en la red de seguridad; "
                             "sin candidatos")
            detectados = []
        umbral_rf = float(_cfg.get("umbral_modelo", 0.4))
        for bb, tx, cf in detectados:
            if cf < umbral_rf:
                intentos_ocr.append(("roboflow_rechazada",
                                     str(tx or ""), float(cf)))
                continue
            candidatos_rf.append((bb, cf))
            texto_rf = texto_de_api(tx)
            if texto_rf:
                lecturas_rf[bb] = (texto_rf, cf)
            elif tx:
                intentos_ocr.append(("roboflow_rechazada",
                                     str(tx), float(cf)))
                logger.info("Roboflow: texto rechazado (%s), "
                            "lee la cadena local: %s", tx, ruta.name)
        if detectados and not lecturas_rf:
            # RF gasto cuota y no confirmo NINGUNA placa: si la foto
            # termina descartada o leida por el motor local, queda
            # copiada en rechazadas_api para diagnostico. (detectados
            # vacio = calle vacia, no es un rechazo que interese.)
            api_rechazado = True
        if detectados and not candidatos_rf:
            logger.info("Roboflow: %d deteccion(es) bajo umbral %.2f: %s",
                        len(detectados), umbral_rf, ruta.name)
        candidatos = candidatos_rf
        if candidatos:
            logger.debug("Roboflow (red de seguridad): %d candidato(s) "
                         "en %s", len(candidatos), ruta.name)
    if not candidatos:
        _anotar_rechazo("sin_placa")
        _descartar(ruta, "sin_placa")
        _cooldown_sin_placa()
        return
    candidatos = sorted(candidatos,
                        key=lambda par: par[0][2] * par[0][3], reverse=True)

    min_texto = _cfg.get("umbral_texto_min", 5.0)
    max_texto = _cfg.get("umbral_texto_max", 55.0)
    umbral_nitidez = _cfg.get("umbral_nitidez", 120.0)
    umbral_quemada = _cfg.get("umbral_sobreexpuesto", 2.0)
    umbral_bordes = _cfg.get("umbral_bordes", 2.0)
    umbral_rojo_max = _cfg.get("umbral_rojo_max", 15.0)
    umbral_ancho_bbox = _cfg.get("umbral_ancho_bbox", 0.20)
    ocr_requerido = _cfg.get("ocr_requerido", "si") == "si"
    # gates_fp = no -> la unica validacion de la foto es la franja
    # verde (decision 2026-11: camara mira solo la linea; los filtros
    # FP clasicos quedan apagados con esta llave en [filtros]).
    gates_fp = bool(_cfg.get("gates_fp", True))

    alto, ancho = img.shape[:2]
    elegido = None
    ultimo_fallo: tuple[str, str] = ("sin_placa", "")
    bbox_ocr: tuple[int, int, int, int] | None = None
    huella = 0

    for bbox, confianza in candidatos:
        x, y, w, h = bbox
        recorte = img[y:y + h, x:x + w]
        if recorte.size == 0:
            continue

        toca_borde = (x <= 4 or y <= 4 or x + w >= ancho - 4
                      or y + h >= alto - 4)
        corte_borde = (f"(toca el borde: x {x}..{x + w} de {ancho}, "
                       f"y {y}..{y + h} de {alto})")

        # Gate de franja verde: la placa debe quedar TOTALMENTE entre las
        # dos lineas verdes; la que cruza la de abajo (auto paso de mas)
        # no es la toma que queremos. El borde de arriba ya lo cubre
        # toca_borde (el corte ES la linea verde superior).
        if not _placa_en_franja(y, h, alto):
            # Naranja = cruza (o queda por debajo de) la linea de abajo
            # (auto paso de mas / demasiado cerca); rojo = por encima
            # de la linea de arriba / corte superior.
            limite = _franja_inf_px(alto)
            if limite is not None and y + h > limite + 4:
                variante = "naranja"
            else:
                variante = "rojo"
            ultimo_fallo = (f"fuera_de_area/{variante}",
                            f"(fuera de franja verde: y {y}..{y + h} "
                            f"de {alto})")
            continue

        # Corte vertical: la placa debe quedar TOTALMENTE entre
        # linea_verde_izq y der; la que roza el borde del encuadre
        # (mitad afuera) no es la toma que queremos. Vertical = rojo.
        if not _placa_en_franja_x(x, w, ancho):
            ultimo_fallo = ("fuera_de_area/rojo",
                            f"(fuera de franja vertical: x {x}..{x + w} "
                            f"de {ancho})")
            continue

        # Centro de la placa sobre la FOTO COMPLETA (mismas coords que
        # el .txt y _dibujar_roi): base del gate de posicion y del
        # ranking de la ventana.
        cy_pos = ((y + h / 2.0 + dy_roi) / alto_save
                  if alto_save else 0.0)

        # Gate de POSICION (2026-11-10, pedido FELIPE): con la linea de
        # prioridad calibrada solo pasa la placa dentro de
        # linea_prioridad +/- tolerancia_ventana ("un poquito lejos, no
        # tan lejos"). Naranja si quedo pegada, rojo si quedo lejos;
        # se descarta SIN gastar API. Sin linea -> gate apagado.
        ref_pos = _linea_prioridad()
        if ref_pos is not None:
            dist_pos = abs(cy_pos - ref_pos)
            tol_pos = _tolerancia_ventana()
            if dist_pos > tol_pos:
                variante = "naranja" if dist_pos <= 2 * tol_pos else "rojo"
                ultimo_fallo = (
                    f"fuera_de_area/{variante}",
                    f"(fuera de ventana de linea: cy {cy_pos:.3f} vs "
                    f"{ref_pos:.3f} +/-{tol_pos:.3f})")
                continue

        # Ventana best-of: 1 foto por roi_eval_ventana_seg; mientras la
        # ventana esta abierta la foto NO se evalua (queda como
        # candidata o se descarta interna). "procesar" = es la elegida
        # (reproceso tras el cierre) o seg=0 (directo) -> ahi SI pasa
        # por la cadena OCR/API completa.
        if _eval_ventana_probar(vista, ruta, cy_pos, w * h) != "procesar":
            return

        # nitidez se calcula siempre: elegido la guarda en el registro.
        nitidez = _nitidez(recorte)

        # Gates FP (oscuros, nitidez, sobreexpuesta, bordes, rojo, ancho):
        # con gates_fp = no se saltan TODOS: la validacion es solo la
        # franja verde de arriba.
        if gates_fp:
            # El check de oscuros filtra basura del detector geometrico;
            # las detecciones del YOLO de Roboflow no lo necesitan.
            if _detector is None and not candidatos_rf:
                fraccion = _fraccion_oscuros(recorte)
                if not min_texto <= fraccion <= max_texto:
                    ultimo_fallo = ("sin_texto",
                                    f"({fraccion:.1f}% de oscuros)")
                    continue

            if nitidez < umbral_nitidez:
                ultimo_fallo = ("baja_nitidez", f"(nitidez {nitidez:.1f})")
                continue

            quemada = _fraccion_sobreexpuesta(recorte)
            if quemada > umbral_quemada:
                ultimo_fallo = ("sobreexpuesta",
                                f"({quemada:.1f}% saturada)")
                continue

            bordes = _fraccion_bordes(recorte)
            if bordes < umbral_bordes:
                ultimo_fallo = ("ocluida", f"({bordes:.1f}% de bordes)")
                continue

            # Gate de color: cintas reflectoras y luces no son placas.
            # Reales miden <4% de rojo; la cinta reflejo del FP conocido
            # ~32%.
            rojo = _fraccion_rojo(recorte)
            if rojo > umbral_rojo_max:
                ultimo_fallo = ("color_no_placa", f"({rojo:.1f}% rojo)")
                continue

            # Gate de ancho: texto lateral de omnibus/camion (letreto,
            # nombre, cinta) mide 0.24-0.46 del ancho de la imagen; placa
            # real 0.04-0.14 (GT-29 max 0.082). Umbral 0.20 deja ~2.4x
            # de margen.
            if not _pasa_gate_ancho(w, ancho, umbral_ancho_bbox):
                rel_ancho = w / ancho if ancho else 1.0
                ultimo_fallo = ("texto_lateral",
                                f"({rel_ancho:.2f} del ancho de imagen)")
                continue

        # 2026-11-10 (pedido FELIPE): SIN atajos de API. El pre-chequeo
        # por hash (reutilizaba lecturas viejas) y _api_confirmada_hoy
        # ("1 API por vehiculo por dia") se eliminaron: TODA foto que se
        # guarda pasa por la cadena OCR/API completa y queda etiquetada
        # con su fuente (_api-*). La ventana best-of de arriba es la que
        # ahora evita el gasto de cuota en la rafaga.
        texto_placa = ""
        if bbox in lecturas_rf:
            # Red de seguridad: el modelo no vio nada y Roboflow hallo la
            # placa en la foto completa; su texto ya viene listo.
            texto_placa, conf_ocr = lecturas_rf[bbox]
            intentos_ocr = [("roboflow", texto_placa, conf_ocr)]
            bbox_ocr = bbox
            _trazar_aoi(ruta, img, bbox, texto_placa, conf_ocr, intentos_ocr)
        else:
            # API 1: Roboflow lee el recorte; si no da placa valida, API 2
            # PlateRecognizer; si falla, motor local. Primera valida gana.
            rf_texto, rf_conf = _rf_leer_recorte(img, bbox)
            if _ROBOFLOW_ACTIVO and not rf_texto:
                api_rechazado = True
            if rf_texto:
                texto_placa, conf_ocr = rf_texto, rf_conf
                intentos_ocr = [("roboflow", rf_texto, rf_conf)]
                bbox_ocr = bbox
                _trazar_aoi(ruta, img, bbox, rf_texto, rf_conf, intentos_ocr)
            elif leer_placa_6 is not None:
                texto_placa, conf_ocr, intentos_ocr = leer_placa_6(
                    img, bbox, enmascarar=not ya_enmascarada)
                bbox_ocr = bbox
                _trazar_aoi(ruta, img, bbox, texto_placa, conf_ocr, intentos_ocr)
                if not texto_placa:
                    logger.info(
                        "Sin lectura solida: %s | bbox (%d,%d,%d,%d) | "
                        "intentos: %s", ruta.name, x, y, w, h,
                        " | ".join(f"{n}={(t or 'sin lectura')}({c:.2f})"
                                   for n, t, c in intentos_ocr))
                    if toca_borde:
                        ultimo_fallo = ("texto_invalido", corte_borde)
                        continue
                    if ocr_requerido:
                        ultimo_fallo = ("texto_invalido",
                                        "(sin consenso en los intentos)")
                        continue
            elif leer_placa is not None:
                texto_placa = leer_placa(_recorte_con_margen(img, bbox))
                if not texto_placa:
                    if toca_borde:
                        ultimo_fallo = ("texto_invalido", corte_borde)
                        continue
                    if ocr_requerido:
                        ultimo_fallo = ("texto_invalido",
                                        "(el recorte no se lee como placa)")
                        continue
            elif toca_borde:
                ultimo_fallo = ("texto_invalido", corte_borde)
                continue

        huella = _hash_perceptual(recorte)
        # Lectura de API -> matching EXACTO (Levenshtein 0): AOM-361 y
        # AOM-861 son placas distintas aunque la OCR local las acerque
        # (2026-11-10: falsos positivos de pasadas cruzadas).
        confiable = fuente_de_intentos(intentos_ocr) in (
            "roboflow", "platerecognizer")
        previa = _buscar_misma(vista, huella, texto_placa,
                               confiable=confiable)
        if previa is not None:
            # Texto similar pero hash lejano = placa del costado con otra
            # lectura: no reemplazar la placa guardada; archivar sin
            # registrar pasada nueva.
            if _es_lectura_lateral(huella, texto_placa, previa[0],
                                   previa[1], _cfg.get("umbral_dedup", 18)):
                _anotar_rechazo("texto_invalido", "lectura lateral")
                _descartar(
                    ruta, "texto_invalido",
                    f"(lectura lateral {texto_placa} ~ {previa[1]})")
                return
            if texto_placa and evidencia_y_color is not None:
                ev_nueva, color_nuevo = evidencia_y_color(
                    texto_placa, conf_ocr, intentos_ocr)
                if _reemplazar_si_mejor(vista, previa[0], previa[1],
                                        foto_save, _bbox_full(bbox),
                                        texto_placa, ev_nueva,
                                        color_nuevo, confianza, huella,
                                        etiqueta=etiqueta_fuente(
                                            intentos_ocr),
                                        fuente=fuente_de_intentos(
                                            intentos_ocr),
                                        origen=ruta.name):
                    _anotar_rechazo("guardado_reemplazo")
                    ruta.unlink(missing_ok=True)
                    _cooldown_placa()
                    return
            # Misma placa ya guardada: NO se descarta (2026-11-09).
            # Sigue al conteo de pasadas: mismo dia = pasada 2/3.

        pasada, minutos = (1, float("inf"))
        if texto_placa:
            pasada, minutos = _pasada_de_placa(texto_placa, vista,
                                               confiable=confiable)
            ventana = float(_cfg.get("ventana_reinspeccion_min", 0.0))
            if ventana > 0 and minutos < ventana:
                _anotar_rechazo("duplicado_placa",
                                "ventana reinspeccion")
                _descartar(ruta, "duplicado_placa",
                           f"(placa {texto_placa} registrada hace "
                           f"{minutos:.0f} min, ventana {ventana:.0f} min)")
                return

        max_pasadas = int(_cfg.get("max_pasadas", 3))
        excedida = texto_placa and pasada > max_pasadas
        elegido = (bbox, confianza, nitidez, texto_placa, pasada,
                   bool(excedida), _sufijo_pasada(pasada))
        break

    dudosa = False
    elegibles = [cbx for cbx, _cf in candidatos if cbx[2] > 0 and cbx[3] > 0]
    if (elegido is None and elegibles
            and ultimo_fallo[0] not in FALLOS_SIN_REEVAL
            and not ultimo_fallo[0].startswith("fuera_de_area")
            and (leer_placa_6 is not None or leer_placa is not None)):
        logger.info("REEVALUACION (hasta 3 intentos) de %s: %s",
                    ruta.name, ultimo_fallo[0])
        for intento in range(3):
            cbx = elegibles[intento % len(elegibles)]
            bx, by, bw, bh = cbx
            extra = 0.10 + 0.04 * intento
            ex, ey = int(bw * extra), int(bh * extra)
            rx0, ry0 = max(0, bx - ex), max(0, by - ey)
            rx1, ry1 = min(ancho, bx + bw + ex), min(alto, by + bh + ey)
            bbox_re = (rx0, ry0, rx1 - rx0, ry1 - ry0)
            rf_re, rf_conf_re = _rf_leer_recorte(img, bbox_re)
            if _ROBOFLOW_ACTIVO and not rf_re:
                api_rechazado = True
            if rf_re:
                texto_re = rf_re
                conf_re = rf_conf_re
                intentos_re = [("roboflow", rf_re, rf_conf_re)]
                _trazar_aoi(ruta, img, bbox_re, rf_re, rf_conf_re,
                            intentos_re)
            elif leer_placa_6 is not None:
                texto_re, conf_re, intentos_re = leer_placa_6(
                    img, bbox_re, reintento=False,
                    enmascarar=not ya_enmascarada)
                _trazar_aoi(ruta, img, bbox_re, texto_re, conf_re,
                            intentos_re)
            elif leer_placa is not None:
                texto_re = leer_placa(_recorte_con_margen(img, bbox_re))
                conf_re, intentos_re = 0.0, []
            else:
                continue
            intentos_ocr.extend(intentos_re)
            if not texto_re:
                continue
            rec_re = img[bbox_re[1]:bbox_re[1] + bbox_re[3],
                         bbox_re[0]:bbox_re[0] + bbox_re[2]]
            huella_re = _hash_perceptual(rec_re)
            re_confiable = fuente_de_intentos(intentos_re) in (
                "roboflow", "platerecognizer")
            if _es_misma_placa(vista, huella_re, texto_re,
                               confiable=re_confiable):
                # Misma placa en reeval: NO se descarta; cuenta como
                # pasada siguiente (2026-11-09).
                pasada_re, minutos_re = _pasada_de_placa(
                    texto_re, vista, confiable=re_confiable)
                ventana = float(_cfg.get("ventana_reinspeccion_min", 0.0))
                if ventana > 0 and minutos_re < ventana:
                    _anotar_rechazo("duplicado_placa",
                                    "reeval: ventana")
                    _descartar(ruta, "duplicado_placa",
                               f"(reeval: placa {texto_re} registrada hace "
                               f"{minutos_re:.0f} min, ventana {ventana:.0f} min)")
                    return
                max_pasadas = int(_cfg.get("max_pasadas", 3))
                nit_re = _nitidez(rec_re)
                elegido = (bbox_re, conf_re, nit_re, texto_re, pasada_re,
                           bool(pasada_re > max_pasadas),
                           _sufijo_pasada(pasada_re))
                bbox, conf_ocr, intentos_ocr = bbox_re, conf_re, intentos_re
                bbox_ocr, huella = bbox_re, huella_re
                dudosa = True
                logger.info("REEVALUACION EXITOSA (intento %d/3): %s <- %s",
                            intento + 1, texto_re, ruta.name)
                break
            pasada_re, minutos_re = _pasada_de_placa(
                texto_re, vista, confiable=re_confiable)
            ventana = float(_cfg.get("ventana_reinspeccion_min", 0.0))
            if ventana > 0 and minutos_re < ventana:
                _anotar_rechazo("duplicado_placa",
                                "reeval: ventana")
                _descartar(ruta, "duplicado_placa",
                           f"(reeval: placa {texto_re} registrada hace "
                           f"{minutos_re:.0f} min, ventana {ventana:.0f} min)")
                return
            max_pasadas = int(_cfg.get("max_pasadas", 3))
            nit_re = _nitidez(rec_re)
            elegido = (bbox_re, conf_re, nit_re, texto_re, pasada_re,
                       bool(pasada_re > max_pasadas),
                       _sufijo_pasada(pasada_re))
            bbox, conf_ocr, intentos_ocr = bbox_re, conf_re, intentos_re
            bbox_ocr, huella = bbox_re, huella_re
            dudosa = True
            logger.info("REEVALUACION EXITOSA (intento %d/3): %s <- %s",
                        intento + 1, texto_re, ruta.name)
            break

    if elegido is None:
        mejor_texto, mejor_conf = "", 0.0
        for _n, t, c in intentos_ocr:
            if t and c > mejor_conf:
                mejor_texto, mejor_conf = t, c
        if mejor_texto and mejor_conf >= UMBRAL_ETIQUETA_AUTO:
            try:
                encolada = _encolar_entrenamiento(ruta, img, bbox_ocr,
                                                  mejor_texto, intentos_ocr)
                if encolada:
                    logger.info(
                        "Foto fallida encolada para entrenar (mejor intento "
                        "%.2f, fallo %s): %s: %s",
                        mejor_conf, ultimo_fallo[0], mejor_texto, ruta.name)
            except Exception:
                logger.exception("No se pudo encolar la foto fallida %s",
                                 ruta.name)
        try:
            import log_sin_lectura
            log_sin_lectura.registrar_worker(
                ruta=ruta, img=img, candidatos=candidatos,
                intentos=intentos_ocr, bbox=bbox_ocr,
                filtro=ultimo_fallo[0], detalle=ultimo_fallo[1])
        except Exception:
            logger.debug("log_sin_lectura no disponible", exc_info=True)
        _anotar_rechazo(ultimo_fallo[0], ultimo_fallo[1])
        _descartar(ruta, ultimo_fallo[0], ultimo_fallo[1])
        _cooldown_sin_placa()
        return

    bbox, confianza, nitidez, texto_placa, pasada, excedida, sufijo = elegido

    # Gate de formato: solo placas peruanas reales. Los carteles y
    # stickers leidos como placa (5TC-000) no pasan de aca.
    # 2026-11-09: se archiva como texto_invalido (uno de los 3 motivos).
    if (texto_placa and _cfg.get("validar_formato", True)
            and not _formato_placa_valido(texto_placa)):
        detalle_fmt = f"(texto {texto_placa} no es formato peruano)"
        _anotar_rechazo("texto_invalido", detalle_fmt)
        _descartar(ruta, "texto_invalido", detalle_fmt)
        return

    # Confianza minima de la deteccion: placas reales 0.83+; placards
    # (UN1-910 0.64) y stickers (WO3-330 0.65) quedan afuera. 0 = sin
    # informacion (rutas sin scoring) no se rechaza.
    # 2026-11-09: se archiva como texto_invalido (uno de los 3 motivos).
    min_conf = float(_cfg.get("confianza_minima", 0.66))
    if confianza and confianza < min_conf:
        detalle_conf = f"(confianza {confianza:.2f} < {min_conf})"
        _anotar_rechazo("texto_invalido", detalle_conf)
        _descartar(ruta, "texto_invalido", detalle_conf)
        return

    evidencia, color = (evidencia_y_color(texto_placa, conf_ocr, intentos_ocr)
                        if evidencia_y_color is not None and texto_placa
                        else (0, "rojo"))

    if excedida:
        logger.warning("REINSPECCION EXCEDIDA: placa %s en pasada %d "
                       "(maximo %d por %d dias) -> %s",
                       texto_placa, pasada, max_pasadas,
                       int(_cfg.get("ventana_reinspeccion_dias", 30)),
                       ruta.name)

    _anotar_rechazo("guardado")
    
    # NUEVO: Sufijo condicional si el ROI está habilitado
    sufijo_roi = "_roi-evaluado" if _cfg.get("roi_habilitado") else ""

    destino = _guardar(foto_save, _bbox_full(bbox), vista, texto_placa,
                       sufijo=etiqueta_fuente(intentos_ocr) + sufijo_roi,
                       dudosa=dudosa, pasada=pasada)
    relativo = str(destino.relative_to(_cfg["procesadas"]))
    confirmada = _detector is not None or bool(candidatos_rf)

    via_roboflow = any(n == "roboflow" for n, _t, _c in intentos_ocr)
    via_api = any(n == "api" for n, _t, _c in intentos_ocr)
    placa_api = next((t for n, t, _c in intentos_ocr
                      if n in ("api", "roboflow") and t), None)
    provisional = (bool(texto_placa)
                   and (_AOI_ACTIVA or _ROBOFLOW_ACTIVO)
                   and not via_api and not via_roboflow)
    motivo = None
    if provisional:
        try:
            if _ROBOFLOW_ACTIVO:
                import roboflow
                motivo = roboflow.motivo_sin_api()
            elif _AOI_ACTIVA:
                import platerecognizer
                motivo = platerecognizer.motivo_sin_api()
            else:
                motivo = "sin_lectura_api"
        except Exception:
            motivo = "sin_lectura_api"
    registrar(relativo, placa=texto_placa or None, confianza=confianza,
              estado="guardado" if confirmada else "pendiente", phash=huella,
              evidencia=evidencia, semaforo=color,
              fuente=(fuente_de_intentos(intentos_ocr) if texto_placa
                      else None),
              provisional=provisional, placa_api=placa_api,
              placa_modelo=(None if (via_api or via_roboflow)
                            else (texto_placa or None)),
              motivo_sin_api=motivo)

    if texto_placa:
        try:
            _encolar_entrenamiento(ruta, img, bbox, texto_placa,
                                   intentos_ocr)
        except Exception:
            logger.exception("No se pudo encolar la foto %s para entrenar",
                             ruta.name)

    _hashes.setdefault(vista, {})[huella] = texto_placa
    if texto_placa:
        _registrar_pasada(texto_placa, pasada, excedida)
        _placas.setdefault(vista, set()).add(texto_placa)
        _marcar_json()

    origen = "conf" if confirmada else "geom"
    leida = f" (placa {texto_placa}" if texto_placa else ""
    cierre_leida = f", pasada {pasada}{sufijo})" if texto_placa else ")"
    logger.info("Placa detectada%s%s: %s (nitidez %.1f, %s %.2f, "
                "%.2f%% de cambio) <- %s", leida, cierre_leida, relativo,
                nitidez, origen, confianza, porcentaje, ruta.name)

    _ultima_imagen[vista] = img.copy()
    ts_guardada = _ts_de_foto(ruta.name)
    if ts_guardada is not None:
        _ultima_placa_foto_ts = ts_guardada
    ruta.unlink(missing_ok=True)
    _cooldown_placa()


def _debe_entrar_en_reposo() -> bool:
    """True solo si toca reposo Y no queda nada en la cola.

    Con cola pendiente el worker nunca se duerme: el reposo descarga
    detector/OCR y bajaria a prioridad IDLE en medio del drenaje.
    """
    return (_reposo_seg > 0 and not _en_reposo
            and COLA.qsize() == 0
            and time.time() - _t_ultima_actividad >= _reposo_seg)


def bucle_worker() -> None:
    global _t_ultima_actividad
    _t_ultima_actividad = time.time()
    _prioridad(True)
    logger.info("Bucle del worker iniciado")
    while True:
        try:
            ruta = COLA.get(timeout=1.0)
            with _encolados_lock:
                _encolados.discard(ruta)
        except queue.Empty:
            _vaciar_json_pendiente()
            if _debe_entrar_en_reposo():
                _entrar_en_reposo()
            continue
        _despertar()
        try:
            procesar_imagen(Path(ruta))
        except Exception:
            logger.exception("Error procesando %s", ruta)
            try:
                Path(ruta).unlink(missing_ok=True)
            except OSError:
                pass
        finally:
            COLA.task_done()
            _t_ultima_actividad = time.time()