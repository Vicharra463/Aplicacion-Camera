#!/usr/bin/env python3
r"""Cola de trabajo de la camara CYTV Ancon.

Responsabilidades:
  - Anti-duplicados estrictos (cooldown + diferencia visual)
  - Filtros de calidad de la placa, en orden de menor a mayor costo
  - Dedup global por hash perceptual: una sola imagen por placa
  - Enrutado por VISTA de la camara: canal 01 (PT) -> carpeta placa\
    (modelo + OCR + resalte rojo), canal 02 (panoramico) -> carpeta
    vehiculo\ (foto cruda, nombre por hora), cada una con su fecha
  - Registro en SQLite de cada decision (aceptada o descartada, y por que)

Orden de la cadena de filtros:

    cooldown -> diferencia visual -> candidato -> hay texto -> nitidez del
    recorte -> sobreexposicion -> oclusion -> dedup -> guardar

Fase detector (hecha):
    si existe ``modelo\\placas.onnx`` se carga ``modelo.DetectorPlacas`` y la
    deteccion es la del modelo ONNX (estado ``guardado``, carpeta de dia
    limpia). Si no existe o falla al cargar, se cae al detector geometrico
    (OpenCV puro) y la imagen queda ``pendiente`` en la subcarpeta
    ``pendientes``. El servicio jamas se cae por el modelo.

Extension pendiente (Fase OCR):
    el texto de la placa se escribe en ``registros.placa`` con
    ``registrar(..., placa=...)``; ``ftp_server.py`` no se toca.
"""
from __future__ import annotations

import json
import logging
import queue
import re
import shutil
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

# Import perezoso y a prueba de fallos: si onnxruntime o rapidocr no estan
# instalados, el worker sigue funcionando sin esa pieza en vez de no arrancar.
try:
    from modelo import DetectorPlacas, RUTA_ONNX
except Exception:                                     # pragma: no cover
    DetectorPlacas = None                             # type: ignore[assignment]
    RUTA_ONNX = None                                   # type: ignore[assignment]

try:
    from ocr import leer_placa, leer_placa_6
except Exception:                                     # pragma: no cover
    leer_placa = None                                 # type: ignore[assignment]
    leer_placa_6 = None                               # type: ignore[assignment]

logger = logging.getLogger("cytv.worker")

# Cola entre el hilo FTP y el worker. El FTP jamas espera: si se llena,
# el archivo queda en disco y el barrido de huerfanos lo reintenta.
COLA: "queue.Queue[str]" = queue.Queue(maxsize=200)

# Motivos de volumen: se loguean a DEBUG para no llenar servidor.log.
# Hoy solo "sin_texto" (solo existe en el fallback geometrico). Todo lo demas
# --cooldown, duplicado, sin_placa, duplicado_placa, texto_invalido-- va a INFO
# porque el operador tiene que ver que le pasa a cada imagen que recibe la
# camara: Imagen recibida -> No se detecta placa / Placa detectada
# -> Imagen eliminada.
_MOTIVOS_VOLUMEN = {"sin_texto"}

# ---------------------------------------------------------------------------
# Vistas de la camara (TandemVu de 2 canales). El nombre que sube por FTP
# codifica el canal: 192.168.18.42_01_20261009171056107_MD_WITH_TARGET.jpg
#   canal 01 -> PT 8 mm (44°): ~2x pixeles en la placa -> carpeta placa\
#               (modelo + OCR + resalte rojo; nombre = placa canonica)
#   canal 02 -> panoramico 2.8 mm (94°): el vehiculo COMPLETO
#               -> carpeta vehiculo\ (foto cruda, nombre por hora, sin OCR)
# Mapeo corregido 2026-10-16: antes estaba invertido respecto a la
# realidad fisica de la camara (causa de que la "placa" llegara al
# panoramico y viceversa).
# ---------------------------------------------------------------------------
FOLDER_POR_CANAL = {"01": "placa", "02": "vehiculo"}
CARPETA_POR_DEFECTO = "placa"
_RE_CANAL = re.compile(r"_(\d{2})_")

# Estado anti-duplicados. Solo lo escribe este hilo (unico consumidor),
# por eso no necesita bloqueo. Es POR VISTA: las dos fotos de un mismo
# evento llegan ~1,4 s separadas y ninguna debe matar a la otra.
_ultima_imagen: dict[str, np.ndarray] = {}
_ultimo_timestamp: dict[str, float] = {}

# Dedup cargado de la base: por vista -> huella perceptual -> texto de placa
# ("" si no se pudo leer en su momento), mas las placas ya vistas para el
# dedup EXACTO por texto que da el OCR. Una misma placa puede y DEBE guardarse
# en las dos vistas (son carpetas distintas con propositos distintos).
_hashes: dict[str, dict[int, str]] = {}
_placas: dict[str, set[str]] = {}
# Dia al que pertenecen _hashes y _placas. El dedup NO cruza de fecha: si el
# mismo vehiculo vuelve despues de tiempo (otra fecha) se registra de nuevo.
_dia_dedup: str = ""

_cfg: dict = {}
_ruta_registro: Path | None = None
# Detector ONNX cargado (None = modo geometrico). Se carga una sola vez.
_detector = None


def _carpeta_de(nombre_archivo: str) -> str:
    """Vista de la camara a la que pertenece un archivo recibido.

    Extrae el canal del nombre (…_01_20261009….jpg) y lo traduce a la
    carpeta (01 -> placa, 02 -> vehiculo); sin patron de canal (subidas
    manuales de prueba) va a la vista por defecto (placa).
    """
    encontrado = _RE_CANAL.search(nombre_archivo)
    canal = encontrado.group(1) if encontrado else ""
    return FOLDER_POR_CANAL.get(canal, CARPETA_POR_DEFECTO)


def _carpeta_de_ruta(relativo: str) -> str:
    """Vista que corresponde a una ruta ya guardada en el registro.

    Formato nuevo: ``vehiculo\\2026-10-09\\vehiculo_16-32-28.jpg``.
    Formato viejo (antes del enrutado): ``2026-10-09\\vehiculo_…jpg`` ->
    sin carpeta al frente, se asume la vista por defecto.
    """
    parte = Path(relativo).parts[0] if relativo else ""
    return parte if parte in FOLDER_POR_CANAL.values() else CARPETA_POR_DEFECTO


def iniciar(cfg: dict) -> None:
    """Prepara carpetas, crea la tabla, carga hashes/placas y el modelo."""
    global _cfg, _ruta_registro, _hashes, _placas, _dia_dedup, _detector
    _cfg = cfg
    _ruta_registro = cfg["registro"]
    cfg["procesadas"].mkdir(parents=True, exist_ok=True)
    # Con modo_descarte=eliminar no se crea la carpeta de descartes: nada
    # se queda ahi. Con retener se crea para poder auditar.
    if cfg.get("modo_descarte", "retener") == "retener":
        _carpeta_descartes().mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(_ruta_registro)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS registros (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha_hora TEXT    NOT NULL,
                archivo    TEXT    NOT NULL,
                placa      TEXT,
                confianza  REAL,
                estado     TEXT    NOT NULL DEFAULT 'guardado'
            )
            """
        )
        # Migracion suave: columnas que se sumaron despues de la tabla inicial.
        columnas = {fila[1] for fila in con.execute("PRAGMA table_info(registros)")}
        if "phash" not in columnas:
            con.execute("ALTER TABLE registros ADD COLUMN phash TEXT")
        # Control de pasadas (Fase 1): una placa vuelve a registrar despues
        # de ventana_reinspeccion_min; max_pasadas por ventana_reinspeccion_dias;
        # mas alla de max_pasadas se guarda igual con sufijo EXCEDIDA + alerta.
        # CREATE IF NOT EXISTS = migracion idempotente (re-servicio seguro).
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS pasadas (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                placa     TEXT    NOT NULL,
                pasada    INTEGER NOT NULL,
                fecha_hora TEXT   NOT NULL,
                excedida  INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_pasadas_placa "
            "ON pasadas (placa, fecha_hora)"
        )
        con.commit()
        # Dedup cargado solo de imagenes que SIGUEN en disco. Si la carpeta
        # se vacio (limpieza manual), arrastrar esas huellas haria que la
        # proxima foto de esa misma placa se descartara como duplicada
        # aunque no exista ni una imagen. La fila queda en la base; solo se
        # ignora su huella.
        # Ademas solo del DIA DE HOY: una placa vista ayer no bloquea la de
        # hoy (el vehiculo que vuelve despues de tiempo es un evento nuevo).
        _dia_dedup = datetime.now().strftime("%Y-%m-%d")
        limite_dia = f"{_dia_dedup}T00:00:00"
        carpeta_dia = _cfg["procesadas"]
        _hashes = {}
        _placas = {}
        for valor, archivo, placa in con.execute(
                "SELECT phash, archivo, placa FROM registros "
                "WHERE phash IS NOT NULL AND fecha_hora >= ?",
                (limite_dia,)):
            if archivo and not (carpeta_dia / archivo).is_file():
                continue
            vista = _carpeta_de_ruta(archivo)
            _hashes.setdefault(vista, {})[int(valor, 16)] = placa or ""
        for texto, archivo in con.execute(
                "SELECT placa, archivo FROM registros "
                "WHERE placa IS NOT NULL AND fecha_hora >= ?",
                (limite_dia,)
        ):
            if not texto:
                continue
            if archivo and not (carpeta_dia / archivo).is_file():
                continue
            _placas.setdefault(_carpeta_de_ruta(archivo), set()).add(texto)
    finally:
        con.close()
    total_huellas = sum(len(v) for v in _hashes.values())
    total_placas = sum(len(v) for v in _placas.values())
    logger.info("Worker listo: registro en %s, %d huellas y %d placas cargadas "
                "(dedup del dia %s, vistas: %s)",
                _ruta_registro.name, total_huellas, total_placas, _dia_dedup,
                ", ".join(f"{k}={len(v)} placas"
                          for k, v in sorted(_placas.items())) or "ninguna")

    # Fase detector: si hay pesos ONNX publicados, la deteccion la hace el
    # modelo. Cualquier fallo deja el detector en None y seguimos con la
    # geometria: el servicio no puede quedarse sin arrancar por esto.
    if DetectorPlacas is not None and RUTA_ONNX is not None and RUTA_ONNX.is_file():
        try:
            _detector = DetectorPlacas(RUTA_ONNX,
                                       umbral=float(cfg.get("umbral_modelo", 0.25)))
            logger.info("Deteccion: MODELO ONNX activo (%s)", RUTA_ONNX.name)
        except Exception:
            logger.exception("No se pudo cargar el modelo ONNX; "
                             "sigo con el detector geometrico")
            _detector = None
    else:
        logger.info("Deteccion: sin modelo ONNX, detector geometrico")

    # Caliento el OCR: la primera llamada tarda ~0.45 s cargando los pesos,
    # mejor pagarlo aqui que en la primera placa que llegue.
    if leer_placa is not None:
        try:
            leer_placa(np.zeros((48, 160, 3), dtype=np.uint8))
            logger.info("OCR: RapidOCR listo (dedup exacto por texto de placa)")
        except Exception:
            logger.exception("No se pudo iniciar el OCR; sigo con el dedup "
                             "por hash perceptual")
    else:
        logger.info("OCR: no disponible, dedup solo por hash perceptual")

    # Reportes JSON: el dia en curso se regenera desde la base (un
    # reinicio a media jornada no deja el archivo viejo) y los dias
    # ANTERIORES con placas guardadas tambien: si el servicio estuvo
    # apagado y no cruzo la medianoche vivo, ese dia nunca se "cerraba"
    # y su placas-AAAA-MM-DD.json quedaba sin generar.
    try:
        ruta_json = generar_json_dia(_dia_dedup)
        logger.info("Reporte JSON del dia %s actualizado: %s",
                    _dia_dedup, ruta_json)
    except Exception:
        logger.exception("No se pudo generar el reporte JSON del dia %s",
                         _dia_dedup)
    try:
        con = sqlite3.connect(_ruta_registro)
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


def _actualizar_dia() -> None:
    """Cruza la medianoche: el dedup vive solo dentro del dia.

    Un vehiculo visto ayer NO puede bloquear el de hoy. Si el servicio corre
    por mas de un dia, al cambiar de fecha se vacian huellas y placas de las
    DOS vistas y el proximo paso de esa matricula vuelve a contar.
    """
    global _dia_dedup, _hashes, _placas, _ultima_imagen, _ultimo_timestamp
    hoy = datetime.now().strftime("%Y-%m-%d")
    if hoy == _dia_dedup:
        return
    if _dia_dedup:
        # Cierre del dia que se va: el JSON de ayer queda final.
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
    _ultimo_timestamp = {}


def registrar(archivo: str, placa: str | None = None,
              confianza: float | None = None, estado: str = "guardado",
              phash: int | None = None) -> None:
    """Agrega una fila al registro."""
    if _ruta_registro is None:
        return
    con = sqlite3.connect(_ruta_registro)
    try:
        con.execute(
            "INSERT INTO registros (fecha_hora, archivo, placa, confianza, estado, phash) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), archivo, placa,
             confianza, estado, None if phash is None else f"{phash:016x}"),
        )
        con.commit()
    finally:
        con.close()


def _carpeta_reportes() -> Path:
    propio = _cfg.get("reportes")
    if propio:
        return Path(propio)
    return _cfg["procesadas"].parent / "reportes"


def generar_json_dia(fecha: str) -> Path:
    """Volcado JSON de todas las placas GUARDADAS del dia ``fecha``.

    Una entrada por placa unica (texto del OCR) con primera/ultima hora,
    los archivos de la vista placa y de la panoramica, y la confianza
    maxima del modelo. Se REESCRIBE completo (idempotente): sirve para
    refrescar el dia en curso y para cerrar el dia al cruzar medianoche.

    Salida: ``reportes\\placas-<fecha>.json`` (absoluta, lista para que
    cualquier otro sistema la consuma).
    """
    if _ruta_registro is None:
        raise RuntimeError("worker no inicializado")
    con = sqlite3.connect(_ruta_registro)
    try:
        filas = con.execute(
            "SELECT fecha_hora, archivo, placa, confianza FROM registros "
            "WHERE fecha_hora LIKE ? AND estado = 'guardado' "
            "AND placa IS NOT NULL AND placa != '' ORDER BY fecha_hora",
            (f"{fecha}%",),
        ).fetchall()
        # Ultima pasada registrada el dia ``fecha`` por placa (control de
        # pasadas Fase 1): si una placa volvio, el JSON lo refleja.
        pasadas_dia: dict[str, tuple[int, bool]] = {}
        for placa, pasada, excedida in con.execute(
                "SELECT placa, pasada, excedida FROM pasadas "
                "WHERE fecha_hora LIKE ? ORDER BY fecha_hora",
                (f"{fecha}%",)):
            pasadas_dia[placa] = (pasada, bool(excedida))
    finally:
        con.close()

    por_placa: dict[str, dict] = {}
    for fecha_hora, archivo, placa, confianza in filas:
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
                "imagenes": {"placa": [], "panoramica": []},
            }
        entrada["ultima_vez"] = fecha_hora
        entrada["detecciones"] += 1
        if confianza is not None:
            entrada["confianza_max"] = max(entrada["confianza_max"], confianza)
        vista = _carpeta_de_ruta(archivo)
        clave = "placa" if vista == "placa" else "panoramica"
        entrada["imagenes"][clave].append(str(_cfg["procesadas"] / archivo))

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
    # Escritura atomica: un lector del JSON nunca ve el archivo a medias.
    temporal = carpeta / f"placas-{fecha}.json.tmp"
    temporal.write_text(json.dumps(datos, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    temporal.replace(destino)
    return destino


def calcular_diferencia(img1: np.ndarray, img2: np.ndarray) -> float:
    """Porcentaje de pixeles que cambian entre dos tomas (mismo criterio que FTP.py)."""
    img1_peq = cv2.resize(img1, (640, 480))
    img2_peq = cv2.resize(img2, (640, 480))
    gris1 = cv2.cvtColor(img1_peq, cv2.COLOR_BGR2GRAY)
    gris2 = cv2.cvtColor(img2_peq, cv2.COLOR_BGR2GRAY)
    diferencia = cv2.absdiff(gris1, gris2)
    _, umbral = cv2.threshold(diferencia, 35, 255, cv2.THRESH_BINARY)
    return (np.count_nonzero(umbral) / (umbral.shape[0] * umbral.shape[1])) * 100


def encolar(ruta: str) -> bool:
    """Pone una imagen en la cola sin bloquear al llamante."""
    try:
        COLA.put_nowait(ruta)
        return True
    except queue.Full:
        logger.warning("Cola llena: %s queda para el barrido de huerfanos", Path(ruta).name)
        return False


# --------------------------------------------------------------------------
# Filtros
# --------------------------------------------------------------------------

def _candidatos_placa(img: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
    """Busca las regiones con forma de plancha, ordenadas de mejor a peor.

    Con modelo ONNX cargado (Fase detector) la geometria no se usa: si el
    modelo no ve placa, devuelve lista vacia y la imagen se descarta. Sin
    modelo, devuelve un score geometrico (0..1) y prueba VARIAS regiones
    porque si el mejor candidato es ruido, se prueba el siguiente en vez de
    tirar la foto.

    ((x, y, w, h), score)
    """
    if _detector is not None:
        return _detector.detectar(img)

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
        # Placa peruana ~460x100 mm -> relacion 4.6; se tolera 1.6..6.0 para
        # camiones, angulos y placas de motos (3L+4N, mas cuadradas).
        if not 1.6 <= relacion <= 6.0:
            continue
        if not 0.0015 * area_img <= w * h <= 0.30 * area_img:
            continue
        if x < 0 or y < 0 or x + w > ancho or y + h > alto:
            continue
        ordenados.append((1.0 - abs(relacion - 4.6) / 4.6, (x, y, w, h)))

    ordenados.sort(key=lambda par: par[0], reverse=True)

    # Suprime cajas anidadas del mismo objeto (contornos internos/externos).
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
    """% de pixeles oscuros (texto) dentro del recorte.

    Es lo que separa una placa de un rectangulo cualquiera: la placa tiene
    caracteres negros sobre fondo claro. Un blob de ruido o una region lisa
    queda por fuera del rango.
    """
    gris = _a_gris(recorte)
    _, texto = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return float(np.count_nonzero(texto)) / texto.size * 100.0


def _nitidez(recorte: np.ndarray) -> float:
    """Varianza del Laplaciano SOBRE EL RECORTE. Foto nitta con placa movida
    no pasa este filtro, que es justamente el error clasico."""
    return float(cv2.Laplacian(_a_gris(recorte), cv2.CV_64F).var())


def _fraccion_sobreexpuesta(recorte: np.ndarray) -> float:
    """% de pixeles saturados. Las placas retrorreflectivas se queman con IR."""
    gris = _a_gris(recorte)
    return float(np.count_nonzero(gris > 250)) / gris.size * 100.0


def _fraccion_bordes(recorte: np.ndarray) -> float:
    """% de pixeles con borde. Proxy de oclusion: una placa descubierta tiene
    bordes de caracteres; piel o ropa quedan mucho mas lisos. Heuristica: se
    reemplaza por detector de personas cuando entre el YOLO."""
    gris = _a_gris(recorte)
    bordes = cv2.Canny(gris, 80, 200)
    return float(np.count_nonzero(bordes)) / bordes.size * 100.0


def _hash_perceptual(img: np.ndarray) -> int:
    """pHash de 64 bits (DCT 8x8). Dedup sin necesidad de leer la placa."""
    gris = _a_gris(img)
    pequena = np.float32(cv2.resize(gris, (32, 32), interpolation=cv2.INTER_AREA))
    coef = np.float32(cv2.dct(pequena))[:8, :8].reshape(-1)
    mediana = float(np.median(coef))
    valor = 0
    for bit in coef > mediana:
        valor = (valor << 1) | int(bit)
    return valor


# ---------------------------------------------------------------------------
# Resalte en la foto guardada (Fase 0): rojo = deteccion ganadora (placa
# leida); amarillo = detecciones candidatas fallidas en sin_placa (diagnostico
# visual sin abrir el log). El original en ftp_recibidos\ jamas se toca:
# el resalte se dibuja sobre una COPIA que es la que se escribe en
# imagenes_procesadas\.
# ---------------------------------------------------------------------------
_COLOR_ROJO = (0, 0, 255)        # BGR
_COLOR_AMARILLO = (0, 255, 255)  # BGR


def _dibujar_caja(img: np.ndarray, bbox: tuple[int, int, int, int],
                  color: tuple[int, int, int], padding: float = 0.08) -> np.ndarray:
    """Copia de img con un rectangulo sobre bbox.

    Padding ~8% para que el borde NO tape los caracteres de la placa y
    grosor proporcional al tamanio de la imagen (1 px no se ve en una foto
    de 2 MP): se escala con el lado menor.
    """
    salida = img.copy()
    x, y, w, h = bbox
    dx, dy = int(w * padding), int(h * padding)
    alto, ancho = salida.shape[:2]
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(ancho, x + w + dx), min(alto, y + h + dy)
    grosor = max(2, round(min(alto, ancho) / 300))
    cv2.rectangle(salida, (x0, y0), (x1, y1), color, grosor)
    return salida


# ---------------------------------------------------------------------------
# Control de pasadas (Fase 1): ventana entre visitas + tope por periodo.
# ---------------------------------------------------------------------------

def _pasada_de_placa(placa: str) -> tuple[int, float]:
    """(pasada a registrar, minutos desde la ultima visita dentro de la
    ventana). Primera vez => (1, inf)."""
    dias = int(_cfg.get("ventana_reinspeccion_dias", 30))
    limite = (datetime.now() - timedelta(days=dias)).isoformat(timespec="seconds")
    con = sqlite3.connect(_ruta_registro)
    try:
        filas = [fila[0] for fila in con.execute(
            "SELECT fecha_hora FROM pasadas WHERE placa = ? AND fecha_hora >= ? "
            "ORDER BY fecha_hora", (placa, limite))]
    finally:
        con.close()
    if not filas:
        return 1, float("inf")
    ultima = datetime.fromisoformat(filas[-1])
    minutos = (datetime.now() - ultima).total_seconds() / 60.0
    return len(filas) + 1, minutos


def _registrar_pasada(placa: str, pasada: int, excedida: bool) -> None:
    con = sqlite3.connect(_ruta_registro)
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
    """Sufijo del nombre de archivo por pasada (1a sin sufijo)."""
    return {1: "", 2: "_2da-pasada", 3: "_3ra-pasada"}.get(pasada, "_EXCEDIDA")


def _recorte_con_margen(img: np.ndarray, bbox: tuple[int, int, int, int],
                        margen: float = 0.35) -> np.ndarray:
    """Recorte del bbox con aire alrededor, para el OCR.

    El OCR necesita contexto: con el recorte apretado leia CBN-449 como
    BNE-449 (se come el borde de la C). Con 35% de margen las 4 capturas
    reales del dia se leyeron bien.
    """
    x, y, w, h = bbox
    dx, dy = int(w * margen), int(h * margen)
    alto, ancho = img.shape[:2]
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(ancho, x + w + dx), min(alto, y + h + dy)
    return img[y0:y1, x0:x1]


def _cerca_texto(a: str, b: str) -> bool:
    """Dos lecturas son 'la misma placa' si difieren en a lo sumo 2
    caracteres (Levenshtein): el OCR a veces lee CBN-448 por CBN-449.

    Si difieren mas, son placas distintas aunque los recortes se parezcan:
    ahi manda el texto y NO se fusionan."""
    if a == b:
        return True
    if not a or not b:
        return False
    if abs(len(a) - len(b)) > 2:
        return False
    # Levenshtein completo: las cadenas son de 5-8 caracteres, el costo es nulo.
    anterior = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        actual = [i]
        for j, cb in enumerate(b, 1):
            actual.append(min(anterior[j] + 1, actual[-1] + 1,
                              anterior[j - 1] + (ca != cb)))
        anterior = actual
    return anterior[-1] <= 2


def _sin_guion(texto: str) -> str:
    return texto.replace("-", "")


# Equivalencias que el OCR confunde y que para DEDUP no distinguen:
# B1Y == BIY (I/1), 4O == 40 (O/0), Z1S == ZIS (Z/7 no aplica aqui pero si
# 1/I). Se usan SOLO para comparar dos lecturas, nunca para canonicas.
_EQUIV_COMPARAR = str.maketrans({"I": "1", "O": "0", "Z": "7"})


def _forma_comparable(texto: str) -> str:
    return _sin_guion(texto).upper().translate(_EQUIV_COMPARAR)


def _es_misma_placa(vista: str, huella: int, texto: str = "") -> bool:
    """Dedup en dos niveles, SCOPADO A LA VISTA (Fase 1, fix de la causa
    raiz de los duplicados):

    1) TEXTO PRIMERO: si el OCR leyo la placa, manda el texto (sin guion,
       Levenshtein <= 2 con equivalencias I/1 O/0 Z/7). Asi CRV-25 y
       CRW-025, o B1Y-756 y BIY-756, se fusionan AUNQUE el vehiculo este
       en un angulo distinto (el pHash queda lejos y el dedup viejo jamas
       comparaba el texto: por eso se guardaban como placas distintas).
    2) pHash SOLO de respaldo: mismo recorte visual cuando no hay texto
       (o las dos lecturas quedaron vacias)."""
    guardados = _hashes.get(vista, {})
    if texto:
        forma = _forma_comparable(texto)
        for texto_guardado in guardados.values():
            if not texto_guardado:
                continue
            if _cerca_texto(forma, _forma_comparable(texto_guardado)):
                return True
    umbral = _cfg.get("umbral_dedup", 18)
    for guardada in guardados:
        if bin(huella ^ guardada).count("1") <= umbral:
            return True
    return False


def _carpeta_descartes() -> Path:
    propio = _cfg.get("descartes")
    if propio:
        return Path(propio)
    return _cfg["procesadas"].parent / "descartadas"


def _destino_libre(carpeta: Path, nombre: str) -> Path:
    destino = carpeta / nombre
    indice = 1
    while destino.exists():
        destino = carpeta / f"{Path(nombre).stem}_{indice}{Path(nombre).suffix}"
        indice += 1
    return destino


# Windows no acepta <>:"/\|?* en nombres: el texto del OCR se limpia a
# letras, numeros y guion (los separadores raros se vuelven guion y los
# guiones dobles se colapsan). Si tras limpiar queda vacio, se usa el
# nombre por hora de siempre (no se pierde la imagen por un OCR raro).
_RE_NOMBRE_PLACA = re.compile(r"[^A-Za-z0-9-]+")


def _nombre_con_placa(vista: str, texto_placa: str, ahora: datetime,
                      sufijo: str = "") -> str:
    """Nomenclatura de archivo pedida por el usuario:

    - vista placa    -> ``ABC-123.jpg`` (+ ``_2da-pasada``/``_3ra-pasada``/
      ``_EXCEDIDA`` cuando corresponde, ver _sufijo_pasada)
    - vista vehiculo -> ``placa-panoramica_ABC-123.jpg`` (solo con OCR;
      la panoramica cruda del canal 02 va con nombre por hora)
    - sin lectura    -> ``<vista>_HH-MM-SS_mmm.jpg`` (nombre clasico)

    Devuelve solo la BASE (sin extension); ``_guardar`` le pone .jpg y
    el .txt de la etiqueta YOLO con el mismo stem.
    """
    base = _RE_NOMBRE_PLACA.sub("-", texto_placa or "").strip("-")
    base = re.sub(r"-{2,}", "-", base).upper()
    if not base:
        return f"{vista}_{ahora.strftime('%H-%M-%S')}_{ahora.microsecond:03d}"
    if vista == "placa":
        return base + sufijo
    return f"placa-panoramica_{base}" + sufijo


def _descartar(ruta: Path, motivo: str, detalle: str = "",
               auditar: bool = True,
               img_anotada: np.ndarray | None = None) -> None:
    """Aplica modo_descarte (retener/eliminar) y deja constancia en el registro.

    Con ``img_anotada`` (resalte amarillo de las detecciones fallidas) la
    copia que queda en descartadas\\ lleva los rectangulos; el original en
    ftp_recibidos\\ no se toca nunca (se borra o se reemplaza tras copiar).
    """
    nombre = ruta.name
    if _cfg.get("modo_descarte", "retener") == "eliminar":
        ruta.unlink(missing_ok=True)
        accion = "Imagen eliminada"
    else:
        carpeta = _carpeta_descartes() / datetime.now().strftime("%Y-%m-%d") / motivo
        carpeta.mkdir(parents=True, exist_ok=True)
        destino = _destino_libre(carpeta, nombre)
        try:
            if img_anotada is not None:
                cv2.imwrite(str(destino), img_anotada,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                ruta.unlink(missing_ok=True)
            else:
                shutil.move(str(ruta), str(destino))
            accion = "Imagen movida a descartadas"
        except OSError:
            logger.warning("No se pudo mover el descarte %s", nombre)
            ruta.unlink(missing_ok=True)
            accion = "Imagen eliminada"

    if auditar:
        registrar(nombre, estado=motivo)

    sufijo = f" {detalle}" if detalle else ""
    nivel = logging.DEBUG if motivo in _MOTIVOS_VOLUMEN else logging.INFO

    # Mensajes en claro para el operador: el flujo completo en el log es
    #   Imagen recibida -> (No se detecta placa) -> Imagen eliminada
    if motivo == "sin_placa":
        logger.info("No se detecta placa: %s", nombre)
    logger.log(nivel, "%s: %s | motivo: %s%s", accion, nombre, motivo, sufijo)


def _guardar(img: np.ndarray, bbox: tuple[int, int, int, int] | None,
             vista: str, texto_placa: str = "", sufijo: str = "") -> Path:
    """Escribe imagen en calidad maxima + etiqueta YOLO (.txt) en
    ``<vista>\\<fecha>\\``. Si el OCR ya leyo la placa, el nombre del
    archivo ES la placa (ver _nombre_con_placa); si no, por hora.

    Resalte (Fase 0): en vista placa se dibuja un rectangulo ROJO sobre la
    region de la deteccion ganadora (padding 8%, grosor proporcional). La
    panoramica del canal 02 se guarda SIN caja (no hay deteccion). El
    original en ftp_recibidos\\ queda intacto: esto escribe una copia.
    """
    ahora = datetime.now()
    dia = _cfg["procesadas"] / vista / ahora.strftime("%Y-%m-%d")
    # Carpeta limpia: con modelo activo la imagen ya esta confirmada y va
    # directo al dia; sin modelo queda en pendientes hasta la revision.
    carpeta = dia if _detector is not None else dia / "pendientes"
    carpeta.mkdir(parents=True, exist_ok=True)

    nombre = _nombre_con_placa(vista, texto_placa, ahora, sufijo)
    destino = _destino_libre(carpeta, f"{nombre}.jpg")
    salida = img
    if vista == "placa" and bbox is not None:
        salida = _dibujar_caja(img, bbox, _COLOR_ROJO)
    cv2.imwrite(str(destino), salida, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    if bbox is not None:
        x, y, w, h = bbox
        alto, ancho = img.shape[:2]
        etiqueta = (f"0 {(x + w / 2) / ancho:.6f} {(y + h / 2) / alto:.6f} "
                    f"{w / ancho:.6f} {h / alto:.6f}\n")
        (destino.with_suffix(".txt")).write_text(etiqueta, encoding="utf-8")
    return destino


def procesar_imagen(ruta: Path) -> None:
    """Aplica la cadena completa de filtros y guarda si todo pasa."""
    global _ultima_imagen, _ultimo_timestamp

    if not ruta.exists():
        return

    # Antes de cualquier dedup: si cambio la fecha, empieza memoria nueva.
    _actualizar_dia()

    img = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("Imagen ilegible, se elimina: %s", ruta.name)
        ruta.unlink(missing_ok=True)
        return

    # A que vista pertenece (canal 01/02 del nombre de la camara):
    #   01 -> placa (PT): pipeline completo (modelo + OCR + resalte rojo)
    #   02 -> vehiculo (panoramica): foto cruda, nombre por hora, sin OCR
    vista = _carpeta_de(ruta.name)
    logger.debug("Imagen %s -> vista %s (%dx%d)", ruta.name, vista,
                 img.shape[1], img.shape[0])

    tiempo_actual = time.time()

    # 1) Bloqueo estricto por tiempo, POR VISTA: la foto del panoramico no
    #    tapa a la del PT (llegan ~1,4 s separadas y las dos se guardan).
    #    Se loguea a INFO: cada imagen que recibe la camara debe ver un
    #    desenlace claro en el log.
    if (tiempo_actual - _ultimo_timestamp.get(vista, 0.0)) < _cfg["cooldown"]:
        logger.info("Imagen eliminada: %s | motivo: cooldown (%.0f s desde "
                    "la ultima guardada en %s)", ruta.name,
                    tiempo_actual - _ultimo_timestamp.get(vista, 0.0), vista)
        ruta.unlink(missing_ok=True)
        return

    # 2) Filtro por diferencia visual (tambien por vista)
    porcentaje = 0.0
    ultima = _ultima_imagen.get(vista)
    if ultima is not None:
        porcentaje = calcular_diferencia(ultima, img)
        if porcentaje < _cfg["umbral"]:
            _descartar(ruta, "duplicado", f"({porcentaje:.2f}% de cambio)")
            return

    # 2b) Panoramica cruda (canal 02): el vehiculo completo para comparar
    #     precios por categoria. Sin modelo, sin OCR, sin caja y con nombre
    #     por hora; solo pasa cooldown + diferencia visual.
    if vista == "vehiculo":
        destino = _guardar(img, None, "vehiculo")
        relativo = str(destino.relative_to(_cfg["procesadas"]))
        registrar(relativo, estado="guardado")
        logger.info("Panoramica guardada: %s (%.2f%% de cambio)",
                    relativo, porcentaje)
        _ultima_imagen[vista] = img.copy()
        _ultimo_timestamp[vista] = tiempo_actual
        ruta.unlink(missing_ok=True)
        return

    # 3) Candidatos de placa. GANA EL DE MAYOR AREA (leccion E5A947: entre
    #    varias detecciones, la placa del vehiculo de fondo se perdia si
    #    mandaba el orden por confianza del modelo). Los filtros y el OCR
    #    siguen descartando candidatos malos; el orden solo decide quien
    #    se prueba primero.
    candidatos = _candidatos_placa(img)
    if not candidatos:
        _descartar(ruta, "sin_placa")
        return
    candidatos = sorted(candidatos,
                        key=lambda par: par[0][2] * par[0][3], reverse=True)

    min_texto = _cfg.get("umbral_texto_min", 5.0)
    max_texto = _cfg.get("umbral_texto_max", 55.0)
    umbral_nitidez = _cfg.get("umbral_nitidez", 120.0)
    umbral_quemada = _cfg.get("umbral_sobreexpuesto", 2.0)
    umbral_bordes = _cfg.get("umbral_bordes", 2.0)
    ocr_requerido = _cfg.get("ocr_requerido", "si") == "si"

    alto, ancho = img.shape[:2]
    # elegido: (bbox, confianza_modelo, nitidez, texto_placa, pasada,
    #           excedida, sufijo)
    elegido: tuple[tuple[int, int, int, int], float, float, str, int,
                   bool, str] | None = None
    ultimo_fallo: tuple[str, str] = ("sin_placa", "")

    for bbox, confianza in candidatos:
        x, y, w, h = bbox
        recorte = img[y:y + h, x:x + w]
        if recorte.size == 0:
            continue

        # 4) ¿Toca el borde del cuadro? Solo es DIAGNOSTICO: se usa recien
        #    mas abajo si el OCR no pudo leerla. En produccion 10 de 14
        #    imagenes llegaban con el bbox pegado al borde (la camara pone
        #    la placa muy abajo) y rechazarlas a ciegas vaciaba la cola.
        toca_borde = (x <= 4 or y <= 4 or x + w >= ancho - 4
                      or y + h >= alto - 4)
        corte_borde = (f"(toca el borde: x {x}..{x + w} de {ancho}, "
                       f"y {y}..{y + h} de {alto})")

        # 5) Texto: SOLO con el detector geometrico. Con el modelo activo,
        #    "esto es una placa" ya lo decide el modelo (conf >= umbral_modelo)
        #    y el bbox del modelo trae fondo, asi que el umbral maximo de
        #    oscuros mataba placas validas (7 de 17 en la medicion real).
        if _detector is None:
            fraccion = _fraccion_oscuros(recorte)
            if not min_texto <= fraccion <= max_texto:
                ultimo_fallo = ("sin_texto", f"({fraccion:.1f}% de oscuros)")
                continue

        # 6) Nitidez del recorte (no de la foto completa)
        nitidez = _nitidez(recorte)
        if nitidez < umbral_nitidez:
            ultimo_fallo = ("baja_nitidez", f"(nitidez {nitidez:.1f})")
            continue

        # 7) Sobreexposicion: placa quemada por el IR no sirve para entrenar
        quemada = _fraccion_sobreexpuesta(recorte)
        if quemada > umbral_quemada:
            ultimo_fallo = ("sobreexpuesta", f"({quemada:.1f}% saturada)")
            continue

        # 8) Oclusion (proxy: muy pocos bordes = algo la tapa)
        bordes = _fraccion_bordes(recorte)
        if bordes < umbral_bordes:
            ultimo_fallo = ("ocluida", f"({bordes:.1f}% de bordes)")
            continue

        # 9) OCR con motor de 6 intentos (Fase 0): variantes del recorte +
        #    votacion ponderada; consenso >= 2 = lectura solida. Sin
        #    consenso, se loguean los 6 candidatos (diagnostico en el log)
        #    y se prueba el siguiente bbox. Si no hay motor de 6 (fallback),
        #    se usa la lectura simple de siempre.
        texto_placa = ""
        if leer_placa_6 is not None:
            texto_placa, _conf_ocr, intentos = leer_placa_6(img, bbox)
            if not texto_placa:
                logger.info(
                    "Sin lectura solida: %s | bbox (%d,%d,%d,%d) | "
                    "intentos: %s", ruta.name, x, y, w, h,
                    " | ".join(f"{n}={(t or 'sin lectura')}({c:.2f})"
                               for n, t, c in intentos))
                if toca_borde:
                    ultimo_fallo = ("placa_cortada", corte_borde)
                    continue
                if ocr_requerido:
                    ultimo_fallo = ("texto_invalido",
                                    "(sin consenso en 6 intentos)")
                    continue
        elif leer_placa is not None:
            texto_placa = leer_placa(_recorte_con_margen(img, bbox))
            if not texto_placa:
                if toca_borde:
                    ultimo_fallo = ("placa_cortada", corte_borde)
                    continue
                if ocr_requerido:
                    ultimo_fallo = ("texto_invalido",
                                    "(el recorte no se lee como placa)")
                    continue
        elif toca_borde:
            # Sin OCR no hay quien confirme la lectura: el borde vuelve a
            # ser filtro duro (comportamiento anterior, sin riesgo de
            # guardar basura).
            ultimo_fallo = ("placa_cortada", corte_borde)
            continue

        # 10) Dedup (Fase 1): texto primero, pHash de respaldo. Mata los
        #     duplicados CRV-25/CRW-025 y B1Y/BIY-756 aunque el angulo
        #     cambie el pHash.
        huella = _hash_perceptual(recorte)
        if _es_misma_placa(vista, huella, texto_placa):
            detalle = f"(placa {texto_placa} ya guardada hoy en {vista})" \
                if texto_placa else f"(misma placa ya guardada hoy en {vista})"
            _descartar(ruta, "duplicado_placa", detalle)
            return

        # 11) Control de pasadas (Fase 1): una misma placa puede volver a
        #     registrar despues de ventana_reinspeccion_min minutos; con
        #     mas de max_pasadas visitas en ventana_reinspeccion_dias se
        #     guarda igual pero EXCEDIDA y con alerta WARNING.
        pasada, minutos = (1, float("inf"))
        if texto_placa:
            pasada, minutos = _pasada_de_placa(texto_placa)
            ventana = float(_cfg.get("ventana_reinspeccion_min", 60))
            if minutos < ventana:
                _descartar(ruta, "duplicado_placa",
                           f"(placa {texto_placa} registrada hace "
                           f"{minutos:.0f} min, ventana {ventana:.0f} min)")
                return

        max_pasadas = int(_cfg.get("max_pasadas", 3))
        excedida = texto_placa and pasada > max_pasadas
        elegido = (bbox, confianza, nitidez, texto_placa, pasada,
                   bool(excedida), _sufijo_pasada(pasada))
        break

    if elegido is None:
        # Resalte amarillo de las detecciones candidatas fallidas: en
        # descartadas\ el operador ve QUE se intento leer sin abrir el log.
        img_am = None
        if (_cfg.get("modo_descarte", "retener") == "retener"
                and candidatos):
            img_am = img.copy()
            for bbox_am, _conf_am in candidatos:
                img_am = _dibujar_caja(img_am, bbox_am, _COLOR_AMARILLO)
        _descartar(ruta, ultimo_fallo[0], ultimo_fallo[1], img_anotada=img_am)
        return

    bbox, confianza, nitidez, texto_placa, pasada, excedida, sufijo = elegido

    if excedida:
        logger.warning("REINSPECCION EXCEDIDA: placa %s en pasada %d "
                       "(maximo %d por %d dias) -> %s",
                       texto_placa, pasada, max_pasadas,
                       int(_cfg.get("ventana_reinspeccion_dias", 30)),
                       ruta.name)

    # 12) Aceptada: con modelo es placa confirmada; sin modelo queda pendiente
    destino = _guardar(img, bbox, vista, texto_placa, sufijo=sufijo)
    relativo = str(destino.relative_to(_cfg["procesadas"]))
    confirmada = _detector is not None
    registrar(relativo, placa=texto_placa or None, confianza=confianza,
              estado="guardado" if confirmada else "pendiente", phash=huella)
    _hashes.setdefault(vista, {})[huella] = texto_placa
    if texto_placa:
        _registrar_pasada(texto_placa, pasada, excedida)
        _placas.setdefault(vista, set()).add(texto_placa)
        # Reporte JSON del dia, siempre al dia: cada placa guardada lo refresca.
        try:
            generar_json_dia(_dia_dedup)
        except Exception:
            logger.exception("No se pudo actualizar el reporte JSON del dia")

    origen = "conf" if confirmada else "geom"
    leida = f" (placa {texto_placa}" if texto_placa else ""
    cierre_leida = f", pasada {pasada}{sufijo})" if texto_placa else ")"
    logger.info("Placa detectada%s%s: %s (nitidez %.1f, %s %.2f, "
                "%.2f%% de cambio)", leida, cierre_leida, relativo, nitidez,
                origen, confianza, porcentaje)

    _ultima_imagen[vista] = img.copy()
    _ultimo_timestamp[vista] = tiempo_actual
    ruta.unlink(missing_ok=True)


def bucle_worker() -> None:
    """Consume la cola de forma infinita; un error jamas detiene el servicio."""
    logger.info("Bucle del worker iniciado")
    while True:
        try:
            ruta = COLA.get(timeout=1.0)
        except queue.Empty:
            continue
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
