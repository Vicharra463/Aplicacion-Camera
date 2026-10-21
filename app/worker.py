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
    from modelo import DetectorPlacas, RUTA_ONNX
except Exception:
    DetectorPlacas = None
    RUTA_ONNX = None

try:
    from ocr import leer_placa, leer_placa_6, evidencia_y_color
except Exception:
    leer_placa = None
    leer_placa_6 = None
    evidencia_y_color = None

logger = logging.getLogger("cytv.worker")

COLA: "queue.Queue[str]" = queue.Queue(maxsize=200)

_MOTIVOS_VOLUMEN = {"sin_texto"}

FOLDER_POR_CANAL = {"01": "placa", "02": "vehiculo"}
CARPETA_POR_DEFECTO = "placa"
_RE_CANAL = re.compile(r"_(\d{2})_")

FALLOS_SIN_REEVAL = {"baja_nitidez", "sobreexpuesta", "ocluida"}
LETRAS_REVISAR = "WQH"

_ultima_imagen: dict[str, np.ndarray] = {}

_hashes: dict[str, dict[int, str]] = {}
_placas: dict[str, set[str]] = {}

_dia_dedup: str = ""

_cfg: dict = {}
_ruta_registro: Path | None = None

_detector = None


def _carpeta_de(nombre_archivo: str) -> str:
    encontrado = _RE_CANAL.search(nombre_archivo)
    canal = encontrado.group(1) if encontrado else ""
    return FOLDER_POR_CANAL.get(canal, CARPETA_POR_DEFECTO)


def _carpeta_de_ruta(relativo: str) -> str:
    parte = Path(relativo).parts[0] if relativo else ""
    return parte if parte in FOLDER_POR_CANAL.values() else CARPETA_POR_DEFECTO


def iniciar(cfg: dict) -> None:
    global _cfg, _ruta_registro, _hashes, _placas, _dia_dedup, _detector
    _cfg = cfg
    _ruta_registro = cfg["registro"]
    cfg["procesadas"].mkdir(parents=True, exist_ok=True)
    if str(cfg.get("modo_descarte", "eliminar")).strip().lower() == "retener":
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
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS candidatas (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
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


def registrar(archivo: str, placa: str | None = None,
              confianza: float | None = None, estado: str = "guardado",
              phash: int | None = None, evidencia: int | None = None,
              semaforo: str | None = None) -> None:
    if _ruta_registro is None:
        return
    con = sqlite3.connect(_ruta_registro)
    try:
        con.execute(
            "INSERT INTO registros (fecha_hora, archivo, placa, confianza, "
            "estado, phash, evidencia, semaforo) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), archivo, placa,
             confianza, estado, None if phash is None else f"{phash:016x}",
             evidencia, semaforo),
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
    con = sqlite3.connect(_ruta_registro)
    try:
        filas = con.execute(
            "SELECT fecha_hora, archivo, placa, confianza, evidencia, "
            "semaforo FROM registros "
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
    for fecha_hora, archivo, placa, confianza, evidencia, semaforo in filas:
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
                "imagenes": {"placa": [], "panoramica": []},
            }
        entrada["ultima_vez"] = fecha_hora
        entrada["detecciones"] += 1
        if confianza is not None:
            entrada["confianza_max"] = max(entrada["confianza_max"], confianza)
        if evidencia is not None and evidencia > entrada["evidencia_max"]:
            entrada["evidencia_max"] = evidencia
            if semaforo:
                entrada["semaforo"] = semaforo
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
    temporal = carpeta / f"placas-{fecha}.json.tmp"
    temporal.write_text(json.dumps(datos, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    temporal.replace(destino)
    return destino


def calcular_diferencia(img1: np.ndarray, img2: np.ndarray) -> float:
    img1_peq = cv2.resize(img1, (640, 480))
    img2_peq = cv2.resize(img2, (640, 480))
    gris1 = cv2.cvtColor(img1_peq, cv2.COLOR_BGR2GRAY)
    gris2 = cv2.cvtColor(img2_peq, cv2.COLOR_BGR2GRAY)
    diferencia = cv2.absdiff(gris1, gris2)
    _, umbral = cv2.threshold(diferencia, 35, 255, cv2.THRESH_BINARY)
    return (np.count_nonzero(umbral) / (umbral.shape[0] * umbral.shape[1])) * 100


def encolar(ruta: str) -> bool:
    try:
        COLA.put_nowait(ruta)
        return True
    except queue.Full:
        logger.warning("Cola llena: %s queda para el barrido de huerfanos", Path(ruta).name)
        return False


_MOTIVOS_SIN_REPLAY = {"duplicado", "duplicado_placa"}
_EXT_FOTO = (".jpg", ".jpeg", ".png")


def _limpiar_carpetas_vacias(raiz: Path) -> int:
    borradas = 0
    for fecha in sorted((d for d in raiz.iterdir() if d.is_dir()),
                        reverse=True):
        for motivo in sorted((m for m in fecha.iterdir() if m.is_dir()),
                             reverse=True):
            try:
                next(motivo.iterdir())
            except StopIteration:
                motivo.rmdir()
                borradas += 1
            except OSError:
                continue
        try:
            next(fecha.iterdir())
        except StopIteration:
            fecha.rmdir()
            borradas += 1
        except OSError:
            continue
    if borradas:
        logger.info("Replay: %d carpeta(s) vacia(s) eliminadas de "
                    "descartadas", borradas)
    return borradas


def _fotos_descartadas_para_replay(raiz: Path) -> list[Path]:
    pendientes: list[Path] = []
    for dia in sorted(d for d in raiz.iterdir() if d.is_dir()):
        for hijo in sorted(dia.iterdir()):
            if hijo.is_dir():
                if hijo.name.lower() in _MOTIVOS_SIN_REPLAY:
                    continue
                pendientes.extend(f for f in sorted(hijo.iterdir())
                                  if f.is_file()
                                  and f.suffix.lower() in _EXT_FOTO)
            elif hijo.is_file() and hijo.suffix.lower() in _EXT_FOTO:
                pendientes.append(hijo)
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


def _pasada_de_placa(placa: str) -> tuple[int, float]:
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
    return {1: "", 2: "_2da-pasada", 3: "_3ra-pasada"}.get(pasada, "_EXCEDIDA")


def _recorte_con_margen(img: np.ndarray, bbox: tuple[int, int, int, int],
                        margen: float = 0.35) -> np.ndarray:
    x, y, w, h = bbox
    dx, dy = int(w * margen), int(h * margen)
    alto, ancho = img.shape[:2]
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(ancho, x + w + dx), min(alto, y + h + dy)
    return img[y0:y1, x0:x1]


def _cerca_texto(a: str, b: str) -> bool:
    if a == b:
        return True
    if not a or not b:
        return False
    if abs(len(a) - len(b)) > 2:
        return False
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


_EQUIV_COMPARAR = str.maketrans({"I": "1", "O": "0", "Z": "7"})


def _forma_comparable(texto: str) -> str:
    return _sin_guion(texto).upper().translate(_EQUIV_COMPARAR)


def _buscar_misma(vista: str, huella: int, texto: str = ""):
    guardados = _hashes.get(vista, {})
    if texto:
        forma = _forma_comparable(texto)
        for h, t in guardados.items():
            if t and _cerca_texto(forma, _forma_comparable(t)):
                return h, t
    umbral = _cfg.get("umbral_dedup", 18)
    for h, t in guardados.items():
        if bin(huella ^ h).count("1") <= umbral:
            return h, t
    return None


def _es_misma_placa(vista: str, huella: int, texto: str = "") -> bool:
    return _buscar_misma(vista, huella, texto) is not None


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


_RE_NOMBRE_PLACA = re.compile(r"[^A-Za-z0-9-]+")


def _nombre_con_placa(vista: str, texto_placa: str, ahora: datetime,
                      sufijo: str = "") -> str:
    base = _RE_NOMBRE_PLACA.sub("-", texto_placa or "").strip("-")
    base = re.sub(r"-{2,}", "-", base).upper()
    if not base:
        return f"{vista}_{ahora.strftime('%H-%M-%S')}_{ahora.microsecond:03d}"
    if vista == "placa":
        return base + sufijo
    return f"placa-panoramica_{base}" + sufijo


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
             dudosa: bool = False) -> Path:
    ahora = datetime.now()
    dia = _cfg["procesadas"] / vista / ahora.strftime("%Y-%m-%d")
    carpeta = dia if _detector is not None else dia / "pendientes"
    carpeta.mkdir(parents=True, exist_ok=True)

    nombre = _nombre_con_placa(vista, texto_placa, ahora, sufijo)
    if dudosa:
        nombre = f"{nombre} (DUDOSA)"
    destino = _destino_libre(carpeta, f"{nombre}.jpg")
    salida = img
    if vista == "placa" and bbox is not None:
        if dudosa:
            fino = max(1, round(min(img.shape[:2]) / 600))
            salida = _dibujar_caja(img, bbox, _COLOR_VERDE, padding=0.18,
                                   grosor=fino)
        else:
            salida = _dibujar_caja(img, bbox, _COLOR_VERDE)
    cv2.imwrite(str(destino), salida, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    if bbox is not None:
        x, y, w, h = bbox
        alto, ancho = img.shape[:2]
        etiqueta = (f"0 {(x + w / 2) / ancho:.6f} {(y + h / 2) / alto:.6f} "
                    f"{w / ancho:.6f} {h / alto:.6f}\n")
        (destino.with_suffix(".txt")).write_text(etiqueta, encoding="utf-8")
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
                         huella: int) -> bool:
    if not t_viejo:
        return False
    desde = f"{_dia_dedup}T00:00:00"
    con = sqlite3.connect(_ruta_registro)
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
    if not (area_nueva >= 1.2 * area_vieja
            and evidencia >= (ev_vieja or 0)):
        return False
    sufijo = next((s for s in ("_2da-pasada", "_3ra-pasada", "_EXCEDIDA")
                   if viejo.stem.endswith(s)), "")
    viejo.unlink(missing_ok=True)
    viejo.with_suffix(".txt").unlink(missing_ok=True)
    nuevo = _guardar(img, bbox, vista, texto, sufijo=sufijo)
    relativo = str(nuevo.relative_to(_cfg["procesadas"]))
    con = sqlite3.connect(_ruta_registro)
    try:
        con.execute("UPDATE registros SET archivo = ?, placa = ?, "
                    "confianza = ?, evidencia = ?, semaforo = ?, phash = ? "
                    "WHERE id = ?",
                    (relativo, texto, confianza, evidencia, color,
                     f"{huella:016x}", id_registro))
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
    try:
        generar_json_dia(_dia_dedup)
    except Exception:
        logger.exception("No se pudo actualizar el reporte JSON del dia")
    logger.info("MEJOR FOTO: %s -> %s (area %.4f vs %.4f): %s",
                t_viejo, texto, area_nueva, area_vieja, relativo)
    return True


def procesar_imagen(ruta: Path) -> None:
    global _ultima_imagen

    if not ruta.exists():
        return

    _actualizar_dia()

    img = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("Imagen ilegible, se elimina: %s", ruta.name)
        ruta.unlink(missing_ok=True)
        return

    vista = _carpeta_de(ruta.name)
    logger.debug("Imagen %s -> vista %s (%dx%d)", ruta.name, vista,
                 img.shape[1], img.shape[0])

    porcentaje = 0.0
    ultima = _ultima_imagen.get(vista)
    if ultima is not None:
        porcentaje = calcular_diferencia(ultima, img)
        if porcentaje < _cfg["umbral"]:
            _descartar(ruta, "duplicado", f"({porcentaje:.2f}% de cambio)")
            return

    if vista == "vehiculo":
        destino = _guardar(img, None, "vehiculo")
        relativo = str(destino.relative_to(_cfg["procesadas"]))
        registrar(relativo, estado="guardado")
        logger.info("Panoramica guardada: %s (%.2f%% de cambio)",
                    relativo, porcentaje)
        _ultima_imagen[vista] = img.copy()
        ruta.unlink(missing_ok=True)
        return

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
    elegido = None
    ultimo_fallo: tuple[str, str] = ("sin_placa", "")
    conf_ocr, intentos_ocr = 0.0, []
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

        if _detector is None:
            fraccion = _fraccion_oscuros(recorte)
            if not min_texto <= fraccion <= max_texto:
                ultimo_fallo = ("sin_texto", f"({fraccion:.1f}% de oscuros)")
                continue

        nitidez = _nitidez(recorte)
        if nitidez < umbral_nitidez:
            ultimo_fallo = ("baja_nitidez", f"(nitidez {nitidez:.1f})")
            continue

        quemada = _fraccion_sobreexpuesta(recorte)
        if quemada > umbral_quemada:
            ultimo_fallo = ("sobreexpuesta", f"({quemada:.1f}% saturada)")
            continue

        bordes = _fraccion_bordes(recorte)
        if bordes < umbral_bordes:
            ultimo_fallo = ("ocluida", f"({bordes:.1f}% de bordes)")
            continue

        texto_placa = ""
        if leer_placa_6 is not None:
            texto_placa, conf_ocr, intentos_ocr = leer_placa_6(img, bbox)
            bbox_ocr = bbox
            if not texto_placa:
                logger.info(
                    "Sin lectura solida: %s | bbox (%d,%d,%d,%d) | "
                    "intentos: %s", ruta.name, x, y, w, h,
                    " | ".join(f"{n}={(t or 'sin lectura')}({c:.2f})"
                               for n, t, c in intentos_ocr))
                if toca_borde:
                    ultimo_fallo = ("placa_cortada", corte_borde)
                    continue
                if ocr_requerido:
                    ultimo_fallo = ("texto_invalido",
                                    "(sin consenso en los intentos)")
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
            ultimo_fallo = ("placa_cortada", corte_borde)
            continue

        huella = _hash_perceptual(recorte)
        previa = _buscar_misma(vista, huella, texto_placa)
        if previa is not None:
            if texto_placa and evidencia_y_color is not None:
                ev_nueva, color_nuevo = evidencia_y_color(
                    texto_placa, conf_ocr, intentos_ocr)
                if _reemplazar_si_mejor(vista, previa[0], previa[1], img,
                                        bbox, texto_placa, ev_nueva,
                                        color_nuevo, confianza, huella):
                    ruta.unlink(missing_ok=True)
                    return
            detalle = f"(placa {texto_placa} ya guardada hoy en {vista})" \
                if texto_placa else f"(misma placa ya guardada hoy en {vista})"
            _descartar(ruta, "duplicado_placa", detalle)
            return

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

    dudosa = False
    elegibles = [cbx for cbx, _cf in candidatos if cbx[2] > 0 and cbx[3] > 0]
    if (elegido is None and elegibles
            and ultimo_fallo[0] not in FALLOS_SIN_REEVAL
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
            if leer_placa_6 is not None:
                texto_re, conf_re, intentos_re = leer_placa_6(
                    img, bbox_re, reintento=False)
            else:
                texto_re = leer_placa(_recorte_con_margen(img, bbox_re))
                conf_re, intentos_re = 0.0, []
            intentos_ocr.extend(intentos_re)
            if not texto_re:
                continue
            rec_re = img[bbox_re[1]:bbox_re[1] + bbox_re[3],
                         bbox_re[0]:bbox_re[0] + bbox_re[2]]
            huella_re = _hash_perceptual(rec_re)
            if _es_misma_placa(vista, huella_re, texto_re):
                _descartar(ruta, "duplicado_placa",
                           f"(reeval: placa {texto_re} ya guardada hoy "
                           f"en {vista})")
                return
            pasada_re, minutos_re = _pasada_de_placa(texto_re)
            ventana = float(_cfg.get("ventana_reinspeccion_min", 60))
            if minutos_re < ventana:
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
        _descartar(ruta, ultimo_fallo[0], ultimo_fallo[1])
        return

    bbox, confianza, nitidez, texto_placa, pasada, excedida, sufijo = elegido

    evidencia, color = (evidencia_y_color(texto_placa, conf_ocr, intentos_ocr)
                        if evidencia_y_color is not None and texto_placa
                        else (0, "rojo"))

    if excedida:
        logger.warning("REINSPECCION EXCEDIDA: placa %s en pasada %d "
                       "(maximo %d por %d dias) -> %s",
                       texto_placa, pasada, max_pasadas,
                       int(_cfg.get("ventana_reinspeccion_dias", 30)),
                       ruta.name)

    destino = _guardar(img, bbox, vista, texto_placa, sufijo=sufijo,
                       dudosa=dudosa)
    relativo = str(destino.relative_to(_cfg["procesadas"]))
    confirmada = _detector is not None
    registrar(relativo, placa=texto_placa or None, confianza=confianza,
              estado="guardado" if confirmada else "pendiente", phash=huella,
              evidencia=evidencia, semaforo=color)

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
    ruta.unlink(missing_ok=True)


def bucle_worker() -> None:
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