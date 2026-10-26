#!/usr/bin/env python3
"""Log aditivo de SIN LECTURA para diagnostico del pipeline.

ADITIVO Y REVERSIBLE: mientras LOG_SIN_LECTURA este apagado (defecto),
registrar() retorna tras un unico chequeo de entorno y no hace nada mas:
cero I/O, cero hilos, cero imports pesados. No toca registro.db ni
carpetas servidas por el FTP/portal.

Variables de entorno (todas opcionales):
    LOG_SIN_LECTURA            1/true/si/on = activa el log.
                               Defecto: APAGADO.
    LOG_SIN_LECTURA_DIR        Carpeta de salida. Defecto:
                               C:\\CytvAncon\\diagnostico\\sin_lectura
                               (fuera de ftp_recibidos, reportes e
                               imagenes_procesadas; nada la sirve).
    LOG_SIN_LECTURA_MAX_MB     Maximo MB por archivo JSONL. Defecto: 50.
    LOG_SIN_LECTURA_ARCHIVOS   Archivos JSONL en rotacion (actual +
                               historicos). Defecto: 3 (tope 150 MB).
    LOG_SIN_LECTURA_RETENCION_DIAS  Borra recortes/JSONL mas viejos que
                               estos dias. Defecto: 30.
    LOG_SIN_LECTURA_RECORTES   1 = guarda el recorte JPEG del candidato.
                               Defecto: 0 (apagado).
    LOG_SIN_LECTURA_RECORTES_MAX    Maximo de recortes. Defecto: 200.
    LOG_SIN_LECTURA_RECORTES_MAX_MB Maximo MB de recortes. Defecto: 200.
    LOG_SIN_LECTURA_CRNN       1 = si el pipeline no dejo lectura CRNN,
                               corre el CRNN UNA vez solo para el log
                               (no participa en la decision).
                               Defecto: 1.
    LOG_SIN_LECTURA_MOTIVOS    Motivos del worker a registrar, separados
                               por coma. Ausente = defecto
                               "texto_invalido,placa_cortada"; vacio o
                               "*" = todos los motivos.

Concurrencia: el camino de lectura arma el dict y hace put_nowait a una
cola en memoria; un hilo daemon aparte escribe en append por linea. El
servicio puede morirse o reiniciarse a mitad de linea: al leer, las
lineas que no parseen se ignoran. Fallos (disco, permisos, ruta mala)
se capturan y "cytv.sinlectura" avisa UNA sola vez y sigue.
Dedupe: antes de encolar se compara (nombre de archivo, motivo) contra
las ultimos ~4 MB del JSONL actual, asi el replay de arranque no repite
eventos ya registrados en sesiones anteriores. El campo "imagen" es la
ruta de ENTRADA: tras _descartar el archivo ya fue borrado (modo
eliminar) o movido a descartadas/fecha/motivo/ (modo retener); el
nombre de archivo es el identificador estable para ubicarlo.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("cytv.sinlectura")

DIR_DEFECTO = r"C:\CytvAncon\diagnostico\sin_lectura"
MAX_MB_DEFECTO = 50
ARCHIVOS_DEFECTO = 3
RETENCION_DIAS_DEFECTO = 30
RECORTES_MAX_DEFECTO = 200
RECORTES_MAX_MB_DEFECTO = 200

_NOMBRE = "eventos.jsonl"
MOTIVOS_DEFECTO = "texto_invalido,placa_cortada"
DEDUP_TAIL_BYTES = 4 * 1024 * 1024

_cola: "queue.Queue[tuple[dict, object]]" = queue.Queue(maxsize=200)
_hilo = None
_candado = threading.Lock()
_avisos: set[str] = set()
_podado = False
_vistos: set[tuple[str, str]] = set()
_vistos_cargado = False


def activo() -> bool:
    """True solo si LOG_SIN_LECTURA esta encendido en el entorno."""
    return os.environ.get("LOG_SIN_LECTURA", "").strip().lower() in (
        "1", "true", "si", "sí", "on", "yes")


def _env_int(nombre: str, defecto: int) -> int:
    try:
        valor = int(os.environ.get(nombre, "").strip() or defecto)
        return valor if valor > 0 else defecto
    except ValueError:
        return defecto


def _env_flag(nombre: str, defecto: int) -> int:
    """Bandera env: ausente -> defecto; 0/false/no/off -> 0; resto -> 1."""
    bruto = (os.environ.get(nombre, "") or "").strip().lower()
    if not bruto:
        return 1 if defecto else 0
    return 0 if bruto in ("0", "false", "no", "off") else 1


def _ruta_base() -> Path:
    return Path(os.environ.get("LOG_SIN_LECTURA_DIR", "").strip()
                or DIR_DEFECTO)


def _motivos_permitidos() -> "frozenset[str] | None":
    """Motivos del worker a registrar. None = sin filtro (todos)."""
    bruto = os.environ.get("LOG_SIN_LECTURA_MOTIVOS")
    if bruto is None:
        return frozenset(m.strip().lower()
                         for m in MOTIVOS_DEFECTO.split(",") if m.strip())
    bruto = bruto.strip().lower()
    if not bruto or bruto in ("*", "todos", "all"):
        return None
    return frozenset(m.strip() for m in bruto.split(",") if m.strip())


def _cargar_vistos(base: Path) -> None:
    """Precarga (nombre, motivo) del JSONL actual para no repetir
    eventos al reiniciar (replay de descartes)."""
    global _vistos_cargado
    if _vistos_cargado:
        return
    _vistos_cargado = True
    archivo = base / _NOMBRE
    try:
        if not archivo.is_file():
            return
        tam = archivo.stat().st_size
        with archivo.open("rb") as f:
            if tam > DEDUP_TAIL_BYTES:
                f.seek(tam - DEDUP_TAIL_BYTES)
                f.readline()
            datos = f.read().decode("utf-8", "replace")
    except OSError:
        _avisar_una_vez("dedup", "Log SIN LECTURA: no se pudo leer el "
                        "log previo para deduplicar")
        return
    for linea in datos.splitlines():
        try:
            ev = json.loads(linea)
        except (ValueError, TypeError):
            continue
        clave = (os.path.basename(str(ev.get("imagen", ""))),
                 str(ev.get("motivo", "")))
        if clave != ("", ""):
            _vistos.add(clave)


def _avisar_una_vez(clave: str, texto: str) -> None:
    if clave in _avisos:
        return
    _avisos.add(clave)
    logger.warning("%s (se avisa una sola vez)", texto)


def clasificar_motivo(intentos: list, filtro: str | None = None) -> str:
    """Motivo legible a partir de los intentos del pipeline.

    filtro: motivo de descarte del worker cuando ni siquiera hubo
    intento OCR (baja_nitidez, sobreexpuesta, ocluida...).
    """
    if not intentos:
        return filtro or "sin_intento_ocr"
    if any(_n == "calidad" for _n, _t, _c in intentos):
        return "bbox_bajo_minimo"
    if not any(t for _n, t, _c in intentos):
        return "todas_variantes_vacias"
    if filtro:
        return filtro
    return "sin_consenso_o_fuera_de_flota"


def _crnn_de_intentos(intentos: list) -> dict | None:
    for nombre, texto, conf in intentos:
        if nombre == "crnn_flota" and texto:
            return {"texto": texto, "conf": round(float(conf), 3),
                    "diag": False}
    return None


def _arrancar_hilo() -> None:
    global _hilo
    if _hilo is not None and _hilo.is_alive():
        return
    _hilo = threading.Thread(target=_bucle_escritor,
                             name="sinlectura-writer", daemon=True)
    _hilo.start()


def registrar(evento: dict, recorte=None) -> None:
    """Pone un evento en la cola del log. No bloquea jamas.

    Si el flag esta apagado o algo falla, no hace nada visible para el
    servicio salvo el aviso unico del logger.
    """
    if not activo():
        return
    try:
        evento = dict(evento)
        clave = (os.path.basename(str(evento.get("imagen", ""))),
                 str(evento.get("motivo", "")))
        with _candado:
            _cargar_vistos(_ruta_base())
            if clave != ("", "") and clave in _vistos:
                return
            if clave != ("", ""):
                _vistos.add(clave)
            _arrancar_hilo()
        evento.setdefault(
            "ts", datetime.now().isoformat(timespec="seconds"))
        _cola.put_nowait((evento, recorte))
    except queue.Full:
        _avisar_una_vez("cola", "Log SIN LECTURA: cola llena, evento "
                        "descartado")
    except Exception:
        _avisar_una_vez("encolar", "Log SIN LECTURA: no se pudo encolar "
                        "el evento")


def agregar_crnn_diag(img, bbox, evento: dict) -> None:
    """Completa evento['crnn'] corriendo el CRNN UNA vez, solo para log.

    Nunca participa en la decision de lectura: su resultado unicamente
    se serializa en el JSONL. Se omite si el pipeline ya dejo un CRNN o
    si LOG_SIN_LECTURA_CRNN=0.
    """
    if not activo() or evento.get("crnn"):
        return
    if _env_flag("LOG_SIN_LECTURA_CRNN", 1) == 0 or img is None or not bbox:
        return
    try:
        import ocr
        import reconocedor
        crops = []
        for margen in ocr.MARGENES_CRNN:
            r = ocr._recorte(img, tuple(bbox), margen)
            if r is not None and r.size:
                crops.append(r)
        if not crops:
            return
        medio = ocr._recorte(img, tuple(bbox), ocr.MARGENES_CRNN[1])
        if medio is not None and medio.size:
            crops += [ocr._clahe_gamma(medio), ocr._noche(medio)]
        res = reconocedor.leer_consenso(crops)
        if res and res.get("texto"):
            evento["crnn"] = {
                "texto": str(res.get("texto", "")),
                "conf": round(float(res.get("conf", 0.0)), 3),
                "diag": True,
            }
    except Exception:
        _avisar_una_vez("crnn", "Log SIN LECTURA: CRNN de diagnostico "
                        "no disponible")


def _recorte_de(img, bbox):
    if img is None or not bbox:
        return None
    try:
        x, y, w, h = (int(v) for v in bbox)
        r = img[y:y + h, x:x + w]
        if not getattr(r, "size", 0):
            return None
        return r.copy()
    except Exception:
        return None


def registrar_worker(ruta, img, candidatos: list, intentos: list,
                     bbox, filtro: str | None, detalle: str = "") -> None:
    """Evento desde el worker de produccion (foto que no llego a placa)."""
    if not activo():
        return
    try:
        motivo = clasificar_motivo(intentos, filtro)
        permitidos = _motivos_permitidos()
        if (permitidos is not None and motivo not in permitidos
                and (filtro or "").lower() not in permitidos):
            return
        evento = {
            "origen": "worker",
            "imagen": str(ruta),
            "motivo": motivo,
            "filtro": filtro or "",
            "detalle": detalle or "",
            "candidatos": [
                {"bbox": [int(v) for v in bb], "conf": round(float(c), 3)}
                for bb, c in (candidatos or [])[:5]
            ],
            "bbox_ocr": [int(v) for v in bbox] if bbox else None,
            "variantes": [
                {"n": n, "t": t, "c": round(float(c), 3)}
                for n, t, c in (intentos or [])
            ],
            "crnn": _crnn_de_intentos(intentos or []),
        }
        agregar_crnn_diag(img, bbox, evento)
        registrar(evento, _recorte_de(img, bbox))
    except Exception:
        if "arma_worker" not in _avisos:
            _avisos.add("arma_worker")
            logger.exception("Log SIN LECTURA: no se pudo armar el "
                             "evento del worker")


def registrar_validador(ruta, img, bboxes: list, intentos: list,
                        real: str = "", confs: list | None = None) -> None:
    """Evento desde la evaluacion (validador GT)."""
    if not activo():
        return
    try:
        ultimo = bboxes[-1] if bboxes else None
        detecciones = []
        for i, bb in enumerate(bboxes[:5]):
            c = confs[i] if confs and i < len(confs) else None
            detecciones.append({"bbox": [int(v) for v in bb],
                                "conf": None if c is None
                                else round(float(c), 3)})
        evento = {
            "origen": "validador",
            "imagen": str(ruta),
            "motivo": clasificar_motivo(intentos),
            "real": real,
            "candidatos": detecciones,
            "bbox_ocr": [int(v) for v in ultimo] if ultimo else None,
            "variantes": [
                {"n": n, "t": t, "c": round(float(c), 3)}
                for n, t, c in (intentos or [])
            ],
            "crnn": _crnn_de_intentos(intentos or []),
        }
        agregar_crnn_diag(img, ultimo, evento)
        registrar(evento, _recorte_de(img, ultimo))
    except Exception:
        if "arma_validador" not in _avisos:
            _avisos.add("arma_validador")
            logger.exception("Log SIN LECTURA: no se pudo armar el "
                             "evento del validador")


# ---------------------------------------------------------------- escritor

def _podar(base: Path) -> None:
    """Borra JSONL y recortes mas viejos que la retencion configurada."""
    dias = _env_int("LOG_SIN_LECTURA_RETENCION_DIAS", RETENCION_DIAS_DEFECTO)
    limite = datetime.now().timestamp() - dias * 86400.0
    try:
        for p in base.glob("eventos*.jsonl"):
            if p.stat().st_mtime < limite:
                p.unlink(missing_ok=True)
    except OSError:
        pass
    recortes = base / "recortes"
    if recortes.is_dir():
        try:
            for p in recortes.glob("*.jpg"):
                if p.stat().st_mtime < limite:
                    p.unlink(missing_ok=True)
        except OSError:
            pass


def _podar_recortes_por_tamano(recortes: Path) -> None:
    max_n = _env_int("LOG_SIN_LECTURA_RECORTES_MAX", RECORTES_MAX_DEFECTO)
    max_mb = _env_int("LOG_SIN_LECTURA_RECORTES_MAX_MB",
                      RECORTES_MAX_MB_DEFECTO)
    try:
        archivos = sorted(recortes.glob("*.jpg"),
                          key=lambda p: p.stat().st_mtime)
    except OSError:
        return
    total_mb = 0.0
    for p in archivos:
        try:
            total_mb += p.stat().st_size / (1024.0 * 1024.0)
        except OSError:
            continue
    excedentes = len(archivos) - max_n
    for p in archivos:
        if excedentes > 0 or total_mb > max_mb:
            try:
                total_mb -= p.stat().st_size / (1024.0 * 1024.0)
                p.unlink(missing_ok=True)
                excedentes -= 1
            except OSError:
                pass


def _rotar_si_toca(base: Path) -> None:
    max_mb = _env_int("LOG_SIN_LECTURA_MAX_MB", MAX_MB_DEFECTO)
    n_arch = _env_int("LOG_SIN_LECTURA_ARCHIVOS", ARCHIVOS_DEFECTO)
    actual = base / _NOMBRE
    try:
        if not actual.is_file() or actual.stat().st_size < max_mb * 1024 * 1024:
            return
        (base / f"{_NOMBRE}.{n_arch - 1}").unlink(missing_ok=True)
        for i in range(n_arch - 2, 0, -1):
            origen = base / f"{_NOMBRE}.{i}"
            if origen.is_file():
                origen.replace(base / f"{_NOMBRE}.{i + 1}")
        actual.replace(base / f"{_NOMBRE}.1")
    except OSError:
        _avisar_una_vez("rotar", "Log SIN LECTURA: no se pudo rotar el "
                        "archivo JSONL")


def _guardar_recorte(base: Path, recorte) -> None:
    if _env_flag("LOG_SIN_LECTURA_RECORTES", 0) == 0:
        return
    try:
        import cv2
        recortes = base / "recortes"
        recortes.mkdir(parents=True, exist_ok=True)
        marca = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        (recortes / f"{marca}.jpg")  # solo para fallo temprano de ruta
        ok = cv2.imwrite(str(recortes / f"{marca}.jpg"), recorte)
        if not ok:
            raise OSError("cv2.imwrite devolvio False")
        _podar_recortes_por_tamano(recortes)
    except Exception:
        _avisar_una_vez("recorte", "Log SIN LECTURA: no se pudo guardar "
                        "el recorte")


def _escribir(evento: dict, recorte) -> None:
    base = _ruta_base()
    global _podado
    if not _podado:
        _podado = True
        _podar(base)
    base.mkdir(parents=True, exist_ok=True)
    if recorte is not None:
        _guardar_recorte(base, recorte)
    linea = json.dumps(evento, ensure_ascii=False, default=str) + "\n"
    with open(base / _NOMBRE, "a", encoding="utf-8", errors="replace") as f:
        f.write(linea)
        f.flush()
    _rotar_si_toca(base)


def _bucle_escritor() -> None:
    while True:
        try:
            evento, recorte = _cola.get()
            _escribir(evento, recorte)
        except Exception:
            _avisar_una_vez("escritura", "Log SIN LECTURA: fallo la "
                            "escritura (disco/permisos/ruta); el "
                            "servicio sigue")
