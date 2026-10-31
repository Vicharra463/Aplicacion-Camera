#!/usr/bin/env python3
"""Servidor FTP de la camara Hikvision - CYTV Ancon.

Diseno:
    hilo FTP   -> solo recibe el archivo y lo encola (nunca hace trabajo pesado)
    worker     -> anti-duplicados + guardado + registro  (ver worker.py)
    heartbeat  -> toca heartbeat.txt cada 15 s para el watchdog externo
    barrido    -> recupera archivos que quedaron a medias si el servicio se cayo

Para correr 24/7 en Windows ver scripts\\instalar_servicio.bat (NSSM).
"""
from __future__ import annotations

import configparser
from importlib import import_module
import inspect
import logging
import os
import signal
import socket
import sys
import threading
import time
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

try:
    DummyAuthorizer = import_module("pyftpdlib.authorizers").DummyAuthorizer
    FTPHandler = import_module("pyftpdlib.handlers").FTPHandler
    FTPServer = import_module("pyftpdlib.servers").FTPServer
except ModuleNotFoundError:
    DummyAuthorizer = FTPHandler = FTPServer = None



sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))

import worker

BASE = Path(os.environ.get("CYTV_BASE") or Path(__file__).resolve().parent)
logger = logging.getLogger("cytv.ftp")


def cargar_config() -> dict:
    ruta = BASE / "config.ini"
    if not ruta.exists():
        print(f"[FATAL] No se encontro el archivo de configuracion: {ruta}", file=sys.stderr)
        raise SystemExit(1)

    
    ini = configparser.ConfigParser()
    ini.read(ruta, encoding="utf-8-sig")

    
    
    
    relativas: list[str] = []

    def ruta_rel(seccion: str, clave: str) -> Path:
        p = Path(ini.get(seccion, clave))
        if p.is_absolute():
            return p
        relativas.append(f"[{seccion}] {clave} = {p}")
        return (BASE / p)

    p_ini, p_fin = (int(x) for x in ini.get("ftp", "puertos_pasivos").split("-"))

    def opt(seccion: str, clave: str, defecto: str) -> Path:
        """Ruta opcional: si la seccion/clave no existe, usa el defecto."""
        p = Path(ini.get(seccion, clave, fallback=defecto))
        if p.is_absolute():
            return p
        relativas.append(f"[{seccion}] {clave} = {p}")
        return (BASE / p)

    return {
        "puerto": ini.getint("ftp", "puerto"),
        "usuario": ini.get("ftp", "usuario"),
        "password": ini.get("ftp", "password"),
        "pasivos": (p_ini, p_fin),
        "timeout": ini.getint("ftp", "timeout_segundos"),
        "max_conexiones": ini.getint("ftp", "max_conexiones"),
        "recibidos": ruta_rel("rutas", "recibidos"),
        "procesadas": ruta_rel("rutas", "procesadas"),
        "registro": ruta_rel("rutas", "registro"),
        "logs": ruta_rel("rutas", "logs"),
        "heartbeat": ruta_rel("rutas", "heartbeat"),
        "reportes": opt("rutas", "reportes", "reportes"),
        "aoi_tokens": ini.get("aoi", "tokens", fallback=""),
        "umbral": ini.getfloat("duplicados", "umbral_diferencia"),
        "replay_inicio": ini.get("inicio", "replay_descartes", fallback="si"),
        "replay_pausa_seg": ini.getfloat("inicio", "replay_pausa_seg",
                                         fallback=1.0),
        
        
        "modo_descarte": ini.get("filtros", "modo_descarte", fallback="retener"),
        "umbral_modelo": ini.getfloat("filtros", "umbral_modelo", fallback=0.25),
        "ocr_requerido": ini.get("filtros", "ocr_requerido", fallback="si"),
        "umbral_texto_min": ini.getfloat("filtros", "umbral_texto_min", fallback=5.0),
        "umbral_texto_max": ini.getfloat("filtros", "umbral_texto_max", fallback=55.0),
        "umbral_nitidez": ini.getfloat("filtros", "umbral_nitidez", fallback=120.0),
        "umbral_sobreexpuesto": ini.getfloat("filtros", "umbral_sobreexpuesto",
                                             fallback=2.0),
        "umbral_bordes": ini.getfloat("filtros", "umbral_bordes", fallback=2.0),
        "umbral_dedup": ini.getint("filtros", "umbral_dedup_dist", fallback=12),
        "descartes": opt("filtros", "descartes", "descartadas"),
        
        
        "ventana_reinspeccion_min": ini.getfloat("camara", "ventana_reinspeccion_min",
                                                 fallback=60.0),
        "ventana_reinspeccion_dias": ini.getint("camara", "ventana_reinspeccion_dias",
                                                fallback=30),
        "max_pasadas": ini.getint("camara", "max_pasadas", fallback=3),
        "cooldown_placa_seg": ini.getfloat("camara", "cooldown_placa_seg",
                                           fallback=5.0),
        "reposo_minutos": ini.getfloat("recursos", "reposo_minutos",
                                       fallback=10.0),
        "nivel_log": ini.get("log", "nivel"),
        "retencion": ini.getint("log", "retencion_dias"),
        "_relativas": relativas,
    }


def preparar_log(cfg: dict) -> None:
    cfg["logs"].mkdir(parents=True, exist_ok=True)
    nivel = getattr(logging, str(cfg["nivel_log"]).upper(), logging.INFO)
    formato = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    archivo = TimedRotatingFileHandler(
        cfg["logs"] / "servidor.log",
        when="midnight",
        backupCount=cfg["retencion"],
        encoding="utf-8",
    )
    archivo.setFormatter(formato)

    consola = logging.StreamHandler()
    consola.setFormatter(formato)

    raiz = logging.getLogger()
    raiz.setLevel(nivel)
    raiz.addHandler(archivo)
    raiz.addHandler(consola)


class HandlerCytv(FTPHandler or object):
    """Recibe el archivo y lo pasa a la cola: cero trabajo pesado en este hilo."""

    def on_file_received(self, file_path: str) -> None:
        ruta = Path(file_path)
        if not ruta.exists():
            return
        if worker.encolar(str(ruta)):
            logger.info("Imagen recibida: %s (encolada)", ruta.name)


def bucle_heartbeat(ruta: Path) -> None:
    """Senal de vida externa: si supera los 60 s sin actualizarse, algo se colgo."""
    while True:
        try:
            ruta.write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
        except OSError as exc:
            logger.debug("No se pudo actualizar el heartbeat: %s", exc)
        time.sleep(15)


def bucle_barrido(recibidos: Path) -> None:
    """Reintenta archivos que quedaron en la cola de entrada sin procesar."""
    while True:
        time.sleep(60)
        try:
            for p in recibidos.rglob("*"):
                if p.is_file() and (time.time() - p.stat().st_mtime) > 30:
                    if worker.encolar(str(p)):
                        logger.info("Barrido: archivo recuperado %s", p.name)
        except OSError as exc:
            logger.debug("Barrido interrumpido: %s", exc)


def _ip_local() -> str:
    """IP por la que esta PC es alcanzable en la red local.

    Truco del socket UDP: no manda paquetes de verdad, solo deja que el
    sistema elija la ruta por la que saldria el trafico. Si no hay red,
    devuelve 127.0.0.1.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        finally:
            sock.close()
    except OSError:
        return "127.0.0.1"


def escribir_info_camara(cfg: dict) -> Path:
    """Escribe INFO_CAMARA.txt: todo lo necesario para configurar la camara.

    Se reescribe en CADA arranque porque la IP puede cambiar si el router no
    tiene reserva DHCP hecha. Es el archivo "de frente" que el operador abre
    para saber donde corre el servidor y que poner en la camara.
    """
    ip = _ip_local()
    ruta = BASE / "INFO_CAMARA.txt"
    inicio, fin = cfg["pasivos"]

    
    
    texto_vistas = (
        "  IMPORTANTE - las DOS vistas tienen que subir:\n"
        "    Canal 01 (PT) y Canal 02 (panoramico): en\n"
        "    Storage -> Schedule Settings -> Picture Capture y en el linkage de\n"
        "    movimiento (MD_WITH_TARGET), active la subida por FTP en AMBOS\n"
        "    canales. Ambos van a la carpeta placa\\."
    )
    texto_garantia = (
        "  imagenes_procesadas\\placa\\AAAA-MM-DD\\      ambos canales: modelo\n"
        "      + OCR (y AOI si hay token), nombre = placa canonica\n"
        "      (ABC-123.jpg) con rectangulo ROJO sobre la deteccion.\n"
        "      El crop enviado a la AOI queda en reportes\\aoi\\AAAA-MM-DD\\."
    )

    contenido = f"""\
================================================================
 INFO DE CONEXION - Servidor FTP Camara CYTV Ancon
 Generado: {datetime.now().isoformat(timespec='seconds')}
================================================================

--- DONDE ESTA CORRIENDO ESTE SERVIDOR -------------------------
  PC / IP ............ {ip}
  Puerto FTP ......... {cfg['puerto']}
  Usuario ............ {cfg['usuario']}
  Password ........... {cfg['password']}
  Modo pasivo ........ si, puertos TCP {inicio}-{fin}
  Carpeta de entrada . {cfg['recibidos']}
  Carpeta de salidas . {cfg['procesadas']}
  Registro (SQLite) .. {cfg['registro']}
  Log ................ {cfg['logs'] / 'servidor.log'}
  Heartbeat .......... {cfg['heartbeat']}

--- COMO PONERLO EN LA CAMARA (Hikvision) ----------------------
  En la web de la camara (http://192.168.18.42):
    Servidor (Server) ... {ip}
    Puerto (Port) ....... {cfg['puerto']}
    Usuario ............. {cfg['usuario']}
    Password ............ {cfg['password']}
    Protocolo ........... FTP
    Ruta destino ........ /   (o el subcarpeta que prefiera)
    Modo ............... Pasivo  (si la camara lo pide)

{texto_vistas}

--- QUE GUARDA EL SERVIDOR ------------------------------------
{texto_garantia}

--- CAMBIOS DE CONFIGURACION ----------------------------------
  scripts\\instalar_servicio.bat config              ver config.ini
  scripts\\instalar_servicio.bat config set CLAVE V  cambiar un valor
      (usuario password umbral_modelo ...)

  NOTA: si cambia la password aqui, cambielo TAMBIEN en la camara.
  NOTA: para que la IP no cambie, pida al router una reserva DHCP
        para esta PC ({ip}).
================================================================
"""
    ruta.write_text(contenido, encoding="utf-8")
    logger.info("Info de conexion para la camara: %s (IP %s, puerto %s, "
                "usuario %s)", ruta, ip, cfg["puerto"], cfg["usuario"])
    return ruta


def asegurar_carpetas(cfg: dict) -> None:
    """Verifica con rutas ABSOLUTAS y crea lo que falte, avisando la ruta.

    Regla del sistema: si no existe -> se crea; si existe -> se verifica;
    siempre se dice en que ruta absoluta quedo. Asi un arranque desde el
    escritorio, bajo NSSM (arranca en System32) o tras una limpieza manual
    nunca deja al servicio sin donde escribir.
    """
    for rel in cfg.get("_relativas", []):
        logger.warning("Ruta RELATIVA en config.ini (%s) - use rutas "
                       "absolutas (resuelta igual esta vez)", rel)

    carpetas: list[tuple[str, Path]] = [
        ("recibidos", cfg["recibidos"]),
        ("procesadas", cfg["procesadas"]),
        ("logs", cfg["logs"]),
        ("heartbeat", Path(cfg["heartbeat"]).parent),
    ]
    for vista in sorted(set(worker.FOLDER_POR_CANAL.values())):
        carpetas.append((f"vista {vista}", cfg["procesadas"] / vista))
    if cfg.get("modo_descarte", "retener") == "retener":
        carpetas.append(("descartes", Path(cfg["descartes"])))

    for nombre, ruta in carpetas:
        if ruta.is_dir():
            logger.info("Carpeta verificada (%s): %s", nombre, ruta)
        else:
            ruta.mkdir(parents=True, exist_ok=True)
            logger.info("Carpeta CREADA (%s): %s", nombre, ruta)

    registro = Path(cfg["registro"])
    if registro.is_file():
        logger.info("Registro verificado: %s", registro)
    else:
        logger.info("Registro se creara en el primer guardado: %s", registro)

    onnx = getattr(worker, "RUTA_ONNX", None)
    if onnx is not None:
        onnx = Path(onnx)
        if not onnx.parent.is_dir():
            onnx.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Carpeta CREADA (modelo): %s", onnx.parent)
        if onnx.is_file():
            logger.info("Modelo verificado: %s (%d bytes)", onnx,
                        onnx.stat().st_size)
        else:
            logger.warning("MODELO NO ENCONTRADO: %s - hasta que coloque "
                           "placas.onnx se usara el detector geometrico", onnx)


def main() -> int:
    if DummyAuthorizer is None or FTPHandler is None or FTPServer is None:
        print("[FATAL] Falta la dependencia 'pyftpdlib'. Instalela con: "
              "python -m pip install pyftpdlib", file=sys.stderr)
        return 1

    cfg = cargar_config()
    preparar_log(cfg)

    
    
    asegurar_carpetas(cfg)

    worker.iniciar(cfg)
    threading.Thread(target=worker.bucle_worker, name="worker", daemon=True).start()
    threading.Thread(target=bucle_heartbeat, args=(cfg["heartbeat"],),
                     name="heartbeat", daemon=True).start()
    threading.Thread(target=bucle_barrido, args=(cfg["recibidos"],),
                     name="barrido", daemon=True).start()

    authorizer = DummyAuthorizer()
    authorizer.add_user(cfg["usuario"], cfg["password"], str(cfg["recibidos"]),
                        perm="elradfmwM")

    HandlerCytv.authorizer = authorizer
    HandlerCytv.passive_ports = range(cfg["pasivos"][0], cfg["pasivos"][1] + 1)
    HandlerCytv.timeout = cfg["timeout"]

    servidor = FTPServer(("0.0.0.0", cfg["puerto"]), HandlerCytv)
    servidor.max_cons = cfg["max_conexiones"]
    servidor.max_cons_per_ip = 5

    
    
    def cerrar(signum, _frame):
        logger.info("Senal %s: cerrando servidor...", signum)
        try:
            servidor.close_all()
        except Exception:
            logger.exception("Error al cerrar el servidor")

    signal.signal(signal.SIGINT, cerrar)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, cerrar)

    logger.info("=" * 56)
    logger.info("Servidor FTP CYTV Ancon - IP %s, puerto %s",
                _ip_local(), cfg["puerto"])
    logger.info("Entrada : %s", cfg["recibidos"])
    logger.info("Salida : %s  (placa\\ = ambos canales)", cfg["procesadas"])
    logger.info("Registro : %s", cfg["registro"])
    try:
        escribir_info_camara(cfg)
    except OSError:
        logger.exception("No se pudo escribir INFO_CAMARA.txt")
    logger.info("=" * 56)

    try:
        
        
        if "timeout" in inspect.signature(FTPServer.serve_forever).parameters:
            servidor.serve_forever(timeout=1, handle_exit=False)
        else:
            servidor.serve_forever(poll_timeout=1, handle_exit=False)
    except Exception:
        logger.exception("El servidor FTP termino con error")
        return 1

    logger.info("Servidor FTP detenido correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
