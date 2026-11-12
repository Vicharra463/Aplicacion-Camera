#!/usr/bin/env python3
"""Consola interactiva del servidor de placas CYTV Ancon.

Estilo artisan: lista de comandos navegable con las flechas, Enter para
ejecutar, o numero directo. Salida con :wq (estilo vim) + Enter.

Uso:
    consola.py              -> menu interactivo
    consola.py estado       -> ejecuta un comando y sale (para scripts)
"""
from __future__ import annotations

import configparser
import ftplib
import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RUTA = Path(__file__).resolve().parent
RUTA_CONFIG = RUTA / "config.ini"
SERVICIO = "CamaraCytvAncon"
_INTERACTIVO = True          # False al correr con argumento (consola.py estado)
SIN_VENTANA = 0x08000000          # CREATE_NO_WINDOW: no abrir consolas fantasma
DESACOPLADO = 0x00000008          # DETACHED_PROCESS para node

# --------------------------------------------------------------------------
# Banner de bloques (5 filas, glifos de 6 columnas)
# --------------------------------------------------------------------------
_LETRAS = {
    "C": [" ████ ", "█     ", "█     ", "█     ", " ████ "],
    "Y": ["█   █ ", "█   █ ", " █ █  ", "  █   ", "  █   "],
    "T": ["█████ ", "  █   ", "  █   ", "  █   ", "  █   "],
    "V": ["█   █ ", "█   █ ", "█   █ ", " █ █  ", "  █   "],
    "A": [" ███  ", "█   █ ", "█████ ", "█   █ ", "█   █ "],
    "N": ["█   █ ", "██  █ ", "█ █ █ ", "█  ██ ", "█   █ "],
    "O": ["████  ", "█  █  ", "█  █  ", "█  █  ", "████  "],
    " ": ["      "] * 5,
}


def banner(texto: str) -> list[str]:
    filas = ["" ] * 5
    for i, ch in enumerate(texto.upper()):
        glifo = _LETRAS.get(ch, _LETRAS[" "])
        for r in range(5):
            filas[r] += glifo[r] + (" " if i < len(texto) - 1 else "")
    return filas


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def habilitar_ansi() -> None:
    """Windows 10+ acepta colores ANSI en la consola (Windows Terminal o
    conhost con VT habilitado). Si falla, la consola sigue en texto plano."""
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetConsoleMode(k32.GetStdHandle(-11), 7)
    except Exception:
        pass


def color(txt: str, codigo: str) -> str:
    return f"\033[{codigo}m{txt}\033[0m"


def limpiar() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def leer_config() -> dict:
    p = configparser.ConfigParser(inline_comment_prefixes=(";",))
    # utf-8-sig: el config.ini tiene BOM (\ufeff) que rompe configparser
    p.read(RUTA_CONFIG, encoding="utf-8-sig")
    return {
        "puerto": p.getint("ftp", "puerto", fallback=2121),
        "usuario": p.get("ftp", "usuario", fallback="admin"),
        "password": p.get("ftp", "password", fallback=""),
        "reportes": p.get("rutas", "reportes", fallback=str(RUTA / "reportes")),
        "logs": p.get("rutas", "logs", fallback=str(RUTA / "logs")),
    }


def tcp_abierto(puerto: int, host: str = "127.0.0.1", espera: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, puerto), timeout=espera):
            return True
    except OSError:
        return False


def estado_servicio() -> str:
    r = subprocess.run(["sc", "query", SERVICIO], capture_output=True,
                       text=True, creationflags=SIN_VENTANA)
    # Windows en espanol localiza la salida: ESTADO en vez de STATE.
    for linea in r.stdout.splitlines():
        if "STATE" in linea or "ESTADO" in linea:
            if "RUNNING" in linea:
                return "Running"
            return linea.split(":", 1)[1].strip()
    return "desconocido"


def ultimas_lineas(ruta: Path, n: int = 30) -> list[str]:
    try:
        with open(ruta, "rb") as f:
            f.seek(0, 2)
            tam = f.tell()
            f.seek(max(0, tam - 16384))
            datos = f.read().decode("utf-8", "replace")
        return datos.splitlines()[-n:]
    except OSError as e:
        return [f"(no se pudo leer {ruta}: {e})"]


def confirmar(pregunta: str) -> bool:
    try:
        return input(pregunta).strip().lower() in ("s", "si", "sí", "y", "yes")
    except EOFError:
        return False


def pausa() -> None:
    if not _INTERACTIVO:
        return
    try:
        input("\nPresione Enter para volver al menu...")
    except EOFError:
        pass


# --------------------------------------------------------------------------
# Comandos
# --------------------------------------------------------------------------
def cmd_estado(_args: list[str] | None = None) -> None:
    cfg = leer_config()
    serv = estado_servicio()
    ftp_ok = tcp_abierto(cfg["puerto"])
    onnx = RUTA / "modelo" / "placas.onnx"
    hoy = datetime.now().strftime("%Y-%m-%d")
    total = "?"
    try:
        rep = json.loads((Path(cfg["reportes"]) / f"placas-{hoy}.json").read_text("utf-8"))
        total = len(rep.get("placas", []))
    except OSError:
        pass

    ok = color("OK", "32")
    mal = color("FALLA", "31")
    print()
    print(f"  Servicio {SERVICIO:<18} {ok if serv == 'Running' else mal}  ({serv})")
    print(f"  FTP 127.0.0.1:{cfg['puerto']:<10} {ok if ftp_ok else mal}")
    if onnx.exists():
        mb = onnx.stat().st_size / 1024 / 1024
        print(f"  Modelo ONNX{'':<16} {ok}  ({mb:.1f} MB)")
    else:
        print(f"  Modelo ONNX{'':<16} {mal}  (no existe {onnx})")
    print(f"  Placas hoy ({hoy}): {total}")


def cmd_placas(_args: list[str] | None = None) -> None:
    cfg = leer_config()
    hoy = datetime.now().strftime("%Y-%m-%d")
    ruta = Path(cfg["reportes"]) / f"placas-{hoy}.json"
    try:
        rep = json.loads(ruta.read_text("utf-8"))
    except OSError:
        print(f"\n  No hay reporte de hoy todavia: {ruta}")
        return
    placas = rep.get("placas", [])
    print(f"\n  Placas del dia {rep.get('fecha', hoy)}  (total {len(placas)})")
    if not placas:
        print("  Todavia ninguna placa guardada hoy.")
        return
    print(f"\n  {'PLACA':<12} {'HORA':<10} {'CONFIANZA':>9}")
    print("  " + "-" * 34)
    for p in placas:
        print(f"  {p.get('placa', '?'):<12} {p.get('hora', '?'):<10} "
              f"{p.get('confianza', 0):>8.0%}")


def cmd_historial(_args: list[str] | None = None) -> None:
    con = sqlite3.connect(str(RUTA / "registro.db"))
    try:
        filas = con.execute(
            "SELECT fecha_hora, placa, confianza, estado, archivo "
            "FROM registros ORDER BY id DESC LIMIT 15").fetchall()
    finally:
        con.close()
    print("\n  Ultimos 15 registros (aceptados y descartados):")
    print(f"  {'FECHA/HORA':<20} {'PLACA':<12} {'CONF':>6}  {'ESTADO':<16} ARCHIVO")
    print("  " + "-" * 74)
    for fh, placa, conf, estado, archivo in filas:
        c = f"{conf:.0%}" if conf is not None else "-"
        print(f"  {fh:<20} {placa or '-':<12} {c:>6}  {estado:<16} {archivo}")


def cmd_log(_args: list[str] | None = None) -> None:
    cfg = leer_config()
    ruta = Path(cfg["logs"]) / "servidor.log"
    print(f"\n  Ultimas lineas de {ruta}:")
    print("  " + "-" * 74)
    for linea in ultimas_lineas(ruta, 30):
        print(f"  {linea}")


def cmd_fotos(_args: list[str] | None = None) -> None:
    ruta_txt = input("\n  Ruta de la foto: ").strip().strip('"')
    ruta = Path(ruta_txt)
    if not ruta.is_file():
        print(f"  No existe el archivo: {ruta}")
        return
    canal = input("  Canal (01 panoramica / 02 placa) [01]: ").strip() or "01"
    if canal not in ("01", "02"):
        print("  Canal invalido (use 01 o 02).")
        return
    cfg = leer_config()
    # Nombre al estilo de la camara: el worker elige la vista con _(\\d{2})_
    nombre = f"web_{canal}_{time.strftime('%Y%m%d%H%M%S')}_MD_WITH_TARGET.jpg"
    try:
        with ftplib.FTP() as ftp:
            ftp.connect("127.0.0.1", cfg["puerto"], timeout=10)
            ftp.login(cfg["usuario"], cfg["password"])
            with open(ruta, "rb") as f:
                ftp.storbinary(f"STOR {nombre}", f)
    except Exception as e:
        print(f"  Fallo la subida por FTP: {e}")
        print(f"  Revise que el servicio este Running (comando: servicios).")
        return
    print(f"  Enviada como {nombre} (canal {canal}).")
    print("  El worker la evalua en 1-3 s; mire el desenlace abajo.")
    time.sleep(3.5)
    log = Path(cfg["logs"]) / "servidor.log"
    coincidencias = [l for l in ultimas_lineas(log, 40) if nombre in l]
    print()
    for l in coincidencias:
        print(f"  {l}")
    if not coincidencias:
        print("  (todavia no aparece en el log; espere unos segundos y use 'log')")


def cmd_servicios(_args: list[str] | None = None) -> None:
    serv = estado_servicio()
    print(f"\n  Servicio {SERVICIO}: {serv}")
    print("  (reiniciar corta la cola de imagenes ~2 s y pide UAC)")
    if not confirmar("  ¿Reiniciar el servicio ahora? (s/N): "):
        print("  Cancelado.")
        return
    # NSSM corre como servicio: reiniciar necesita elevacion. Se lanza un
    # powershell elevado y se espera en la consola actual.
    inner = f"Restart-Service {SERVICIO} -Force"
    outer = (
        "Start-Process powershell -Verb RunAs -Wait -ArgumentList "
        f"'-NoProfile','-Command','{inner}'"
    )
    subprocess.Popen(["powershell", "-NoProfile", "-Command", outer],
                     creationflags=SIN_VENTANA)
    print("  Acepte el UAC si aparece; esperando el reinicio...")
    for _ in range(20):
        time.sleep(1)
        serv = estado_servicio()
        if serv == "Running":
            print(f"  Servicio {color('Running', '32')}.")
            return
    print(f"  Estado actual: {serv}. Si no arranco, revise el UAC o el log.")


def cmd_config(_args: list[str] | None = None) -> None:
    print(f"\n  {RUTA_CONFIG}:\n")
    print("  " + "-" * 74)
    for linea in RUTA_CONFIG.read_text("utf-8-sig", errors="replace").splitlines():
        print(f"  {linea}")
    if confirmar("\n  ¿Abrir en el bloc de notas? (s/N): "):
        os.startfile(str(RUTA_CONFIG))


def cmd_ayuda(_args: list[str] | None = None) -> None:
    print("""
  Consola del servidor de placas CYTV Ancon
  -----------------------------------------
  Flechas arriba/abajo ..... navega la lista
  Enter .................... ejecuta el comando resaltado
  1-9 ...................... ejecuta ese comando directo
  :wq + Enter .............. sale de la consola (estilo vim)

  En la linea ':' tambien puede escribir el nombre del comando
  (p. ej. ':placas') sin usar el menu.

  Cada comando corre tambien sin menu:
      consola.py estado
      consola.py log

  Las fotos rechazadas por el worker se borran; para revisar por que
  una placa no guardo, busque su nombre en 'log' o 'historial'.""")


OPCIONES = [
    ("estado",     "Servicio, FTP y modelo",              cmd_estado),
    ("placas",     "Placas detectadas hoy (JSON)",         cmd_placas),
    ("historial",  "Ultimos registros de la base SQLite",  cmd_historial),
    ("log",        "Ultimas lineas del log del servidor",  cmd_log),
    ("fotos",      "Subir una foto de prueba por el FTP",  cmd_fotos),
    ("servicios",  f"Reiniciar el servicio {SERVICIO}",    cmd_servicios),
    ("config",     "Ver o editar el config.ini",           cmd_config),
    ("ayuda",      "Ayuda de la consola",                  cmd_ayuda),
]


# --------------------------------------------------------------------------
# Menu interactivo
# --------------------------------------------------------------------------
def imprimir_cabecera() -> None:
    print()
    for fila in banner("CYTV ANCON"):
        print("  " + color(fila, "36"))
    print("  " + color("Servidor de placas - consola de control", "90"))
    print()


def imprimir_menu(sel: int) -> None:
    print("  Comandos:\n")
    for i, (nombre, desc, _) in enumerate(OPCIONES):
        marca = color(">", "33") if i == sel else " "
        texto = f"{i + 1}) {nombre:<12} {desc}"
        print(f"   {marca} {color(texto, '7') if i == sel else texto}")
    print()
    print(color("  Flechas navegan  |  Enter ejecuta  |  :wq sale", "90"))


def leer_linea_comando() -> str:
    """Mini-linea estilo vim: muestra ':' y devuelve lo tecleado."""
    caracteres: list[bytes] = []
    while True:
        t = msvcrt.getch()
        if t in (b"\r", b"\n"):
            print()
            return b"".join(caracteres).decode("utf-8", "replace")
        if t in (b"\x08", b"\x7f"):          # backspace
            if caracteres:
                caracteres.pop()
                sys.stdout.write("\b \b")
                sys.stdout.flush()
            continue
        if t.isprintable():
            caracteres.append(t)
            sys.stdout.write(t.decode("utf-8", "replace"))
            sys.stdout.flush()


def ejecutar(nombre: str, args: list[str] | None = None) -> None:
    for n, _, fn in OPCIONES:
        if n == nombre:
            limpiar()
            imprimir_cabecera()
            fn(args)
            pausa()
            return
    print(f"\n  Comando desconocido: {nombre!r} (use 'ayuda')")


def menu() -> None:
    import msvcrt                      # solo Windows; el proyecto vive ahi
    sel = 0
    while True:
        limpiar()
        imprimir_cabecera()
        imprimir_menu(sel)
        t = msvcrt.getch()
        if t in (b"\x00", b"\xe0"):         # tecla extendida (flechas)
            sig = msvcrt.getch()
            if sig == b"H":
                sel = (sel - 1) % len(OPCIONES)
            elif sig == b"P":
                sel = (sel + 1) % len(OPCIONES)
            continue
        if t == b"\r":
            ejecutar(OPCIONES[sel][0])
        elif t == b":":
            limpiar()
            imprimir_cabecera()
            linea = input("  :").strip().lstrip(":").strip()
            if linea == "wq":
                limpiar()
                print("  Hasta luego.\n")
                return
            if linea:
                ejecutar(linea)
        elif t.isdigit() and t != b"0":
            idx = int(t) - 1
            if idx < len(OPCIONES):
                sel = idx
                ejecutar(OPCIONES[idx][0])


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    habilitar_ansi()
    if len(sys.argv) > 1:
        global _INTERACTIVO
        _INTERACTIVO = False
        nombre = sys.argv[1].lower()
        if nombre in ("--help", "-h", "help"):
            nombre = "ayuda"
        ejecutar(nombre, sys.argv[2:])
        return
    try:
        menu()
    except KeyboardInterrupt:
        limpiar()
        print("  Interrumpido. Hasta luego.\n")


if __name__ == "__main__":
    main()
