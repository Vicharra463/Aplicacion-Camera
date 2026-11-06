#!/usr/bin/env python3
"""Calibra la franja verde del ROI (seccion [roi] de config.ini).

La camara es fija: una vez marcada la franja, el worker corta TODO lo
que este por ENCIMA de la linea verde de arriba (fondo, timestamp,
pared) y exige que la placa quede TOTALMENTE dentro de la franja (entre
las dos lineas verdes) para que la toma cuente.

Abre la foto, arrastra el rectangulo de la franja (de la linea de arriba
a la de abajo) con el mouse y acepta con ESPACIO o ENTER. Se escriben
linea_verde_sup y linea_verde_inf en config.ini y habilitado = si.

Uso (en el servidor, con el venv del servicio):
    python scripts\\calibrar_roi.py                    # foto mas reciente
    python scripts\\calibrar_roi.py C:\\ruta\\foto.jpg  # foto elegida

Sin argumento busca el .jpg/.png mas reciente en la carpeta de recibidos
([rutas] recibidos de config.ini, p. ej. C:\\CytvAncon\\ftp_recibidos).
"""
from __future__ import annotations

import argparse
import configparser
import re
import sys
from pathlib import Path

import cv2
import numpy as np

RUTA_RAIZ = Path(__file__).resolve().parents[1]
RUTA_CONFIG = RUTA_RAIZ / "config.ini"
EXT = (".jpg", ".jpeg", ".png")


def leer_recibidos() -> Path:
    p = configparser.ConfigParser(inline_comment_prefixes=(";",))
    p.read(RUTA_CONFIG, encoding="utf-8-sig")
    return Path(p.get("rutas", "recibidos",
                      fallback=str(RUTA_RAIZ / "ftp_recibidos")))


def foto_mas_reciente(carpeta: Path) -> Path | None:
    candidatas = [f for f in carpetas_fotos(carpeta)]
    if not candidatas:
        return None
    return max(candidatas, key=lambda f: f.stat().st_mtime)


def carpetas_fotos(carpeta: Path):
    if not carpeta.is_dir():
        return []
    return (f for f in carpeta.rglob("*") if f.suffix.lower() in EXT)


def leer_imagen(ruta: Path) -> np.ndarray | None:
    # imdecode+fromfile: cv2.imread solo acepta rutas ASCII en Windows
    # y esta foto puede venir de Descargas con acentos.
    try:
        datos = np.fromfile(str(ruta), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(datos, cv2.IMREAD_COLOR)


def escribir_roi(sup: float, inf: float) -> None:
    """Escribe las coordenadas en [roi] preservando el resto del archivo
    (configparser reescribiria el ini y borraria todos los comentarios)."""
    texto = RUTA_CONFIG.read_text(encoding="utf-8-sig")
    if not re.search(r"(?m)^\[roi\]", texto):
        texto = texto.rstrip("\n") + (
            "\n\n[roi]\nhabilitado = no\n"
            "linea_verde_sup = 0.45\nlinea_verde_inf = 0.75\n")
    texto = re.sub(r"(?m)^linea_verde_sup = .*$",
                   f"linea_verde_sup = {sup:.4f}", texto, count=1)
    texto = re.sub(r"(?m)^linea_verde_inf = .*$",
                   f"linea_verde_inf = {inf:.4f}", texto, count=1)
    texto = re.sub(r"(?m)^habilitado = .*$", "habilitado = si",
                   texto, count=1)
    RUTA_CONFIG.write_text(texto, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibra la franja verde del ROI de la camara")
    parser.add_argument("foto", nargs="?", default=None,
                        help="foto de referencia (default: la mas "
                             "reciente de recibidos)")
    args = parser.parse_args(argv)

    if args.foto:
        ruta = Path(args.foto)
        if not ruta.is_file():
            print(f"[FATAL] No existe la foto: {ruta}", file=sys.stderr)
            return 1
    else:
        carpeta = leer_recibidos()
        ruta = foto_mas_reciente(carpeta)
        if ruta is None:
            print(f"[FATAL] Sin fotos en {carpeta}; pase la ruta a mano: "
                  f"python scripts\\calibrar_roi.py C:\\ruta\\foto.jpg",
                  file=sys.stderr)
            return 1

    img = leer_imagen(ruta)
    if img is None:
        print(f"[FATAL] Imagen ilegible: {ruta}", file=sys.stderr)
        return 1

    alto, ancho = img.shape[:2]
    print(f"Foto: {ruta} ({ancho}x{alto})")
    print("Arrastre el rectangulo de la FRANJA VERDE (de la linea de "
          "arriba a la de abajo) y acepte con ESPACIO o ENTER.")
    print("ESC cancela.")
    x, y, w, h = cv2.selectROI("Franja verde", img, showCrosshair=True,
                               fromCenter=False)
    cv2.destroyWindow("Franja verde")
    if w == 0 or h == 0:
        print("Cancelado: no se toco config.ini.")
        return 1

    sup = y / alto
    inf = (y + h) / alto
    if not (0.0 <= sup < inf <= 1.0) or (inf - sup) < 0.05:
        print(f"[FATAL] Franja invalida: sup={sup:.4f} inf={inf:.4f} "
              "(se necesita inf-sup >= 0.05). No se toco config.ini.",
              file=sys.stderr)
        return 1

    escribir_roi(sup, inf)

    # Preview: las dos lineas verdes sobre la foto, para confirmar.
    vista = img.copy()
    for yy in (y, y + h):
        cv2.line(vista, (0, yy), (ancho - 1, yy), (0, 255, 0), 3)
    cv2.imshow("ROI calibrado (ENTER para cerrar)", vista)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print()
    print("ROI escrito en config.ini:")
    print(f"  linea_verde_sup = {sup:.4f}")
    print(f"  linea_verde_inf = {inf:.4f}")
    print("  habilitado = si")
    print("REINICIE el servicio de la camara para que el worker lo aplique.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
