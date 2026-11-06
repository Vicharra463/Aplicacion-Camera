#!/usr/bin/env python3
"""Muestra en la IMAGEN que corta el ROI y donde debe estar el vehículo.

Dibuja sobre la foto (coordenadas de config.ini [roi]):
  * banda ROJA  [0, linea_verde_sup)   -> se corta, NO se detecta nunca
  * banda VERDE [sup, inf)              -> aqui debe estar el vehículo con
                                           la placa TOTALMENTE visible
  * linea naranja en inf                -> lo que cruza la linea de abajo
                                           no cuenta (fuera_franja)

Si existe el .txt YOLO hermano (mismo nombre), dibuja la placa GT en
amarillo e imprime el veredicto: CUMPLE / EN ROJO / FUERA.

Uso:
    python scripts\\vista_roi.py                     # foto mas reciente
    python scripts\\vista_roi.py C:\\ruta\\foto.jpg   # foto elegida
    python scripts\\vista_roi.py foto.jpg --mostrar  # ademas abre ventana

Sale en reportes\\vista_roi\\<nombre>_vista.jpg (o --out DIR).
"""
from __future__ import annotations

import argparse
import configparser
import sys
from pathlib import Path

import cv2
import numpy as np

RUTA_RAIZ = Path(__file__).resolve().parents[1]
RUTA_CONFIG = RUTA_RAIZ / "config.ini"
EXT = (".jpg", ".jpeg", ".png")


def leer_config() -> tuple[float, float, bool]:
    p = configparser.ConfigParser(inline_comment_prefixes=(";",))
    p.read(RUTA_CONFIG, encoding="utf-8-sig")
    sup = p.getfloat("roi", "linea_verde_sup", fallback=0.45)
    inf = p.getfloat("roi", "linea_verde_inf", fallback=0.75)
    hab = p.getboolean("roi", "habilitado", fallback=False)
    return sup, inf, hab


def leer_imagen(ruta: Path) -> np.ndarray | None:
    # imdecode+fromfile: cv2.imread solo acepta rutas ASCII en Windows.
    try:
        datos = np.fromfile(str(ruta), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(datos, cv2.IMREAD_COLOR)


def gt_bbox(ruta: Path, ancho: int, alto: int) -> tuple[float, ...] | None:
    """Lee el .txt YOLO hermano (cls cx cy w h normalizados)."""
    txt = ruta.with_suffix(".txt")
    if not txt.is_file():
        return None
    partes = txt.read_text(encoding="utf-8-sig").split()
    if len(partes) < 5:
        return None
    _, cx, cy, w, h = partes[:5]
    cx, cy, w, h = float(cx), float(cy), float(w), float(h)
    return ((cx - w / 2) * ancho, (cy - h / 2) * alto,
            (cx + w / 2) * ancho, (cy + h / 2) * alto)


def veredicto(y0: float, y1: float, sup: float, inf: float) -> str:
    if y1 <= sup:
        return "EN ROJO: se corta, NO se detecta (auto por detras de la linea)"
    if y0 < sup:
        return "EN ROJO: la placa cruza la linea de arriba (se corta)"
    if y1 <= inf:
        return "CUMPLE: placa completa dentro de la franja verde"
    if y0 < inf:
        return "FUERA: la placa cruza la linea de abajo (fuera_franja)"
    return "FUERA: placa por debajo de la franja (demasiado cerca)"


def pintar(img: np.ndarray, sup: float, inf: float) -> np.ndarray:
    alto, ancho = img.shape[:2]
    y_sup = int(round(sup * alto))
    y_inf = int(round(inf * alto))
    vista = img.copy()

    # Banda roja: se corta. Translucida para ver que hay atras.
    rojo = vista[0:y_sup]
    if rojo.size:
        vista[0:y_sup] = (rojo * 0.45 + np.array([0, 0, 180]) * 0.55
                          ).astype(np.uint8)

    # Banda verde: donde debe estar el vehículo.
    verde = vista[y_sup:y_inf]
    if verde.size:
        vista[y_sup:y_inf] = (verde * 0.55 + np.array([0, 150, 0]) * 0.45
                              ).astype(np.uint8)

    # Bajo la linea de abajo: naranja (no cuenta).
    bajo = vista[y_inf:]
    if bajo.size:
        vista[y_inf:] = (bajo * 0.6 + np.array([0, 140, 255]) * 0.4
                         ).astype(np.uint8)

    cv2.line(vista, (0, y_sup), (ancho - 1, y_sup), (0, 255, 0), 3)
    cv2.line(vista, (0, y_inf), (ancho - 1, y_inf), (0, 200, 255), 3)

    cv2.putText(vista, "SE CORTA - no se detecta", (12, max(28, y_sup - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(vista, "AQUI DEBE ESTAR EL VEHICULO",
                (12, y_sup + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)
    if y_inf + 32 < alto:
        cv2.putText(vista, "placa cruza esta linea = no cuenta",
                    (12, y_inf + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return vista


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Vista del ROI: que se corta y donde va el vehículo")
    parser.add_argument("foto", nargs="?", default=None,
                        help="foto a anotar (default: la mas reciente "
                             "de recibidos)")
    parser.add_argument("--out", default=None,
                        help="carpeta de salida "
                             "(default: reportes\\vista_roi)")
    parser.add_argument("--mostrar", action="store_true",
                        help="ademas abre la ventana con la foto anotada")
    args = parser.parse_args(argv)

    if args.foto:
        ruta = Path(args.foto)
        if not ruta.is_file():
            print(f"[FATAL] No existe la foto: {ruta}", file=sys.stderr)
            return 1
    else:
        p = configparser.ConfigParser(inline_comment_prefixes=(";",))
        p.read(RUTA_CONFIG, encoding="utf-8-sig")
        carpeta = Path(p.get("rutas", "recibidos",
                             fallback=str(RUTA_RAIZ / "ftp_recibidos")))
        candidatas = [f for f in carpeta.rglob("*")
                      if f.suffix.lower() in EXT] if carpeta.is_dir() else []
        if not candidatas:
            print(f"[FATAL] Sin fotos en {carpeta}; pase la ruta a mano.",
                  file=sys.stderr)
            return 1
        ruta = max(candidatas, key=lambda f: f.stat().st_mtime)

    img = leer_imagen(ruta)
    if img is None:
        print(f"[FATAL] Imagen ilegible: {ruta}", file=sys.stderr)
        return 1

    sup, inf, hab = leer_config()
    alto, ancho = img.shape[:2]
    print(f"Foto:   {ruta} ({ancho}x{alto})")
    print(f"ROI:    sup={sup:.4f}  inf={inf:.4f}  "
          f"(habilitado = {'si' if hab else 'NO - solo vista previa'})")

    g = gt_bbox(ruta, ancho, alto)
    vista = pintar(img, sup, inf)
    if g is not None:
        x0, y0, x1, y1 = g
        cv2.rectangle(vista, (int(x0), int(y0)), (int(x1), int(y1)),
                      (0, 255, 255), 3)
        cv2.putText(vista, "GT", (int(x0), max(24, int(y0) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                    cv2.LINE_AA)
        ny0, ny1 = y0 / alto, y1 / alto
        print(f"Placa:  y={ny0:.3f}..{ny1:.3f}")
        print(f"Veredicto: {veredicto(ny0, ny1, sup, inf)}")

    salida_dir = Path(args.out) if args.out else (
        RUTA_RAIZ / "reportes" / "vista_roi")
    salida_dir.mkdir(parents=True, exist_ok=True)
    salida = salida_dir / f"{ruta.stem}_vista.jpg"
    ok, buf = cv2.imencode(".jpg", vista, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if ok:
        buf.tofile(str(salida))
    print(f"Vista:  {salida}")

    if args.mostrar:
        cv2.imshow("Vista ROI", vista)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
