"""Barrido de la linea verde de corte: cuantas placas GT sobreviven.

Medida SIN API (solo deteccion local): para cada valor de
linea_verde_sup, recorta las fotos de GT-29 igual que el worker y
cuenta en cuantas el detector todavia encuentra un bbox que solapa la
placa real (IoU >= 0.3). La columna "cortada" son fotos cuya placa
queda por ENCIMA del corte (imposible: el pixel ya no existe).

Sirve para elegir linea_verde_sup antes de calibrar con
scripts\\calibrar_roi.py: mas corte = menos FP de fondo, pero se
abandonan los autos que no llegaron a la posicion ideal (en produccion
la RAFAGA salva esas fotos con un frame posterior; el banco es foto
por foto).

Uso:
    venv\\Scripts\\python.exe medir_roi.py [modelo.onnx]
"""
from __future__ import annotations

import sys
from pathlib import Path

PROYECTO = Path(r"D:\Aplicacion-Camera")
sys.path.insert(0, str(PROYECTO))
sys.path.insert(0, str(PROYECTO / "app"))

import cv2  # noqa: E402

from gt_placas import DIAGNOSTICOS, RAIZ  # noqa: E402
from modelo import DetectorPlacas  # noqa: E402
import worker  # noqa: E402

ONNX = Path(sys.argv[1]) if len(sys.argv) > 1 else PROYECTO / "modelo" / "placas.onnx"
UMBRALES = (0.0, 0.10, 0.15, 0.20, 0.30, 0.40, 0.45)
IOU_MIN = 0.3


def gt_bbox(txt: Path, ancho: int, alto: int) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) normalizado sobre la imagen ORIGINAL."""
    p = txt.read_text(encoding="utf-8").split()
    cx, cy, w, h = (float(v) for v in p[1:5])
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter) if (area_a + area_b - inter) else 0.0


def main() -> int:
    detector = DetectorPlacas(ONNX)
    print(f"modelo={ONNX.name}  fotos={len(DIAGNOSTICOS)}  IoU>={IOU_MIN}\n")
    print("  corte | placa OK | placa cortada | sin deteccion")
    print("  ------+----------+---------------+--------------")
    for sup in UMBRALES:
        worker._cfg = {"roi_habilitado": sup > 0.0,
                       "roi_linea_verde_sup": sup,
                       "roi_linea_verde_inf": 1.0}
        ok = cortada = perdida = 0
        fallos: list[str] = []
        for dia, nombre, real, _ in DIAGNOSTICOS:
            ruta = RAIZ / dia / nombre
            img = cv2.imread(str(ruta))
            if img is None:
                continue
            g = gt_bbox(ruta.with_suffix(".txt"), img.shape[1], img.shape[0])
            if g[3] <= sup:                     # placa entera sobre el corte
                cortada += 1
                fallos.append(f"{nombre}(cortada)")
                continue
            img, _ = worker._preparar_imagen(img)
            # GT convertido a coords del recorte
            escala = 1.0 - sup
            gc = ((g[0], (g[1] - sup) / escala, g[2], (g[3] - sup) / escala))
            mejores = []
            for (x, y, w, h), _conf in detector.detectar(
                    img, enmascarar=not worker._cfg.get(
                        "roi_habilitado", False)):
                bx0, by0 = x / img.shape[1], y / img.shape[0]
                bx1, by1 = (x + w) / img.shape[1], (y + h) / img.shape[0]
                mejores.append(iou(gc, (bx0, by0, bx1, by1)))
            if mejores and max(mejores) >= IOU_MIN:
                ok += 1
            else:
                perdida += 1
                fallos.append(nombre)
        marca = "  <-- base" if sup == 0.0 else ""
        print(f"  {sup:4.2f}  |    {ok:2d}    |      {cortada:2d}      "
              f"|     {perdida:2d}{marca}")
        if sup > 0.0 and (cortada or perdida):
            print(f"          pierde: {', '.join(fallos)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
