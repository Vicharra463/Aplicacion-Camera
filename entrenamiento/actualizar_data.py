#!/usr/bin/env python3
"""Actualiza la data: lee con el motor OFICIAL, reevalua, renombra, deduplica
y descarta lo que no tiene placa. Cotejo en carpeta nueva.

Corre el MISMO pipeline que produccion (app\\ocr.leer_placa_6):

  1. Bbox: del .txt YOLO si existe (mismo camino que la validacion); si no,
     detector ONNX modelo\\placas.onnx (camino de produccion).
  2. Motor: CRNN+flota (closed-set) con fallback al motor de 6 intentos
     con votacion ponderada. Devuelve ademas el REGISTRO de cada intento,
     que es el 'pensamiento de la IA' y se vuelca al CSV y a la imagen.

Y decide por foto:

  - Con GT (etiquetas_reales.csv / gt_placas.DIAGNOSTICOS verificados a
    mano): el nombre VERDADERO es el GT -> renombra al GT. El motor se
    contrasta contra el GT (OK/FALLO) y queda en el CSV.
  - Sin GT: renombra a lo que leyó el motor (marcado sin_gt en el CSV).
  - Motor no leyó y no hay GT -> descartados\\<dia>\\ (la imagen no sirve
    como placa legible).
  - Misma placa en varios frames del mismo dia -> repetidos\\<dia>\\
    (gana la de mayor confianza; a igual confianza, la que tiene GT).

Todo queda en evaluacion-<fecha>.csv: decision, GT, lectura del motor,
confianza, destino y RAZONAMIENTO (registro completo de intentos).

Modos:
    SIMULACION (defecto)   solo imprime + CSV, no escribe imagenes
    -Salida <carpeta>      COPIA ANOTADA a carpeta nueva (contorno ROJO
                           sobre la placa + decision escrita). NO toca la
                           data original. Este es el cotejo.
    -Aplicar               mueve/renombra EN EL LUGAR (produccion).

Uso (venv del servicio):
    C:\\CytvAncon\\venv\\Scripts\\python.exe actualizar_data.py ^
        -Salida "C:\\...\\FOTOS PLACA\\COTEJO 10-19 v2"
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "app"))
sys.path.insert(0, str(BASE / "entrenamiento"))

from ocr import leer_placa_6                       
from gt_placas import DIAGNOSTICOS                 

RAIZ_DEFECTO = Path(
    r"C:\Users\felip\OneDrive\Desktop\Motorgas\APP CAMARA"
    r"\FOTOS PLACA\DATA-NUEVA 10-19"
)
CSV_ETIQUETAS = BASE / "entrenamiento" / "etiquetas_reales.csv"
MARGEN = 0.35
ROJO = (0, 0, 255)
VERDE = (0, 200, 0)
AMARILLO = (0, 215, 255)

_detector_cache = []





def cargar_etiquetas() -> dict[tuple[str, str], str]:
    etiquetas: dict[tuple[str, str], str] = {}
    if CSV_ETIQUETAS.is_file():
        with CSV_ETIQUETAS.open(encoding="utf-8", newline="") as fh:
            for fila in csv.DictReader(fh):
                if fila.get("dia") and fila.get("archivo") and fila.get("placa"):
                    etiquetas[(fila["dia"].strip().lower(),
                               fila["archivo"].strip().lower())] = \
                        fila["placa"].strip()
    
    for dia, nombre, real, _nota in DIAGNOSTICOS:
        etiquetas[(dia.lower(), nombre.lower())] = real
    return etiquetas





def bbox_desde_txt(ruta_txt: Path, ancho: int, alto: int):
    try:
        linea = ruta_txt.read_text(encoding="utf-8").strip().splitlines()[0]
        _c, cx, cy, w, h = linea.split()
        cx, cy, w, h = (float(v) for v in (cx, cy, w, h))
    except (OSError, ValueError, IndexError):
        return None
    bw, bh = w * ancho, h * alto
    return int(cx * ancho - bw / 2), int(cy * alto - bh / 2), int(bw), int(bh)


def obtener_detector():
    if not _detector_cache:
        from modelo import DetectorPlacas, RUTA_ONNX
        if not RUTA_ONNX.is_file():
            return None
        _detector_cache.append(DetectorPlacas(RUTA_ONNX))
    return _detector_cache[0]


def bbox_modelo(img: np.ndarray):
    det = obtener_detector()
    if det is None:
        return None
    detecciones = det.detectar(img)
    if not detecciones:
        return None
    
    return max(detecciones, key=lambda par: par[0][2] * par[0][3])[0]





def razonamiento(registro, texto, conf) -> str:
    if not registro:
        return "sin registros del motor"
    partes = []
    for nombre, t, c in registro:
        lectura = t if t else "(vacio)"
        partes.append(f"{nombre}={lectura}({c:.2f})")
    via = ("CRNN+flota (closed-set)" if registro[0][0] == "crnn_flota"
           else "votos 6 intentos")
    final = texto if texto else "SIN LECTURA"
    return f"via {via} | " + " ; ".join(partes) + \
        f" => motor='{final}' conf={conf:.2f}"





def anotar(img: np.ndarray, bbox, etiqueta: str, color=ROJO) -> np.ndarray:
    salida = img.copy()
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(salida, (x, y), (x + w, y + h), color, 4)
    lineas = [l for l in etiqueta.split("\n") if l]
    y_text = (bbox[1] - 16) if bbox and bbox[1] > 90 else 48
    for linea in lineas:
        (tw, th), _ = cv2.getTextSize(linea, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        x_text = max(0, (bbox[0] if bbox else 20))
        x_text = min(x_text, salida.shape[1] - tw - 14)
        cv2.rectangle(salida, (x_text - 8, y_text - th - 10),
                      (x_text + tw + 8, y_text + 10), (0, 0, 0), -1)
        cv2.putText(salida, linea, (x_text, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
        y_text += th + 20
    return salida





def destino_libre(carpeta: Path, base: str) -> Path:
    destino = carpeta / f"{base}.jpg"
    n = 2
    while destino.exists():
        destino = carpeta / f"{base}-{n}.jpg"
        n += 1
    return destino


def dias_disponibles(raiz: Path) -> list[Path]:
    patron = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    return sorted(d for d in raiz.iterdir()
                  if d.is_dir() and patron.match(d.name))



def main() -> int:
    ap = argparse.ArgumentParser(
        description="Lee, reevalua, renombra, deduplica y descarta por dia.")
    ap.add_argument("-Raiz", type=Path, default=RAIZ_DEFECTO,
                    help="carpeta con las carpetas de dia")
    ap.add_argument("-Dia", default="",
                    help="solo una fecha (AAAA-MM-DD)")
    ap.add_argument("-Aplicar", action="store_true",
                    help="cambios EN EL LUGAR (destructivo)")
    ap.add_argument("-Salida", type=Path, default=None,
                    help="carpeta NUEVA de cotejo: copia anotada con "
                         "contorno rojo; NO modifica la original")
    args = ap.parse_args()

    raiz: Path = args.Raiz
    if not raiz.is_dir():
        print(f"[ERROR] no existe la raiz: {raiz}")
        return 1
    if args.Aplicar and args.Salida:
        print("[ERROR] -Aplicar y -Salida no van juntos.")
        return 1

    dias = dias_disponibles(raiz)
    if args.Dia:
        dias = [d for d in dias if d.name == args.Dia]
        if not dias:
            print(f"[ERROR] dia no encontrado: {args.Dia}")
            return 1
    if not dias:
        print(f"[ERROR] sin carpetas de dia en {raiz}")
        return 1

    modo = "cotejo" if args.Salida else ("aplicar" if args.Aplicar else "sim")
    raiz_salida = args.Salida if modo == "cotejo" else raiz
    mover = (modo == "aplicar")

    etiquetas = cargar_etiquetas()

    print("=" * 62)
    print(" ACTUALIZAR DATA - motor oficial (CRNN+flota / 6 intentos)")
    print(f" Raiz:    {raiz}")
    if modo == "cotejo":
        print(f" Salida:  {raiz_salida} (original intacta)")
    elif modo == "aplicar":
        print(" Salida:  EN EL LUGAR (destructivo)")
    else:
        print(" Salida:  SIMULACION (nada se mueve)")
    print(f" GT:      {len(etiquetas)} etiquetas (csv + diagnosticos)")
    print("=" * 62)

    filas: list[dict] = []
    resumen: dict[str, dict[str, int]] = {}
    t0 = time.time()

    for dia_dir in dias:
        fotos = sorted(dia_dir.glob("*.jpg"))
        if not fotos:
            print(f"\n{dia_dir.name}: sin fotos")
            continue
        print(f"\n{dia_dir.name}: {len(fotos)} fotos")

        
        registros: list[dict] = []
        for jpg in fotos:
            txt = jpg.with_suffix(".txt")
            img = cv2.imread(str(jpg))
            fila = {"jpg": jpg, "txt": txt, "img": img,
                    "bbox": None, "origen_bbox": "", "gt": None,
                    "texto": "", "conf": 0.0, "registro": [],
                    "motivo": "", "destino_dir": None, "base": "",
                    "accion": ""}
            clave = (dia_dir.name.lower(), jpg.name.lower())
            fila["gt"] = etiquetas.get(clave)

            if img is None:
                fila["motivo"] = "imagen_corrupta (cv2 no pudo abrirla)"
                registros.append(fila)
                continue

            alto, ancho = img.shape[:2]
            b = bbox_desde_txt(txt, ancho, alto) if txt.is_file() else None
            if b:
                fila["bbox"], fila["origen_bbox"] = b, "txt"
            else:
                b = bbox_modelo(img)
                if b:
                    fila["bbox"], fila["origen_bbox"] = b, "onnx"
                else:
                    fila["motivo"] = "sin_placa (sin .txt y el ONNX no ve placa)"
                    registros.append(fila)
                    continue

            texto, conf, registro = leer_placa_6(img, fila["bbox"], MARGEN)
            fila["texto"], fila["conf"], fila["registro"] = texto, conf, registro
            if not texto:
                fila["motivo"] = "motor_no_leyo (bbox hay, lectura no)"
            registros.append(fila)

        
        for fila in registros:
            if fila["gt"]:
                fila["base"] = fila["gt"]                     
            elif fila["texto"]:
                fila["base"] = fila["texto"]                  
            else:
                fila["base"] = ""                             

        
        
        
        grupos: dict[str, list[dict]] = {}
        for fila in registros:
            if fila["base"]:
                grupos.setdefault(fila["base"], []).append(fila)
        repetidas: dict[int, str] = {}
        for base, grupo in grupos.items():
            if len(grupo) == 1:
                continue
            orden = sorted(grupo,
                           key=lambda f: (f["gt"] is not None, f["conf"]),
                           reverse=True)
            for perdedor in orden[1:]:
                repetidas[id(perdedor)] = orden[0]["jpg"].name

        
        cont = {"total": len(fotos), "iguales": 0, "renombradas": 0,
                "repetidas": 0, "descartadas": 0,
                "gt": 0, "gt_ok": 0}

        for fila in registros:
            jpg, txt = fila["jpg"], fila["txt"]
            gt, texto, base = fila["gt"], fila["texto"], fila["base"]

            if gt and texto:
                cont["gt"] += 1
                if texto == gt:
                    cont["gt_ok"] += 1

            if not base:
                accion, motivo = "descartada", fila["motivo"] or "sin_placa"
            elif id(fila) in repetidas:
                accion, motivo = "repetida", \
                    f"misma placa que {repetidas[id(fila)]}"
            elif base == jpg.stem:
                accion, motivo = "sin_cambio", \
                    f"motor vs GT: {'OK' if gt and texto == gt else ('FALLO (motor=' + (texto or 'nada') + ', GT=' + gt + ')' if gt else 'sin_gt')}"
            else:
                accion = "renombrada"
                motivo = (f"GT verificado '{gt}' (nombre '{jpg.stem}')"
                          if gt else
                          f"sin_gt: motor leyó '{texto}' (nombre '{jpg.stem}')")
            fila["accion"], fila["motivo"] = accion, motivo

            
            if accion in ("descartada", "repetida"):
                sub = "descartados" if accion == "descartada" else "repetidos"
                fila["destino_dir"] = (raiz / sub if mover
                                       else raiz_salida / sub) / dia_dir.name
            else:
                fila["destino_dir"] = (dia_dir if mover
                                       else raiz_salida / dia_dir.name)

            if accion == "descartada":
                destino = destino_libre(fila["destino_dir"], jpg.stem)
                cont["descartadas"] += 1
            elif accion == "repetida":
                destino = destino_libre(fila["destino_dir"], jpg.stem)
                cont["repetidas"] += 1
            elif accion == "sin_cambio":
                
                
                destino = fila["destino_dir"] / jpg.name
                cont["iguales"] += 1
            else:
                destino = destino_libre(fila["destino_dir"], base)
                cont["renombradas"] += 1

            
            mismo_archivo = (modo == "aplicar" and destino == jpg)
            if modo == "aplicar" and not mismo_archivo:
                jpg.rename(destino)
                if txt.exists():
                    txt.rename(destino.with_suffix(".txt"))
            elif modo == "cotejo" and fila["img"] is not None:
                color = {"sin_cambio": VERDE, "renombrada": ROJO,
                         "repetida": AMARILLO, "descartada": ROJO}[accion]
                if accion == "descartada":
                    etiqueta = f"DESCARTA: {motivo}"
                elif accion == "repetida":
                    etiqueta = f"REPETIDA: {base} (conf {fila['conf']:.2f})\n" \
                               + motivo
                else:
                    linea1 = f"{base} (conf {fila['conf']:.2f})"
                    linea2 = (f"GT: {gt}" if gt else "GT: sin verificar")
                    linea3 = ("OK motor=GT" if gt and texto == gt else
                              (f"FALLO motor={texto or 'nada'}" if gt
                               else f"renombra desde '{jpg.stem}'"))
                    etiqueta = f"{linea1}\n{linea2}\n{linea3}"
                anotada = anotar(fila["img"], fila["bbox"], etiqueta, color)
                destino.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(destino), anotada,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                if txt.exists():
                    shutil.copy2(txt, destino.with_suffix(".txt"))

            if accion != "sin_cambio" or modo == "sim":
                marca = {"sin_cambio": "OK     ", "renombrada": "RENOMBRA",
                         "repetida": "REPITE ", "descartada": "DESCARTA"}[accion]
                print(f"  [{marca}] {jpg.name} -> {destino.name}  {motivo}")

            filas.append({
                "dia": dia_dir.name, "imagen": jpg.name, "accion": accion,
                "motivo": motivo, "gt": gt or "", "motor": texto,
                "coincide_motor_gt": ("si" if gt and texto == gt
                                      else ("no" if gt else "sin_gt")),
                "confianza": f"{fila['conf']:.3f}",
                "bbox_origen": fila["origen_bbox"], "destino": destino.name,
                "razonamiento": razonamiento(fila["registro"], texto,
                                             fila["conf"]),
            })

        resumen[dia_dir.name] = cont

    
    
    
    marca_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    carpeta_csv = raiz_salida if modo != "sim" else raiz
    carpeta_csv.mkdir(parents=True, exist_ok=True)
    csv_path = carpeta_csv / f"evaluacion-{marca_ts}.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "dia", "imagen", "accion", "motivo", "gt", "motor",
            "coincide_motor_gt", "confianza", "bbox_origen", "destino",
            "razonamiento"])
        w.writeheader()
        w.writerows(filas)

    
    
    
    tot = {k: 0 for k in ("total", "iguales", "renombradas", "repetidas",
                          "descartadas", "gt", "gt_ok")}
    print("\n" + "=" * 78)
    print(f"{'DIA':<12} {'FOT':>4} {'GT':>4} {'OKGT':>5} {'%GT':>7} "
          f"{'OK':>4} {'RENO':>5} {'REP':>4} {'DESC':>5}")
    print("-" * 78)
    for dia, c in sorted(resumen.items()):
        pct = (c["gt_ok"] / c["gt"] * 100) if c["gt"] else 0.0
        print(f"{dia:<12} {c['total']:>4} {c['gt']:>4} {c['gt_ok']:>5} "
              f"{pct:>6.1f}% {c['iguales']:>4} {c['renombradas']:>5} "
              f"{c['repetidas']:>4} {c['descartadas']:>5}")
        for k in tot:
            tot[k] += c[k]
    print("-" * 78)
    pct_total = (tot["gt_ok"] / tot["gt"] * 100) if tot["gt"] else 0.0
    print(f"{'TOTAL':<12} {tot['total']:>4} {tot['gt']:>4} {tot['gt_ok']:>5} "
          f"{pct_total:>6.1f}% {tot['iguales']:>4} {tot['renombradas']:>5} "
          f"{tot['repetidas']:>4} {tot['descartadas']:>5}")
    print("=" * 78)
    fallos_gt = [f for f in filas if f["coincide_motor_gt"] == "no"]
    if fallos_gt:
        print(f"\nFALLOS motor vs GT ({len(fallos_gt)}):")
        for f in fallos_gt:
            print(f"  {f['dia']}/{f['imagen']}: GT={f['gt']} "
                  f"motor={f['motor'] or 'SIN LECTURA'}")
    print(f"\nCSV (con razonamiento): {csv_path}")
    print(f"Tiempo: {time.time() - t0:.1f} s")
    if modo == "sim":
        print("\nSIMULACION: -Salida <carpeta> para cotejo con contorno rojo.")
    elif modo == "cotejo":
        print(f"\nCOTEJO listo: {raiz_salida}")
        print("  <dia>\\            contorno ROJO + decision (verde=OK, "
              "rojo=renombrada/descartada, amarillo=repetida)")
        print("  repetidos\\<dia>\\  frames de mas de la misma placa")
        print("  descartados\\<dia>\\ sin placa o ilegibles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
