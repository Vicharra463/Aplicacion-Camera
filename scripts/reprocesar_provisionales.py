"""Reprocesa con Plate Recognizer las lecturas provisionales (hechas por el modelo local).

Uso:
    python scripts\\reprocesar_provisionales.py [TOPE] [--dry-run]

Requiere la variable de entorno PLATERECOGNIZER_TOKEN.
"""
import sqlite3
import sys
import time
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "app"))
import ocr  # noqa: E402
import platerecognizer as pr  # noqa: E402

DB = RAIZ / "servicio_local" / "registro.db"   # ajusta si config.ini apunta a otra ruta
PROCESADAS = RAIZ / "imagenes_procesadas"
TOPE = int(next((a for a in sys.argv[1:] if a.isdigit()), 200))
DRY = "--dry-run" in sys.argv

if not DB.is_file():
    sys.exit(f"No existe la base de datos: {DB}")

con = sqlite3.connect(DB)
con.execute("PRAGMA busy_timeout=5000")
pend = con.execute(
    "SELECT id, archivo, placa, confianza, fecha_hora FROM registros "
    "WHERE provisional = 1 AND estado = 'guardado' "
    "ORDER BY confianza ASC").fetchall()
print(f"{len(pend)} provisionales; se procesan hasta {TOPE}")
if DRY:
    con.close()
    sys.exit()
if not pr.api_disponible():
    con.close()
    sys.exit("API no disponible (falta PLATERECOGNIZER_TOKEN o esta apagada)")

usadas = 0
for rid, archivo, vieja, _conf, fecha in pend:
    if usadas >= TOPE or not pr.api_disponible():
        break
    foto = PROCESADAS / archivo
    crop = foto.with_name(foto.stem + "_crop.jpg")   # la foto principal lleva la caja dibujada
    img = cv2.imread(str(crop))
    if img is None:
        continue
    h, w = img.shape[:2]
    data = pr.leer_api(img, (0, 0, w, h))
    usadas += 1
    if not data:
        continue
    placa, conf = pr._elegir(data, ocr._en_flota)
    if not placa:
        continue
    bruta = pr._con_guion(str(data["results"][0].get("plate", "")).upper())
    nueva = pr._con_guion(placa)
    con.execute(
        "UPDATE registros SET placa=?, confianza=?, fuente='platerecognizer', "
        "provisional=0, placa_api=?, motivo_sin_api=NULL WHERE id=?",
        (nueva, conf, bruta, rid))
    if nueva != vieja:
        con.execute("UPDATE pasadas SET placa=? WHERE placa=? AND fecha_hora>=?",
                    (nueva, vieja, fecha[:10] + "T00:00:00"))
    con.commit()
    print(f"OK {archivo}: {vieja} -> {nueva}")
    time.sleep(0.5)

con.close()
print(f"llamadas usadas: {usadas}")