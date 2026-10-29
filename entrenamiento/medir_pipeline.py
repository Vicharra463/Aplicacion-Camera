# Banco de medidas del pipeline completo (leer_placa_6) sobre GT-29.
# Correr tras CADA cambio: python medir_pipeline.py [--dump]
import sys
from pathlib import Path

import cv2

PROYECTO = Path(r"D:\Aplicacion-Camera")
sys.path.insert(0, str(PROYECTO / "app"))
import ocr  # noqa: E402
from gt_placas import DIAGNOSTICOS, RAIZ  # noqa: E402

DUMP = "--dump" in sys.argv
DIR_DUMP = PROYECTO / "entrenamiento" / "eval_fallos"
if DUMP:
    DIR_DUMP.mkdir(exist_ok=True)


def canon(t: str) -> str:
    return (t or "").upper().replace("-", "").strip()


def dist(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def bbox_de(ruta: Path, ancho: int, alto: int):
    txt = ruta.with_suffix(".txt")
    if not txt.is_file():
        return None
    linea = next((ln.strip() for ln in
                  txt.read_text(encoding="utf-8").splitlines() if ln.strip()), "")
    p = linea.split()
    if len(p) < 5:
        return None
    cx, cy, w, h = (float(v) for v in p[1:5])
    if max(cx, cy, w, h) <= 1.0:               # normalizado -> píxeles
        cx, cy, w, h = cx * ancho, cy * alto, w * ancho, h * alto
    cx, cy, w, h = (int(round(v)) for v in (cx, cy, w, h))
    return (cx - w // 2, cy - h // 2, w, h)


casos = []
for dia, nombre, real, _nota in DIAGNOSTICOS:
    ruta = RAIZ / dia / nombre
    img = cv2.imread(str(ruta))
    if img is None:
        continue
    bbox = bbox_de(ruta, img.shape[1], img.shape[0])
    if bbox is None:
        continue
    casos.append((f"{dia}/{nombre}", img, bbox, real))

aciertos = 0
err, nch = 0, 0
fallos = []
por_real: dict[str, list[bool]] = {}
for caso, img, bbox, real in casos:
    texto, conf, registro = ocr.leer_placa_6(img, bbox)
    rc, lc = canon(real), canon(texto)
    err += dist(lc, rc)
    nch += len(rc)
    ok = lc == rc and bool(lc)
    aciertos += ok
    por_real.setdefault(rc, []).append(ok)
    if not ok:
        fallos.append((caso, real, texto, conf, registro))

unicas = len(por_real)
resueltas = sum(1 for v in por_real.values() if any(v))
cer = err / max(1, nch)
print(f"pipeline: {aciertos}/{len(casos)} fotos = {100*aciertos/len(casos):.1f}%   "
      f"placas unicas: {resueltas}/{unicas} = {100*resueltas/unicas:.1f}%   "
      f"CER={cer:.3f}")

print(f"\nfallos ({len(fallos)}):")
for caso, real, texto, conf, registro in fallos:
    rc, lc = canon(real), canon(texto)
    if not lc:
        tipo = "VACIO"
    elif len(lc) != len(rc):
        tipo = "LARGO"
    elif dist(lc, rc) == 1:
        tipo = "1CHAR"
    else:
        tipo = f"MULTI({dist(lc, rc)})"
    print(f"  [{tipo}] {caso:<32} real={real:<9} leido={texto or '(nada)':<10} conf={conf:.2f}")
    if DUMP:
        resumen = [f"{n}={t or '-'}({c:.2f})" for n, t, c in registro
                   if n in ("crnn", "crnn_flota", "snap_flota", "recorte",
                            "clahe_gamma", "margen+5%",
                            "margen+10%", "upscale2x", "noche", "umbral_inv",
                            "decision", "arbitraje", "arbitraje_descartado",
                            "desempate")]
        print(f"         {' | '.join(resumen)}")
        img = dict((c, i) for c, i, _b, _r in casos)[caso]
        _x, _y, w, h = dict((c, b) for c, _i, b, _r in casos)[caso]
        base = ocr._recorte(img, dict((c, b) for c, _i, b, _r in casos)[caso], 0.35)
        if base is not None:
            cv2.imwrite(str(DIR_DUMP / f"PIPELINE__real-{real}__leido-{texto or 'nada'}__{caso.replace('/', '_')}.png"), base)
