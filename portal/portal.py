#!/usr/bin/env python3
"""Portal web de diagnostico: subi una foto y mira como piensa el pipeline.

Muestra deteccion, filtros de calidad, intentos OCR con confianza,
consenso CRNN, evidencia y semaforo final. Sin dependencias nuevas.

Uso: python portal/portal.py [--puerto 8090]
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "app"))

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("cytv.portal")

_detector = None
_carga: dict = {}
_candado = threading.Lock()
MAX_BYTES = 25 * 1024 * 1024


def _cargar_modelos() -> None:
    global _detector
    t0 = time.perf_counter()
    import worker
    from modelo import DetectorPlacas, RUTA_ONNX
    _detector = DetectorPlacas(RUTA_ONNX, umbral=0.40)
    worker._detector = _detector
    _detector.detectar(np.zeros((720, 1280, 3), dtype=np.uint8))
    import ocr
    ocr._obtener_motor()
    import reconocedor
    reconocedor._obtener_sesion()
    _carga["segundos"] = round(time.perf_counter() - t0, 1)
    logger.info("modelos cargados en %.1fs", _carga["segundos"])


def _b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _dibujar(img: np.ndarray, bboxes: list, mejor_idx: int) -> np.ndarray:
    salida = img.copy()
    for i, (x, y, w, h) in enumerate(bboxes):
        color = (0, 255, 0) if i == mejor_idx else (0, 200, 255)
        grosor = max(3, round(min(img.shape[:2]) / 400))
        cv2.rectangle(salida, (x, y), (x + w, y + h), color, grosor)
        cv2.putText(salida, f"#{i} {color == (0, 255, 0) and 'MEJOR' or ''}",
                    (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    grosor * 0.6, color, grosor)
    return salida


def _crop_zoom(img: np.ndarray, bbox: tuple) -> np.ndarray:
    import ocr
    r = ocr._recorte(img, tuple(bbox), 0.35)
    if r is None or not r.size:
        return img
    escala = max(1.0, min(4.0, 720 / max(1, r.shape[1])))
    if escala > 1.0:
        r = cv2.resize(r, None, fx=escala, fy=escala,
                       interpolation=cv2.INTER_CUBIC)
    return r


def analizar(datos: bytes) -> dict:
    import worker
    import ocr
    tiempos: dict[str, float] = {}

    t0 = time.perf_counter()
    arr = np.frombuffer(datos, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "error": "no se pudo decodificar la imagen"}
    tiempos["decode"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    candidatos = worker._candidatos_placa(img)
    tiempos["deteccion"] = round((time.perf_counter() - t0) * 1000, 1)
    detecciones = [{"bbox": [int(v) for v in b], "conf": round(c, 3)}
                   for b, c in candidatos]

    resultado: dict = {
        "ok": True,
        "foto": {"ancho": img.shape[1], "alto": img.shape[0],
                 "bytes": len(datos)},
        "detecciones": detecciones,
        "mejor": None,
        "imagenes": {},
        "tiempos_ms": tiempos,
    }
    if not candidatos:
        resultado["mensaje"] = "sin detecciones: el modelo no vio ninguna placa"
        resultado["imagenes"]["anotada"] = _b64(img)
        return resultado

    orden = sorted(range(len(candidatos)),
                   key=lambda i: candidatos[i][0][2] * candidatos[i][0][3],
                   reverse=True)
    idx = orden[0]
    bbox, conf_det = candidatos[idx]
    bbox = tuple(int(v) for v in bbox)

    x, y, w, h = bbox
    recorte = img[y:y + h, x:x + w]
    filtros = {
        "nitidez": round(worker._nitidez(recorte), 1),
        "nitidez_min": 120.0,
        "bordes": round(worker._fraccion_bordes(recorte), 2),
        "bordes_min": 2.0,
        "sobreexpuesta_pct": round(worker._fraccion_sobreexpuesta(recorte), 2),
        "sobreexpuesta_max": 2.0,
        "oscuros_pct": round(worker._fraccion_oscuros(recorte), 1),
        "toca_borde": bool(x <= 4 or y <= 4 or x + w >= img.shape[1] - 4
                           or y + h >= img.shape[0] - 4),
    }

    t0 = time.perf_counter()
    consenso = ocr._crnn_consenso(img, bbox, 0.35)
    tiempos["crnn_consenso"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    texto, conf, intentos = ocr.leer_placa_6(img, bbox)
    tiempos["ocr_total"] = round((time.perf_counter() - t0) * 1000, 1)

    evidencia, semaforo = ocr.evidencia_y_color(texto, conf, intentos)

    resultado["mejor"] = {
        "bbox": list(bbox),
        "conf_detector": round(conf_det, 3),
        "filtros": filtros,
        "consenso_crnn": (None if not consenso else {
            "texto": consenso["texto"],
            "normal": consenso["normal"],
            "conf": round(consenso["conf"], 3),
            "conf_min": round(consenso["conf_min"], 3),
            "acuerdo": round(consenso["acuerdo"], 2),
        }),
        "intentos": [{"nombre": n, "texto": t or "(sin lectura)",
                      "conf": round(c, 3)} for n, t, c in intentos],
        "resultado": {"texto": texto or "(sin placa)", "conf": round(conf, 3),
                      "evidencia": evidencia, "semaforo": semaforo},
    }
    resultado["imagenes"]["anotada"] = _b64(_dibujar(img, [b for b, _ in
                                                          candidatos], idx))
    resultado["imagenes"]["crop"] = _b64(_crop_zoom(img, bbox))
    tiempos["total"] = round(sum(tiempos.values()), 1)
    return resultado


HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CYTV - Mirada del modelo</title>
<style>
:root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3;
        --verde:#3fb950; --amarillo:#d29922; --rojo:#f85149; --azul:#58a6ff; }
* { box-sizing:border-box; margin:0; }
body { background:var(--bg); color:var(--text); font-family:Consolas,'Courier New',monospace; padding:18px; }
h1 { font-size:20px; letter-spacing:1px; } h1 span { color:var(--azul); }
.sub { color:#8b949e; font-size:12px; margin:4px 0 16px; }
.zona { border:2px dashed var(--border); border-radius:10px; padding:28px;
        text-align:center; color:#8b949e; cursor:pointer; transition:.15s; }
.zona:hover, .zona.arrastre { border-color:var(--azul); color:var(--azul); }
input[type=file] { display:none; }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:16px; }
.panel { background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:12px; }
.panel h2 { font-size:13px; color:var(--azul); margin-bottom:8px; text-transform:uppercase; letter-spacing:1px; }
img { max-width:100%; border-radius:6px; }
table { width:100%; border-collapse:collapse; font-size:12px; }
td, th { border-bottom:1px solid var(--border); padding:4px 6px; text-align:left; }
.valor { font-size:26px; font-weight:bold; }
.verde { color:var(--verde); } .amarillo { color:var(--amarillo); } .rojo { color:var(--rojo); }
.chip { display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; margin:2px; }
.pasa { background:#12351b; color:var(--verde); } .falla { background:#3d1418; color:var(--rojo); }
.fila { display:flex; justify-content:space-between; font-size:12px; padding:3px 0; border-bottom:1px solid #21262d; }
.oculto { display:none; }
.carga { text-align:center; color:var(--azul); padding:30px; font-size:14px; }
</style>
</head>
<body>
<h1><span>CYTV</span> - Mirada del modelo</h1>
<div class="sub">Subi una foto de la camara y mira paso a paso como el pipeline decide la placa.</div>
<label class="zona" id="zona">
  <input type="file" id="archivo" accept="image/*">
  arrastra una foto aqui o hace clic para elegirla
</label>
<div id="carga" class="carga oculto">analizando... (detector + 8 intentos OCR + consenso)</div>
<div id="resultado"></div>
<script>
const zona = document.getElementById('zona');
const input = document.getElementById('archivo');
const carga = document.getElementById('carga');
const res = document.getElementById('resultado');
zona.addEventListener('dragover', e => { e.preventDefault(); zona.classList.add('arrastre'); });
zona.addEventListener('dragleave', () => zona.classList.remove('arrastre'));
zona.addEventListener('drop', e => { e.preventDefault(); zona.classList.remove('arrastre');
  if (e.dataTransfer.files.length) enviar(e.dataTransfer.files[0]); });
input.addEventListener('change', () => { if (input.files.length) enviar(input.files[0]); });

async function enviar(file) {
  carga.classList.remove('oculto'); res.innerHTML = '';
  const buf = await file.arrayBuffer();
  let r;
  try {
    const resp = await fetch('/analizar', { method:'POST',
      headers:{'Content-Type':'application/octet-stream'}, body: buf });
    r = await resp.json();
  } catch (e) { carga.classList.add('oculto');
    res.innerHTML = '<div class="panel rojo">error: ' + e + '</div>'; return; }
  carga.classList.add('oculto');
  render(r);
}

function render(r) {
  if (!r.ok) { res.innerHTML = '<div class="panel rojo">' + (r.error||'fallo') + '</div>'; return; }
  let html = '';
  if (r.mensaje) html += '<div class="panel"><span class="rojo valor">SIN DETECCION</span><div class="sub">' + r.mensaje + '</div></div>';
  html += '<div class="grid">';
  html += '<div class="panel"><h2>foto ' + r.foto.ancho + 'x' + r.foto.alto + '</h2>' +
          (r.imagenes.anotada ? '<img src="' + r.imagenes.anotada + '">' : '') + '</div>';
  if (r.imagenes.crop)
    html += '<div class="panel"><h2>crop de la placa (zoom)</h2><img src="' + r.imagenes.crop + '"></div>';
  html += '</div>';
  if (r.detecciones.length) {
    html += '<div class="panel" style="margin-top:14px"><h2>detecciones (' + r.detecciones.length + ')</h2><table><tr><th>#</th><th>bbox x,y,w,h</th><th>conf</th></tr>';
    r.detecciones.forEach((d,i) => { html += '<tr><td>' + i + '</td><td>' + d.bbox.join(', ') + '</td><td>' + d.conf + '</td></tr>'; });
    html += '</table></div>';
  }
  if (r.mejor) {
    const m = r.mejor, f = m.filtros, resu = m.resultado;
    html += '<div class="grid" style="margin-top:14px">';
    html += '<div class="panel"><h2>filtros de calidad</h2>' +
      fila('nitidez', f.nitidez + ' (min ' + f.nitidez_min + ')', f.nitidez >= f.nitidez_min) +
      fila('bordes', f.bordes + '% (min ' + f.bordes_min + ')', f.bordes >= f.bordes_min) +
      fila('saturados', f.sobreexpuesta_pct + '% (max ' + f.sobreexpuesta_max + ')', f.sobreexpuesta_pct <= f.sobreexpuesta_max) +
      fila('toca borde', f.toca_borde ? 'SI' : 'no', !f.toca_borde) +
      '</div>';
    let cr = m.consenso_crnn;
    html += '<div class="panel"><h2>consenso CRNN (5 crops)</h2>' +
      (cr ? fila('texto', cr.texto) + fila('canonico', cr.normal) +
            fila('confianza media', cr.conf) + fila('confianza minima', cr.conf_min) +
            fila('acuerdo crops', cr.acuerdo)
          : '<div class="sub">CRNN no alcanzo consenso</div>') + '</div>';
    html += '</div>';
    html += '<div class="panel" style="margin-top:14px"><h2>intentos OCR (como vota)</h2><table><tr><th>variante</th><th>texto</th><th>conf</th></tr>';
    m.intentos.forEach(it => { html += '<tr><td>' + it.nombre + '</td><td>' + it.texto + '</td><td>' + it.conf + '</td></tr>'; });
    html += '</table></div>';
    const cls = resu.semaforo;
    html += '<div class="panel" style="margin-top:14px"><h2>resultado final</h2>' +
      '<div class="valor ' + cls + '">' + resu.texto + '</div>' +
      '<div class="sub">confianza ' + resu.conf + ' &middot; evidencia ' + resu.evidencia + ' &middot; semaforo <b class="' + cls + '">' + resu.semaforo + '</b></div></div>';
  }
  const t = r.tiempos_ms;
  html += '<div class="panel" style="margin-top:14px"><h2>tiempos (ms)</h2><div class="sub">' +
    Object.keys(t).map(k => k + '=' + t[k]).join(' &middot; ') + '</div></div>';
  res.innerHTML = html;
}
function fila(k, v, ok) {
  const badge = (ok === undefined) ? '' :
    (ok ? ' <span class="chip pasa">PASA</span>' : ' <span class="chip falla">FALLA</span>');
  return '<div class="fila"><span>' + k + '</span><span>' + v + badge + '</span></div>';
}
</script>
</body>
</html>
"""


class Portal(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _responder(self, code: int, cuerpo: bytes, tipo: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._responder(200, HTML.encode("utf-8"),
                            "text/html; charset=utf-8")
        else:
            self._responder(404, b"no encontrado", "text/plain")

    def do_POST(self):
        if self.path != "/analizar":
            self._responder(404, b"no encontrado", "text/plain")
            return
        largo = int(self.headers.get("Content-Length", 0))
        if not 0 < largo <= MAX_BYTES:
            self._responder(400, b"tamanio invalido", "text/plain")
            return
        datos = self.rfile.read(largo)
        try:
            with _candado:
                salida = analizar(datos)
        except Exception as exc:
            logger.exception("fallo el analisis")
            salida = {"ok": False, "error": str(exc)}
        cuerpo = json.dumps(salida, ensure_ascii=False).encode("utf-8")
        self._responder(200, cuerpo, "application/json; charset=utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto", type=int, default=8090)
    args = ap.parse_args()
    print("cargando modelos (detector + RapidOCR + CRNN)...", flush=True)
    _cargar_modelos()
    servidor = ThreadingHTTPServer(("0.0.0.0", args.puerto), Portal)
    print(f"portal en http://0.0.0.0:{args.puerto}  (Ctrl+C para cortar)",
          flush=True)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
