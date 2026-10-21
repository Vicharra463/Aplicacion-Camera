#!/usr/bin/env python
"""Depurador visual del pipeline de placas CYTV (solo local).

Levanta una pagina web local (stdlib puro, sin dependencias nuevas) que
corre el worker REAL sobre una foto que subis y muestra paso a paso como
piensa y decide:

  1. candidatos del detector ONNX (todas las cajas, con confianza),
  2. filtros por candidato (borde, nitidez, sobreexpuesta, ocluida),
  3. intentos de cada motor OCR con texto y confianza,
  4. decision final real del worker (guardar / DUDOSA / duplicado / ...).

Modo aislado: NO escribe registro.db, NO guarda fotos y NO toca
imagenes_procesadas: los efectos de lado del worker quedan enchufados a
capturas en memoria. Los umbrales salen de config.ini del repo.

Uso:  ABRIR-DEPURADOR.bat   ->  http://127.0.0.1:8791
"""
from __future__ import annotations

import base64
import configparser
import json
import logging
import os
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "app"))

import cv2  # noqa: E402
import worker  # noqa: E402
from modelo import DetectorPlacas, RUTA_ONNX  # noqa: E402

PUERTO = 8791
LOCK = threading.Lock()
captura: dict | None = None  # buffers del request en curso (bajo LOCK)

# ------------------------------------------------------------------ config


def cargar_cfg() -> dict:
    ini = configparser.ConfigParser()
    ini.read(RAIZ / "config.ini", encoding="utf-8-sig")
    f = ini["filtros"]
    dup = ini["duplicados"]
    cam = ini["camara"]
    tmp = Path(tempfile.mkdtemp(prefix="depurador_"))
    return {
        "procesadas": tmp,
        "registro": str(tmp / "registro.db"),
        "umbral": float(dup.get("umbral_diferencia", 8.0)),
        "umbral_modelo": float(f.get("umbral_modelo", 0.35)),
        "umbral_texto_min": float(f.get("umbral_texto_min", 5.0)),
        "umbral_texto_max": float(f.get("umbral_texto_max", 55.0)),
        "umbral_nitidez": float(f.get("umbral_nitidez", 120.0)),
        "umbral_sobreexpuesto": float(f.get("umbral_sobreexpuesto", 2.0)),
        "umbral_bordes": float(f.get("umbral_bordes", 2.0)),
        "ocr_requerido": f.get("ocr_requerido", "si").strip().lower(),
        "modo_descarte": f.get("modo_descarte", "eliminar").strip().lower(),
        "ventana_reinspeccion_min": float(cam.get("ventana_reinspeccion_min", 60)),
        "ventana_reinspeccion_dias": int(cam.get("ventana_reinspeccion_dias", 30)),
        "max_pasadas": int(cam.get("max_pasadas", 3)),
    }


CFG = cargar_cfg()

# ------------------------------------------------------- estado del worker


def _stub_guardar(img, bbox, vista, texto_placa="", sufijo="", dudosa=False):
    captura["guardados"].append({
        "vista": vista, "placa": texto_placa, "sufijo": sufijo,
        "dudosa": bool(dudosa),
        "bbox": [int(v) for v in bbox] if bbox is not None else None,
    })
    return CFG["procesadas"] / "depuracion" / f"{texto_placa or 'pano'}{sufijo}.jpg"


def _stub_descartar(ruta, motivo, detalle="", auditar=True, img_anotada=None):
    captura["descartes"].append({"motivo": motivo, "detalle": detalle or ""})


def _stub_registrar(archivo, placa=None, confianza=None, estado="guardado",
                    phash=None, evidencia=None, semaforo=None):
    captura["reg"].append({
        "archivo": archivo, "placa": placa,
        "confianza": None if confianza is None else round(float(confianza), 3),
        "estado": estado, "evidencia": evidencia, "semaforo": semaforo,
    })


_l6_real = worker.leer_placa_6


def _l6_traza(img, bbox):
    texto, conf, intentos = _l6_real(img, bbox)
    captura["intentos"].append({
        "bbox": [int(v) for v in bbox], "texto": texto,
        "conf": round(float(conf), 3),
        "intentos": [[n, t or "", round(float(c), 3)] for n, t, c in intentos],
    })
    return texto, conf, intentos


worker._guardar = _stub_guardar
worker._descartar = _stub_descartar
worker.registrar = _stub_registrar
worker._es_misma_placa = lambda *a, **k: False
worker._pasada_de_placa = lambda *a, **k: (1, float("inf"))
worker._registrar_pasada = lambda *a, **k: None
worker._encolar_entrenamiento = lambda *a, **k: None
worker.leer_placa_6 = _l6_traza
worker._cfg = CFG
worker._detector = DetectorPlacas(RUTA_ONNX, umbral=CFG["umbral_modelo"])


class _CapturaLog(logging.Handler):
    def emit(self, record):
        if captura is not None:
            try:
                captura["log"].append(f"{record.levelname}: {record.getMessage()}")
            except Exception:
                pass


logging.getLogger().addHandler(_CapturaLog())
logging.getLogger().setLevel(logging.DEBUG)


# ------------------------------------------------------------ el analisis


def _b64(img) -> str:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def _motivo_filtro(f: dict) -> str:
    if f["toca_borde"]:
        return "toca el borde"
    if not f["nitidez_ok"]:
        return f"baja nitidez ({f['nitidez']})"
    if not f["sobreexp_ok"]:
        return f"sobreexpuesta ({f['sobreexp']}%)"
    if not f["bordes_ok"]:
        return f"ocluida ({f['bordes']}%)"
    return ""


def analizar(ruta: Path) -> dict:
    global captura
    t0 = time.time()
    img = cv2.imread(str(ruta))
    if img is None:
        return {"ok": False, "error": "No se pudo leer la imagen"}

    with LOCK:
        captura = {"intentos": [], "descartes": [], "guardados": [],
                   "reg": [], "log": []}
        try:
            # 1) candidatos reales + espejo de filtros (metricas reales del worker)
            cands = worker._candidatos_placa(img)
            cands = sorted(cands, key=lambda p: p[0][2] * p[0][3], reverse=True)
            alto, ancho = img.shape[:2]
            filas = []
            for i, (bb, sc) in enumerate(cands):
                x, y, w, h = [int(v) for v in bb]
                rec = img[y:y + h, x:x + w]
                if rec.size == 0:
                    continue
                nit = worker._nitidez(rec)
                que = worker._fraccion_sobreexpuesta(rec)
                bor = worker._fraccion_bordes(rec)
                filas.append({
                    "i": i, "bbox": [x, y, w, h], "area": w * h,
                    "conf_det": round(float(sc), 3),
                    "toca_borde": bool(x <= 4 or y <= 4 or x + w >= ancho - 4
                                        or y + h >= alto - 4),
                    "nitidez": round(float(nit), 1),
                    "nitidez_ok": nit >= CFG["umbral_nitidez"],
                    "sobreexp": round(float(que), 1),
                    "sobreexp_ok": que <= CFG["umbral_sobreexpuesto"],
                    "bordes": round(float(bor), 1),
                    "bordes_ok": bor >= CFG["umbral_bordes"],
                    "probado": False, "ganador": False, "motivo_filtro": "",
                })
                filas[-1]["motivo_filtro"] = (
                    _motivo_filtro(filas[-1]) if not filas[-1]["nitidez_ok"]
                    or not filas[-1]["sobreexp_ok"] or not filas[-1]["bordes_ok"]
                    or filas[-1]["toca_borde"] else "")

            # 2) el worker REAL decide (efectos de lado capturados)
            try:
                worker.procesar_imagen(ruta)
                error = None
            except Exception:
                error = traceback.format_exc()

            # 3) marcar candidatos usados/ganador segun lo que paso
            for it in captura["intentos"]:
                for f in filas:
                    if f["bbox"] == it["bbox"]:
                        f["probado"] = True
            if captura["guardados"]:
                bb = captura["guardados"][-1].get("bbox")
                for f in filas:
                    if f["bbox"] == bb:
                        f["ganador"] = True

            # 4) veredicto
            veredicto = None
            if captura["guardados"]:
                g = captura["guardados"][-1]
                r = captura["reg"][-1] if captura["reg"] else {}
                veredicto = {
                    "tipo": "guardada_dudosa" if g["dudosa"] else "guardada",
                    "placa": (g["placa"] + g["sufijo"]) or "",
                    "evidencia": r.get("evidencia"), "semaforo": r.get("semaforo"),
                    "conf": r.get("confianza"),
                    "detalle": "reevaluacion x10 (rescatada)" if g["dudosa"] else
                               "primera pasada",
                }
            elif captura["descartes"]:
                d = captura["descartes"][-1]
                veredicto = {"tipo": d["motivo"], "detalle": d["detalle"],
                             "placa": "", "evidencia": None, "semaforo": None,
                             "conf": None}
            else:
                veredicto = {"tipo": "sin_decision", "detalle":
                             "el worker no produjo veredicto (ver error/log)",
                             "placa": "", "evidencia": None, "semaforo": None,
                             "conf": None}

            # 5) dibujo de candidatos
            salida = img.copy()
            esc = max(1, round(min(alto, ancho) / 700))
            for f in filas:
                x, y, w, h = f["bbox"]
                if f["ganador"]:
                    color, grosor = (46, 200, 70), 4 * esc
                    etq = f"GANADOR {veredicto.get('placa') or ''}".strip()
                elif f["probado"]:
                    color, grosor = (0, 200, 255), 3 * esc
                    etq = f"#{f['i']} probado"
                else:
                    color, grosor = (40, 40, 230), 2 * esc
                    etq = f"#{f['i']} {f['motivo_filtro'] or 'no probado'}"
                cv2.rectangle(salida, (x, y), (x + w, y + h), color, grosor)
                ty = max(14 * esc, y - 6 * esc)
                cv2.putText(salida, etq, (x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55 * esc, color, max(1, 2 * esc), cv2.LINE_AA)
            if not filas:
                cv2.putText(salida, "SIN CANDIDATOS (detector umbral "
                            f"{CFG['umbral_modelo']})", (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8 * esc, (40, 40, 230), 2,
                            cv2.LINE_AA)

            return {
                "ok": True, "ms": int((time.time() - t0) * 1000),
                "img": _b64(salida),
                "candidatos": filas, "intentos": captura["intentos"],
                "veredicto": veredicto,
                "config": {k: (str(v) if isinstance(v, Path) else v)
                           for k, v in CFG.items()},
                "log": captura["log"][-160:], "error": error,
            }
        finally:
            captura = None


# ------------------------------------------------------------------ web


INDEX = Path(__file__).with_name("index.html")


class Manejador(BaseHTTPRequestHandler):
    def _enviar(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._enviar(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/health":
            self._enviar(200, b"ok", "text/plain")
        else:
            self._enviar(404, b"no encontrado", "text/plain")

    def do_POST(self):
        if self.path != "/api/evaluar":
            self._enviar(404, b"no encontrado", "text/plain")
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            datos = self.rfile.read(n)
            if not datos:
                self._enviar(400, b"vacio", "text/plain")
                return
            fd, ruta = tempfile.mkstemp(prefix="foto_", suffix=".jpg")
            os.close(fd)
            Path(ruta).write_bytes(datos)
            try:
                res = analizar(Path(ruta))
            finally:
                Path(ruta).unlink(missing_ok=True)
            cuerpo = json.dumps(res, ensure_ascii=False).encode("utf-8")
            self._enviar(200, cuerpo, "application/json; charset=utf-8")
        except Exception:
            err = {"ok": False, "error": traceback.format_exc()}
            self._enviar(500, json.dumps(err).encode("utf-8"),
                         "application/json; charset=utf-8")

    def log_message(self, *a):
        pass


def main() -> None:
    url = f"http://127.0.0.1:{PUERTO}"
    print(f"Depurador de placas CYTV -> {url}")
    print("Modo aislado: no guarda fotos ni toca registro.db. Ctrl+C = salir.")
    try:
        ThreadingHTTPServer(("127.0.0.1", PUERTO), Manejador).serve_forever()
    except OSError as e:
        print(f"[ERROR] No pude abrir el puerto {PUERTO}: {e}")
        print("¿Quizas ya esta corriendo? Cierra la otra instancia.")


if __name__ == "__main__":
    main()
