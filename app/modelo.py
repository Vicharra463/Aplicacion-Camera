#!/usr/bin/env python
"""Fase 2: deteccion de placas con ONNX dentro del servicio.

Reemplaza al detector geometrico de ``worker._candidatos_placa()`` cuando
existe ``modelo\\placas.onnx``. Usa onnxruntime (CPU) y NO importa torch:
el servicio no arrastra el framework de entrenamiento, solo los pesos.

Si el archivo no existe o algo falla al cargar, worker sigue con la geometria
pura: el servicio nunca se cae por el modelo.

Uso directo (prueba sin levantar el servidor):

    python modelo.py ruta\\a\\imagen.jpg
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("cytv.modelo")

RUTA_ONNX = Path(__file__).resolve().parents[1] / "modelo" / "placas.onnx"
FONDO = 114          
IOU = 0.45           
MAX_CANDIDATOS = 8   







ZONAS_OSD = (
    (0.00, 0.00, 0.34, 0.12),    
    (0.70, 0.85, 1.00, 1.00),    
)


def enmascarar_osd(img: np.ndarray) -> np.ndarray:
    """Devuelve una copia con las zonas del OSD en negro (solo para detectar)."""
    salida = img.copy()
    alto, ancho = salida.shape[:2]
    for x0, y0, x1, y1 in ZONAS_OSD:
        salida[int(y0 * alto):int(y1 * alto), int(x0 * ancho):int(x1 * ancho)] = 0
    return salida


class DetectorPlacas:
    """Carga el ONNX una vez y lo reutiliza en cada imagen."""

    def __init__(self, ruta: Path, umbral: float = 0.25) -> None:
        import onnxruntime as ort  

        self.umbral = float(umbral)
        sesion_op = ort.SessionOptions()
        sesion_op.intra_op_num_threads = 1
        sesion_op.inter_op_num_threads = 1
        sesion_op.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sesion_op.enable_cpu_mem_arena = False
        sesion_op.log_severity_level = 4
        self.sesion = ort.InferenceSession(
            str(ruta), sess_options=sesion_op,
            providers=["CPUExecutionProvider"])
        entrada = self.sesion.get_inputs()[0]
        self.nombre_entrada = entrada.name
        forma = entrada.shape                      
        lado = forma[2] if len(forma) == 4 else 640
        self.lado = int(lado) if isinstance(lado, (int, float)) else 640
        logger.info("Modelo ONNX cargado: %s (entrada %dx%d, umbral %.2f)",
                    ruta.name, self.lado, self.lado, self.umbral)

    
    def detectar(self, img: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        """Devuelve [((x, y, w, h), confianza)] en coordenadas de la imagen."""
        alto, ancho = img.shape[:2]
        lado = self.lado

        
        
        entrada = enmascarar_osd(img)

        
        
        escala = min(lado / ancho, lado / alto)
        nw, nh = max(1, int(round(ancho * escala))), max(1, int(round(alto * escala)))
        redim = cv2.resize(entrada, (nw, nh), interpolation=cv2.INTER_LINEAR)
        lienzo = np.full((lado, lado, 3), FONDO, dtype=np.uint8)
        dx, dy = (lado - nw) // 2, (lado - nh) // 2
        lienzo[dy:dy + nh, dx:dx + nw] = redim

        blob = lienzo[:, :, ::-1].transpose(2, 0, 1)          
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0

        salida = self.sesion.run(None, {self.nombre_entrada: blob[None]})[0]
        return self._postprocesar(salida, escala, dx, dy, ancho, alto)

    
    def _postprocesar(self, salida: np.ndarray, escala: float,
                      dx: int, dy: int, ancho: int, alto: int):
        
        cajas = salida[0]
        if cajas.shape[0] < cajas.shape[1]:
            cajas = cajas.T

        xywh = cajas[:, :4]
        puntuaciones = cajas[:, 4:]
        conf = puntuaciones.max(axis=1) if puntuaciones.shape[1] else np.zeros(0)

        
        
        if conf.size and (conf.min() < 0.0 or conf.max() > 1.0):
            conf = 1.0 / (1.0 + np.exp(-conf))

        candidatos = conf >= self.umbral
        if not candidatos.any():
            return []

        cuadros = xywh[candidatos]
        puntajes = conf[candidatos]
        
        x0 = np.clip(cuadros[:, 0] - cuadros[:, 2] / 2.0, 0, self.lado)
        y0 = np.clip(cuadros[:, 1] - cuadros[:, 3] / 2.0, 0, self.lado)
        boxes = np.stack([x0, y0, cuadros[:, 2], cuadros[:, 3]], axis=1)

        indice = cv2.dnn.NMSBoxes(boxes.astype(np.float32).tolist(),
                                  puntajes.astype(np.float32).tolist(),
                                  self.umbral, IOU)
        if indice is None or len(indice) == 0:
            return []

        resultados = []
        for i in np.array(indice).flatten()[:MAX_CANDIDATOS]:
            bx, by, bw, bh = boxes[i]
            
            x = int(round((bx - dx) / escala))
            y = int(round((by - dy) / escala))
            w = int(round(bw / escala))
            h = int(round(bh / escala))
            x, y = max(0, x), max(0, y)
            w, h = min(w, ancho - x), min(h, alto - y)
            if w <= 0 or h <= 0:
                continue
            resultados.append(((x, y, w, h), round(float(puntajes[i]), 3)))

        resultados.sort(key=lambda par: par[1], reverse=True)
        return resultados





def _probar(ruta_imagen: str) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(message)s")
    if not RUTA_ONNX.is_file():
        print(f"[ERROR] No existe {RUTA_ONNX}")
        return 1

    img = cv2.imread(ruta_imagen)
    if img is None:
        print(f"[ERROR] No se pudo leer {ruta_imagen}")
        return 1

    detector = DetectorPlacas(RUTA_ONNX)
    detecciones = detector.detectar(img)
    print(f"{Path(ruta_imagen).name}: {len(detecciones)} deteccion(es)")
    for caja, conf in detecciones:
        x, y, w, h = caja
        print(f"   conf={conf:.3f}  caja=({x},{y},{w},{h})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python modelo.py ruta\\a\\imagen.jpg")
        raise SystemExit(2)
    raise SystemExit(_probar(sys.argv[1]))
