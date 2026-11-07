"""Script auxiliar e independiente para generar imágenes con la traza visual
(caja de YOLO, franja verde y colores) sin tocar el código de producción.

Uso:
    venv\Scripts\python.exe scripts\auditar_detecciones.py <ruta_imagen_o_carpeta>
"""
from __future__ import annotations

import sys
from pathlib import Path

PROYECTO = Path(r"D:\Aplicacion-Camera")
sys.path.insert(0, str(PROYECTO))
sys.path.insert(0, str(PROYECTO / "app"))

import cv2
import numpy as np
from modelo import DetectorPlacas, RUTA_ONNX
import worker

def color_caja_local(texto_sufijo: str) -> tuple[int, int, int]:
    if "Roboflow" in texto_sufijo:
        return (0, 255, 0)     # Verde
    if "PlateRecognizer" in texto_sufijo:
        return (0, 255, 255)   # Amarillo
    return (255, 0, 0)         # Azul

def auditar_imagen(ruta_img: Path, salida_dir: Path, detector: DetectorPlacas | None) -> None:
    img = cv2.imread(str(ruta_img), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[ERROR] No se pudo leer la imagen: {ruta_img.name}")
        return

    # Intentar buscar el archivo .txt de YOLO gemelo si existe (para pintar la caja real guardada)
    txt_yolo = ruta_img.with_suffix(".txt")
    alto, ancho = img.shape[:2]
    
    anotada = img.copy()
    
    # Si hay archivo YOLO guardado, dibujamos la caja guardada
    if txt_yolo.is_file():
        try:
            partes = txt_yolo.read_text(encoding="utf-8").split()
            if len(partes) >= 5:
                _, x_cent, y_cent, w_norm, h_norm = map(float, partes[:5])
                w = int(w_norm * ancho)
                h = int(h_norm * alto)
                x = int((x_cent * ancho) - (w / 2))
                y = int((y_cent * alto) - (h / 2))
                
                # Dibujar rectángulo
                cv2.rectangle(anotada, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(anotada, "YOLO GT/Detectado", (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except Exception:
            pass

    # Si no hay txt pero tenemos el detector ONNX cargado, detectamos al vuelo
    elif detector is not None:
        detecciones = detector.detectar(img)
        for (x, y, w, h), conf in detecciones:
            cv2.rectangle(anotada, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(anotada, f"Conf: {conf:.2f}", (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Guardar resultado en la carpeta de auditoría local
    salida_dir.mkdir(parents=True, exist_ok=True)
    destino = salida_dir / f"{ruta_img.stem}_auditada.jpg"
    cv2.imwrite(str(destino), anotada, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"[OK] Auditada generada: {destino}")

def main() -> int:
    if len(sys.argv) < 2:
        print("Uso: venv\\Scripts\\python.exe scripts\\auditar_detecciones.py <imagen_o_carpeta>")
        return 2

    objetivo = Path(sys.argv[1]).resolve()
    salida = PROYECTO / "reportes" / "auditoria_visual"
    
    detector = None
    if RUTA_ONNX and RUTA_ONNX.is_file():
        try:
            detector = DetectorPlacas(RUTA_ONNX, umbral=0.4)
        except Exception:
            pass

    if objetivo.is_file():
        auditar_imagen(objetivo, salida, detector)
    elif objetivo.is_dir():
        for foto in objetivo.rglob("*.jpg"):
            if not foto.stem.endswith("_crop") and not foto.stem.endswith("_vista"):
                auditar_imagen(foto, salida, detector)
    else:
        print(f"[ERROR] Ruta no encontrada: {objetivo}")
        return 1

    print(f"\n¡Listo! Revisa las imágenes procesadas en: {salida}")
    return 0

if __name__ == "__main__":
    sys.exit(main())