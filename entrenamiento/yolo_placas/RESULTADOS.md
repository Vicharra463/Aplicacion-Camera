# Entrenamiento YOLOv8 — detección de placas peruanas

Dataset: Roboflow *Peru License Plate v9* (2626 imágenes, 1 clase `placa`;
train 2445 / valid 128 / test 53). El `data.yaml` original de Roboflow trae
rutas `../train/images` que no existen al descomprimir: usar
`data_placas.yaml` (rutas absolutas).

Receta: imgsz 640, 100 épocas, batch 32, AdamW (auto), workers 4, RTX 3060.

## Resultados en GT-29 (29 fotos de producción, verdad visual)

| Modelo | Cadena local (CRNN+RapidOCR) | Cadena API (detectar→recortar→PlateRecognizer) |
|---|---|---|
| yolov8n (nano) | 25/29 @umbral 0.25 · 26/29 @0.05 | — |
| **yolov8s (small)** | **28/29 = 96.6%** | **26/29 = 89.7%** |
| `modelo/placas.onnx` (prod) | 28/29 = 96.6% | — |

- **Techo real = 28/29**: `1G-777` es ilegible en la foto; ningún detector,
  umbral ni lector la salva.
- Cadena API del small: 26 resueltas vía API (24 correctas), 2 vía motor
  local, 1 sin lectura.
- Fallos API: `871-923` (API devolvió B7I-923), `A7-935` (leyó TVI-986,
  placa del fondo), `1G-777` (el small no la detecta, cand=0).
- El small detectó las3 fotos duras que el nano no veía (AFH-559, 4O-820,
  E5A947): la cadena API las leyó bien.

Latencias CPU 1 hilo (informativo): prod 90 ms · nano 92 ms · small 255 ms.
Cuantización INT8 dinámico descartada: 3795 ms (kernels de referencia).

## Cómo correr

```bat
:: entrenar (python con torch CUDA)
python train.py yolov8n.pt yolov8n
python train.py yolov8s.pt yolov8s

:: exportar ONNX (opset 12, mismo contrato que DetectorPlacas)
python -c "from ultralytics import YOLO; YOLO(r'runs\yolov8s\weights\best.pt').export(format='onnx', imgsz=640, opset=12)"

:: evaluar (venv de produccion, con onnxruntime)
venv\Scripts\python.exe evaluar_candidato.py runs\yolov8s\weights\best.onnx
venv\Scripts\python.exe evaluar_candidato.py runs\yolov8s\weights\best.onnx --umbral 0.05
venv\Scripts\python.exe evaluar_api.py runs\yolov8s\weights\best.onnx
```

## Artefactos (NO versionados)

Pesos y runs quedan en esta máquina, fuera de git:

- `runs/yolov8n/weights/best.{pt,onnx}` — nano (12.7 MB ONNX)
- `runs/yolov8s/weights/best.{pt,onnx}` — small (44.7 MB ONNX)
- `runs/yolov8s/weights/best_int8.onnx` — INT8 dinámico (descartado)

Para regenerar: `train.py` + `data_placas.yaml` (la receta está arriba).
