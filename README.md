# Cytv — Reconocimiento de placas desde CCTV

Cámara Hikvision → FTP → detección (YOLO ONNX) + OCR (RapidOCR + CRNN + cierre de flota) → SQLite → portal web.
Windows, solo CPU, sin GPU.

---

## Requisitos

- Windows 10/11 o Server (producción: Server 2012 R2 con Python 3.9)
- Python 3.13 (recomendado) — o Python 3.9 para el legacy
- ~2 GB de disco (modelos ONNX + dependencias)

---

## Instalación

```bat
cd /d D:\Aplicacion-Camera
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Windows legacy (Python 3.9):** usar `pip install -r requirements-py39.txt`
(fija `opencv-python-headless==4.5.3.56`, sin Media Foundation).

**Configuración:** editar `config.ini` — solo rutas **absolutas**.
Credenciales de la cámara: `servicio_local\INFO_CAMARA.txt` (nunca subir al repo).

**Servicio Windows (NSSM):**

```bat
scripts\instalar_servicio.bat          :: menú de instalación
scripts\instalar_servicio.bat config   :: ver el config.ini que usa el servicio
```

---

## Comandos

| Comando | Qué hace |
|---|---|
| `cytv.cmd` | Menú de utilidades: `estado`, `placas`, `historial`, `log`, `fotos`, `replay`, `entrenador`, `heatmap`, `servicios`, `config`, `limpiar`, `ayuda` |
| `scripts\consola.bat` | Consola interactiva del sistema |
| `scripts\reevaluar.bat --dia 2026-10-14 [--modo auto\|txt\|modelo]` | Reevalúa un día contra las etiquetas → OK/FALLO y % de acierto |
| `python entrenamiento\medir_pipeline.py [--dump]` | Banco GT-29: acierto por placa sobre las 29 fotos de referencia |
| `python entrenamiento\evaluacion\validar_diagnosticos.py [--detector txt\|modelo]` | Validador oficial (`modelo` = camino de producción) |
| `python portal\portal.py [--puerto 8090]` | Portal web de consultas |
| `python ftp_server.py` | Servicio FTP + pipeline (en producción lo corre NSSM) |
| `python entrenamiento\crnn\entrenar.py [--epocas 12]` | Entrenar CRNN con datos sintéticos |
| `python entrenamiento\crnn\entrenar_real.py [--epocas 20]` | Fine-tune con `etiquetas_reales.csv` |
| `python scripts\replay_descartes.py [--fecha YYYY-MM-DD] [--dry-run]` | Reprocesar fotos descartadas |
| `python scripts\heatmap_placas.py [--salida out.csv]` | Banda espacial típica de las placas |
| `cytv ayuda` | Lista completa de subcomandos |

---

## Estructura

```
D:\Aplicacion-Camera\
├── README.md                    este archivo
├── cytv.cmd                     lanzador del menú de utilidades
├── ftp_server.py                ÚNICA entrada del servicio (la referencia NSSM; no mover)
├── config.ini                   rutas absolutas de datos (lo lee ftp_server.py; no mover)
├── requirements.txt             Python 3.13
├── requirements-py39.txt        legacy Python 3.9
│
├── app\                         MOTOR OCR
│   ├── ocr.py                   pipeline: bbox → CRNN → votos → snap de flota
│   ├── modelo.py                detector YOLO (ONNX)
│   ├── reconocedor.py           CRNN (ONNX)
│   ├── decodificador_flota.py   diccionario de placas + corrección por vecinos
│   ├── worker.py                cola de fotos, reposo, prioridad de proceso
│   ├── consola.py               consola interactiva
│   └── gt_placas.py             GT-29 (29 fotos de referencia)
│
├── modelo\                      modelos activos
│   ├── placas.onnx / .pt        detector
│   ├── reconocedor.onnx + .data CRNN en producción
│   ├── respaldo\                backups históricos
│   └── candidatos\              experimentos no adoptados (48×240, duro)
│
├── portal\                      web (portal.py + logs)
├── servicio_local\              datos del servicio local (registro.db, config, INFO_CAMARA)
│
├── scripts\                     utilidades
│   ├── cytv.ps1                 implementación de cytv.cmd
│   ├── instalar*.ps1/.bat       instalación NSSM
│   ├── reevaluar.bat            reevaluar un día
│   ├── replay_descartes.py      reprocesar descartes
│   ├── heatmap_placas.py, diag_recursos.ps1, limpiar_*, reiniciar_*
│   └── nssm.exe
│
├── entrenamiento\               experimentos y datos
│   ├── pesos\                   pesos .pt del CRNN (placas, reales, ideales, 48, duro)
│   ├── respaldo\                backups de pesos y CSVs
│   ├── crnn\                    código del CRNN + entrenadores
│   ├── evaluacion\              validar_diagnosticos, reevaluar, eval_rapido, gap
│   ├── herramientas\            actualizar_data, exportar_data_dura
│   ├── detector\                entrenar_detector
│   ├── etiquetas_reales.csv     flota verificada a mano (datos, no mover)
│   ├── etiquetas_ideales.csv    transcripciones ideales del GT
│   └── medir_pipeline.py        banco oficial GT-29
│
├── imagenes_procesadas\         fotos por día (datos)
├── cola_entrenamiento\          logs del entrenamiento continuo
├── docs\                        README-DETALLE-TECNICO.md, GUIA-COMANDOS.txt
└── venv\                        entorno virtual dev (no mover)
```

**Por qué quedan 7 archivos en la raíz:** `.git` los exige (`.gitignore`, `.gitattributes`),
el instalador y NSSM apuntan a `ftp_server.py` + `config.ini`, `cytv.cmd` es el lanzador
documentado, y `requirements*.txt` es la convención de pip. Todo lo demás vive en carpetas.

---

## Métricas actuales (GT-29, 2026-10-22)

| Escenario | Acierto |
|---|---|
| Producción (detector ONNX + diccionario de flota) | **28/29 = 96,6 %** |
| Validación por bbox etiquetado (pipeline clásico) | 27/29 = 93,1 % |
| Sin diccionario de flota (placas nuevas para el sistema) | 14/29 = 48,3 % |

Meta del proyecto: ≥ 90 % por placa. El punto débil son las placas que todavía no
están en el diccionario: cada foto etiquetada nueva sube el dato de entrenamiento
**y** la entrada de flota.

---

## Documentación completa

- `docs\README-DETALLE-TECNICO.md` — arquitectura, motor paso a paso, contrato de features
- `docs\GUIA-COMANDOS.txt` — guía de instalación y comandos, paso a paso
- `cytv ayuda` — comandos del día a día
