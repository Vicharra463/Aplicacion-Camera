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
| `python entrenamiento\evaluacion\evaluar_holdout.py [--detector txt\|modelo]` | Evaluación del holdout: fotos nuevas fuera de GT-29, 4 modos separados (solo lectura) |
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
│   ├── log_sin_lectura.py       log JSONL de SIN LECTURA (apagado por defecto)
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
│   ├── evaluacion\              validar_diagnosticos, evaluar_holdout + holdout\, reevaluar, gap
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

## Log de SIN LECTURA — diagnóstico de fotos sin placa (APAGADO)

Cuando una foto no da placa, el sistema puede registrar el caso en un JSONL
aparte (material para mejorar el dataset) **sin tocar `registro.db` ni las
rutas de producción**. Por defecto está **apagado**: sin la variable de
entorno, no escribe nada.

| Variable | Default | Qué controla |
|---|---|---|
| `LOG_SIN_LECTURA` | `0` (apagado) | enciende el log |
| `LOG_SIN_LECTURA_DIR` | `C:\CytvAncon\diagnostico\sin_lectura` | carpeta del JSONL + `recortes\` |
| `LOG_SIN_LECTURA_MAX_MB` | `50` | rotación: tamaño máximo por archivo |
| `LOG_SIN_LECTURA_ARCHIVOS` | `3` | archivos conservados (rotados + actual) |
| `LOG_SIN_LECTURA_RETENCION_DIAS` | `30` | **retención: borra eventos con más de N días** |
| `LOG_SIN_LECTURA_RECORTES` | `0` | guarda el recorte JPEG de la placa |
| `LOG_SIN_LECTURA_RECORTES_MAX` | `200` | máximo de recortes en disco |
| `LOG_SIN_LECTURA_RECORTES_MAX_MB` | `200` | máximo de MB de recortes |
| `LOG_SIN_LECTURA_CRNN` | `1` | diagnóstico CRNN del evento (`0` = más liviano) |
| `LOG_SIN_LECTURA_MOTIVOS` | `texto_invalido,placa_cortada` | motivos del worker a registrar (`vacío` o `*` = todos) |

- **Rendimiento** (medido sobre GT-29): escritura en hilo aparte con cola en
  memoria → ~11 µs por foto con el flag encendido; con diagnóstico CRNN,
  ~34 ms **solo** en fotos sin lectura (≤ 0,2 % sobre la media total).
- **Privacidad**: los recortes viven en `LOG_SIN_LECTURA_DIR\recortes\`,
  fuera de las carpetas servidas; ni el portal ni el FTP publican esa ruta.
- **Fallos** (ruta inválida, sin permisos, disco lleno): aviso único en el
  log del servicio y continúa — nunca corta el proceso.
- **Sin duplicados**: al encolar se deduplica por (nombre de archivo,
  motivo) contra los últimos ~4 MB del JSONL — el replay de arranque no
  repite eventos de sesiones anteriores. El campo `imagen` es la ruta de
  *entrada* (tras `_descartar` el archivo quedó en
  `descartadas/<fecha>/<motivo>/` o fue borrado según `modo_descarte`).
- **Apagado**: no definir la variable (o `=0`). Para desactivar la
  recolección al servicio hay que deshacer los dos bloques aditivos en
  `app\worker.py` y `entrenamiento\evaluacion\validar_diagnosticos.py`.

---

## Documentación completa

- `docs\README-DETALLE-TECNICO.md` — arquitectura, motor paso a paso, contrato de features
- `docs\GUIA-COMANDOS.txt` — guía de instalación y comandos, paso a paso
- `cytv ayuda` — comandos del día a día
