# Cytv — Reconocimiento de placas desde CCTV (CYTV Ancón)

Cámara Hikvision TandemVu → FTP → cola → detección (YOLO ONNX) → **lectura: AOI
externa (PlateRecognizer) con caída automática al motor local (CRNN+RapidOCR)**
→ SQLite → reportes JSON diarios → portal web.

Windows, solo CPU, sin GPU. Python 3.13 (recomendado) o 3.9 legacy.

---

## Requisitos

- Windows 10/11 o Server (producción: Server 2012 R2 con Python 3.9)
- Python 3.13 — o Python 3.9 para el legacy
- ~2 GB de disco (modelos ONNX + dependencias)
- Conexión a internet **opcional**: solo si se usan claves AOI (`config.ini [aoi]`)

---

## Instalación

```bat
cd /d D:\Aplicacion-Camera
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Legacy (Python 3.9):** `pip install -r requirements-py39.txt`
(fija `opencv-python-headless==4.5.3.56`, sin Media Foundation).

**Configuración:** editar `config.ini` — solo rutas **absolutas**.
Credenciales de la cámara: `servicio_local\INFO_CAMARA.txt` (nunca subir al repo).

**AOI (PlateRecognizer) — opcional:** claves en `config.ini`:

```ini
[aoi]
tokens = sk_primera, sk_segunda, sk_tercera
```

El servicio prueba cada clave **en orden**: si una se queda sin cuota la
aparta y sigue con la siguiente; solo cuando **todas** fallan interviene el
motor local. Sin claves (o sin sección), el sistema trabaja 100 % con el
motor local (idéntico al comportamiento previo). También se pueden sumar
claves con `PLATERECOGNIZER_TOKEN` (se concatenan). Ver «Arquitectura de
lectura».

**Servicio Windows (NSSM):**

```bat
scripts\instalar_servicio.bat          :: menú de instalación
scripts\instalar_servicio.bat config   :: ver el config.ini que usa el servicio
```

---

## Arquitectura de lectura: AOI + motor local

El sistema tiene DOS fuentes de lectura de placa, orquestadas con prioridad
y caída automática:

```
                    ┌──────────────────────────────────┐
  foto + bbox ────► │ leer_placa_6_api (wrapper)       │
                    │   claves de config.ini [aoi]     │
                    └─────────────┬────────────────────┘
                                  │
              ¿alguna clave vigente (sin cuota)?
                         │                        │
                        SÍ                       NO
                         │                        │
                         ▼                        ▼
        por cada clave vigente, EN ORDEN:   ocr.leer_placa_6
        POST crop.jpg a la API              (motor local completo:
        (regions=pe, timeout 10s)            CRNN consenso + RapidOCR
                         │                  + votos + snap de flota)
      ┌──────┬──────┬────┴─────┐                     │
    200    402/403  429    red/timeout                │
   con     │        │        │                       │
  results  │        │        └──► apagado global     │
     │     │        │            2 min → None ◄──────┘
     ▼     ▼        ▼
  _elegir  clave   clave           ◄── (fallback local:
  candidates  │     60 s fuera        recién cuando TODAS
  vs flota    │     y sigue          las claves agotaron)
     │        ▼
     │    siguiente clave...
     ▼
  registro fuente=
  "platerecognizer"
```

### Rotación de claves — `config.ini [aoi]`

```ini
[aoi]
tokens = sk_uno, sk_dos, sk_tres
```

- **Orden de prueba**: la primera clave vigente que responda gana; las
  claves se prueban **antes** de caer al motor local.
- **Estado por clave** (circuit breaker individual):

  | Condición | Clave apartada | Motivo |
  |---|---|---|
  | HTTP 402 (cuota agotada) | 1 h | `http_402` |
  | HTTP 403 (clave inválida) | 1 h | `http_403` |
  | HTTP 429 (rate limit, 3 intentos) | 1 min | `http_429` |
  | Sin conexión / timeout | **global** 2 min (no rota: sin red todas fallan) | `sin_conexion` |
  | Otras fallas HTTP | 3 intentos con 1 s y vuelve al motor local | — |

- **Cuando todas las claves están en cuota**: `api_disponible()` es `False`
  → **cero llamadas HTTP**, directo al motor local hasta que una clave
  recupere su ventana.
- **`motivo_sin_api()`** para el registro: `sin_token`, `http_402`,
  `http_403`, `sin_conexion`, `sin_lectura_api` o claves separadas por coma.

### Registro por lectura (`registrar`, worker.py:1272–1289)

Cada guardado lleva en SQLite:

- `fuente`: `"platerecognizer"` (API confirmó) o `"modelo"` (motor local).
- `provisional`: `1` cuando la AOI está activa pero la lectura vino del motor
  local (la API no estuvo disponible o no leyó) — el registro es «mejor esfuerzo».
- `placa_api` / `placa_modelo`: texto crudo de cada fuente (auditoría).
- `motivo_sin_api`: por qué se usó el motor local (`http_402`, `sin_token`…).

### Traza AOI (`reportes\aoi\AAAA-MM-DD\`)

Cada envío a la AOI deja, en `aoi-AAAA-MM-DD.jsonl`: timestamp, foto, bbox,
lectura, confianza, `via_api` (bool), lista de intentos y ruta del crop enviado
(`*.jpg`). El portal (`/recortes`) lee ese JSONL y muestra eficacia API vs local.

### Reprocesar provisionales

```bat
python scripts\reprocesar_provisionales.py [TOPE] [--dry-run]
```

Toma los registros con `provisional=1`, reenvía el crop a la API y actualiza
`placa`, `confianza`, `fuente='platerecognizer'`, `provisional=0`,
`placa_api` y `motivo_sin_api=NULL`. Rate limit 0.5 s por llamada.

---

## Motor local (cuando no hay API o la API falla)

### Detección — `app/modelo.py`

YOLO (`modelo\placas.onnx`) vía `onnxruntime` CPU, 1 hilo, sin torch.
Enmascara las zonas del OSD de la Hikvision (timestamp + watermark) antes de
detectar. Letterbox a 640, NMS IoU 0.45, umbral 0.40 (configurable
`[filtros] umbral_modelo`). Si el ONNX falta o falla, worker sigue con un
detector geométrico de respaldo — el servicio nunca se cae por el modelo.

### OCR multicapa — `app/ocr.py` (`leer_placa_6`)

Pipeline de consenso sobre variantes del recorte:

1. **CRNN gateado** (`app/reconocedor.py`): consenso de 5 crops (3 márgenes +
   clahe+gamma + noche). Carácter a carácter con softmax; solo pasa si
   normaliza a `ABC-123` y el snap de flota lo deja en el diccionario.
   Si `conf ≥ 0.95` → salida temprana (no corre RapidOCR).
2. **RapidOCR** (`rapidocr-onnxruntime==1.2.3`, 1 hilo) sobre variantes
   (margen base/+5/+10, upscale 2x, clahe+gamma, noche, umbral invertido).
   Votos agrupados por forma equivalente (`I↔1`, `O↔0`, …).
3. **Arbitraje** CRNN vs RapidOCR en pares confusos (`WH`, `WV`, `QO`, `OD`,
   `MN`, `UV`) usando la probabilidad por carácter del CRNN.
4. **Snap de flota** (`app/decodificador_flota.py`): distancia de edición
   ponderada (I↔1 = 0.2, W↔H/N/M/V = 0.4) contra el diccionario cerrado de
   placas CYTV; remapea solo si hay UNA vecina a distancia ≤ k.
5. **Rescate** y **retry ONNX** (re-detectar con el modelo y releer).

### Semaforo — `evidencia_y_color`

Puntaje 0–100 por evidencia (apoyos del consenso + confianza); fuera de flota
techo 89. Verde ≥ 90, amarillo ≥ 50, rojo < 50.

---

## Flujo del servicio (`ftp_server.py` → `app/worker.py`)

```
Hikvision sube por FTP (pyftpdlib, puerto 2121, pasivo 60000-60099)
        │  on_file_received: solo encola (cero trabajo pesado)
        ▼
worker.COLA (queue 200)
        │  hilo worker: prioridad a lo vivo, reposo a los 10 min sin fotos
        ▼
procesar_imagen():
  1. dedup fotográfico vs última imagen de la vista (% absdiff, umbral 8%)
  2. candidatos de placa (ONNX o geométrico), ordenados por área
  3. filtros del recorte: nitidez (Laplaciano ≥ 120), sobreexpuesto (≤2%),
     bordes (≥2%), texto oscuro (5–55%) — solo sin modelo
  4. leer_placa_6 (→ AOI si hay token, si no motor local)
  5. dedup semántico: hash perceptual 64-bit (hamming ≤ 18) + texto cercano
  6. control de pasadas: ventana 60 min / 30 días / máx 3 pasadas
  7. guardar: fotos\placa\AAAA-MM-DD\<PLACA-123>[_crop.jpg|_2da-pasada|…]
  8. registrar() en SQLite + marcar JSON del día
  9. traza AOI (JSONL) si la AOI está activa
 10. si todo falla: descartar a descartadas\ (modo retener) + log_sin_lectura
```

- **Replay al arranque** (`[inicio] replay_descartes = si`): re-evalúa lo
  descartado con la lógica actual; lo rescatado entra como PLACA (DUDOSA).
- **Mejor foto**: si una placa ya está guardada y llega una foto con mayor
  área/nitidez, reemplaza el archivo y actualiza el registro.
- **Reevaluación**: si el primer bbox falla los filtros, hasta 3 reintentos
  con bbox expandidos (+10 %, +14 %, +18 %).
- **Cola de entrenamiento**: fotos con lectura fuerte pero descartadas por
  filtro se encolan para reetiquetado (`cola_entrenamiento\`).

---

## Reportes JSON diarios

`reportes\placas-YYYY-MM-DD.json` — regenerado con cada guardado y al
arrancar/cerrar día. Escritura atómica (`.tmp` + `replace`). Contenido:
fecha, totales y por placa: pasada, primera/última vez, detecciones,
`confianza_max`, `evidencia_max`, semáforo, `fuente`, `provisional` y rutas
de foto + crop. El portal y `cytv placas` lo consumen.

---

## Límites de CPU (diseño para correr junto a todo en el mismo server)

| Mecanismo | Dónde | Qué hace |
|---|---|---|
| ONNX a **1 hilo** | `modelo.py`, `reconocedor.py`, `ocr.py` (`intra_op/inter_op_num_threads=1`, `ORT_SEQUENTIAL`, sin arena) | detector, CRNN y RapidOCR no usan más de 1 hilo de cada modelo |
| Prioridad **BELOW_NORMAL** activo / **IDLE** en reposo | `worker._prioridad` | la cámara cede CPU al resto de apps; en idle no compite |
| **Reposo** a los `reposo_minutos` (10) sin fotos | `worker._entrar_en_reposo` | libera detector+OCR+CRNN de memoria y baja a IDLE; la próxima foto los recarga |
| 1 solo hilo de worker + cola de 200 | `worker.bucle_worker` | el FTP solo encola; el procesamiento es secuencial |
| Timeout HTTP 10 s + cortes rápidos | `platerecognizer._post_clave` | sin red: **un solo** intento (10 s) y la AOI se apaga 2 min; claves sin cuota fallan con 402/403 rápidos (sin esperas) |

---

## Estructura de archivos

```
D:\Aplicacion-Camera\
├── README.md                    este archivo
├── cytv.cmd                     lanzador del menú de utilidades
├── ftp_server.py                ÚNICA entrada del servicio (la referencia NSSM; no mover)
│                                └ pyftpdlib: HandlerCytv.on_file_received → worker.encolar
│                                  heartbeat cada 15 s, barrido de huérfanos cada 60 s,
│                                  INFO_CAMARA.txt regenerado en cada arranque
├── config.ini                   rutas absolutas + umbrales (lo lee ftp_server.py)
│                                └ [ftp] puerto/usuario/password/pasivos
│                                  [rutas] recibidos/procesadas/registro/logs/heartbeat/reportes
│                                  [aoi] tokens = claves PlateRecognizer en orden de prueba
│                                  [camara] ventana y máximo de pasadas
│                                  [recursos] reposo_minutos (libera modelos)
│                                  [duplicados] umbral_diferencia (% absdiff)
│                                  [inicio] replay_descartes, replay_pausa_seg
│                                  [filtros] modo_descarte, umbral_modelo, ocr_requerido,
│                                    nitidez, sobreexpuesto, bordes, dedup, texto_min/max
│                                  [log] nivel, retencion_dias
├── requirements.txt             Python 3.13 (incluye requests para la AOI)
├── requirements-py39.txt        legacy Python 3.9
│
├── app\                         MOTOR (sys.path lo agrega ftp_server.py)
│   ├── worker.py                orquestador: cola, dedup, filtros, pasadas, guardado,
│   │                            registro SQLite, reportes JSON, traza AOI, reposo,
│   │                            replay, cola de entrenamiento, mejor foto, reevaluación
│   ├── ocr.py                   pipeline: variantes → CRNN consenso → RapidOCR →
│   │                            votos → arbitraje → snap de flota → rescate → retry ONNX
│   │                            + evidencia_y_color (semáforo 0–100)
│   ├── modelo.py                detector YOLO ONNX (enmascara OSD, letterbox, NMS)
│   ├── reconocedor.py           CRNN ONNX (CTC decode, consenso multi-crop,
│   │                            recarga en caliente si cambian los pesos)
│   ├── platerecognizer.py       adaptador AOI: rotación de claves [aoi] + circuit
│   │                            breaker por clave (402/403/429/red) y fallback
│   ├── decodificador_flota.py   diccionario cerrado + distancia edición ponderada
│   ├── gt_placas.py             GT-29: 29 diagnósticos verificados visualmente
│   ├── log_sin_lectura.py       JSONL diagnóstico de fotos sin placa (apagado por defecto)
│   └── consola.py               consola interactiva (scripts\consola.bat)
│
├── modelo\                      modelos activos
│   ├── placas.onnx              detector YOLO
│   └── reconocedor.onnx(+.data) CRNN en producción
│
├── portal\portal.py             web de diagnóstico (stdlib HTTP + threading):
│                                  / subir foto → traza del pipeline
│                                  /recortes  envíos AOI del día (eficacia API vs local)
│                                  /pasadas   control de pasadas por placa
│
├── servicio_local\              datos del servicio (registro.db, INFO_CAMARA)
│
├── scripts\
│   ├── cytv.ps1                 implementación del menú cytv.cmd
│   ├── instalar_servicio.bat/.ps1  instalación NSSM
│   ├── reprocesar_provisionales.py reprocesa lecturas locales vía AOI
│   ├── replay_descartes.py      reprocesar descartadas\ bajo demanda
│   ├── heatmap_placas.py        banda espacial típica de las placas
│   ├── consola.bat              consola interactiva del sistema
│   ├── reevaluar.bat            reevalúa un día contra las etiquetas
│   ├── diag_recursos.ps1, medir_recursos.ps1, gpu_fix.ps1
│   ├── limpiar_almacenamiento.ps1, registrar_tarea_limpieza.ps1
│   ├── reiniciar_servicio.ps1, reinstalar_servicio.ps1, instalar.ps1
│   └── nssm.exe
│
├── entrenamiento\               experimentos y datos
│   ├── detector\                entrenar_detector.py (YOLO) + convertir_coco_a_yolo.py
│   ├── evaluacion\              validar_diagnosticos.py, evaluar_holdout.py, reevaluar.py,
│   │                            eval_rapido.py, diagnostico_gap.py + holdout\, GT29\
│   ├── medir_pipeline.py        banco oficial GT-29 (leer_placa_6 completo)
│   └── etiquetas_reales.csv     flota verificada a mano (datos; .gitignore)
│
│   Nota: el código de entrenamiento del CRNN (crnn.py, entrenar.py,
│   generador_placas.py) se retiró del repo en la limpieza 2026-10; los
│   pesos entrenados siguen en modelo\reconocedor.onnx y el respaldo de
│   datos está en D:\Escritorio\CAMARA PLACAS.
│
├── tests\                       pytest (sin red, sin modelos reales):
│   ├── test_platerecognizer.py  AOI: parseo, rotación de claves 402/403, 429, red,
│   │                            fallback local, orden clave→modelo
│   ├── test_worker_aoi.py       cableado AOI desde config.ini [aoi] / env
│   ├── test_worker_estructura.py traza AOI JSONL
│   ├── test_worker_crop_json.py crop + reporte JSON
│   └── test_portal_recortes.py  listado de traza AOI del portal
│
├── imagenes_procesadas\         fotos por día (datos; .gitignore)
└── venv\                        entorno virtual dev (no mover)
```

---

## Base de datos (`registro.db`, SQLite WAL)

| Tabla | Columnas | Uso |
|---|---|---|
| `registros` | fecha_hora, archivo, placa, confianza, estado, evidencia, semaforo, phash, fuente, provisional, placa_api, placa_modelo, motivo_sin_api | cada foto procesada |
| `pasadas` | placa, pasada, fecha_hora, excedida | control de reinspección |
| `candidatas` | fecha_hora, placa, archivo, evidencia, votos, estado | candidatas a flota |

Las columnas de AOI (`fuente`, `provisional`, `placa_api`, `placa_modelo`,
`motivo_sin_api`) se agregan automáticamente con `ALTER TABLE` si faltan
(migración en `worker.iniciar`).

---

## Comandos

| Comando | Qué hace |
|---|---|
| `cytv.cmd` | menú: `estado`, `placas`, `historial`, `log`, `fotos`, `replay`, `entrenador`, `heatmap`, `servicios`, `config`, `limpiar`, `ayuda` |
| `python ftp_server.py` | servicio FTP + pipeline (en producción lo corre NSSM) |
| `python portal\portal.py [--puerto 8090]` | portal web de consultas |
| `python scripts\reprocesar_provisionales.py [TOPE] [--dry-run]` | reprocesa provisionales vía AOI |
| `python scripts\replay_descartes.py [--fecha] [--dry-run]` | reprocesa descartadas |
| `python entrenamiento\medir_pipeline.py [--dump]` | banco GT-29 |
| `python entrenamiento\evaluacion\validar_diagnosticos.py [--detector txt\|modelo]` | validador oficial |
| `python entrenamiento\evaluacion\evaluar_holdout.py [--detector txt\|modelo]` | holdout fuera de GT-29 |
| `python -m pytest tests\ -q` | suite (23 tests: AOI, worker, portal) |

---

## Variables de entorno

| Variable | Efecto |
|---|---|
| `PLATERECOGNIZER_TOKEN` | suma claves AOI a las de `config.ini [aoi]` (se concatenan) |
| `CYTV_BASE` | raíz del proyecto (default: carpeta de `ftp_server.py`) |
| `CYTV_CRNN_ONNX` | ruta alternativa del CRNN |
| `CYTV_FLOTA_SIN_GT` | `1` = flota sin GT-29 (para medir holdout honesto) |
| `LOG_SIN_LECTURA` (+ `_DIR/_MAX_MB/_ARCHIVOS/_RETENCION_DIAS/_RECORTES/_MOTIVOS/_CRNN`) | log diagnóstico de fotos sin placa — apagado por defecto; ver detalle abajo |

### Log de SIN LECTURA (APAGADO)

Registra en JSONL (fuera de `registro.db`) las fotos sin placa, como material
para mejorar el dataset. Escritura en hilo aparte (~11 µs/foto; ~34 ms solo en
fotos sin lectura con diagnóstico CRNN). Retención 30 días por defecto.
Deduplica por (archivo, motivo) contra los últimos ~4 MB del JSONL.

---

## Métricas actuales (GT-29, 2026-10-22)

| Escenario | Acierto |
|---|---|
| Producción (detector ONNX + diccionario de flota) | **28/29 = 96,6 %** |
| Validación por bbox etiquetado (pipeline clásico) | 27/29 = 93,1 % |
| Sin diccionario de flota (placas nuevas) | 14/29 = 48,3 % |

Meta: ≥ 90 % por placa. Con la AOI activa, las placas fuera de flota pasan a
depender del score de PlateRecognizer (que no usa el diccionario local).

---

## Documentación completa

- `cytv ayuda` — comandos del día a día
- `scripts\consola.bat` — consola interactiva
- Este README + comentarios inline en cada módulo (`app\`, `portal\`, `scripts\`)
