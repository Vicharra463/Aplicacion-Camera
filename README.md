# Cytv — Reconocimiento de placas desde CCTV (CYTV Ancón)

Cámara Hikvision TandemVu → FTP → cola → detección → **lectura en cadena:
Roboflow (1ra fuente) → PlateRecognizer (2da) → motor local (3ra)**
→ SQLite → reportes JSON diarios → portal web.

Windows, solo CPU, sin GPU. Python 3.13 (recomendado) o 3.9 legacy
(legacy = sin Roboflow, ver «Instalación»).

---

## Requisitos

- Windows 10/11 o Server (producción: Server 2012 R2 con Python 3.9)
- Python 3.13 — o Python 3.9 para el legacy
- ~2 GB de disco (modelos ONNX + dependencias)
- Conexión a internet **opcional**: solo si se configura una fuente externa
  (`config.ini [roboflow]` o `[aoi]`). Sin claves, todo funciona 100 % con
  el motor local.

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
En legacy **no** se instala `inference-sdk` (arrastra opencv/scipy que
rompen los pins): la cadena Roboflow queda desactivada por diseño y
siguen PlateRecognizer + motor local.

**Configuración:** editar `config.ini` — solo rutas **absolutas**.
Credenciales de la cámara: `servicio_local\INFO_CAMARA.txt` (nunca subir al repo).

**Roboflow (cadena PRIMERA) — opcional:**

```ini
[roboflow]
; api_key VACIA en el repo: ponga la clave SOLO en el servidor (o en la
; variable de entorno ROBOFLOW_API_KEY). Nunca la commitee.
api_key =
workspace = felipe-guerrero-eiqzh
workflow_id = license-plate-detection-1793468546778
```

**PlateRecognizer (cadena SEGUNDA) — opcional:** claves en `config.ini`:

```ini
[aoi]
tokens = sk_primera, sk_segunda, sk_tercera
```

El servicio prueba cada clave **en orden**: si una se queda sin cuota la
aparta y sigue con la siguiente. También se pueden sumar claves con las
variables de entorno `ROBOFLOW_API_KEY` y `PLATERECOGNIZER_TOKEN` (se
concatenan). Ver «Arquitectura de lectura».

**Servicio Windows (NSSM):**

```bat
scripts\instalar_servicio.bat          :: menú de instalación
scripts\instalar_servicio.bat config   :: ver el config.ini que usa el servicio
```

---

## Regla de oro: una placa son EXACTAMENTE 6 alfanuméricos

Toda lectura — de cualquier fuente — debe cumplir la misma regla antes de
aceptarse: **exactamente 6 caracteres alfanuméricos, con letras Y dígitos
mezclados** (`ABC-123`). La implementa `ocr.es_placa_6()` (y
`ocr._normalizar()`, que además canoniza `I↔1`, `O↔0`, `S↔5`, `D→0` en
posiciones 4–6 y exige la mezcla letras+dígitos).

Dónde se aplica:

| Punto de entrada | Función | Qué hace si no cumple |
|---|---|---|
| Roboflow (`plate_text`) | `worker.texto_de_api` | línea descartada; si no queda ninguna, cae la cadena local sobre la misma bbox |
| PlateRecognizer | `platerecognizer.leer_placa_6_api` | no acepta la lectura de la API; delega en el motor local |
| Motor local | `ocr._normalizar` dentro de `leer_placa_6` | no hay consenso → reevaluación / descarte |

Una lectura de 5, 7 u 8 caracteres **nunca** llega a la base de datos.

---

## Arquitectura de lectura: Roboflow → PlateRecognizer → motor local

Tres fuentes en cadena, con caída automática y cortacircuitos propios:

```
                    ┌──────────────────────────────────────────┐
  foto completa ───►│ 1) roboflow.detectar_leer()              │
                    │    1 request: bbox + plate_text          │
                    └────────────────┬─────────────────────────┘
                                     │
              ¿texto con forma de placa (es_placa_6)?
                     │                              │
                    SÍ                            NO / sin clave
                     │                              │
                     ▼                              ▼
        texto listo, bbox listo          detector local (ONNX)
        (_api-Roboflow)                            │
                                                   ▼
                    ┌──────────────────────────────────────────┐
   por cada bbox ──►│ 2) leer_placa_6_api()  [si hay claves]   │
                    │    POST crop.jpg → PlateRecognizer        │
                    └────────────────┬─────────────────────────┘
                                     │
                        ¿leyó con forma de placa?
                     │                              │
                    SÍ                            NO / sin claves
                     │                              │
                     ▼                              ▼
        texto listo (_api-PlateRecognizer)   3) ocr.leer_placa_6()
                                             (CRNN + RapidOCR + votos
                                              + snap de flota)
```

- **Roboflow** es un solo request por foto: detecta **y** lee sobre la
  imagen completa (no consume cuota de PlateRecognizer si leyó).
- **PlateRecognizer** solo se invoca si Roboflow no produjo lectura
  válida para esa bbox.
- **Motor local** solo interviene si las APIs no leyeron o no están.
- Cada guardado queda etiquetado con su fuente (ver «Colores de caja»).

### Roboflow — primera fuente (`app/roboflow.py`)

- Workflow serverless (`inference-sdk >= 1.5.0`, `serverless.roboflow.com`)
  sobre la foto completa: devuelve `predictions` (bbox) + `plate_text`
  en el mismo request; `use_cache=True` (Roboflow cachea respuestas
  idénticas server-side).
- `parsear()` convierte centro→esquina, acota a los bordes de la imagen y
  empareja `plate_text` con las detecciones **por posición**; si hay
  desalineo, el texto queda vacío y la lectura cae a la cadena local.
- Cortacircuito: sin red → **2 min** fuera; error de API → **5 min** fuera;
  sin clave / SDK ausente → cadena desactivada (`sin_clave`).
- El texto pasado por `texto_de_api` aplica la regla de 6 (multi-línea:
  se valida cada línea y solo se acepta si hay **exactamente una**
  candidata con forma de placa; ambigüedad → cadena local).

### Rotación de claves AOI — `config.ini [aoi]`

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

### Registro por lectura (`registrar`, worker.py)

Cada guardado lleva en SQLite:

- `fuente`: `"roboflow"`, `"platerecognizer"` o `"modelo"` (motor local).
- `provisional`: `1` cuando hay una API activa pero la lectura vino del
  motor local (la API no estuvo disponible o no leyó) — el registro es
  «mejor esfuerzo».
- `placa_api` / `placa_modelo`: texto crudo de cada fuente (auditoría).
- `motivo_sin_api`: por qué se usó el motor local (`sin_clave`,
  `http_402`, `sin_token`…).

### Traza de APIs (`reportes\aoi\AAAA-MM-DD\`)

Cada envío a una API deja, en `aoi-AAAA-MM-DD.jsonl`: timestamp, foto,
bbox, lectura, confianza, `via_api`, `via_roboflow`, lista de intentos y
ruta del crop enviado (`*.jpg`). El portal (`/recortes`) lee ese JSONL y
muestra eficacia API vs local.

### Reprocesar provisionales

```bat
python scripts\reprocesar_provisionales.py [TOPE] [--dry-run]
```

Toma los registros con `provisional=1`, reenvía el crop a la API y
actualiza `placa`, `confianza`, `fuente`, `provisional=0`, `placa_api` y
`motivo_sin_api=NULL`. Rate limit 0.5 s por llamada.

---

## Colores de caja (fuente de la lectura)

Al guardar, el recorte de la placa se dibuja con un color según la
fuente que leyó (`worker.color_caja`):

| Color | BGR | Sufijo en el nombre del archivo | Fuente |
|---|---|---|---|
| **Verde** | `(0, 255, 0)` | `_api-Roboflow` | Roboflow |
| **Ámbar** | `(0, 255, 255)` | `_api-PlateRecognizer` | PlateRecognizer |
| **Azul** | `(255, 0, 0)` | *(sin sufijo)* | motor local |

El color se deriva del sufijo de la fuente, así que sobrevive a los
sufijos de pasada (`_api-Roboflow_2da-pasada` sigue verde). Azul en vez
de rojo para no chocar con el semáforo de evidencia (rojo = mala
evidencia, no = fuente local).

---

## Duplicados: cómo NO consumir la API de más

Orden de protección en `worker.procesar_imagen`:

1. **Dedup fotográfico — ANTES de cualquier API.**
   `% absdiff` de la foto contra la **última imagen procesada de la misma
   vista** (`config.ini [duplicados] umbral_diferencia = 8`). Si el cambio
   es menor → se descarta como `duplicado` y la función termina **ahí**:
   ni Roboflow, ni PlateRecognizer ni el motor local se enteran. Esta es
   la principal red: en CCTV la cámara manda frames cada pocos segundos y
   la gran mayoría son casi idénticos.

2. **Un solo request por foto que pasa el filtro.**
   - Roboflow: **1 request full-image** con detección + lectura juntas.
   - PlateRecognizer: **solo** si Roboflow no leyó esa bbox.
   - Motor local: **solo** si las APIs no dieron lectura (o no están).

3. **Dedup semántico — DESPUÉS de leer** (no ahorra la llamada ya
   hecha, evita guardar/repetir):
   - hash perceptual 64-bit, hamming ≤ `umbral_dedup_dist` (18), o
     texto cercano → `duplicado_placa`;
   - ventana de reinspección: misma placa registrada hace < 60 min →
     descartada;
   - máximo **3 pasadas** por placa cada 30 días (sufijos
     `_2da-pasada`, `_3ra-pasada`, `_EXCEDIDA`);
   - **mejor foto**: si la nueva tiene ≥ 1.2× el área y no menos
     evidencia, reemplaza el archivo guardado.

4. **Caché server-side de Roboflow**: `use_cache=True` — la misma imagen
   no se re-evalúa del lado de Roboflow.

5. **Cortacircuitos = 0 HTTP**: sin clave, sin cuota o apagado por
   error/red → ninguna llamada sale de la máquina.

> **Limitación a conocer**: el dedup semántico (pHash + texto) carga al
> arrancar **solo los registros del día en curso** (`_dia_dedup`), y el
> absdiff compara contra la última imagen **en memoria** (se vacía al
> reiniciar). Una foto guardada **ayer** vuelve a consumir la API si se
> reenvía **hoy**. Para re-lotes grandes conviene revisar qué falta antes
> de mandar todo.

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
  1. dedup fotográfico vs última imagen de la vista (% absdiff, 8%)
     → duplicado: RETURN antes de tocar cualquier API
  2. Roboflow (si hay clave): 1 request full-image → bbox + plate_text
     → texto validado por la regla de 6 (es_placa_6)
  3. candidatos: de Roboflow si detectó; si no, ONNX/geométrico
  4. filtros del recorte: nitidez (Laplaciano ≥ 120), sobreexpuesto (≤2%),
     bordes (≥2%), texto oscuro (5–55%)
  5. lectura por bbox: texto Roboflow → PlateRecognizer → motor local
     (cada una con su circuit breaker; regla de 6 en cada entrada)
  6. dedup semántico: pHash 64-bit (hamming ≤ 18) + texto cercano
  7. control de pasadas: ventana 60 min / 30 días / máx 3 pasadas
  8. guardar: imagenes_procesadas\<vista>\AAAA-MM-DD\<PLACA>[sufijos].jpg
     + .txt (etiqueta YOLO) + _crop.jpg, caja coloreada por fuente
  9. registrar() en SQLite + marcar JSON del día
 10. traza aoi-AAAA-MM-DD.jsonl (via_api / via_roboflow)
 11. si todo falla: descartar a descartadas\ (modo retener) + log_sin_lectura
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
`confianza_max`, `evidencia_max`, semáforo, `fuente`
(`roboflow`/`platerecognizer`/`modelo`), `provisional` y rutas de foto +
crop. El portal y `cytv placas` lo consumen.

---

## Límites de CPU (diseño para correr junto a todo en el mismo server)

| Mecanismo | Dónde | Qué hace |
|---|---|---|
| ONNX a **1 hilo** | `modelo.py`, `reconocedor.py`, `ocr.py` (`intra_op/inter_op_num_threads=1`, `ORT_SEQUENTIAL`, sin arena) | detector, CRNN y RapidOCR no usan más de 1 hilo de cada modelo |
| Prioridad **BELOW_NORMAL** activo / **IDLE** en reposo | `worker._prioridad` | la cámara cede CPU al resto de apps; en idle no compite |
| **Reposo** a los `reposo_minutos` (10) sin fotos | `worker._entrar_en_reposo` | libera detector+OCR+CRNN de memoria y baja a IDLE; la próxima foto los recarga |
| 1 solo hilo de worker + cola de 200 | `worker.bucle_worker` | el FTP solo encola; el procesamiento es secuencial |
| Timeout HTTP 10 s + cortes rápidos | `platerecognizer._post_clave` | sin red: **un solo** intento (10 s) y la AOI se apaga 2 min; claves sin cuota fallan con 402/403 rápidos (sin esperas) |
| Cortacircuitos Roboflow 2/5 min | `roboflow._apagar` | sin red o error de API → la cadena no reintenta en cada foto |

---

## Estructura de archivos

```
D:\Aplicacion-Camera\
├── README.md                    este archivo
├── cytv.cmd                     lanzador del menú de utilidades
├── ftp_server.py                ÚNICA entrada del servicio (la referencia NSSM; no mover)
│                                └ pyftpdlib: HandlerCytv.on_file_received → worker.encolar
│                                  carga config.ini (rutas, [aoi], [roboflow], umbrales),
│                                  heartbeat cada 15 s, barrido de huérfanos cada 60 s,
│                                  INFO_CAMARA.txt regenerado en cada arranque
├── config.ini                   rutas absolutas + umbrales (lo lee ftp_server.py)
│                                └ [ftp] puerto/usuario/password/pasivos
│                                  [rutas] recibidos/procesadas/registro/logs/heartbeat/reportes
│                                  [roboflow] api_key (VACIA en repo), workspace, workflow_id
│                                  [aoi] tokens = claves PlateRecognizer en orden de prueba
│                                  [camara] ventana y máximo de pasadas
│                                  [recursos] reposo_minutos (libera modelos)
│                                  [duplicados] umbral_diferencia (% absdiff, ANTES de la API)
│                                  [inicio] replay_descartes, replay_pausa_seg
│                                  [filtros] modo_descarte, umbral_modelo, ocr_requerido,
│                                    nitidez, sobreexpuesto, bordes, umbral_dedup_dist,
│                                    texto_min/max
│                                  [log] nivel, retencion_dias
├── requirements.txt             Python 3.13 (requests + inference-sdk para las APIs)
├── requirements-py39.txt        legacy Python 3.9 (sin Roboflow por diseño)
│
├── app\                         MOTOR (sys.path lo agrega ftp_server.py)
│   ├── worker.py                orquestador: cola, dedup, cadena de fuentes, filtros,
│   │                            pasadas, guardado con colores, registro SQLite,
│   │                            reportes JSON, traza de APIs, reposo, replay,
│   │                            cola de entrenamiento, mejor foto, reevaluación
│   ├── roboflow.py              1ra fuente: cliente del workflow serverless
│   │                            (parseo, cortacircuitos 2/5 min, plate_text → es_placa_6)
│   ├── ocr.py                   pipeline: variantes → CRNN consenso → RapidOCR →
│   │                            votos → arbitraje → snap de flota → rescate → retry ONNX
│   │                            + evidencia_y_color (semáforo 0–100)
│   │                            + es_placa_6 / _normalizar (regla de 6)
│   ├── modelo.py                detector YOLO ONNX (enmascara OSD, letterbox, NMS)
│   ├── reconocedor.py           CRNN ONNX (CTC decode, consenso multi-crop,
│   │                            recarga en caliente si cambian los pesos)
│   ├── platerecognizer.py       adaptador 2da fuente: rotación de claves [aoi] +
│   │                            circuit breaker por clave (402/403/429/red),
│   │                            regla de 6 antes de aceptar, fallback local
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
│                                  /recortes  envíos a APIs del día (eficacia API vs local)
│                                  /pasadas   control de pasadas por placa
│
├── servicio_local\              datos del servicio (registro.db, INFO_CAMARA)
│
├── scripts\
│   ├── cytv.ps1                 implementación del menú cytv.cmd
│   ├── instalar_servicio.bat/.ps1  instalación NSSM
│   ├── reprocesar_provisionales.py reenvía provisionales a la API
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
├── tests\                       pytest (sin red, sin modelos reales; 34 tests):
│   ├── conftest.py              agrega app\ y portal\ al sys.path
│   ├── test_roboflow.py         parseo del workflow (fixture real), configurar,
│   │                            cortacircuitos de red/error, workspace/workflow
│   ├── test_platerecognizer.py  regla de 6: acepta 6, rechaza 5/7 y todo-dígitos
│   │                            o todo-letras, fallback local, sin lectura API
│   └── test_worker_etiquetas.py etiquetas de fuente, texto_de_api (multilinea,
│                                ambigüedad, regla de 6), colores de caja
│
├── imagenes_procesadas\         fotos por día (datos; .gitignore)
│   └── <vista>\AAAA-MM-DD\<PLACA>[sufijos].jpg + .txt + _crop.jpg
└── venv\                        entorno virtual dev (no mover)
```

---

## Base de datos (`registro.db`, SQLite WAL)

| Tabla | Columnas | Uso |
|---|---|---|
| `registros` | fecha_hora, archivo, placa, confianza, estado, evidencia, semaforo, phash, fuente, provisional, placa_api, placa_modelo, motivo_sin_api | cada foto procesada |
| `pasadas` | placa, pasada, fecha_hora, excedida | control de reinspección |
| `candidatas` | fecha_hora, placa, archivo, evidencia, votos, estado | candidatas a flota |

`fuente` ∈ `roboflow` | `platerecognizer` | `modelo`. Las columnas de
API (`fuente`, `provisional`, `placa_api`, `placa_modelo`,
`motivo_sin_api`) se agregan automáticamente con `ALTER TABLE` si faltan
(migración en `worker.iniciar`).

---

## Comandos

| Comando | Qué hace |
|---|---|
| `cytv.cmd` | menú: `estado`, `placas`, `historial`, `log`, `fotos`, `replay`, `entrenador`, `heatmap`, `servicios`, `config`, `limpiar`, `ayuda` |
| `python ftp_server.py` | servicio FTP + pipeline (en producción lo corre NSSM) |
| `python portal\portal.py [--puerto 8090]` | portal web de consultas |
| `python scripts\reprocesar_provisionales.py [TOPE] [--dry-run]` | reenvía provisionales a la API |
| `python scripts\replay_descartes.py [--fecha] [--dry-run]` | reprocesa descartadas |
| `python entrenamiento\medir_pipeline.py [--dump]` | banco GT-29 |
| `python entrenamiento\evaluacion\validar_diagnosticos.py [--detector txt\|modelo]` | validador oficial |
| `python entrenamiento\evaluacion\evaluar_holdout.py [--detector txt\|modelo]` | holdout fuera de GT-29 |
| `python -m pytest tests\ -q` | suite (34 tests: Roboflow, AOI, etiquetas y colores) |

---

## Variables de entorno

| Variable | Efecto |
|---|---|
| `ROBOFLOW_API_KEY` | clave del workflow Roboflow (alternativa a `config.ini [roboflow] api_key`; se concatenan) |
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

## Métricas actuales

### GT-29 (2026-10-22) — motor local

| Escenario | Acierto |
|---|---|
| Producción (detector ONNX + diccionario de flota) | **28/29 = 96,6 %** |
| Validación por bbox etiquetado (pipeline clásico) | 27/29 = 93,1 % |
| Sin diccionario de flota (placas nuevas) | 14/29 = 48,3 % |

### Lote D:\NUEVAS FOTOS → servicio local (2026-11-01, 87 fotos)

| Fuente ganadora | Fotos |
|---|---|
| Roboflow | 68 |
| Motor local | 19 |
| PlateRecognizer | 0 (AOI sin claves en producción) |

| Coincidencia con GT | Fotos |
|---|---|
| Exacta | 54 |
| Parcial | 17 |
| Fallo | 15 |
| Sin GT | 1 |

Revisión visual: ~80/87 correctas. Fallos: `UN3-077`, `SES-05A`,
`CHX-518`, `ERU-562`; lecturas borrosas: `F5U-884`, `T8G-903`, `BWM-926`.

Meta: ≥ 90 % por placa. Con fuentes externas activas, las placas fuera de
flota pasan a depender del score de Roboflow/PlateRecognizer (que no usan
el diccionario local).

---

## Documentación completa

- `cytv ayuda` — comandos del día a día
- `scripts\consola.bat` — consola interactiva
- Este README + comentarios inline en cada módulo (`app\`, `portal\`, `scripts\`)
