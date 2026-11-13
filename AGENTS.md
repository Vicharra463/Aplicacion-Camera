# AGENTS.md — D:\aplicacion-camera

Guía de contexto para cualquier IA (opencode, Claude Code, etc.).
Leer ANTES de tocar nada. Si un punto cambió (commits, configs, fixes),
ACTUALIZAR este archivo en la misma sesión.

---

## Qué es este proyecto

Sistema de vigilancia de placas **CYTV Ancon** (Perú): cámara TandemVu
envía fotos por FTP → `ftp_server.py` las encola → `app/worker.py`
(detección ONNX + OCR/API) → guardan la placa en
`imagenes_procesadas\placa\<dia>\<PLACA>\` + reporte JSON diario +
`registro.db`.

| Rol | Ruta | Quién la toca |
|---|---|---|
| Desarrollo (repo, git) | `D:\aplicacion-camera` | La IA, con TDD |
| Producción (servicio real) | `C:\CytvAncon` | **Solo FELIPE** (la IA NO la toca sin pedido; lectura solo si la pide) |
| Transferencia server | `C:\Users\FELIPE\Music\MOVER AL SERVIDOR\`, `...\COPIAR A SERVIDOR\`, `...\Nueva carpeta\` | La IA solo llena, FELIPE copia |

Regla de oro: **el diagnóstico SIEMPRE pasando fotos REALES por el
pipeline de la app** — nunca scripts nuevos inventados, nunca la
carpeta "PRUEBAS ROI" (FELIPE la descartó).

---

## Primero: memoria persistente (Engram)

Antes de responder preguntas sobre trabajo previo ("qué hicimos",
"cómo lo resolví", "recordar…") o al empezar cualquier tarea:

1. `mem_context(project: "d:\aplicacion-camera")` — historial reciente.
2. `mem_search(query: ...)` — FTS5 por palabras clave.
3. `mem_get_observation(id)` — contenido completo (los resultados de
   búsqueda vienen truncados).
4. Guardar DESPUÉS de cada hito con `mem_save` (bugfix, decisión,
   descubrimiento, preferencia) — no esperar a que lo pidan.

El project key se normaliza a minúsculas: `d:\aplicacion-camera`.

---

## Estructura del repo

```
app/worker.py        Núcleo: pipeline, dedup, pasadas, guardado (~2700 líneas)
app/ocr.py           CRNN local
app/platerecognizer.py / app/roboflow.py   APIs de lectura
ftp_server.py        Carga config (línea ~46: BASE/config.ini),
                     cargar_config() UNA sola vez al arrancar (línea ~414),
                     lanza worker (bucle_worker), FTP y barrido
portal/portal.py     Panel web (recortes AOI, pasadas 1/2/3, AOI)
scripts/             calibrar_roi.py, replay_descartes.py, migrar_pasadas_carpetas.py, …
tests/               pytest (281 tests) — fixture principal en
                     tests/test_descartes_minimos.py (proc, _VeRojo, _guardadas)
config.ini           Config del repo (rutas a D:\)
README.md            Documentación larga del sistema
```

Servidor `C:\CytvAncon`: mismo árbol pero clonado; el servicio corre
`ftp_server.py` de la RAÍZ y hace `sys.path.insert(app)` → el worker
que importa es **`C:\CytvAncon\app\worker.py`**. Arranque: `cytv.cmd`
→ `scripts\cytv.ps1`. Logs en `logs\`, DB `registro.db`.

---

## Pipeline REAL (orden verificado, worker.py)

1. **`cooldown_evento` ANTES de `cv2.imread`** (~línea 2050): si la foto
   (timestamp de nombre de cámara) está dentro de ±`cooldown_evento_seg`
   de la última placa guardada → se BORA sin gastar API/CPU. Al revés
   de lo que parece: borra si NO pasó el tiempo.
2. Decode → máscara OSD → corte superior (ROI sup 0.35).
3. Detector ONNX (`modelo/placas.onnx`), umbral `umbral_modelo`.
4. Gate franja verde (ROI sup/inf, motivo naranja/rojo) + corte
   vertical (`linea_verde_izq/der`).
5. Gate de posición: `linea_prioridad ± tolerancia_ventana`
   (0.60 ± 0.20) → fuera = `fuera_de_area` SIN gastar API.
6. Ventana best-of (`eval_ventana_seg=15`): de una ráfaga evalúa UNA
   foto (más cerca de la línea, más grande si empate).
7. Gates FP → apagados en prod (`gates_fp = no`).
8. OCR/API: Roboflow (recorte) → PlateRecognizer `[aoi]` → motor local
   CRNN. `fuente_de_intentos` decide si es "confiable".
9. Dedup: `_buscar_misma` (huella perceptual + texto con
   `_forma_comparable`), luego pasadas `_pasada_de_placa`
   (DB → fallback disco `_placa_existe_hoy`), ventana de reinspección.
10. Guardado (MEJOR FOTO `_reemplazar_si_mejor` solo reemplaza si
    mejora área 1.2x o posición con evidencia ≥) → `_cooldown_placa`
    (solo `time.sleep` si la cola está vacía; NO descarta).

`cy_pos` se calcula sobre la **foto COMPLETA** — todas las fracciones
(0.35/0.85/0.60/0.20) son sobre la imagen original.

---

## Config: regla de reinicio y desincronías

- **NO hay hot-reload**: `cargar_config()` corre una vez al arrancar
  (ftp_server.py ~414). **Cualquier cambio en config.ini → reiniciar
  `CamaraCYTVAncon`** para que se aplique.
- El servicio lee **`C:\CytvAncon\config.ini`** (BASE = dir de
  ftp_server.py, línea ~46). Los configs "MOVER AL SERVIDOR" /
  "Nueva carpeta" NO los lee nadie hasta que FELIPE los copia.
- Estado de claves (2026-11-12): repo = `cooldown_placa 90`,
  `cooldown_evento 90`, `ventana_reinspeccion_min 5`; server =
  `cooldown_placa 90`, **`cooldown_evento 15` (elección DECLARADA de
  FELIPE — NO "corregirla" sin preguntar)**, `ventana 5` +
  7 claves que le faltaban (umbral_leven_pasada, revisar_carpetas_dia,
  tolerancia_ventana, tolerancia_prioridad, linea_verde_izq/der,
  eval_ventana_seg) parcheadas por chat el 2026-11-12.
- Desincronía repo↔server conocida (propuesta pendiente "sync config").
- `config.ini` del repo tiene tokens/claves de API: NO commitear
  cambios que los expongan ni tocar los del server.

---

## Comandos (los ÚNICOS válidos)

```bash
# Tests COMPLETOS (281) — SIEMPRE con --basetemp (ver Gotchas):
C:\Users\FELIPE\anaconda3\python.exe -m pytest tests\ -q --basetemp="C:\Users\FELIPE\AppData\Local\Temp\pytest-fresh-20261112"

# Un test / archivo:
C:\Users\FELIPE\anaconda3\python.exe -m pytest tests\test_descartes_minimos.py -q --basetemp="C:\Users\FELIPE\AppData\Local\Temp\pytest-fresh-20261112"
```

- Pipeline con fotos reales: `C:\CytvAncon\venv\Scripts\python.exe`
  (el venv de CytvAncon no tiene pytest; anaconda NO sirve para
  inferencia real — ver Gotchas).
- NO correr `scripts\migrar_pasadas_carpetas.py` en prod.
- NO hacer build tras cambios (regla de FELIPE: "never build").
- NO hay ruff ni mypy. `rg` NO está instalado (usar grep/Select-String).

---

## Preferencias de FELIPE (no negociable)

- **Español rioplatense con voseo** si él escribe en español; inglés
  con la misma energía si escribe en inglés.
- **TDD estricto**: test ROJO primero, fix, verde, suite completa.
- **Commit/push SOLO cuando él lo pida explícitamente.** Nunca
  atribución a IA en commits; conventional commits.
- **NO tocar `C:\CytvAncon`** sin pedido (leer solo si pide "hurgar").
- **UNA pregunta a la vez** con la tool `question` y ESPERAR la
  respuesta. Nunca continuar asumiendo.
- Siempre ofrece primero la opción **recomendada** (él casi siempre
  la elige).
- Verificar TODO antes de afirmar: "dejame verificar" + leer código.
  Si él está mal, explicarle POR QUÉ con evidencia.
- Está apurado/ directo: respuestas cortas, cero sermoneos largos.
- Reglas extra en `C:\Users\FELIPE\.claude\CLAUDE.md` (opencode la
  carga como global fallback): sin cat/grep/sed/ls (usar rg/fd/bat),
  sin build, commits conventional.

---

## Estado del proyecto (2026-11-12 — ACTUALIZAR AL CAMBIAR)

**Último fix (Opción A, placas REPETIDAS B7B-820/BIB-820)** — TDD
281/281 verde, **SIN COMMITEAR**:
- `app/worker.py`: `_EQUIV_COMPARAR = {"I":"1","O":"0","Z":"1","7":"1"}`
  (familia visual {Z,7,I,1} unida; maketrans es de UNA pasada) +
  canónica en el bloque de previa de `procesar_imagen` (si la
  relectura tiene la misma forma → gana el texto previo, carpeta
  `B7B-820\segunda\`, jamás `BIB-820`).
- `tests/test_descartes_minimos.py`: 2 tests nuevos
  (`test_forma_comparable_une_7_con_i`,
  `test_relectura_ocr_equivalente_usa_lectura_previa` — restaura
  `_buscar_misma_real` del guardado línea ~34) + test ventana 5.
- `config.ini` (repo) + test de ventana 5 (`ventana_reinspeccion_min = 5`).

Archivos de transferencia al server ya cargados (2026-11-12):
`Nueva carpeta\app\worker.py`, `Nueva carpeta\tests\...`,
`MOVER AL SERVIDOR\app\worker.py` (+ tests). Falta que FELIPE copie
`worker.py` a `C:\CytvAncon\app\`, pegar el config parcheado y
**reiniciar el servicio**.

Pendientes/decisiones viejas sin resolver:
- 5 cambios huérfanos en CytvAncon sin decisión de FELIPE.
- Cámara `192.168.18.42` caída; origen de carpetas sin responder.
- Propuesta "Panel de descartes" ofrecida y no elegida aún.
- Desincronía de configs repo↔server (propuesta #2).

---

## Gotchas (ahorran horas)

- **ACL roto**: `C:\Users\FELIPE\AppData\Local\Temp\pytest-of-FELIPE`
  da WinError 5 → 12 tests de `tmp_path` reventarán SIN `--basetemp`.
  SIEMPRE usar el flag (ojo: `--basetemp` es flag de CLI, `-o basetemp=`
  NO funciona porque es opción ini).
- **anaconda tiene onnxruntime ROTO** (0xC0000005) — los tests
  parchean el detector (`_VeRojo`), por eso sirve para pytest pero NO
  para fotos reales. Para fotos reales: `C:\CytvAncon\venv`.
- `_buscar_misma` con `confiable=True` (fuente API roboflow/
  platerecognizer) = Levenshtein **0** y SIN fallback por hash.
  El fixture `proc` de tests parchea `_buscar_misma`→None (línea ~138):
  tests de matching real deben restaurarlo con `_buscar_misma_real`.
- El fixture `proc.state["texto"]` dicta lo que "lee" el OCR mock;
  `proc.foto(nombre, y0, y1, color=...)` crea fotos sintéticas
  (ROJO/AZUL_CLARO para 2ª foto) y `_guardadas(proc)` lista guardados.
- `_hashes` solo se llena al guardar (worker.py ~2640) y en MEJOR
  FOTO (~1994); se recarga del disco al arrancar.
- Reemplazos MEJOR FOTO hacen UPDATE en `registro.db` (NO suman
  `detetecciones` en el JSON); `pasada`/`excedida` salen de la tabla
  `pasadas`. JSON se regenera al arrancar, al cambiar de día y por
  `_marcar_json`.
- `reconocedor.onnx` usa pesos en archivo EXTERNO `reconocedor.onnx.data`
  — copiar el .onnx solo deja el modelo incompleto.
- Servicio viejo en CytvAncon puede tener worker de versión anterior
  (hash ≠ repo) — verificar con `certutil -hashfile` antes de asumir
  que corre el código del repo.
