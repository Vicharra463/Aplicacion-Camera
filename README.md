# Servidor FTP de Placas CYTV Ancon

Sistema 24/7 que recibe fotos de una cámara Hikvision TandemVu (2 canales), detecta y lee placas peruanas con un motor propio (detector YOLOv8n + CRNN con votación de variantes y límite de formato peruano; sin flota cerrada), y organiza todo en carpetas diarias con reportes JSON.

**Precisión actual en el set de validación (29 diagnósticos): 44,8 % sin flota cerrada** (medido 2026-10-21). Con la flota cerrada (ya eliminada) daba 86,2 %; el motor anterior (RapidOCR solo) daba 55,2 %. Meta del proyecto: ≥ 90 % con datos reales vía fine-tune.

Este README tiene dos caminos:

| Soy... | Leo... |
|--------|--------|
| **Operador / instalador** | Instalación → Cámara → Operación → Salida → Problemas |
| **Desarrollador** | Arquitectura → Motor de lectura → Desarrollo y reentrenamiento |

---

# Para operadores

## Instalación (una sola vez)

### Requisitos

- **Windows** 8.1 / Server 2012 R2 o superior (64 bits)
- **Python** 3.9 (Windows ≤ 10 build 14393) o 3.13 (Windows 10 1607+) — lo instala el script si falta
- **Firewall**: TCP **2121** (FTP) y **60000–60099** (pasivo)
- **Reserva DHCP** en el router para la IP fija del servidor

### Pasos

```cmd
REM 1. Copia la carpeta COMPLETA del proyecto a C:\CytvAncon
REM 2. Doble clic en scripts\instalar_servicio.bat  (pide UAC)
REM 3. Espera: crea venv, instala deps, registra servicio NSSM, abre firewall
```

El instalador es **idempotente**: volver a correrlo reutiliza lo que ya esté bien.

### Verificar sin tocar nada

```cmd
scripts\instalar_servicio.bat --check
```

Salida típica:

```
Carpeta:   C:\CytvAncon\
Elevacion: SI
Sistema:   Windows 11 build 22631
Soporte:   SI - se instalara Python 3.13
Python:    presente (C:\Program Files\Python313)
NSSM:      presente en C:\CytvAncon
Proyecto:  instalado en C:\CytvAncon
Servicio:  Running
Puerto 2121: libre
Firewall:  regla presente
```

### Server 2012 R2: Media Foundation y OpenCV (resuelto, por si aparece)

El problema de fondo: **Server 2012 R2 no trae Media Foundation activa** (`mfplat.dll`). El instalador la habilita con DISM, Windows responde **3010 = reinicio pendiente** y las DLL no existen hasta reiniciar. OpenCV normal las necesita (`DLL load failed while importing cv2`) y, encima, `rapidocr-onnxruntime` declara dependencia de `opencv-python`: si pip instala esa versión normal encima del headless, `cv2` queda corrupto (`partially initialized module 'cv2'`).

**El instalador ya lo resuelve solo** — no hace falta ningún parche ni script extra:

1. El **3010 ya no corta** la instalación: solo avisa y continúa.
2. `requirements-py39.txt` fija `opencv-python-headless==4.5.3.56`, que carga **sin** Media Foundation (y sigue funcionando después del reinicio).
3. Antes de `pip install`, crea un metadato falso de `opencv-python 4.5.3.56` en el venv para que pip **no** instale el normal encima.
4. Si igual apareciera el paquete normal, una guarda lo desinstala y repone el headless.
5. El paso final de humo prueba los imports reales (`cv2`, `onnxruntime`, `RapidOCR`) y solo dice `imports OK` si todo cargó.

Si tenés que recrear el venv **a mano** sin el instalador (o verificar a posteriori):

```powershell
$py = "C:\CytvAncon\venv\Scripts\python.exe"

# metadato falso (solo si el venv es nuevo y sin él):
$d = "C:\CytvAncon\venv\Lib\site-packages\opencv_python-4.5.3.56.dist-info"
New-Item -ItemType Directory $d -Force | Out-Null
Set-Content "$d\METADATA" "Metadata-Version: 2.1`nName: opencv-python`nVersion: 4.5.3.56" -Encoding ASCII
Set-Content "$d\INSTALLER" "manual" -Encoding ASCII
Set-Content "$d\RECORD" "" -Encoding ASCII

# verificación final (debe imprimir TODO OK):
& $py -c "import pyftpdlib, cv2, numpy, onnxruntime; from rapidocr_onnxruntime import RapidOCR; print('TODO OK', cv2.__version__)"
```

### Cómo reconocer cada fallo

| Error en la consola del instalador | Causa |
|---|---|
| `DLL load failed while importing cv2` | OpenCV normal sin Media Foundation (no debería pasar con el headless pineado) |
| `partially initialized module 'cv2'` | Se mezclaron opencv normal y headless — la guarda del instalador lo corrige |
| Se detiene en "pendiente de REINICIO" con error | Instalador viejo sin el parche del 3010 — actualizá `scripts\` |
| `imports OK` pero el servicio no levanta | Mirá el detalle que imprime justo debajo (stderr se vuelca en la misma ventana) |

Reinicio opcional: cuando reinicies el servidor, `mfplat.dll` queda disponible para siempre; el instalador lo detecta en el paso 5.6 y omite el aviso. Con el headless pineado no hace falta apurarse a reiniciar.

## Puesta en marcha de la cámara (Hikvision)

1. Abre `C:\CytvAncon\INFO_CAMARA.txt` (se regenera en cada arranque; **no se sube a git** porque contiene credenciales).
2. En la web de la cámara (`http://<IP-camera>`):
   - **Storage → Schedule Settings → Picture Capture** → activa FTP en **AMBOS canales**
   - **Event → Motion Detection (MD_WITH_TARGET)** → linkage FTP en **AMBOS canales**
3. Datos del servidor FTP:

   ```
   Server:     <IP de INFO_CAMARA.txt>
   Port:       2121
   User:       admin
   Password:   <la de config.ini>
   Protocol:   FTP
   Remote Dir: /
   Mode:       Passive
   ```

4. **Importante**: si solo sube un canal, la carpeta `placa\` o `vehiculo\` no recibe fotos. El mapeo de canales se configura en `[camara]` de `config.ini`.

## Operación diaria

### Comandos (`cytv` — funciona desde cualquier carpeta)

```powershell
cytv estado              # servicio, puerto, heartbeat, últimas placas
cytv placas              # placas de hoy desde el reporte JSON
cytv historial           # historial de una placa
cytv log -Lineas 50      # tail del log principal
cytv fotos C:\fotos\auto.jpg -Canal 02   # probar una foto al pipeline
cytv servicios reiniciar -Fuerza
cytv config -Abrir
cytv limpiar             # borra imágenes con más de 14 días (manual)
cytv ayuda
```

### Limpieza automática de almacenamiento

Hay una **tarea de Windows** (`\felip\CYTV_LimpiezaAlmacenamiento`) que corre **cada 2 semanas (domingo 3:00 AM)** y borra:

- `imagenes_procesadas\placa\YYYY-MM-DD\` y `vehiculo\YYYY-MM-DD\` con más de **14 días**
- `descartadas\` (si existe `modo_descarte = retener`)
- archivos sueltos viejos en `ftp_recibidos\`

Log en `logs\limpieza.log`. Ejecución manual: `cytv limpiar`. Si el equipo está apagado el domingo 3 AM, Windows la ejecuta al próximo encendido (`StartWhenAvailable`).

## Estructura de salida

```
C:\CytvAncon\imagenes_procesadas\
├── placa\<AAAA-MM-DD>\
│   ├── CBN-449.jpg          ← nombre = texto de la placa (OCR)
│   ├── CBN-449.txt          ← etiqueta YOLO (mismo stem)
│   └── placa_16-32-28_107.jpg  ← si OCR falla, nombre por hora
└── vehiculo\<AAAA-MM-DD>\
    ├── placa-panoramica_CBN-449.jpg
    ├── placa-panoramica_CBN-449.txt
    └── vehiculo_16-32-28_107.jpg
```

**Reporte JSON diario** (`reportes/placas-YYYY-MM-DD.json`):

```json
{
  "fecha": "2026-11-11",
  "generado": "2026-11-11T17:20:45",
  "total_placas": 3,
  "total_detecciones": 5,
  "placas": [
    {
      "placa": "CBN-449",
      "primera_vez": "2026-11-11T17:20:45",
      "ultima_vez": "2026-11-11T17:20:47",
      "detecciones": 2,
      "confianza_max": 0.91,
      "imagenes": {
        "placa": ["C:\\CytvAncon\\imagenes_procesadas\\placa\\2026-11-11\\CBN-449.jpg"],
        "panoramica": ["C:\\CytvAncon\\imagenes_procesadas\\vehiculo\\2026-11-11\\placa-panoramica_CBN-449.jpg"]
      }
    }
  ]
}
```

Se refresca con cada placa guardada; a medianoche se cierra el día anterior.

### Consumir desde PowerShell

```powershell
Get-Content C:\CytvAncon\reportes\placas-2026-11-11.json -Raw |
  ConvertFrom-Json | ForEach-Object { $_.placas.placa }
```

## Configuración (`config.ini`)

**Rutas SIEMPRE absolutas** — el servicio corre como SYSTEM y no ve tu perfil.

```ini
[ftp]
puerto = 2121
usuario = admin
password = TU_CLAVE_AQUI
puertos_pasivos = 60000-60099
timeout_segundos = 30
max_conexiones = 20

[rutas]
recibidos = C:\CytvAncon\ftp_recibidos
procesadas = C:\CytvAncon\imagenes_procesadas
registro = C:\CytvAncon\registro.db
logs = C:\CytvAncon\logs
heartbeat = C:\CytvAncon\heartbeat.txt
reportes = C:\CytvAncon\reportes

[camara]
; Mapeo de canales de la TandemVu (ver comentarios dentro del config)
; y control de re-inspección:
ventana_reinspeccion_min = ...   ; minutos mínimos entre visitas de la misma placa
ventana_reinspeccion_dias = ...  ; ventana de días para contar pasadas
max_pasadas = ...                ; máximo de pasadas por ventana

[duplicados]
cooldown_segundos = 10.0
umbral_diferencia = 8.0

[filtros]
modo_descarte = eliminar          ; retener | eliminar
umbral_modelo = 0.5               ; confianza mínima ONNX
ocr_requerido = si                ; si | no
umbral_texto_min = 5.0            ; % píxeles oscuros en el recorte
umbral_texto_max = 55.0
umbral_nitidez = 120.0            ; varianza Laplaciano del recorte
umbral_sobreexpuesto = 2.0        ; % píxeles > 250
umbral_bordes = 2.0               ; % bordes Canny (proxy oclusión)
umbral_dedup_dist = 18            ; distancia Hamming pHash 64-bit
descartes = C:\CytvAncon\descartadas

[log]
nivel = INFO
retencion_dias = 30
```

Los comentarios dentro de tu `config.ini` real documentan cómo se midió cada umbral (producción 2026-10-09). **No edites `[camara]` ni `[filtros]` sin leerlos.**

### Cambiar un valor y reiniciar en un comando

```cmd
scripts\instalar_servicio.bat config set password MiNuevaClave
scripts\instalar_servicio.bat config set umbral_modelo 0.45
scripts\instalar_servicio.bat config editar
```

Luego: `Restart-Service CamaraCytvAncon`.

## Solución de problemas (en este orden)

1. `Get-Service CamaraCytvAncon` → ¿está **Running**?
2. `Get-Content C:\CytvAncon\logs\stderr.log -Tail 20`
3. `Get-Content C:\CytvAncon\logs\servidor.log -Tail 30`
4. `netstat -ano | findstr ":2121"` → ¿alguien más en el puerto?
5. Visor de eventos → System → filtrar por `CamaraCytvAncon` (eventos 7000/7024/7031/7034)

| Síntoma | Causa probable | Solución |
|---------|----------------|----------|
| Servicio **Paused** + "no tiene acceso al archivo" en stderr | El `.exe` del servicio quedó en `C:\Users\...` | Todo debe estar en `C:\CytvAncon`; reinstala |
| `WinError 10048` / "dirección en uso" | Dos instancias peleando por 2121 | `Stop-Service` el viejo antes de arrancar el nuevo |
| Servicio Running pero cámara no sube | IP cambió (sin reserva DHCP) | Mira `INFO_CAMARA.txt`, pon reserva DHCP |
| Modelo no carga | `placas.onnx` ausente o `onnxruntime` no instalado | Reexporta el modelo o reinstala deps en el venv |

### Logs

| Archivo | Qué contiene |
|---------|--------------|
| `logs/servidor.log` | Flujo principal, rotado a medianoche (30 días) |
| `logs/stderr.log` | Errores de arranque (imports, puerto ocupado, permisos) |
| `logs/stdout.log` | Salida estándar del proceso |

El instalador no escribe ningún archivo: todo el detalle se imprime en su ventana de CMD.

Tail en vivo: `Get-Content C:\CytvAncon\logs\servidor.log -Wait -Tail 20`

## Actualizar / reinstalar

```cmd
Stop-Service CamaraCytvAncon
REM Copia los archivos nuevos encima de C:\CytvAncon
scripts\instalar_servicio.bat
```

En otra máquina: copia la carpeta completa, corre el instalador, ajusta las rutas absolutas de `config.ini`, `Restart-Service`, y reconfigura la cámara con la IP de `INFO_CAMARA.txt`.

## Reglas de oro

- **NUNCA** corras `scripts\instalar_servicio.bat` desde una copia vieja: re-apuntaría el servicio.
- **NUNCA** muevas `C:\CytvAncon` con el servicio corriendo.
- **NUNCA** pongas rutas relativas en `config.ini`.
- Puerto **2121 = UN solo servidor**. Para migrar: para el viejo, recién arranca el nuevo.
- Sin reserva DHCP, la IP del servidor puede cambiar y la cámara deja de subir.

---

# Para desarrolladores

## Arquitectura

```
┌───────────────────┐   FTP 2121    ┌────────────────────────────────────────┐
│ Cámara Hikvision  │ ────────────▶ │ ftp_server.py (raíz)                   │
│ TandemVu 2 canales│   STOR/226    │  └─ hilo FTP: solo recibe y encola     │
└───────────────────┘               │        │ worker.COLA                  │
                                    │        ▼                              │
                                    │ app\worker.py                         │
                                    │  cadena de filtros → guardado          │
                                    │   ├─ app\modelo.py   YOLOv8n ONNX     │
                                    │   └─ app\ocr.py      variantes + CRNN  │
                                    │       └─ app\reconocedor.py (ONNX)    │
                                    └────────┬───────────────────────────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────┐
              ▼                              ▼                      ▼
     imagenes_procesadas\             reportes\placas-*.json   descartadas\
     (placa\ y vehiculo\              (JSON diario, se         (solo modo
      por día, renombrados             refresca solo)           retener)
      por placa)                 registro.db (SQLite: dedup, historial, pasadas)
```

**Hilos del proceso** (`ftp_server.py` es la única entrada y la referencia NSSM):

| Hilo | Qué hace |
|------|----------|
| FTP | Acepta `STOR`, encola en `worker.COLA`, responde 226 al instante |
| Worker | Único consumidor de la cola; filtros, OCR, guardado, JSON |
| Heartbeat | Toca `heartbeat.txt` cada 15 s (watchdog externo) |
| Barrido | Cada 60 s recupera archivos huérfanos en `ftp_recibidos\` |

**Caducidad del día**: el dedup vive solo dentro del día calendario; a medianoche se cierra el JSON y se reinician las huellas.

### Cadena de filtros del worker (orden exacto)

Tomado del docstring de `app\worker.py`:

```
cooldown → diferencia visual → candidato (ONNX o geométrico) → hay texto
→ nitidez del recorte → sobreexposición → oclusión (proxy) → OCR
→ dedup perceptual (pHash) → control de pasadas → guardar
```

| # | Filtro | Detalle |
|---|--------|---------|
| 1 | Cooldown | Bloqueo estricto por vista (`cooldown_segundos`) |
| 2 | Diferencia visual | ≥ 8 % de cambio de píxeles vs. la última guardada en esa vista |
| 3 | Candidatos | `modelo\placas.onnx` (YOLOv8n); si falta o falla, detector geométrico OpenCV |
| 4 | Texto | 5–55 % píxeles oscuros en el recorte (calidad de la lectura) |
| 5 | Nitidez | Varianza del Laplaciano ≥ 120, **sobre el recorte** |
| 6 | Sobreexposición | ≤ 2 % píxeles saturados (> 250) — IR quemando la placa |
| 7 | Oclusión (proxy) | ≥ 2 % bordes Canny en el recorte |
| 8 | OCR | `ocr.leer_placa_6` (ver motor de lectura abajo) |
| 9 | Dedup | pHash 64-bit, Hamming ≤ 18; **dedup exacto por texto** además del perceptual |
| 10 | Pasadas (Fase 1) | `ventana_reinspeccion_min` / `ventana_reinspeccion_dias` / `max_pasadas`; al exceder se guarda con sufijo `EXCEDIDA` + WARNING |
| 11 | Guardado | JPEG calidad 100 + `.txt` YOLO; renombrado por placa si OCR leyó |

**Dos vistas, dos dedups independientes**: la misma matrícula en `placa\` y `vehiculo\` son dos archivos distintos y queridos.

## Estructura del repo

```
CytvAncon\
├── ftp_server.py        ← ÚNICA entrada (referencia NSSM; no mover)
├── config.ini           ← rutas absolutas de datos (no mover)
├── cytv.cmd             ← acceso rápido: cytv estado | placas | log ...
├── README.md, requirements*.txt
│
├── app\                 ← código del servicio (Python, imports planos)
│   ├── worker.py        cola de trabajo, filtros, pasadas, JSON
│   ├── ocr.py           leer_placa_6: CRNN primero, fallback 6 intentos
│   ├── reconocedor.py   CRNN propio (ONNX numpy/PIL/ort, sin torch)
│   ├── modelo.py        detector ONNX (YOLOv8n) con fallback geométrico
│   ├── consola.py       consola interactiva (scripts\consola.bat)
│   ├── decodificador_flota.py   heredado, sin uso en el pipeline (flota eliminada 2026-10-21)
│   └── gt_placas.py     tabla GT de los 29 diagnósticos
│
├── modelo\              ← pesos ONNX/PT (placas.onnx, reconocedor.onnx, ...)
├── scripts\             ← operación: nssm.exe, instalar*, cytv.ps1,
│                          consola.bat, reiniciar/reinstalar, limpieza
├── docs\                ← GUIA-COMANDOS.txt, INFO_CAMARA.txt*, respaldos\
├── entrenamiento\       ← CRNN, validador, etiquetas, herramientas
│
└── (datos, en rutas absolutas de config.ini — NO mover ni subir):
    ftp_recibidos\  imagenes_procesadas\  reportes\  registro.db  logs\  descartadas\
```

`INFO_CAMARA.txt` lleva `*`: contiene credenciales y está en `.gitignore` (solo local).

El servicio arranca con `sys.path` apuntando a `app\`, de modo que todos los imports internos son planos (`import worker`) sin importar desde qué carpeta se lance.

### Entornos: dos Pythons, sin ambigüedad

| Entorno | Python | Qué corre | Torch |
|---------|--------|-----------|-------|
| **Servicio** (`C:\CytvAncon\venv`) | 3.9 o 3.13 (según el SO) | `ftp_server.py` 24/7; solo ONNX Runtime (1 hilo) | **Nunca** |
| **Entrenamiento** (`C:\CytvAncon\venv`, local) | 3.13 + torch/`onnxscript` | `entrenamiento\*.py`; nunca se despliega al server | N/A (local) |

El servicio **nunca** importa torch: el CRNN se ejecuta exportado en `modelo\reconocedor.onnx` vía onnxruntime + numpy + PIL.

## Motor de lectura: CRNN + votación de variantes (sin flota)

### Cómo lee una placa

1. **Detección** — YOLOv8n (`modelo\placas.onnx`) propone bboxes; score ≥ `umbral_modelo`.
2. **Recorte** — bbox + margen 35 %, se normaliza a la entrada del lector.
3. **Variantes** — `leer_placa_6` (`app\ocr.py`) genera recortes alternativos y vota con RapidOCR sobre ellos; cada intento produce un texto candidato.
4. **CRNN** (`app\reconocedor.py`) — red propia CRNN+CTC exportada a ONNX. Si las variantes vieron texto, el CRNN entra como **un voto más** en el grupo de equivalencia; si es el único que habla, **se abstiene** (no inventa texto).
5. **Límite de formato** — la lectura final debe encajar en un formato de placa peruana real (`_FORMATOS` / `_normalizar` en `app\ocr.py`); si no encaja, se descarta.

La **flota cerrada y el snap de placas se eliminaron el 2026-10-21**: el motor no consulta ninguna lista de placas conocidas ni acepta texto solo por pertenecer a un catálogo.

### Métricas (set de validación = 29 diagnósticos de `app\gt_placas.py`)

| Motor | Acierto | Notas |
|-------|---------|-------|
| RapidOCR 6 intentos (heredado) | 55,2 % | Motor anterior, hoy es el voto principal de `leer_placa_6` |
| CRNN crudo (fine-tune) | 41,4 % | 20 épocas, mix 50/50 real+sintético |
| CRNN + flota cerrada (histórico) | 86,2 % | **Eliminado 2026-10-21** |
| **Pipeline actual (sin flota)** | **44,8 %** | `validar_diagnosticos.py`, medido 2026-10-21 |

Bancos de pasadas (fotos reales de tránsito; métrica "placa correcta por pasada"): **69,8 % (81/116)** en el banco y **72,0 % (67/93)** en el cotejo, con el pipeline sin flota y con los límites de hilos.

Al quitar la flota cerrada se perdió su corrector de letras legales confusas (H↔W, N↔M, U↔V): como ambas letras son válidas en placas peruanas, eso **no se arregla con reglas** — solo con datos reales vía fine-tune.

**Caveat honesto**: 4 camiones del set de validación también aparecen en las fotos de entrenamiento (`etiquetas_reales.csv`), así que el número real sobre placas nuevas es un poco menor. La meta del proyecto es ≥ 90 %.

### Contratos técnicos

| Contrato | Valor |
|----------|-------|
| Entrada del CRNN | `(1, 1, 32, 160)` — PIL `convert("L").resize((160,32), BILINEAR) / 255`; BGR→RGB antes de PIL |
| CHARSET | `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-` (37 glifos) + CTC blank = índice 37; `N_CLASSES = 38` |
| Límite de formato | `_FORMATOS` / `_normalizar` en `app\ocr.py`: la lectura debe parecer una placa peruana (ABC-123, AB-1234, ...); lo que no encaja se descarta |
| Votos | Las variantes de `leer_placa_6` y el CRNN deben apuntar al mismo texto; el CRNN solo no basta (anti-fabricación) |
| Datos reales | `gt_placas.DIAGNOSTICOS` (29, holdout **sagrado**, jamás se entrena) + `entrenamiento\etiquetas_reales.csv` (87 filas, 74 placas) para fine-tune |

## Desarrollo y reentrenamiento

### Correr el validador (la métrica oficial del proyecto)

```powershell
# desde la raíz del repo, con el Python del servicio (sin torch)
python entrenamiento\validar_diagnosticos.py
```

Debe imprimir `TOTAL: 13/29 = 44.8% ... NO PASA` (medido 2026-10-21, sin flota cerrada; con flota daba 86,2 %). Los 29 diagnósticos **jamás se entrenan** — son el holdout permanentemente sagrado.

### Ciclo de mejora con datos reales

```powershell
# 1. Etiquetar a mano los crops nuevos (montajes en entrenamiento\montajes\)
#    → filas nuevas en entrenamiento\etiquetas_reales.csv (PLACA,filename)

# 2. Fine-tune sintetico + real (con torch; imprime GT_TEST por epoca)
& C:\CytvAncon\venv\Scripts\python.exe entrenamiento\entrenar.py
& C:\CytvAncon\venv\Scripts\python.exe entrenamiento\entrenar_real.py --desde entrenamiento\crnn_reales.pt

# 3. Re-validar el pipeline integrado (metrica oficial)
python entrenamiento\validar_diagnosticos.py

# 4. Publicar: el export a ONNX de entrenar_real.py ya escribe
#    modelo\reconocedor.onnx — reiniciar el servicio para cargarlo
Restart-Service CamaraCytvAncon
```

Reglas del dataset:

- Las placas del GT (los 29) **no** se agregan a `etiquetas_reales.csv` como material de entrenamiento sin decidirlo explícitamente.
- Un archivo con dos placas visibles se anota con la del canal de placa; si está al borde y se corta, se descarta (no se inventa el carácter perdido).
- Los pesos se guardan en `entrenamiento\crnn_reales.pt`; el servicio usa solo el `.onnx`.

### Detector de placas (YOLOv8n)

```powershell
# exportar un best.pt ya entrenado a modelo\placas.onnx
python entrenamiento\entrenar_detector.py --solo-export C:\ruta\best.pt
```

Si `placas.onnx` no existe o falla al cargar, el worker cae al detector geométrico (OpenCV puro) y las imágenes quedan en `pendientes/` — el servicio nunca muere por el modelo.

### Convenciones del repo

- Rama oficial: `master` (repo público `Vicharra463/Aplicacion-Camera`).
- Commits convencionales (`feat:`, `fix:`, `chore:`); sin atribuciones de IA.
- **Nunca** subir credenciales: `config.ini` es la única excepción histórica — rotá la passwordFTP y la de la cámara si el repo es público.
- Datos de ejecución fuera de git (`.gitignore`); `modelo\respaldo\` y `docs\respaldos\` son locales.
- Después de editar código del servicio: `python -m py_compile` en cada `.py` tocado y correr el validador antes de dar nada por terminado.

## Licencia

Uso interno — CYTV Ancon. Código basado en Python estándar, OpenCV, ONNX Runtime, pyftpdlib, NSSM; entrenamiento con PyTorch y Ultralytics.
