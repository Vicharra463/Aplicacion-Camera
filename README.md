# Servidor FTP de Placas CYTV Ancon

Sistema 24/7 que recibe imágenes de una cámara Hikvision TandemVu (2 canales), detecta y lee placas peruanas, y las organiza en carpetas diarias con reportes JSON listos para consumir.

---

## Arquitectura rápida

```
┌──────────────────┐     FTP 2121      ┌──────────────────┐
│  Cámara Hikvision │ ───────────────▶ │  ftp_server.py   │  (hilo FTP: solo recibe y encola)
│  Canal 01 (pano)  │                  │  worker.py       │  (anti-dup + filtros + guardado)
│  Canal 02 (placa) │                  │  modelo.py       │  (ONNX YOLOv8n - detector placas)
└──────────────────┘                  │  ocr.py          │  (RapidOCR - lectura texto)
                                      │  registro.db     │  (SQLite - dedup + historial)
                                      └────────┬─────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    ▼                          ▼                          ▼
           ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
           │ imagenes_pro- │          │   reportes/   │          │  descartadas/ │
           │ cesadas/      │          │  placas-      │          │  (solo modo   │
           │  placa/       │          │  YYYY-MM-DD   │          │   retener)    │
           │  vehiculo/    │          │  .json        │          └───────────────┘
           └───────────────┘          └───────────────┘
                    ▲                          ▲
                    │                          │
           ┌────────┴────────┐         ┌───────┴────────┐
           │  webapp/        │         │  consola.py /  │
           │  (React + Node) │         │  cytv.ps1      │
           │  localhost:3000 │         │  (operación)   │
           └─────────────────┘         └────────────────┘
```

**Hilos del servicio:**
- **FTP** (`ftp_server.py`) — recibe `STOR`, encola en `worker.COLA`, devuelve 226 al instante.
- **Worker** (`app/worker.py`) — único consumidor; cadena de filtros estricta, dedup por hash perceptual + texto OCR, renombrado por placa.
- **Heartbeat** — toca `heartbeat.txt` cada 15 s (watchdog externo).
- **Barrido** — cada 60 s recupera archivos huérfanos en `ftp_recibidos/`.

---

## Estructura del proyecto

```
CytvAncon\
├── ftp_server.py        ← ÚNICA entrada (la referencia NSSM; no mover)
├── config.ini           ← rutas absolutas de datos (no mover)
├── cytv.cmd             ← acceso rápido: cytv estado | placas | log ...
├── README.md, requirements*.txt
│
├── app\                 ← código del servicio (Python)
│   ├── worker.py        cola de trabajo y filtros
│   ├── ocr.py           lector de placas (CRNN + motor 6 intentos)
│   ├── reconocedor.py   CRNN propio (ONNX + flota cerrada)
│   ├── modelo.py        detector ONNX (YOLOv8n)
│   ├── consola.py       consola interactiva (scripts\consola.bat)
│   ├── decodificador_flota.py, gt_placas.py
│
├── modelo\              ← pesos ONNX (placas.onnx, reconocedor.onnx)
├── scripts\             ← operación: nssm.exe, instalar*, cytv.ps1,
│                          consola.bat, reiniciar/reinstalar, limpieza
├── docs\                ← GUIA-COMANDOS.txt, INFO_CAMARA.txt, respaldos\
├── entrenamiento\       ← CRNN, validación, herramientas de diagnóstico
│
└── (datos, en rutas absolutas de config.ini — NO mover):
    ftp_recibidos\  imagenes_procesadas\  reportes\  registro.db  logs\  descartadas\
```

---

## Requisitos

- **Windows** 8.1 / Server 2012 R2 o superior (64 bits)
- **Python** 3.9 (Win ≤ 10 build < 14393) o 3.13 (Win 10 1607+) — lo instala el script
- **Node.js** 18+ (solo para la app web opcional)
- **Firewall** puertos TCP **2121** (FTP) y **60000–60099** (pasivo)
- **Reserva DHCP** en el router para la IP fija del servidor

---

## Instalación (una sola vez)

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

---

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

[duplicados]
cooldown_segundos = 10.0
umbral_diferencia = 8.0

[filtros]
modo_descarte = eliminar          ; retener | eliminar
umbral_modelo = 0.5               ; confianza mínima ONNX
ocr_requerido = si                ; si | no
umbral_texto_min = 5.0            ; % pixeles oscuros en recorte
umbral_texto_max = 55.0
umbral_nitidez = 120.0            ; varianza Laplaciano del recorte
umbral_sobreexpuesto = 2.0        ; % pixeles > 250
umbral_bordes = 2.0               ; % bordes Canny (proxy oclusión)
umbral_dedup_dist = 18            ; distancia Hamming pHash 64-bit
descartes = C:\CytvAncon\descartadas

[log]
nivel = INFO
retencion_dias = 30
```

### Cambiar un valor y reiniciar en un comando

```cmd
scripts\instalar_servicio.bat config set password MiNuevaClave
scripts\instalar_servicio.bat config set umbral_modelo 0.45
```

O editar a mano y reiniciar:
```cmd
scripts\instalar_servicio.bat config editar
Restart-Service CamaraCytvAncon
```

---

## Puesta en marcha de la cámara (Hikvision)

1. Abre `C:\CytvAncon\INFO_CAMARA.txt` (se regenera en cada arranque).
2. En la web de la cámara (`http://192.168.18.42` o su IP):
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
4. **Importante**: si solo sube el canal 01, la carpeta `placa\` no recibe fotos.

---

## Operación diaria

### PowerShell (`cytv.ps1`)

```powershell
# Desde cualquier carpeta
cytv estado
cytv placas
cytv historial
cytv log -Lineas 50
cytv fotos C:\fotos\auto.jpg -Canal 02
cytv servicios reiniciar -Fuerza
cytv config -Abrir
cytv web
cytv limpiar          # borra imagenes con mas de 14 dias (manual)
cytv ayuda
```

### Limpieza automática de almacenamiento

Con ~20+ carros/día las fotos llenan el disco en semanas. Hay una **tarea de Windows** (`\felip\CYTV_LimpiezaAlmacenamiento`) que corre **cada 2 semanas (domingo 3:00 AM)** y borra:

- `imagenes_procesadas\placa\YYYY-MM-DD\` y `vehiculo\YYYY-MM-DD\` con más de **14 días**
- `descartadas\` (si existe `modo_descarte = retener`)
- archivos sueltos viejos en `ftp_recibidos\`

Log en `logs\limpieza.log`. Ejecución manual: `cytv limpiar`. El equipo debe estar encendido el domingo 3 AM (si está apagada, Windows la ejecuta al próximo encendido gracias a `StartWhenAvailable`).

### App Web (localhost:3000)

```cmd
cd C:\CytvAncon\webapp
npm install
npm run build
node server.js
```
Abre `http://localhost:3000`. La UI tiene dos filas:

1. **Subir foto** — arrastre o clic; se muestra el preview.
2. **Resultado** — al instante el servidor corre el **mismo modelo ONNX + OCR** que la cámara real:
   - Si hay placa: bandera verde **PLACA DETECTADA**, texto OCR, confianza, **dónde está** (bbox `x / y` en px) y el recorte + foto anotada.
   - Si no: **X grande** roja **SIN DETECTAR PLACA**.

El endpoint es `POST /api/evaluar` (multer → `evaluar_web.py` → JSON con base64). No toca el pipeline del servicio: es solo evaluación sincrónica para feedback inmediato.

---

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
Se **refresca automático** con cada placa guardada; a medianoche se cierra el día anterior.

### Consumir desde PowerShell
```powershell
Get-Content C:\CytvAncon\reportes\placas-2026-11-11.json -Raw |
  ConvertFrom-Json | ForEach-Object { $_.placas.placa }
# CBN-449
# ABC-1234
```

---

## Entrenamiento del detector (Fase 1 → Fase 2)

El servicio usa **solo ONNX** (`modelo/placas.onnx`). El entrenamiento se hace **fuera** del servicio:

```cmd
REM En tu máquina de ML (con GPU y entorno ultralytics):
C:\Users\felip\anpr\venv\Scripts\python.exe entrenar_detector.py --epocas 60
```

- Dataset: `C:\Users\felip\anpr\dataset_fuente\raw\data.yaml` (8 823 imágenes, clase `license_plate`)
- Salida: `runs\placas\weights\best.pt` → se exporta a ONNX y **publica** en `C:\CytvAncon\modelo\placas.onnx`
- Si ya tienes `best.pt`:
  ```cmd
  python entrenar_detector.py --solo-export C:\ruta\best.pt
  ```

El servicio **nunca falla por el modelo**: si `placas.onnx` no existe o falla al cargar, cae al detector geométrico (OpenCV puro) y las imágenes quedan en `pendientes/`.

---

## Filtros del worker (orden exacto)

1. **Cooldown** (10 s por vista) — bloqueo estricto por tiempo.
2. **Diferencia visual** (≥ 8 % de cambio de píxeles vs. última guardada en esa vista).
3. **Candidatos de placa** — detector ONNX (si hay modelo) o geometría (contornos, relación 1.6–6.0).
4. **Texto en recorte** (solo modo geométrico) — 5–55 % píxeles oscuros.
5. **Nitidez del recorte** — varianza Laplaciano ≥ 120.
6. **Sobreexposición** — ≤ 2 % píxeles saturados (> 250).
7. **Oclusión (proxy)** — ≥ 2 % bordes Canny.
8. **OCR** (RapidOCR) — lectura del recorte con margen 35 %; dedup exacto por texto.
9. **Dedup perceptual** — pHash 64-bit, distancia Hamming ≤ 18.
10. **Guardado** — JPEG calidad 100 + `.txt` YOLO; renombrado por placa si OCR leyó.

**Dos vistas, dos dedups independientes**: la misma matrícula en `placa\` y `vehiculo\` son dos archivos distintos y queridos.

---

## Logs y diagnóstico

| Archivo | Qué contiene |
|---------|--------------|
| `logs/servidor.log` | Log principal rotado a medianoche (30 días). Flujo: `Imagen recibida → Placa detectada / No se detecta placa → Imagen eliminada/movida` |
| `logs/stderr.log` | Errores de arranque Python (imports, puerto ocupado, permisos) |
| `logs/stdout.log` | Salida estándar del proceso |
| `instalacion.log` | Última corrida del instalador |

**Tail en vivo:**
```powershell
Get-Content C:\CytvAncon\logs\servidor.log -Wait -Tail 20
```

**Buscar placas:**
```powershell
Select-String -Path C:\CytvAncon\logs\servidor.log -Pattern "placa"
```

---

## Solución de problemas (orden)

1. `Get-Service CamaraCytvAncon` → ¿está **Running**?
2. `Get-Content C:\CytvAncon\logs\stderr.log -Tail 20`
3. `Get-Content C:\CytvAncon\logs\servidor.log -Tail 30`
4. `netstat -ano | findstr ":2121"` → ¿alguien más en el puerto?
5. Visor de eventos → System → filtrar por `CamaraCytvAncon` (eventos 7000/7024/7031/7034)

### Causas frecuentes

| Síntoma | Causa probable | Solución |
|---------|----------------|----------|
| Servicio **Paused** + "no tiene acceso al archivo" en stderr | El `.exe` del servicio quedó en `C:\Users\...` | Todo debe estar en `C:\CytvAncon`; reinstala |
| `WinError 10048` / "dirección en uso" | Dos instancias peleando por 2121 | `Stop-Service` el viejo antes de arrancar el nuevo |
| Servicio Running pero cámara no sube | IP cambió (sin reserva DHCP) | Mira `INFO_CAMARA.txt`, pon reserva DHCP en el router |
| Modelo no carga | `placas.onnx` no existe o `onnxruntime` no instalado | `python entrenar_detector.py --solo-export ...` o `pip install onnxruntime` en el venv |

---

## Actualizar / Reinstalar en la misma máquina

```cmd
Stop-Service CamaraCytvAncon
REM Copia los archivos nuevos encima de C:\CytvAncon
scripts\instalar_servicio.bat
```
El instalador reutiliza venv y Python si ya están bien.

---

## En otra máquina

1. Copia la **carpeta completa del proyecto** a cualquier ruta.
2. Corre `scripts\instalar_servicio.bat` (instala Python, venv, deps, NSSM, firewall, servicio).
3. Edita `config.ini` con las rutas absolutas de **esa** máquina.
4. `Restart-Service CamaraCytvAncon`.
5. Configura la cámara con la nueva IP (`INFO_CAMARA.txt`).

---

## Reglas de oro

- **NUNCA** corras `scripts\instalar_servicio.bat` desde una copia vieja (Escritorio, etc.): re-apuntaría el servicio.
- **NUNCA** muevas `C:\CytvAncon` con el servicio corriendo.
- **NUNCA** pongas rutas relativas en `config.ini`.
- Puerto **2121 = UN solo servidor**. Para migrar: para el viejo, recién arranca el nuevo.
- La cámara apunta a la IP de `INFO_CAMARA.txt`; sin reserva DHCP, esa IP puede cambiar.

---

## Licencia

Uso interno — CYTV Ancon. Código basado en Python estándar, OpenCV, ONNX Runtime, RapidOCR, pyftpdlib, NSSM, React/Vite/Express.