// Servidor web CYTV Ancon - sube fotos por el MISMO FTP que usa la camara.
//
// Por que FTP y no una carpeta compartida: el handler del servidor Python
// encola al worker solo lo que llega por STOR (on_file_received). Subir por
// FTP hace que el modelo, el OCR, los filtros, el renombrado por placa y el
// JSON diario corran IDENTICO a cuando sube la camara real. Cero cambios
// en el pipeline.
//
// Endpoints:
//   POST /api/subir   multipart "fotos" (+ campo "canal" = 01|02) -> FTP 2121
//   GET  /api/estado  servicio + placas del dia + cola del log
//   GET  /            build del frontend React (dist/)
import express from "express";
import multer from "multer";
import ftp from "basic-ftp";
import fs from "fs";
import net from "net";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const RAIZ = path.resolve(__dirname, ".."); // C:\CytvAncon

// --- Config del servidor Python (fuente unica: config.ini) ----------------
function leerConfig() {
  const ini = fs.readFileSync(path.join(RAIZ, "config.ini"), "utf8");
  const seccion = (nombre) => {
    const m = ini.match(new RegExp(`\\[${nombre}\\]([\\s\\S]*?)(?=\\n\\[|$)`));
    return m ? m[1] : "";
  };
  const clave = (seccionTxt, claveTxt, defecto) => {
    const m = seccionTxt.match(new RegExp(`^\\s*${claveTxt}\\s*=\\s*(.+)$`, "m"));
    return m ? m[1].trim() : defecto;
  };
  const ftpTxt = seccion("ftp");
  return {
    puertoFtp: Number(clave(ftpTxt, "puerto", "2121")),
    usuario: clave(ftpTxt, "usuario", "admin"),
    password: clave(ftpTxt, "password", ""),
    reportes: (() => {
      const rutas = seccion("rutas");
      const rel = clave(rutas, "reportes", "reportes");
      return path.isAbsolute(rel) ? rel : path.join(RAIZ, rel);
    })(),
    logs: (() => {
      const rutas = seccion("rutas");
      const rel = clave(rutas, "logs", "logs");
      return path.isAbsolute(rel) ? rel : path.join(RAIZ, rel);
    })(),
  };
}
const cfg = leerConfig();

// --- Subida de archivos (en memoria; el unico destino es el FTP) ----------
const subir = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024, files: 8 },
  fileFilter: (_req, file, cb) => {
    if (/^image\/(jpeg|png|webp|bmp|tiff)$/.test(file.mimetype)) return cb(null, true);
    cb(new Error("Solo se aceptan imagenes (JPG, PNG, WEBP, BMP, TIFF)"));
  },
});

// Nombre al estilo de la camara: web_<canal>_<yyyymmddHHMMSSmmm>_MD_WITH_TARGET.jpg
// El worker elige la vista con el patron _(\\d{2})_ del nombre, igual que
// con la camara real (…_01_… panoramica, …_02_… placa).
function nombreCamara(canal, n) {
  const d = new Date();
  const p = (v, l = 2) => String(v).padStart(l, "0");
  const ts = `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}` +
             `${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}` +
             `${p(d.getMilliseconds(), 3)}`;
  return `web_${canal}_${ts}${n ? `_${n}` : ""}_MD_WITH_TARGET.jpg`;
}

const app = express();
app.use(express.json());

// --- POST /api/subir: recibe fotos y las empuja al FTP 2121 ---------------
app.post("/api/subir", subir.array("fotos", 8), async (req, res) => {
  const canal = req.body.canal === "02" ? "02" : "01";
  if (!req.files || req.files.length === 0) {
    return res.status(400).json({ ok: false, error: "No llego ninguna foto" });
  }
  const cliente = new ftp.Client();
  cliente.timeout = 15000;
  try {
    await cliente.access({
      host: "127.0.0.1",
      port: cfg.puertoFtp,
      user: cfg.usuario,
      password: cfg.password,
      secure: false,
    });
    const enviadas = [];
    for (let i = 0; i < req.files.length; i++) {
      const f = req.files[i];
      const nombre = nombreCamara(canal, req.files.length > 1 ? i + 1 : "");
      const { Readable } = await import("stream");
      await cliente.uploadFrom(Readable.from(f.buffer), nombre);
      enviadas.push({ nombre, bytes: f.size, original: f.originalname });
    }
    res.json({ ok: true, canal, enviadas,
      mensaje: `${enviadas.length} foto(s) enviada(s) al pipeline (canal ${canal}). ` +
               "El modelo las evalua en 1-3 s; vea el resultado abajo." });
  } catch (err) {
    const detalle = String(err && err.message || err);
    const esPuesto = /ECONNREFUSED|timed?\s*out|EHOSTUNREACH/i.test(detalle);
    res.status(502).json({
      ok: false,
      error: esPuesto
        ? `No hay servidor FTP en el puerto ${cfg.puertoFtp}. ` +
          "Revise que el servicio 'CamaraCytvAncon' este Running (Get-Service CamaraCytvAncon)."
        : `Fallo al subir por FTP: ${detalle}`,
    });
  } finally {
    cliente.close();
  }
});

// --- GET /api/estado: para que el front muestre que paso ------------------
function tcpAbierto(puerto, ms = 1500) {
  return new Promise((resuelve) => {
    const s = net.connect({ port: puerto, host: "127.0.0.1" });
    const fin = (v) => { s.destroy(); resuelve(v); };
    s.setTimeout(ms, () => fin(false));
    s.once("connect", () => fin(true));
    s.once("error", () => fin(false));
  });
}

function ultimasLineas(ruta, n = 18) {
  try {
    const fd = fs.openSync(ruta, "r");
    const tam = fs.fstatSync(fd).size;
    const buf = Buffer.alloc(Math.min(tam, 12 * 1024));
    fs.readSync(fd, buf, 0, buf.length, Math.max(0, tam - buf.length));
    fs.closeSync(fd);
    return buf.toString("utf8").trimEnd().split(/\r?\n/).slice(-n);
  } catch {
    return [];
  }
}

function reporteHoy() {
  const d = new Date();
  const p = (v) => String(v).padStart(2, "0");
  const hoy = `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
  try {
    return JSON.parse(fs.readFileSync(path.join(cfg.reportes, `placas-${hoy}.json`), "utf8"));
  } catch {
    return { fecha: hoy, total_placas: 0, placas: [] };
  }
}

// Cache 30 s: cada chequeo TCP crea una sesion FTP que pyftpdlib registra
// en el log; sin cache el log se llena de open/close cada 5 s del polling.
let ftpCache = { v: false, t: 0 };
async function ftpDisponible() {
  const ahora = Date.now();
  if (ahora - ftpCache.t > 30000) {
    ftpCache = { v: await tcpAbierto(cfg.puertoFtp), t: ahora };
  }
  return ftpCache.v;
}

app.get("/api/estado", async (_req, res) => {
  res.json({
    ftp: await ftpDisponible(),
    puertoFtp: cfg.puertoFtp,
    reporte: reporteHoy(),
    log: ultimasLineas(path.join(cfg.logs, "servidor.log")),
  });
});

// --- Frontend buildado (dist/) --------------------------------------------
const dist = path.join(__dirname, "dist");
app.use(express.static(dist));
app.get("*", (_req, res) => res.sendFile(path.join(dist, "index.html")));

const PUERTO_WEB = 3000;
app.listen(PUERTO_WEB, "0.0.0.0", () => {
  console.log(`Web CYTV Ancon: http://localhost:${PUERTO_WEB}`);
  console.log(`FTP destino    : 127.0.0.1:${cfg.puertoFtp} (usuario ${cfg.usuario})`);
});
