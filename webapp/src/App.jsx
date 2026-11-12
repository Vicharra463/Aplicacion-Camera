import React, { useCallback, useEffect, useRef, useState } from "react";

// Pantalla unica: elegir canal (como la camara real), subir fotos, y ver
// que decidio el pipeline (modelo + OCR + filtros) leyendo el log y el
// reporte JSON del dia. Sin frameworks de UI: sencilla y directa.
export default function App() {
  const [canal, setCanal] = useState("01");
  const [archivos, setArchivos] = useState([]);
  const [enviando, setEnviando] = useState(false);
  const [mensaje, setMensaje] = useState(null); // {tipo: ok|error, texto}
  const [estado, setEstado] = useState(null);
  const [arrastrando, setArrastrando] = useState(false);
  const inputRef = useRef(null);

  const leerEstado = useCallback(async () => {
    try {
      const r = await fetch("/api/estado");
      setEstado(await r.json());
    } catch {
      /* el backend esta caido; se reintenta en el proximo tick */
    }
  }, []);

  useEffect(() => {
    leerEstado();
    const t = setInterval(leerEstado, 5000);
    return () => clearInterval(t);
  }, [leerEstado]);

  async function enviar() {
    if (archivos.length === 0 || enviando) return;
    setEnviando(true);
    setMensaje(null);
    const fd = new FormData();
    fd.append("canal", canal);
    for (const f of archivos) fd.append("fotos", f);
    try {
      const r = await fetch("/api/subir", { method: "POST", body: fd });
      const d = await r.json();
      if (d.ok) {
        setMensaje({ tipo: "ok", texto: d.mensaje });
        setArchivos([]);
        if (inputRef.current) inputRef.current.value = "";
        // El worker tarda 1-3 s: refrescamos varias veces para ver el resultado.
        leerEstado();
        setTimeout(leerEstado, 2500);
        setTimeout(leerEstado, 6000);
      } else {
        setMensaje({ tipo: "error", texto: d.error });
      }
    } catch (e) {
      setMensaje({ tipo: "error", texto: "No se pudo contactar al servidor: " + e.message });
    } finally {
      setEnviando(false);
    }
  }

  function alElegir(lista) {
    setArchivos(Array.from(lista).slice(0, 8));
    setMensaje(null);
  }

  const vista = canal === "02" ? "PLACA (canal 02, tele)" : "PANORAMICA (canal 01, 94°)";

  return (
    <div className="app">
      <header>
        <h1>Camara CYTV Ancon</h1>
        <p className="sub">
          Suba fotos y el servidor las procesa <b>por el mismo FTP y el mismo
          modelo</b> que usa la camara real.
        </p>
      </header>

      <section className="tarjeta">
        <h2>1. Canal</h2>
        <div className="canal">
          {[
            { id: "01", txt: "Panoramica (01)" },
            { id: "02", txt: "Placa / tele (02)" },
          ].map((o) => (
            <label key={o.id} className={canal === o.id ? "elegido" : ""}>
              <input
                type="radio"
                name="canal"
                checked={canal === o.id}
                onChange={() => setCanal(o.id)}
              />
              {o.txt}
            </label>
          ))}
        </div>
        <p className="nota">
          Se enviaran como canal <b>{canal}</b> &rarr; carpeta{" "}
          <code>{canal === "02" ? "placa\\" : "vehiculo\\"}</code>. Vista actual: {vista}.
        </p>
      </section>

      <section className="tarjeta">
        <h2>2. Fotos (max 8)</h2>
        <div
          className={"soltar" + (arrastrando ? " activo" : "")}
          onDragOver={(e) => { e.preventDefault(); setArrastrando(true); }}
          onDragLeave={() => setArrastrando(false)}
          onDrop={(e) => {
            e.preventDefault();
            setArrastrando(false);
            alElegir(e.dataTransfer.files);
          }}
          onClick={() => inputRef.current && inputRef.current.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            multiple
            hidden
            onChange={(e) => alElegir(e.target.files)}
          />
          {archivos.length === 0
            ? "Arrastre fotos aqui o haga clic para elegir"
            : `${archivos.length} foto(s) elegida(s): ` +
              archivos.map((f) => f.name).join(", ")}
        </div>
        <button className="boton" disabled={enviando || archivos.length === 0} onClick={enviando ? undefined : enviar}>
          {enviando ? "Enviando al servidor..." : "Subir y evaluar"}
        </button>
        {mensaje && <p className={"msg " + mensaje.tipo}>{mensaje.texto}</p>}
      </section>

      <section className="tarjeta">
        <h2>3. Resultado del dia</h2>
        {estado && (
          <>
            <p className={estado.ftp ? "okline" : "errorline"}>
              FTP {estado.puertoFtp}: {estado.ftp ? "respondiendo" : "NO responde - revise el servicio"}
            </p>
            <h3>
              Placas de hoy ({estado.reporte.fecha}): {estado.reporte.total_placas}
            </h3>
            {estado.reporte.placas.length === 0 ? (
              <p className="nota">Todavia ninguna placa guardada hoy.</p>
            ) : (
              <table>
                <thead>
                  <tr><th>Placa</th><th>1a vez</th><th>Confianza</th><th>Fotos</th></tr>
                </thead>
                <tbody>
                  {estado.reporte.placas.map((p) => (
                    <tr key={p.placa}>
                      <td><b>{p.placa}</b></td>
                      <td>{p.primera_vez.slice(11)}</td>
                      <td>{(p.confianza_max * 100).toFixed(0)}%</td>
                      <td>
                        {p.imagenes.placa.length} placa / {p.imagenes.panoramica.length} pano
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            <h3>Log del servidor (ultimas lineas)</h3>
            <pre className="log">{(estado.log || []).join("\n") || "(sin log aun)"}</pre>
          </>
        )}
      </section>

      <footer>
        Sin camara a mano: estas fotos entran por el FTP 2121 con el nombre de
        la camara, y el worker aplica modelo ONNX, OCR, filtros, renombrado por
        placa y el JSON diario. Todo el pipeline real.
      </footer>
    </div>
  );
}
