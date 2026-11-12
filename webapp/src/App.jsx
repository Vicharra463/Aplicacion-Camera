import React, { useRef, useState } from "react";

// UI de una sola fila por accion:
//   fila 1  -> subir la foto (drag & drop o clic)
//   fila 2  -> resultado: si el modelo hallo placa, muestra "PLACA DETECTADA"
//              con el recorte, la confianza y EN DONDE esta (bbox x/y);
//              si no, muestra una X grande "SIN DETECTAR PLACA".
// Corre el MISMO pipeline que la camara real (modelo ONNX + OSD + OCR).
export default function App() {
  const [foto, setFoto] = useState(null);        // File elegida
  const [preview, setPreview] = useState(null);  // objectURL para mostrar
  const [resultado, setResultado] = useState(null); // null | {detectado, ...}
  const [cargando, setCargando] = useState(false);
  const [error, setError] = useState(null);
  const [arrastrando, setArrastrando] = useState(false);
  const inputRef = useRef(null);

  function elegir(lista) {
    const f = lista && lista[0];
    if (!f) return;
    if (preview) URL.revokeObjectURL(preview);
    setFoto(f);
    setPreview(URL.createObjectURL(f));
    setResultado(null);
    setError(null);
  }

  async function evaluar() {
    if (!foto || cargando) return;
    setCargando(true);
    setError(null);
    setResultado(null);
    const fd = new FormData();
    fd.append("foto", foto);
    try {
      const r = await fetch("/api/evaluar", { method: "POST", body: fd });
      const d = await r.json();
      if (d.ok) setResultado(d);
      else setError(d.error || "Error desconocido");
    } catch (e) {
      setError("No se pudo contactar al servidor: " + e.message);
    } finally {
      setCargando(false);
    }
  }

  function limpiar() {
    if (preview) URL.revokeObjectURL(preview);
    setFoto(null);
    setPreview(null);
    setResultado(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <div className="app">
      <header>
        <h1>Camara CYTV Ancon</h1>
        <p className="sub">
          Suba una foto y el servidor la evalua <b>con el mismo modelo ONNX</b>{" "}
          que usa la camara real (OSD enmascarado + OCR).
        </p>
      </header>

      {/* ---------------- FILA 1: subir ---------------- */}
      <section className="fila">
        <div className="fila-titulo">1 &middot; Subir foto</div>
        <div
          className={"soltar" + (arrastrando ? " activo" : "") + (foto ? " con-foto" : "")}
          onDragOver={(e) => { e.preventDefault(); setArrastrando(true); }}
          onDragLeave={() => setArrastrando(false)}
          onDrop={(e) => {
            e.preventDefault();
            setArrastrando(false);
            elegir(e.dataTransfer.files);
          }}
          onClick={() => inputRef.current && inputRef.current.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => elegir(e.target.files)}
          />
          {preview ? (
            <img src={preview} alt="foto elegida" className="preview" />
          ) : (
            <span className="soltar-texto">
              Arrastre una foto aqui <b>o haga clic para elegir</b>
            </span>
          )}
        </div>

        {foto && (
          <div className="acciones">
            <span className="nombre-archivo">{foto.name}</span>
            <button className="boton" disabled={cargando} onClick={cargando ? undefined : evaluar}>
              {cargando ? "Evaluando..." : "Evaluar placa"}
            </button>
            <button className="boton secundario" onClick={limpiar} disabled={cargando}>
              Limpiar
            </button>
          </div>
        )}

        {cargando && <p className="nota">Corriendo modelo + OCR (1-3 s)...</p>}
        {error && <p className="msg error">{error}</p>}
      </section>

      {/* ---------------- FILA 2: resultado ---------------- */}
      {resultado && (
        <section className={"fila resultado " + (resultado.detectado ? "ok" : "fallo")}>
          <div className="fila-titulo">2 &middot; Resultado</div>

          {resultado.detectado ? (
            <div className="detectado">
              <div className="bandera ok">PLACA DETECTADA</div>

              <div className="info-placa">
                {resultado.placa && (
                  <div className="texto-placa">{resultado.placa}</div>
                )}
                <div className="conf">
                  Confianza: <b>{(resultado.confianza * 100).toFixed(1)}%</b>
                </div>
                <div className="donde">
                  Donde: x {resultado.x0}&hellip;{resultado.x1} / y {resultado.y0}&hellip;{resultado.y1}
                  <span className="tam">
                    {" "}({resultado.bbox.w}&times;{resultado.bbox.h} px)
                  </span>
                </div>
                {resultado.detecciones > 1 && (
                  <div className="nota">+{resultado.detecciones - 1} deteccion(es) adicional(es) en la foto</div>
                )}
              </div>

              <div className="imagenes">
                {resultado.recorte && (
                  <figure>
                    <img src={resultado.recorte} alt="recorte de la placa" />
                    <figcaption>Recorte de la placa</figcaption>
                  </figure>
                )}
                {resultado.anotada && (
                  <figure>
                    <img src={resultado.anotada} alt="foto anotada" />
                    <figcaption>Foto completa (bbox dibujado)</figcaption>
                  </figure>
                )}
              </div>
            </div>
          ) : (
            <div className="no-detectado">
              <div className="x-grande" aria-label="sin detectar placa">&#10005;</div>
              <div className="bandera fallo">SIN DETECTAR PLACA</div>
              <p className="nota">
                El modelo no encontro ninguna placa en esta foto.
                Pruebe con otra imagen, otra distancia o mejor iluminacion.
              </p>
            </div>
          )}
        </section>
      )}

      <footer>
        El evaluador usa <code>modelo\placas.onnx</code> y RapidOCR, igual que
        el servicio <code>CamaraCytvAncon</code>. Sin camara a mano: suba la
        foto y vea al instante si el pipeline la aceptaria.
      </footer>
    </div>
  );
}
