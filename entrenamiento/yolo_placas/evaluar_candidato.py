"""Evalua un ONNX candidato en GT-29 con el validador oficial.

El validador (validar_diagnosticos.py) importa RUTA_ONNX de app\\modelo al
momento de correr; aca parcheamos esa constante ANTES para apuntar al
candidato, sin tocar modelo\\placas.onnx de produccion.

Uso (venv de produccion, que tiene onnxruntime+rapidocr):
    venv\\Scripts\\python.exe evaluar_candidato.py <ruta\\candidato.onnx> [--verbose]
"""
import runpy
import sys
from pathlib import Path

APP = Path(r"D:\Aplicacion-Camera\app")
VALIDAR = Path(r"D:\Aplicacion-Camera\entrenamiento\evaluacion\validar_diagnosticos.py")

if len(sys.argv) < 2:
    print(__doc__)
    raise SystemExit(2)

args = sys.argv[1:]
candidato = Path(args.pop(0)).resolve()
if not candidato.is_file():
    print(f"[ERROR] No existe {candidato}")
    raise SystemExit(1)

umbral = None
if "--umbral" in args:
    i = args.index("--umbral")
    umbral = float(args.pop(i + 1))
    args.pop(i)

sys.path.insert(0, str(APP))
import modelo

modelo.RUTA_ONNX = candidato
if umbral is not None:
    _orig = modelo.DetectorPlacas
    modelo.DetectorPlacas = lambda ruta, _o=_orig: _o(ruta, umbral=umbral)
print(f"[eval] modelo inyectado: {candidato}"
      + (f" | umbral={umbral}" if umbral is not None else ""))

sys.argv = [str(VALIDAR), "--detector", "modelo"] + args
runpy.run_path(str(VALIDAR), run_name="__main__")
