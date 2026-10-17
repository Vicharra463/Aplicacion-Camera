#!/usr/bin/env python3
"""Generador sintetico de placas peruanas para entrenar el reconocedor.

Renders de placas (fondo blanco/hueso, texto negro, banda PERU opcional)
con degradacion realista: blur, perspectiva, brillo, suciedad, ruido JPEG.
El Dataset del entrenamiento llama a generar() on-the-fly: nada se guarda
en disco (OneDrive no se llena, y cada epoca ve augmentacion distinta).

Formatos (frecuencias medidas en la ground truth real de 23 placas):
    ABC-123   3 letras + 3 digitos        ~35%   (AFH-559, WIG-774)
    AB1C-123  patron LNL mixto             ~40%   (F8E-876, C2Q-618, T2K-790)
    AB11-123  LNN (raro)                   ~5%    (B71-923)
    ABC-1234  moto nueva                   ~12%
    AB-1234   moto vieja                   ~8%

Uso:
    python entrenamiento/generador_placas.py            # hoja de contactos
    python entrenamiento/generador_placas.py 8          # 8 muestras
"""
from __future__ import annotations

import random
import string
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

BASE = Path(__file__).resolve().parent
FONTS = [
    r"C:\Windows\Fonts\ARIALNB.TTF",   # Arial Narrow Bold (mas parecida)
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\consolab.ttf",
    r"C:\Windows\Fonts\verdanab.ttf",
    r"C:\Windows\Fonts\tahomabd.ttf",
]
LETRAS = string.ascii_uppercase
DIGITOS = string.digits
CHARSET = DIGITOS + LETRAS + "-"          # CTC: el guion tambien se predice
ANCHO, ALTO = 160, 32                     # entrada del CRNN (grayscale)


def placa_aleatoria(rng: random.Random) -> str:
    """Texto de placa segun la distribucion real peruana."""
    r = rng.random()
    def L(n): return "".join(rng.choice(LETRAS) for _ in range(n))
    def N(n): return "".join(rng.choice(DIGITOS) for _ in range(n))
    if r < 0.35:
        return f"{L(3)}-{N(3)}"
    if r < 0.75:                            # LNL mixto: letra, digito, letra
        return f"{L(1)}{N(1)}{L(1)}-{N(3)}"
    if r < 0.80:                            # LNN (B71-923)
        return f"{L(1)}{N(2)}-{N(3)}"
    if r < 0.92:                            # moto nueva
        return f"{L(3)}-{N(4)}"
    return f"{L(2)}-{N(4)}"                 # moto vieja


def _render_placa(texto: str, rng: random.Random) -> Image.Image:
    """Placa limpia, sin degradar: fondo hueso, texto negro, banda opcional."""
    ancho, alto = 400, 100
    img = Image.new("RGB", (ancho, alto),
                    (rng.randint(225, 255), rng.randint(222, 250),
                     rng.randint(205, 240)))     # blanco/hueso variable
    draw = ImageDraw.Draw(img)

    banda = rng.random() < 0.5                  # banda "PERU" superior
    if banda:
        draw.rectangle([0, 0, ancho, 18], fill=(15, 40, 120))
        try:
            fb = ImageFont.truetype(FONTS[0], 13)
            draw.text((ancho - 55, 2), "PERU", font=fb, fill=(240, 240, 240))
        except OSError:
            pass

    fuente_path = rng.choice(FONTS)
    tam = rng.randint(62, 80)
    try:
        font = ImageFont.truetype(fuente_path, tam)
    except OSError:
        font = ImageFont.load_default()
    # centrado vertical del texto de la placa (respetando la banda si hay)
    bbox = draw.textbbox((0, 0), texto, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (ancho - tw) // 2 - rng.randint(-8, 8)
    tope = 20 if banda else 6
    y = max(tope, tope + (alto - tope - 6 - th) // 2) + rng.randint(-3, 3)
    draw.text((x, y), texto, font=font, fill=(rng.randint(5, 30),) * 3)

    draw.rectangle([0, 0, ancho - 1, alto - 1], outline=(90, 90, 90), width=2)
    return img


def _find_coeffs(pa, pb):
    """Coeficientes del transform perspectiva de PIL.

    pa: esquinas en la imagen de SALIDA (donde deben caer).
    pb: esquinas en la imagen de ENTRADA (de donde salen).
    (Snippet estandar de la documentacion de PIL.Image.transform.)
    """
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
    A = np.array(matrix, dtype=float)
    B = np.array(pb, dtype=float).reshape(8)
    return np.linalg.solve(A, B).tolist()


def _degradar(img: Image.Image, rng: random.Random) -> Image.Image:
    """De lo limpio a algo que se parezca a un crop de camara 4MP a 30 m."""
    arr = np.asarray(img).astype(np.float32)

    # brillo/contraste (acotado: sin esto el blanco de la placa se funde
    # con el fondo y la muestra sale vacia)
    arr *= rng.uniform(0.80, 1.15)
    arr += rng.uniform(-18, 18)
    # ruido gaussiano
    arr += np.random.default_rng(rng.randrange(2**31)).normal(
        0, rng.uniform(2, 12), arr.shape)

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # perspectiva leve + rotacion (fillcolor gris: simula fondo de escena,
    # no blanco que se confunde con la placa)
    fondo = (rng.randint(110, 170),) * 3
    ang = rng.uniform(-4, 4)
    img = img.rotate(ang, resample=Image.BILINEAR, expand=False,
                     fillcolor=fondo)
    if rng.random() < 0.6:
        w, h = img.size
        dx, dy = rng.uniform(3, 12), rng.uniform(2, 7)
        # esquinas de la SALIDA (donde caen las esquinas de la placa)
        pa = [(dx, dy), (w - dx * 0.4, 0), (w, h - dy), (0, h)]
        # esquinas de la ENTRADA (placa completa)
        pb = [(0, 0), (w, 0), (w, h), (0, h)]
        img = img.transform((w, h), Image.PERSPECTIVE, _find_coeffs(pa, pb),
                            resample=Image.BILINEAR, fillcolor=fondo)

    # blur (camara lejana + movimiento) — leve: en GT la mayoria es legible
    if rng.random() < 0.45:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    if rng.random() < 0.10:
        img = img.filter(ImageFilter.GaussianBlur(1.8))   # caso malo

    # suciedad / motas
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(0, 12)):
        x, y = rng.randint(0, img.size[0] - 1), rng.randint(0, img.size[1] - 1)
        rr = rng.randint(1, 3)
        c = rng.randint(60, 170)
        draw.ellipse([x, y, x + rr, y + rr], fill=(c, c, max(0, c - 15)))

    return img


def _escena(rng: random.Random, ancho: int, alto: int) -> Image.Image:
    """Fondo tipo foto de camara: gris oscuro con estructura (asfalto,
    sombra, vehiculo). Estadistica medida de los crops GT reales:
    media_gris ~= 107, desvio ~= 51."""
    base = rng.uniform(35, 95)
    semilla = rng.randrange(2**31)
    arr = np.random.default_rng(semilla).normal(base, rng.uniform(10, 30),
                                                (alto, ancho, 3))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    # manchas oscuras/claras grandes (contornos difusos de la escena)
    for _ in range(rng.randint(2, 6)):
        x0 = rng.randint(-20, ancho)
        y0 = rng.randint(-20, alto)
        x1 = x0 + rng.randint(ancho // 3, ancho)
        y1 = y0 + rng.randint(alto // 3, alto)
        c = int(np.clip(rng.gauss(base, 18), 15, 190))
        draw.ellipse([x0, y0, x1, y1], fill=(c, c, max(5, c - 12)))
    return img


def generar(rng: random.Random | None = None) -> tuple[Image.Image, str]:
    """(imagen de ESCENA degradada, texto de placa).

    Reproduce la geometria de un crop del detector real (medida en los 29
    GT): lienzo aspecto 0.74-1.92, placa ~= ancho del crop y 37-56% de su
    altura, escena oscura alrededor. Eso es lo que el modelo va a ver en
    produccion; una placa que llena el marco (generador v1) daba 0/29.
    """
    rng = rng or random.Random()
    texto = placa_aleatoria(rng)
    placa = _render_placa(texto, rng)

    aspecto = rng.uniform(0.74, 1.92)              # rango real GT
    alto_c = rng.randint(60, 222)
    ancho_c = max(48, int(alto_c * aspecto))
    lienzo = _escena(rng, ancho_c, alto_c)

    # placa: ancho toca/sobrepasa bordes; ALTO como fraccion del crop
    # (medido en GT: la placa ocupa ~55-75% de la altura del crop ->
    # letras grandes, que es lo que el modelo ve en produccion)
    ancho_p = int(ancho_c * rng.uniform(0.90, 1.15))
    alto_p = int(alto_c * rng.uniform(0.52, 0.78))
    placa = placa.resize((ancho_p, alto_p), Image.BILINEAR)
    ang = rng.uniform(-7, 7)
    gris = int(rng.uniform(45, 90))
    placa = placa.rotate(ang, resample=Image.BILINEAR, expand=True,
                         fillcolor=(gris, gris, gris))
    px = int((ancho_c - placa.width) / 2 + rng.uniform(-0.10, 0.10) * ancho_c)
    py = int((alto_c - placa.height) / 2 + rng.uniform(-0.12, 0.12) * alto_c)
    lienzo.paste(placa, (px, py))

    return _degradar(lienzo, rng), texto


def a_tensor(img: Image.Image) -> np.ndarray:
    """Image PIL -> array float32 (1, H, W) [0,1], recortado a ANCHOxALTO."""
    g = img.convert("L").resize((ANCHO, ALTO), Image.BILINEAR)
    arr = np.asarray(g, dtype=np.float32) / 255.0
    return arr[None, :, :]


def hoja_contacto(n: int = 16, ruta: str | None = None) -> Path:
    rng = random.Random(7)
    filas = []
    for _ in range(n):
        img, texto = generar(rng)
        draw = ImageDraw.Draw(img)
        try:
            f = ImageFont.truetype(FONTS[1], 14)
        except OSError:
            f = ImageFont.load_default()
        draw.text((4, img.size[1] - 16), texto, font=f, fill=(200, 0, 0))
        filas.append(img)
    salida = Image.new("RGB", (400, sum(f.size[1] + 4 for f in filas)), (30, 30, 30))
    y = 0
    for f in filas:
        salida.paste(f, (0, y))
        y += f.size[1] + 4
    destino = Path(ruta) if ruta else BASE / "muestras_generadas.png"
    salida.save(destino)
    return destino


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    ruta = hoja_contacto(n)
    print(f"hoja de contactos: {ruta}")
