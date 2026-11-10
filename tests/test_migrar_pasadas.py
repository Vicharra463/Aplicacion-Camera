"""FASE 3 (2026-11-10): migracion de pasadas historicas.

Los archivos con sufijo de pasada EN EL NOMBRE (layout pre-subcarpetas:
`..._2da-pasada.jpg`, `..._3ra-pasada.jpg`, `..._EXCEDIDA.jpg`, con
colision `_N` y `(DUDOSA)` posibles) se mueven a la subcarpeta que hoy
usa el worker (segunda/tercera/excedida) renombrados SIN el sufijo.

Dry-run por defecto (solo planifica e imprime); `--aplicar` ejecuta.
Idempotente: corrida sobre carpetas ya migradas no encuentra nada.
NO se corre en prod sin pedido de FELIPE.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from scripts import migrar_pasadas_carpetas as mig  # noqa: E402

SUF_2DA = "C8K-898_api-Roboflow_roi-evaluado_2da-pasada"
CLEAN_2DA = "C8K-898_api-Roboflow_roi-evaluado"


@pytest.fixture
def base(tmp_path):
    placa = tmp_path / "placa" / "2026-11-01" / "C8K-898"
    placa.mkdir(parents=True)
    # 2da pasada completa (main + txt + crop + original) con sufijo
    (placa / f"{SUF_2DA}.jpg").write_bytes(b"segunda")
    (placa / f"{SUF_2DA}.txt").write_text("0 .5 .5 .1 .1", encoding="utf-8")
    (placa / f"{SUF_2DA}_crop.jpg").write_bytes(b"crop")
    (placa / f"{SUF_2DA}_original.jpg").write_bytes(b"original")
    # 3ra pasada
    (placa / "C8K-898_3ra-pasada.jpg").write_bytes(b"tercera")
    # excedida con colision de nombre (_1 por _destino_libre)
    (placa / "C8K-898_EXCEDIDA.jpg").write_bytes(b"exc1")
    (placa / "C8K-898_EXCEDIDA_1.jpg").write_bytes(b"exc2")
    # dudosa de 2da pasada
    (placa / "C8K-898_2da-pasada (DUDOSA).jpg").write_bytes(b"dudosa")
    # 1a pasada limpia y archivo ajeno: NO se tocan
    (placa / "C8K-898.jpg").write_bytes(b"primera")
    (placa / "otra-cosa.jpg").write_bytes(b"ajena")
    return tmp_path


def _raiz(base: Path) -> Path:
    return base


def test_dry_run_planifica_sin_mover(base):
    pares = mig.migrar(_raiz(base), aplicar=False)
    # 4 (2da) + 1 (3ra) + 2 (excedida) + 1 (dudosa) = 8
    assert len(pares) == 8
    placa = base / "placa" / "2026-11-01" / "C8K-898"
    # nada se movio
    assert (placa / f"{SUF_2DA}.jpg").is_file()
    assert (placa / "C8K-898_3ra-pasada.jpg").is_file()
    # el plan apunta a subcarpetas con nombre limpio
    destinos = {dst.name for _, dst in pares}
    assert f"{CLEAN_2DA}.jpg" in destinos
    assert "C8K-898.jpg" in destinos          # 3ra y excedida (unica por carpeta)
    assert all(dst.parent.name in ("segunda", "tercera", "excedida")
               for _, dst in pares)
    # la 1a pasada y la ajena no entran al plan
    assert not any(src.name in ("C8K-898.jpg", "otra-cosa.jpg")
                   for src, _ in pares)


def test_aplicar_mueve_y_renombra(base):
    pares = mig.migrar(_raiz(base), aplicar=True)
    assert len(pares) == 8
    placa = base / "placa" / "2026-11-01" / "C8K-898"
    # raiz: solo 1a pasada limpia y la ajena
    assert sorted(p.name for p in placa.iterdir() if p.is_file()) == [
        "C8K-898.jpg", "otra-cosa.jpg"]
    segunda = placa / "segunda"
    assert sorted(p.name for p in segunda.iterdir()) == [
        "C8K-898 (DUDOSA).jpg",
        f"{CLEAN_2DA}.jpg",
        f"{CLEAN_2DA}.txt",
        f"{CLEAN_2DA}_crop.jpg",
        f"{CLEAN_2DA}_original.jpg",
    ]
    # el .txt acompana al main en la MISMA carpeta
    assert (segunda / f"{CLEAN_2DA}.txt").read_text(
        encoding="utf-8") == "0 .5 .5 .1 .1"
    tercera = placa / "tercera"
    assert [p.name for p in tercera.iterdir()] == ["C8K-898.jpg"]
    excedida = placa / "excedida"
    # colision resuelta: la segunda excedida queda con _1
    assert sorted(p.name for p in excedida.iterdir()) == [
        "C8K-898.jpg", "C8K-898_1.jpg"]


def test_idempotente(base):
    primero = mig.migrar(_raiz(base), aplicar=True)
    assert len(primero) == 8
    segundo = mig.migrar(_raiz(base), aplicar=True)
    assert segundo == []
    tercero = mig.migrar(_raiz(base), aplicar=False)
    assert tercero == []


def test_cli_dry_run_y_aplicar(base, capsys):
    rc = mig.main([str(base)])
    assert rc == 0
    placa = base / "placa" / "2026-11-01" / "C8K-898"
    assert (placa / f"{SUF_2DA}.jpg").is_file()      # dry-run no mueve
    salida = capsys.readouterr().out
    assert "MOVERIA" in salida
    rc = mig.main([str(base), "--aplicar"])
    assert rc == 0
    assert (placa / "segunda" / f"{CLEAN_2DA}.jpg").is_file()
    salida = capsys.readouterr().out
    assert "MOVIDO" in salida


def test_raiz_inexistente_no_revienta(tmp_path):
    assert mig.migrar(tmp_path / "no-existe", aplicar=True) == []
    assert mig.main([str(tmp_path / "no-existe")]) == 0
