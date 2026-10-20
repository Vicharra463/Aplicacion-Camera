#!/usr/bin/env python3
"""Ground truth CYTV: 29 diagnosticos verificados visualmente.

Tabla unica de la placa REAL de cada foto (fuentes: memorias #610 y #624).
La consumen validar_diagnosticos.py (motor actual) y entrenar.py (test
honesto del CRNN). Sin imports pesados a proposito: debe cargar en el
venv de entrenamiento, que no tiene cv2 ni rapidocr.
"""
from __future__ import annotations

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1] / "imagenes_procesadas" / "placa"


DIAGNOSTICOS: list[tuple[str, str, str, str]] = [
    
    ("2026-10-13", "12-790.jpg",   "T2K-790",  "leia 12-790 (T2K truncado)"),
    ("2026-10-13", "871-923.jpg",  "B71-923",  "leia 871-923 (B7->87)"),
    ("2026-10-13", "B1V-756.jpg",  "B1Y-756",  "leia B1V-756 (Y->V)"),
    ("2026-10-13", "BIY-756.jpg",  "B1Y-756",  "leia BIY-756 (1->I)"),
    ("2026-10-13", "C20-618.jpg",  "C2Q-618",  "leia C20-618 (Q->0)"),
    ("2026-10-13", "CF635.jpg",    "ACF-635",  "leia CF635 (A perdida, crop al borde)"),
    ("2026-10-13", "A3E726.jpg",   "A3E-726",  "OK historico (sin guion)"),
    ("2026-10-13", "AFH-559.jpg",  "AFH-559",  "OK historico"),
    ("2026-10-13", "C6Y-488.jpg",  "C6Y-488",  "OK historico"),
    
    ("2026-10-14", "4O-820.jpg",   "AOR-820",  "mismo camion que AOR-820 (O/0, chars iniciales)"),
    ("2026-10-14", "AR1-537.jpg",  "ARI-537",  "mismo camion que ARI-537 (1->I)"),
    
    
    ("2026-10-15", "1G-777.jpg",   "WIG-774",  "leia 1G-777 (WIG-774 casi ilegible)"),
    ("2026-10-15", "2P-703.jpg",   "F2P-703",  "leia 2P-703 (F perdida)"),
    ("2026-10-15", "F2P-703.jpg",  "F2P-703",  "OK historico"),
    ("2026-10-15", "AV-463.jpg",   "AVY-463",  "leia AV-463 (Y perdida)"),
    ("2026-10-15", "C21-303.jpg",  "C2I-303",  "leia C21-303 (I->1)"),
    ("2026-10-15", "E5A947.jpg",   "F5A-947",  "leia E5A947; ademas placa del FONDO (asociacion)"),
    ("2026-10-15", "NTA703.jpg",   "WIA-703",  "leia NTA703 (W->N, I->T)"),
    ("2026-10-15", "WIG774.jpg",   "WIG-774",  "OK historico (sin guion)"),
    ("2026-10-15", "NTG-774.jpg",  "WIG-774",  "leia NTG-774 (W->N, I->T/G)"),
    ("2026-10-15", "Z1S-508.jpg",  "ZIS-508",  "leia Z1S-508 (I->1)"),
    ("2026-10-15", "ZIS-508.jpg",  "ZIS-508",  "OK historico"),
    
    
    ("2026-10-16", "A7-935.jpg",   "AWI-935",  "leia A7-935 (truncado)"),
    ("2026-10-16", "AHL-935.jpg",  "AWI-935",  "leia AHL-935 (doble fallo del bus)"),
    ("2026-10-16", "B0-716.jpg",   "B8D-716",  "leia B0-716 (8D->0)"),
    ("2026-10-16", "BTU394.jpg",   "BTU-394",  "OK historico (sin guion)"),
    ("2026-10-16", "EZX165.jpg",   "E7X-165",  "leia EZX165 (7->Z)"),
    ("2026-10-16", "F1-953.jpg",   "FIH-953",  "leia F1-953 (FIH->F1)"),
    ("2026-10-16", "F8E876.jpg",   "F8E-876",  "OK historico (sin guion)"),
]
