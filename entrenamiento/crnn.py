#!/usr/bin/env python3
"""CRNN para reconocimiento de placas peruanas (CTC).

Arquitectura clasica CRNN (conv + BiLSTM + CTC), chica a proposit:
la GTX 1650 del servidor entrena en minutos/epoca y la inferencia ONNX
en CPU toma ~20 ms por crop, que es lo que el worker necesita.

Entrada:  1 x 32 x 160  (grayscale, texto de la placa)
Salida:   T x 38 logits (T=~40; 37 chars de CHARSET + blank CTC)
"""
from __future__ import annotations

import torch
import torch.nn as nn

# 10 digitos + 26 letras + "-" ; el blank de CTC es el ultimo indice.
N_CLASSES = 38


class CRNN(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            # 1x32x160 -> 32x16x80
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            # -> 64x8x40
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            # -> 128x4x40
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.MaxPool2d((2, 1), (2, 1)),
            # -> 128x2x40
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.MaxPool2d((2, 1), (2, 1)),
            # -> 128x1x40
            nn.Conv2d(128, 128, 2, padding=0), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # -> 128x1x39
        )
        self.rnn = nn.LSTM(128, 256, num_layers=2, bidirectional=True,
                           batch_first=True, dropout=0.1)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 32, 160)
        feat = self.cnn(x)                       # (B, 128, 1, 39)
        feat = feat.squeeze(2).permute(0, 2, 1)  # (B, T=39, 128)
        seq, _ = self.rnn(feat)                  # (B, T, 512)
        return self.fc(seq)                      # (B, T, n_classes)
