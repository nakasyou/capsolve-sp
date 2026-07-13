"""Standalone architecture and inference helpers for capsolve-sp."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.layers(inputs))


class CaptchaCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 5))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.features(inputs)).squeeze(2).transpose(1, 2)
        return self.classifier(features)


def image_to_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("L")
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        pixels = pixels.reshape(1, 1, image.height, image.width).float()
    return (255.0 - pixels) / 255.0


def load_model(
    model_directory: Path, device: torch.device
) -> tuple[CaptchaCNN, dict]:
    config = json.loads((model_directory / "config.json").read_text())
    model = CaptchaCNN(config["num_classes"]).to(device)
    state_dict = load_file(model_directory / "model.safetensors", device=str(device))
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


@torch.inference_mode()
def predict(
    model: CaptchaCNN,
    charset: str,
    image_path: Path,
    device: torch.device,
) -> tuple[str, list[float]]:
    probabilities = model(image_to_tensor(image_path).to(device)).softmax(dim=-1)[0]
    confidences, indices = probabilities.max(dim=-1)
    text = "".join(charset[index] for index in indices.tolist())
    return text, confidences.cpu().tolist()

