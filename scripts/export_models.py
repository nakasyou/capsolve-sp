#!/usr/bin/env python3
"""Export a training checkpoint to Safetensors and FP32/INT8 ONNX."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from PIL import Image
from safetensors.torch import save_file
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learn import CaptchaCNN


class ExportModel(nn.Module):
    def __init__(self, model: CaptchaCNN) -> None:
        super().__init__()
        self.features = model.features
        self.classifier = model.classifier

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        width = features.shape[-1]
        cells = []
        for index in range(5):
            start = index * width // 5
            end = ((index + 1) * width + 4) // 5
            cells.append(features[:, :, :, start:end].mean(dim=(2, 3)))
        return self.classifier(torch.stack(cells, dim=1))


class CalibrationReader(CalibrationDataReader):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = iter(paths)

    def get_next(self):
        try:
            path = next(self.paths)
        except StopIteration:
            return None
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("L"), dtype=np.float32)
        return {"image": ((255.0 - pixels) / 255.0)[None, None]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--calibration-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--calibration-samples", type=int, default=1024)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {
        name: tensor.detach().contiguous().cpu()
        for name, tensor in checkpoint["model"].items()
    }
    save_file(
        state,
        args.output / "model.safetensors",
        metadata={
            "format": "pt",
            "architecture": "CaptchaCNN",
            "charset": checkpoint["charset"],
        },
    )

    model = CaptchaCNN(len(checkpoint["charset"]))
    model.load_state_dict(state)
    model.eval()
    export_model = ExportModel(model).eval()
    example = torch.zeros(1, 1, 60, 175)
    torch.onnx.export(
        export_model,
        (example,),
        args.output / "model-fp32.onnx",
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )

    paths = sorted(args.calibration_data.glob("*/0.png"))
    paths = random.Random(42).sample(paths, min(args.calibration_samples, len(paths)))
    quantize_static(
        args.output / "model-fp32.onnx",
        args.output / "model.onnx",
        CalibrationReader(paths),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )


if __name__ == "__main__":
    main()
