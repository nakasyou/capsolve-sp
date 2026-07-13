#!/usr/bin/env python3
"""Predict a five-character CAPTCHA with capsolve-sp."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import load_model, predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model, config = load_model(args.model_dir, device)

    for image_path in args.images:
        text, confidences = predict(model, config["charset"], image_path, device)
        confidence_text = " ".join(f"{value:.2%}" for value in confidences)
        print(f"{image_path}: {text} [{confidence_text}]")


if __name__ == "__main__":
    main()
