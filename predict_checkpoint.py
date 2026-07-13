#!/usr/bin/env python3
"""Predict five-character CAPTCHA text with a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from learn import CaptchaCNN, choose_device


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[CaptchaCNN, str]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    charset = checkpoint["charset"]
    model = CaptchaCNN(len(charset)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, charset


def image_to_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("L")
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        pixels = pixels.reshape(1, 1, image.height, image.width).float()
        # 学習時と同じく、白い背景を0、濃い描画部分を1にする。
        return (255.0 - pixels) / 255.0


@torch.inference_mode()
def predict(
    model: CaptchaCNN,
    charset: str,
    image_path: Path,
    device: torch.device,
) -> tuple[str, list[float]]:
    image = image_to_tensor(image_path).to(device)
    probabilities = model(image).softmax(dim=-1)[0]
    confidences, indices = probabilities.max(dim=-1)
    text = "".join(charset[index] for index in indices.tolist())
    return text, confidences.cpu().tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="*", help="PNG/GIFなどの画像")
    parser.add_argument(
        "--model", type=Path, default=Path("captcha-model.pt"),
        help="学習済みcheckpoint (default: captcha-model.pt)",
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or mps"
    )
    parser.add_argument(
        "--jsonl", action="store_true",
        help="標準入力から画像パスを1行ずつ読み、JSON Linesで返す",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"モデルが見つかりません: {args.model}")
    if not args.jsonl and not args.images:
        raise SystemExit("画像を1つ以上指定してください")
    missing = [path for path in args.images if not path.is_file()]
    if missing:
        raise SystemExit(f"画像が見つかりません: {missing[0]}")

    device = choose_device(args.device)
    model, charset = load_model(args.model, device)
    status_output = sys.stderr if args.jsonl else sys.stdout
    print(f"device={device} model={args.model}", file=status_output)
    if args.jsonl:
        for line in sys.stdin:
            image_path = Path(line.rstrip("\r\n"))
            try:
                text, confidences = predict(model, charset, image_path, device)
                result = {
                    "path": str(image_path),
                    "text": text,
                    "confidences": confidences,
                }
            except Exception as error:
                result = {"path": str(image_path), "error": str(error)}
            print(json.dumps(result), flush=True)
        return

    for image_path in args.images:
        text, confidences = predict(model, charset, image_path, device)
        confidence_text = " ".join(f"{value:.2%}" for value in confidences)
        print(f"{image_path}: {text}  [{confidence_text}]")


if __name__ == "__main__":
    main()
