#!/usr/bin/env python3
"""Download capsolve-sp model artifacts from Hugging Face."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import onnxruntime
from huggingface_hub import hf_hub_download


FILES = (
    "config.json",
    "metadata.json",
    "model.safetensors",
    "model.onnx",
    "model-fp32.onnx",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="nakasyou/capsolve-sp")
    parser.add_argument("--output", type=Path, default=Path("models"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        source = Path(hf_hub_download(args.repo, name))
        shutil.copy2(source, args.output / name)
        print(args.output / name)

    runtime_dir = Path(onnxruntime.__file__).parent / "capi"
    runtimes = sorted(runtime_dir.glob("libonnxruntime.so.*"))
    if not runtimes:
        raise FileNotFoundError(f"libonnxruntime.so was not found in {runtime_dir}")
    shutil.copy2(runtimes[-1], args.output / "libonnxruntime.so")
    print(args.output / "libonnxruntime.so")


if __name__ == "__main__":
    main()

