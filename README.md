# capsolve-sp

Synthetic data generation, training, and high-performance CPU inference for the `capsolve-sp` five-character CAPTCHA model.

Model weights are published separately on Hugging Face:

- `https://huggingface.co/nakasyou/capsolve-sp`

This GitHub repository intentionally does not contain model weights.

## License

The source code is licensed under the MIT License. See `LICENSE`.

ONNX Runtime is distributed under its own MIT license; see
`LICENSE-ONNXRUNTIME` for the bundled runtime's license notice.

## Repository layout

```text
crates/
  captcha-gen/   Rust synthetic CAPTCHA dataset generator
  capsolve-sp/   Rust ONNX Runtime CPU inference CLI
scripts/         Model download and export utilities
*.py             PyTorch training and inference tools
```

## Synthetic data generation

Build both Rust tools from the workspace root:

```bash
cargo build --workspace --release
```

Generate one reproducible CAPTCHA:

```bash
./target/release/captcha-gen one abc12 dataset/abc12.gif --seed 42
```

Generate a training dataset in parallel. The generator writes Hugging Face-compatible JSON Lines metadata alongside the images:

```bash
./target/release/captcha-gen random \
  'dataset/images/{prefix}/{id}.{ext}' \
  --count 100000 \
  --seed 42 \
  --metadata dataset/metadata.jsonl
```

The output pattern supports `{prefix}`, `{label}`, `{id}`, and `{ext}`. Omit `--count` for continuous generation and use `--workers` to control CPU parallelism.

## Rust CPU inference

Download the model and Linux x86-64 ONNX Runtime library:

```bash
uv run --with huggingface-hub --with onnxruntime \
  python scripts/download_models.py
```

Build and run the optimized Rust engine:

```bash
cargo build --release -p capsolve-sp
./target/release/capsolve-sp path/to/captcha.png
```

For maximum throughput, pass multiple images in one process:

```bash
./target/release/capsolve-sp --batch-size 8 image1.png image2.png image3.png
```

The default `models/model.onnx` is calibrated INT8. Use the FP32 model when required:

```bash
./target/release/capsolve-sp --model models/model-fp32.onnx image.png
```

On an AMD Ryzen AI 7 PRO 350, INT8 reached a median of approximately 1,573 images/second at batch size 8 and eight threads. See the Hugging Face model card for accuracy results.

The release profile uses `target-cpu=native`. Build on the target machine or remove the rustflag from `.cargo/config.toml` for a portable binary.

## Python inference

```bash
uv run --with-requirements requirements.txt python scripts/download_models.py
uv run --with-requirements requirements.txt python predict.py image.png
```

`predict_checkpoint.py` reads the training `.pt` checkpoint directly. `predict.py` reads the downloaded Safetensors model.

## Training

`learn.py` downloads `nakasyou/captcha-like-suica` from Hugging Face and trains from its `metadata.jsonl` image/label pairs:

```bash
uv run --with-requirements requirements.txt python learn.py \
  --device cuda \
  --output captcha-model.pt
```

Use `--data-dir dataset` to retain a visible local snapshot, or `--dataset-repo owner/name` to train from another compatible Hugging Face dataset.

## Fine-tuning

Fine-tune a checkpoint with local `captchas/<id>/0.png` and `label.txt` pairs:

```bash
uv run --with-requirements requirements.txt python finetune_real.py \
  --model captcha-model.pt \
  --output captcha-model-finetuned.pt \
  --data captchas
```

The split is performed by CAPTCHA directory, keeping frames from the same CAPTCHA together.

## Export

Export Safetensors, FP32 ONNX, and calibrated INT8 ONNX:

```bash
uv run --with-requirements requirements.txt python scripts/export_models.py \
  captcha-model-finetuned.pt \
  --calibration-data captchas
```

Generated files are written under `models/`, which is ignored by Git.

## Tests

```bash
cargo test --workspace --locked
```
