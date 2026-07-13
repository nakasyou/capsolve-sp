#!/usr/bin/env python3
"""Train the CAPTCHA recognizer with a dataset hosted on Hugging Face."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

HEIGHT = 60
WIDTH = 175


class CaptchaDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Image and label pairs listed in Hugging Face metadata.jsonl."""

    def __init__(
        self,
        samples: list[tuple[Path, str]],
        charset: str,
    ) -> None:
        self.samples = samples
        self.char_to_index = {char: i for i, char in enumerate(charset)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, text = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("L")
            if image.size != (WIDTH, HEIGHT):
                raise ValueError(f"expected {WIDTH}x{HEIGHT} image: {path}")
            pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            pixels = (255.0 - pixels.float()).reshape(1, HEIGHT, WIDTH) / 255.0
        target = torch.tensor(
            [self.char_to_index[char] for char in text], dtype=torch.long
        )
        return pixels, target


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.layers(x))


class CaptchaCNN(nn.Module):
    """CNN followed by one classifier shared by the five character cells."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x)).squeeze(2).transpose(1, 2)
        return self.classifier(x)  # (batch, 5, num_classes)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    loss_sum = 0.0
    char_correct = 0
    sequence_correct = 0
    samples = 0
    criterion = nn.CrossEntropyLoss()

    for images, targets in tqdm(loader, desc="validation", unit="batch", leave=False):
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits.flatten(0, 1), targets.flatten())
        predictions = logits.argmax(dim=-1)
        batch_size = targets.size(0)
        loss_sum += loss.item() * batch_size
        char_correct += (predictions == targets).sum().item()
        sequence_correct += (predictions == targets).all(dim=1).sum().item()
        samples += batch_size

    return loss_sum / samples, char_correct / (samples * 5), sequence_correct / samples


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def download_dataset(repo_id: str, data_dir: Path | None) -> Path:
    """Download or update a Hugging Face dataset snapshot in the local cache."""
    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=data_dir,
        )
    )


def load_samples(root: Path, charset: str) -> list[tuple[Path, str]]:
    metadata = root / "metadata.jsonl"
    if not metadata.is_file():
        raise FileNotFoundError(f"metadata.jsonl was not found in {root}")
    samples = []
    with metadata.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            label = row["text"]
            path = root / row["file_name"]
            if len(label) != 5 or any(char not in charset for char in label):
                raise ValueError(f"invalid label on line {line_number}: {label!r}")
            if not path.is_file():
                raise FileNotFoundError(f"missing image on line {line_number}: {path}")
            samples.append((path, label))
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo", default="nakasyou/captcha-like-suica")
    parser.add_argument(
        "--data-dir", type=Path, help="download directory (default: Hugging Face cache)"
    )
    parser.add_argument("--charset", default="0123456789abcdefghijklmnopqrstuvwxyz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--validation-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--output", type=Path, default=Path("captcha-model.pt"))
    parser.add_argument("--resume", type=Path, help="resume from a checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.charset or len(set(args.charset)) != len(args.charset):
        raise SystemExit("--charset must contain unique characters")
    if min(args.epochs, args.validation_samples, args.batch_size) < 1:
        raise SystemExit("epochs, validation samples, and batch size must be positive")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = choose_device(args.device)
    root = download_dataset(args.dataset_repo, args.data_dir)
    samples = load_samples(root, args.charset)
    if len(samples) <= args.validation_samples:
        raise SystemExit("dataset must contain more samples than --validation-samples")
    random.Random(args.seed).shuffle(samples)
    validation_samples = samples[: args.validation_samples]
    training_samples = samples[args.validation_samples :]
    model = CaptchaCNN(len(args.charset)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    start_epoch = 0
    best_sequence_accuracy = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["charset"] != args.charset:
            raise SystemExit("checkpoint charset does not match --charset")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_sequence_accuracy = checkpoint.get("best_sequence_accuracy", 0.0)

    validation = CaptchaDataset(validation_samples, args.charset)
    validation_loader = DataLoader(
        validation,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    criterion = nn.CrossEntropyLoss()
    print(
        f"device={device} train={len(training_samples):,} "
        f"validation={len(validation_samples):,} "
        f"parameters={sum(p.numel() for p in model.parameters()):,}"
    )

    for epoch in range(start_epoch, args.epochs):
        training = CaptchaDataset(training_samples, args.charset)
        training_loader = DataLoader(
            training,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        model.train()
        train_loss_sum = 0.0
        samples = 0
        progress = tqdm(
            training_loader,
            desc=f"epoch {epoch + 1}/{args.epochs}",
            unit="batch",
        )
        for images, targets in progress:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits.flatten(0, 1), targets.flatten())
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * targets.size(0)
            samples += targets.size(0)
            progress.set_postfix(loss=f"{train_loss_sum / samples:.4f}")

        val_loss, char_accuracy, sequence_accuracy = evaluate(
            model, validation_loader, device
        )
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"train_loss={train_loss_sum / samples:.4f} val_loss={val_loss:.4f} "
            f"char_acc={char_accuracy:.2%} sequence_acc={sequence_accuracy:.2%}"
        )

        if sequence_accuracy >= best_sequence_accuracy:
            best_sequence_accuracy = sequence_accuracy
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "charset": args.charset,
                    "width": WIDTH,
                    "height": HEIGHT,
                    "best_sequence_accuracy": best_sequence_accuracy,
                },
                args.output,
            )
            print(f"saved {args.output}")


if __name__ == "__main__":
    main()
