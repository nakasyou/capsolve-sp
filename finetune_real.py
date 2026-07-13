#!/usr/bin/env python3
"""Fine-tune captcha-model.pt on locally labeled real CAPTCHA images."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from learn import CaptchaCNN, choose_device


class RealCaptchaDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: list[tuple[Path, str]], charset: str) -> None:
        self.samples = samples
        self.char_to_index = {char: index for index, char in enumerate(charset)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, text = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("L")
            pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            pixels = (255.0 - pixels.reshape(1, image.height, image.width).float()) / 255.0
        target = torch.tensor(
            [self.char_to_index[char] for char in text], dtype=torch.long
        )
        return pixels, target


def load_groups(
    root: Path, charset: str
) -> tuple[list[list[tuple[Path, str]]], int, int]:
    groups: list[list[tuple[Path, str]]] = []
    rejected = 0
    pseudo_labeled = 0
    for label_path in sorted(root.glob("*/label.txt")):
        label = "".join(label_path.read_text().split()).lower()
        if len(label) != 5 or any(char not in charset for char in label):
            pseudo_path = label_path.with_name("pseudo-label.txt")
            if pseudo_path.is_file():
                label = "".join(pseudo_path.read_text().split()).lower()
                pseudo_labeled += 1
        image_paths = sorted(
            label_path.parent.glob("*.png"), key=lambda path: path.name
        )
        if len(label) != 5 or any(char not in charset for char in label) or not image_paths:
            rejected += 1
            continue
        groups.append([(image_path, label) for image_path in image_paths])
    return groups, rejected, pseudo_labeled


@torch.inference_mode()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    char_correct = 0
    sequence_correct = 0
    count = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits.flatten(0, 1), targets.flatten())
        predictions = logits.argmax(dim=-1)
        batch_size = targets.size(0)
        loss_sum += loss.item() * batch_size
        char_correct += (predictions == targets).sum().item()
        sequence_correct += (predictions == targets).all(dim=1).sum().item()
        count += batch_size
    return loss_sum / count, char_correct / (count * 5), sequence_correct / count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("captcha-model.pt"))
    parser.add_argument("--output", type=Path, default=Path("captcha-model.pt"))
    parser.add_argument("--data", type=Path, default=Path("captchas"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    charset = checkpoint["charset"]
    groups, rejected, pseudo_labeled = load_groups(args.data, charset)
    random.Random(args.seed).shuffle(groups)
    validation_group_count = max(1, round(len(groups) * args.validation_ratio))
    validation_groups = groups[:validation_group_count]
    training_groups = groups[validation_group_count:]
    validation_samples = [sample for group in validation_groups for sample in group]
    training_samples = [sample for group in training_groups for sample in group]
    if not training_samples:
        raise SystemExit("not enough valid labeled samples")

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    training_loader = DataLoader(
        RealCaptchaDataset(training_samples, charset), shuffle=True, **loader_options
    )
    validation_loader = DataLoader(
        RealCaptchaDataset(validation_samples, charset), shuffle=False, **loader_options
    )

    model = CaptchaCNN(len(charset)).to(device)
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    baseline = evaluate(model, validation_loader, device)
    best_accuracy = baseline[2]
    best_loss = baseline[0]
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    print(
        f"device={device} train={len(training_samples)} validation={len(validation_samples)} "
        f"rejected={rejected} pseudo_labeled={pseudo_labeled}"
    )
    print(
        f"baseline val_loss={baseline[0]:.4f} char_acc={baseline[1]:.2%} "
        f"sequence_acc={baseline[2]:.2%}"
    )

    stale_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        progress = tqdm(training_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for images, targets in progress:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits.flatten(0, 1), targets.flatten())
            loss.backward()
            optimizer.step()
            batch_size = targets.size(0)
            loss_sum += loss.item() * batch_size
            count += batch_size
            progress.set_postfix(loss=f"{loss_sum / count:.4f}")
        scheduler.step()

        val_loss, char_accuracy, sequence_accuracy = evaluate(
            model, validation_loader, device
        )
        print(
            f"epoch={epoch + 1} train_loss={loss_sum / count:.4f} "
            f"val_loss={val_loss:.4f} char_acc={char_accuracy:.2%} "
            f"sequence_acc={sequence_accuracy:.2%}"
        )
        improved = sequence_accuracy > best_accuracy or (
            sequence_accuracy == best_accuracy and val_loss < best_loss
        )
        if improved:
            best_accuracy = sequence_accuracy
            best_loss = val_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stop after {epoch + 1} epochs")
                break

    checkpoint.update(
        {
            "model": best_state,
            "optimizer": optimizer.state_dict(),
            "width": 175,
            "height": 60,
            "real_finetune_samples": len(training_samples),
            "real_validation_samples": len(validation_samples),
            "real_validation_sequence_accuracy": best_accuracy,
            "real_validation_loss": best_loss,
            "real_finetune_seed": args.seed,
            "real_label_smoothing": args.label_smoothing,
            "real_pseudo_labeled_groups": pseudo_labeled,
            "real_finetune_source": str(args.model),
            "real_training_groups": len(training_groups),
            "real_validation_groups": len(validation_groups),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"saved={args.output} source={args.model} best_sequence_acc={best_accuracy:.2%}")


if __name__ == "__main__":
    main()
