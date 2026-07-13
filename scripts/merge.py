#!/usr/bin/env python3
"""Merge every captcha PNG into near-square grid batches."""

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


CAPTCHAS_DIR = Path(__file__).resolve().parent / "captchas"
OUTPUT_DIR = Path(__file__).resolve().parent / "merged_captchas"
PADDING = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="captchas/ 内の PNG を番号付きの格子状画像へまとめます。"
    )
    parser.add_argument("n", type=int, help="選択数（--all 時は1ファイルの最大枚数）")
    parser.add_argument(
        "--all",
        action="store_true",
        help="全画像を n 枚ずつ merged_captchas/ に出力する",
    )
    args = parser.parse_args()
    if args.n < 1:
        parser.error("n は 1 以上を指定してください")
    return args


def merge_batch(paths: list[Path], output_path: Path) -> None:
    images = []
    try:
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))

        font = ImageFont.load_default()
        labels = [str(i) for i in range(1, len(paths) + 1)]
        label_width = max(
            font.getbbox(label)[2] - font.getbbox(label)[0] for label in labels
        )
        number_column_width = PADDING + label_width + PADDING
        cell_image_width = max(image.width for image in images)
        cell_height = max(image.height for image in images)
        cell_width = number_column_width + cell_image_width
        columns = min(
            range(1, len(paths) + 1),
            key=lambda candidate: abs(
                math.log(
                    (candidate * cell_width)
                    / (math.ceil(len(paths) / candidate) * cell_height)
                )
            ),
        )
        rows = math.ceil(len(paths) / columns)
        canvas_width = columns * cell_width
        canvas_height = rows * cell_height

        merged = Image.new("RGB", (canvas_width, canvas_height), "white")
        draw = ImageDraw.Draw(merged)
        for index, (label, image) in enumerate(zip(labels, images)):
            column = index % columns
            row = index // columns
            x = column * cell_width
            y = row * cell_height
            bbox = draw.textbbox((0, 0), label, font=font)
            text_height = bbox[3] - bbox[1]
            text_y = y + (image.height - text_height) // 2 - bbox[1]
            draw.text((x + PADDING, text_y), label, fill="black", font=font)
            merged.paste(image, (x + number_column_width, y))

        merged.save(output_path)
        merged.close()
    finally:
        for image in images:
            image.close()


def main() -> None:
    args = parse_args()
    paths = sorted(CAPTCHAS_DIR.rglob("*.png"), key=lambda path: path.as_posix())
    if not paths:
        raise SystemExit(f"{CAPTCHAS_DIR} に PNG がありません")

    if not args.all:
        if args.n > len(paths):
            raise SystemExit(
                f"PNG は {len(paths)} 枚しかありません（指定: {args.n} 枚）"
            )
        output_path = Path(__file__).resolve().parent / "merged.png"
        merge_batch(random.sample(paths, args.n), output_path)
        print(f"{args.n} 枚を結合して {output_path} を作成しました")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tqdm(total=len(paths), unit="枚", desc="結合中") as progress:
        for start in range(0, len(paths), args.n):
            batch = paths[start : start + args.n]
            start_uuid = batch[0].parent.name
            end_uuid = batch[-1].parent.name
            output_path = OUTPUT_DIR / f"{start_uuid}-{end_uuid}.png"
            merge_batch(batch, output_path)
            progress.update(len(batch))

    print(f"{len(paths)} 枚を {OUTPUT_DIR} に出力しました")


if __name__ == "__main__":
    main()
