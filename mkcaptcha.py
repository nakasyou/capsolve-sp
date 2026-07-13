#!/usr/bin/env python3
"""Generate a CAPTCHA compatible with the samples in ``captchas/``.

The files in the sample set have a .png suffix, but are actually 175x60
GIF89a images.  This implementation is fitted directly to repeated samples
in the corpus: fixed character cells, independently distorted glyph layers,
and two unwarped noise layers.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


WIDTH = 175
HEIGHT = 60
FONT_SIZE = 28
DOT_NOISE = 100
LINE_NOISE = 3


def _find_font() -> str:
    """Return the closest commonly available match to the sample font."""
    candidates = (
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    )
    for name in candidates:
        if Path(name).is_file():
            return name

    # NixOS and other systems keep fonts outside /usr/share/fonts.
    system_fonts = Path("/run/current-system/sw/share/fonts")
    for filename in ("FreeSansBold.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"):
        matches = list(system_fonts.glob(f"**/{filename}"))
        if matches:
            return str(matches[0])
    nix_store = Path("/nix/store")
    if nix_store.exists():
        patterns = (
            "*-freefont-ttf-*/share/fonts/truetype/FreeSansBold.ttf",
            "*-dejavu-fonts-*/share/fonts/truetype/DejaVuSans-Bold.ttf",
            "*-liberation-fonts-*/share/fonts/truetype/LiberationSans-Bold.ttf",
        )
        for pattern in patterns:
            matches = list(nix_store.glob(pattern))
            if matches:
                return str(matches[0])
    raise FileNotFoundError(
        "a suitable bold sans-serif font was not found; specify one with --font"
    )


def _noise(
    image: Image.Image,
    rng: random.Random,
    *,
    dot_size: int,
    draw_lines: bool,
) -> None:
    scale = 4
    dots = Image.new("L", (WIDTH * scale, HEIGHT * scale), 255)
    dot_draw = ImageDraw.Draw(dots)
    for _ in range(DOT_NOISE):
        # Quarter-pixel centres and circular coverage produce the coloured
        # one-pixel fringe visible around corpus speckles when magnified.
        cx = rng.randrange(0, WIDTH * scale + 1)
        cy = rng.randrange(0, HEIGHT * scale + 1)
        radius = dot_size * scale / 2
        dot_draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=0,
        )
    # Corpus lines are not binary Bresenham staircases: their black core is
    # surrounded by partially covered pixels.  Draw at quarter-pixel endpoint
    # precision and filter down to reproduce that coverage antialiasing.
    lines = Image.new("L", (WIDTH * scale, HEIGHT * scale), 255)
    line_draw = ImageDraw.Draw(lines)
    if draw_lines:
        for _ in range(LINE_NOISE):
            end_x = rng.randrange(0, WIDTH * scale + 1)
            end_y = rng.randrange(0, HEIGHT * scale + 1)
            points = [(0, 0)]
            if rng.randint(0, 1):
                # The bend may point anywhere; x is intentionally not
                # monotonic, so the second segment can turn back to the left.
                bend_x = rng.randrange(0, WIDTH * scale + 1)
                bend_y = rng.randrange(0, HEIGHT * scale + 1)
                points.append((bend_x, bend_y))
            points.append((end_x, end_y))
            line_draw.line(
                points,
                fill=0,
                # One geometric pixel supplies the black core.  BOX filtering
                # keeps its coverage fringe to the immediately adjacent pixel.
                width=scale,
                joint="curve",
            )
    dots = dots.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    # BOX represents pixel coverage directly and does not add the two-pixel
    # ringing halo that LANCZOS creates around a thin line.
    lines = lines.resize((WIDTH, HEIGHT), Image.Resampling.BOX)
    image.paste(ImageChops.darker(image, ImageChops.darker(dots, lines)))


def _wave(source: Image.Image, rng: random.Random) -> Image.Image:
    """Apply the smooth two-axis displacement fitted to corpus samples."""
    freq = [rng.randint(700_000, 1_000_000) / 15_000_000 for _ in range(4)]
    phase = [rng.randint(0, 3_141_592) / 1_000_000 for _ in range(4)]
    size_x = rng.randint(300, 700) / 100
    size_y = rng.randint(300, 700) / 100
    src = source.load()
    result = Image.new("L", (WIDTH, HEIGHT), 255)
    dst = result.load()

    for x in range(WIDTH):
        for y in range(HEIGHT):
            sx = x + (math.sin(x * freq[0] + phase[0]) +
                      math.sin(y * freq[2] + phase[2])) * size_x
            sy = y + (math.sin(x * freq[1] + phase[1]) +
                      math.sin(y * freq[3] + phase[3])) * size_y
            if sx < 0 or sy < 0 or sx >= WIDTH - 1 or sy >= HEIGHT - 1:
                continue

            ix, iy = math.floor(sx), math.floor(sy)
            c, cx = src[ix, iy], src[ix + 1, iy]
            cy, cxy = src[ix, iy + 1], src[ix + 1, iy + 1]
            if c == cx == cy == cxy == 255:
                continue
            if c == cx == cy == cxy == 0:
                dst[x, y] = 0
                continue
            fx, fy = sx - ix, sy - iy
            dst[x, y] = round(
                c * (1 - fx) * (1 - fy)
                + cx * fx * (1 - fy)
                + cy * (1 - fx) * fy
                + cxy * fx * fy
            )
    return result


def _websafe_palette() -> list[int]:
    # This is the exact 6x6x6 palette ordering found in every sample.
    levels_rb = (0, 51, 102, 153, 204, 255)
    levels_g = (0, 43, 85, 128, 170, 213, 255)
    colors = [(r, g, b) for r in levels_rb for g in levels_g for b in levels_rb]
    colors.extend([(0, 0, 0)] * (256 - len(colors)))
    return [component for color in colors[:256] for component in color]


def _to_sample_gif(image: Image.Image) -> Image.Image:
    palette = _websafe_palette()
    colors = [tuple(palette[i:i + 3]) for i in range(0, 252 * 3, 3)]
    # Explicit lookup avoids the unused transparent index 252.  It also mimics
    # the source encoder's nearest-colour conversion without introducing dither.
    lookup = []
    for grey in range(256):
        lookup.append(min(
            range(252),
            key=lambda i: sum((component - grey) ** 2 for component in colors[i]),
        ))
    result = Image.new("P", image.size)
    result.putpalette(palette)
    result.putdata([lookup[pixel] for pixel in image.tobytes()])
    return result


def make_captcha(text: str, *, seed: int | None = None,
                 font_path: str | None = None) -> Image.Image:
    if len(text) != 5:
        raise ValueError("text must contain exactly 5 characters")
    rng = random.Random(seed)
    font = ImageFont.truetype(font_path or _find_font(), FONT_SIZE)

    result = Image.new("L", (WIDTH, HEIGHT), 255)
    # The source font has unusually wide metrics: glyph centres are almost
    # exactly 34 px apart.  Explicit cells reproduce that metric consistently
    # even when the exact source font is unavailable.
    cell_width = 34
    left = (WIDTH - cell_width * len(text)) / 2
    for index, char in enumerate(text):
        # Keep every glyph on its own layer.  Its displacement parameters are
        # independent from the other four glyphs and from the background.
        char_layer = Image.new("L", (WIDTH, HEIGHT), 255)
        draw = ImageDraw.Draw(char_layer)
        box = draw.textbbox((0, 0), char, font=font)
        cw, ch = box[2] - box[0], box[3] - box[1]
        x_jitter = rng.uniform(-3.0, 3.0)
        x = (
            left
            + index * cell_width
            + (cell_width - cw) / 2
            - box[0]
            + x_jitter
        )
        y = (HEIGHT - ch) / 2 - box[1]
        draw.text((x, y), char, font=font, fill=0)
        result = ImageChops.darker(result, _wave(char_layer, rng))

    # Noise is drawn only after the independently distorted glyphs have been
    # composed.  It therefore remains straight and does not warp the canvas.
    _noise(result, rng, dot_size=2, draw_lines=True)
    _noise(result, rng, dot_size=1, draw_lines=False)
    return _to_sample_gif(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="the exact five-character CAPTCHA text")
    parser.add_argument("-o", "--output", default="captcha.png",
                        help="output path (default: captcha.png; data is GIF89a)")
    parser.add_argument("--seed", type=int,
                        help="make the random noise and distortion reproducible")
    parser.add_argument("--font", help="path to a TrueType/OpenType font")
    args = parser.parse_args()
    try:
        image = make_captcha(args.text, seed=args.seed, font_path=args.font)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    image.save(
        args.output,
        format="GIF",
        transparency=252,
        duration=0,
        optimize=False,
        bits=8,
    )
    # Pillow writes color-resolution=1 (0x87).  The corpus encoder marks the
    # same 256-entry global table as color-resolution=8 (0xf7).  This field is
    # descriptive only, so patching it gives an exact GIF logical descriptor.
    output = Path(args.output)
    with output.open("r+b") as stream:
        stream.seek(10)
        stream.write(b"\xf7")


if __name__ == "__main__":
    main()
