"""Generate a deterministic three-band demonstration scene."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo-scene.png"))
    parser.add_argument("--size", type=int, default=960)
    return parser.parse_args()


def generate(size: int) -> Image.Image:
    rng = np.random.default_rng(2026)
    y, x = np.mgrid[0:size, 0:size]
    texture = rng.normal(0, 8, size=(size, size))
    green = 74 + 35 * np.sin(x / 57) + texture
    red = 83 + 28 * np.cos(y / 71) + texture * 0.7
    nir = 118 + 42 * np.sin((x + y) / 83) + texture * 0.5
    composite = np.stack([nir, red, green], axis=-1)
    image = Image.fromarray(np.clip(composite, 0, 255).astype(np.uint8), "RGB")

    roads = Image.new("RGBA", image.size)
    road_draw = ImageDraw.Draw(roads)
    for offset in range(-size, size * 2, size // 5):
        road_draw.line((offset, 0, offset + size // 2, size), fill=(220, 207, 161, 85), width=5)
    image = Image.alpha_composite(image.convert("RGBA"), roads)

    clouds = Image.new("RGBA", image.size)
    cloud_draw = ImageDraw.Draw(clouds)
    for _ in range(22):
        cx = int(rng.normal(size * 0.56, size * 0.17))
        cy = int(rng.normal(size * 0.43, size * 0.14))
        radius_x = int(rng.uniform(size * 0.05, size * 0.15))
        radius_y = int(rng.uniform(size * 0.035, size * 0.09))
        cloud_draw.ellipse(
            (cx - radius_x, cy - radius_y, cx + radius_x, cy + radius_y),
            fill=(250, 252, 249, int(rng.uniform(185, 235))),
        )
    clouds = clouds.filter(ImageFilter.GaussianBlur(size / 80))
    return Image.alpha_composite(image, clouds).convert("RGB")


if __name__ == "__main__":
    arguments = parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    generate(arguments.size).save(arguments.output, optimize=True)
    print(f"Wrote demonstration scene to {arguments.output}")

