"""Convert clear multispectral GeoTIFFs into normalized training patches."""

import argparse
from pathlib import Path

import numpy as np
import rasterio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=192)
    parser.add_argument("--bands", type=int, default=3)
    parser.add_argument("--minimum-valid", type=float, default=0.92)
    return parser.parse_args()


def normalize(array: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(array, dtype=np.float32)
    for index, band in enumerate(array.astype(np.float32)):
        valid = np.isfinite(band)
        if not valid.any():
            continue
        low, high = np.percentile(band[valid], [1, 99])
        normalized[index] = np.clip((band - low) / (high - low + 1e-6), 0, 1)
        normalized[index][~valid] = 0
    return normalized


def prepare(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    written = 0

    for scene_path in sorted(args.input.glob("*.tif*")):
        with rasterio.open(scene_path) as source:
            indexes = list(range(1, min(source.count, args.bands) + 1))
            raw_scene = source.read(indexes)
            nodata = source.nodata
        valid_scene = np.isfinite(raw_scene)
        if nodata is not None:
            valid_scene &= raw_scene != nodata
        scene = normalize(raw_scene)
        height, width = scene.shape[1:]

        for top in range(0, height - args.patch_size + 1, args.stride):
            for left in range(0, width - args.patch_size + 1, args.stride):
                patch = scene[
                    :,
                    top : top + args.patch_size,
                    left : left + args.patch_size,
                ]
                valid_patch = valid_scene[
                    :,
                    top : top + args.patch_size,
                    left : left + args.patch_size,
                ]
                valid_fraction = float(np.mean(np.all(valid_patch, axis=0)))
                if valid_fraction < args.minimum_valid:
                    continue
                output_name = f"{scene_path.stem}_{top:05d}_{left:05d}.npz"
                np.savez_compressed(args.output / output_name, clear=patch)
                written += 1

    return written


if __name__ == "__main__":
    arguments = parse_args()
    count = prepare(arguments)
    print(f"Wrote {count} training patches to {arguments.output}")
