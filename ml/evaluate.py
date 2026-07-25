import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an exported generator.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def evaluate(model_path: Path, data_path: Path) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(model_path), map_location=device).eval()
    totals = {"mae": 0.0, "mse": 0.0, "psnr": 0.0, "sam_degrees": 0.0}
    files = sorted(data_path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No evaluation patches found in {data_path}")

    for path in files:
        with np.load(path) as sample:
            clear = torch.from_numpy(sample["clear"]).float().unsqueeze(0).to(device)
            cloudy = torch.from_numpy(sample["cloudy"]).float().unsqueeze(0).to(device)
            mask = torch.from_numpy(sample["mask"]).float().reshape(1, 1, *clear.shape[-2:]).to(device)
        prediction = model(torch.cat([cloudy, mask], dim=1))
        active = mask.expand_as(clear) >= 0.5
        difference = (prediction - clear)[active]
        mae = difference.abs().mean()
        mse = difference.square().mean()

        dot = (prediction * clear).sum(dim=1)
        denominator = (
            prediction.square().sum(dim=1).sqrt() * clear.square().sum(dim=1).sqrt()
        ).clamp_min(1e-6)
        angles = torch.rad2deg(torch.acos((dot / denominator).clamp(-0.9999, 0.9999)))
        totals["mae"] += float(mae)
        totals["mse"] += float(mse)
        totals["psnr"] += float(20 * torch.log10(1.0 / torch.sqrt(mse + 1e-12)))
        totals["sam_degrees"] += float(angles[mask[:, 0] >= 0.5].mean())

    return {key: value / len(files) for key, value in totals.items()}


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(evaluate(arguments.model, arguments.data), indent=2))

