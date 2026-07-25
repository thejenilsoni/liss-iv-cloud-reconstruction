import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from ml.data import LissCloudDataset
from ml.losses import ReconstructionLoss
from ml.models import MaskGuidedGenerator, PatchDiscriminator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the mask-guided LISS-IV generator.")
    parser.add_argument("--data", type=Path, required=True, help="Directory containing .npz patches")
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--bands", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LissCloudDataset(
        args.data,
        patch_size=args.patch_size,
        bands=args.bands,
        seed=args.seed,
    )
    validation_size = max(1, round(len(dataset) * 0.12))
    training_size = len(dataset) - validation_size
    training, validation = random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    loaders = {
        "train": DataLoader(
            training,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        ),
        "validation": DataLoader(
            validation,
            batch_size=args.batch_size,
            num_workers=args.workers,
        ),
    }

    generator = MaskGuidedGenerator(bands=args.bands).to(device)
    discriminator = PatchDiscriminator(bands=args.bands).to(device)
    reconstruction_loss = ReconstructionLoss()
    generator_optimizer = AdamW(generator.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    discriminator_optimizer = AdamW(
        discriminator.parameters(),
        lr=args.learning_rate * 0.5,
        betas=(0.5, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        generator_optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.05,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_training_epoch(
            loaders["train"],
            generator,
            discriminator,
            reconstruction_loss,
            generator_optimizer,
            discriminator_optimizer,
            device,
        )
        validation_metrics = run_validation_epoch(
            loaders["validation"],
            generator,
            reconstruction_loss,
            device,
        )
        scheduler.step()

        record = {
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        print(json.dumps(record))

        checkpoint = {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer": generator_optimizer.state_dict(),
            "validation_loss": validation_metrics["loss"],
            "bands": args.bands,
        }
        torch.save(checkpoint, args.output / "latest.pt")
        if validation_metrics["loss"] < best_validation:
            best_validation = validation_metrics["loss"]
            torch.save(checkpoint, args.output / "best.pt")
            export_torchscript(generator, args.output / "generator.ts", args.bands, args.patch_size)


def run_training_epoch(
    loader: DataLoader,
    generator: MaskGuidedGenerator,
    discriminator: PatchDiscriminator,
    reconstruction_loss: ReconstructionLoss,
    generator_optimizer: AdamW,
    discriminator_optimizer: AdamW,
    device: torch.device,
) -> dict[str, float]:
    generator.train()
    discriminator.train()
    totals = {"generator": 0.0, "discriminator": 0.0, "masked_l1": 0.0}

    for batch in loader:
        clear, cloudy, mask = (batch[key].to(device) for key in ("clear", "cloudy", "mask"))
        model_input = torch.cat([cloudy, mask], dim=1)

        with torch.no_grad():
            detached_prediction = generator(model_input)
        real_logits = discriminator(cloudy, clear, mask)
        fake_logits = discriminator(cloudy, detached_prediction, mask)
        discriminator_loss = (
            nn.functional.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            + nn.functional.binary_cross_entropy_with_logits(
                fake_logits, torch.zeros_like(fake_logits)
            )
        ) / 2.0
        discriminator_optimizer.zero_grad(set_to_none=True)
        discriminator_loss.backward()
        discriminator_optimizer.step()

        prediction = generator(model_input)
        adversarial_logits = discriminator(cloudy, prediction, mask)
        breakdown = reconstruction_loss(prediction, clear, mask, adversarial_logits)
        generator_optimizer.zero_grad(set_to_none=True)
        breakdown.total.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), max_norm=2.0)
        generator_optimizer.step()

        totals["generator"] += float(breakdown.total.detach())
        totals["discriminator"] += float(discriminator_loss.detach())
        totals["masked_l1"] += float(breakdown.masked_l1.detach())

    return {key: value / max(len(loader), 1) for key, value in totals.items()}


@torch.inference_mode()
def run_validation_epoch(
    loader: DataLoader,
    generator: MaskGuidedGenerator,
    reconstruction_loss: ReconstructionLoss,
    device: torch.device,
) -> dict[str, float]:
    generator.eval()
    total_loss = 0.0
    total_masked_l1 = 0.0
    for batch in loader:
        clear, cloudy, mask = (batch[key].to(device) for key in ("clear", "cloudy", "mask"))
        prediction = generator(torch.cat([cloudy, mask], dim=1))
        breakdown = reconstruction_loss(prediction, clear, mask)
        total_loss += float(breakdown.total)
        total_masked_l1 += float(breakdown.masked_l1)
    divisor = max(len(loader), 1)
    return {"loss": total_loss / divisor, "masked_l1": total_masked_l1 / divisor}


def export_torchscript(
    generator: MaskGuidedGenerator,
    path: Path,
    bands: int,
    patch_size: int,
) -> None:
    generator.eval()
    example = torch.zeros(1, bands + 1, patch_size, patch_size, device=next(generator.parameters()).device)
    traced = torch.jit.trace(generator, example)
    torch.jit.save(traced, path)


if __name__ == "__main__":
    train(parse_args())

