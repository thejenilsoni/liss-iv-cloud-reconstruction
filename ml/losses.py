from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(slots=True)
class LossBreakdown:
    total: torch.Tensor
    masked_l1: torch.Tensor
    context_l1: torch.Tensor
    spectral: torch.Tensor
    structure: torch.Tensor
    edge: torch.Tensor
    adversarial: torch.Tensor


class ReconstructionLoss(nn.Module):
    def __init__(
        self,
        masked_weight: float = 6.0,
        context_weight: float = 1.0,
        spectral_weight: float = 1.4,
        structure_weight: float = 0.8,
        edge_weight: float = 0.6,
        adversarial_weight: float = 0.08,
    ) -> None:
        super().__init__()
        self.weights = {
            "masked": masked_weight,
            "context": context_weight,
            "spectral": spectral_weight,
            "structure": structure_weight,
            "edge": edge_weight,
            "adversarial": adversarial_weight,
        }

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        discriminator_logits: torch.Tensor | None = None,
    ) -> LossBreakdown:
        masked_l1 = _region_mean(torch.abs(prediction - target), mask)
        context_l1 = _region_mean(torch.abs(prediction - target), 1.0 - mask)
        spectral = spectral_angle_loss(prediction, target, mask)
        structure = multiscale_structure_loss(prediction, target, mask)
        edge = gradient_loss(prediction, target, mask)
        adversarial = (
            F.binary_cross_entropy_with_logits(
                discriminator_logits,
                torch.ones_like(discriminator_logits),
            )
            if discriminator_logits is not None
            else prediction.new_zeros(())
        )
        total = (
            masked_l1 * self.weights["masked"]
            + context_l1 * self.weights["context"]
            + spectral * self.weights["spectral"]
            + structure * self.weights["structure"]
            + edge * self.weights["edge"]
            + adversarial * self.weights["adversarial"]
        )
        return LossBreakdown(
            total=total,
            masked_l1=masked_l1,
            context_l1=context_l1,
            spectral=spectral,
            structure=structure,
            edge=edge,
            adversarial=adversarial,
        )


def spectral_angle_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    dot = (prediction * target).sum(dim=1, keepdim=True)
    denominator = (
        prediction.square().sum(dim=1, keepdim=True).sqrt()
        * target.square().sum(dim=1, keepdim=True).sqrt()
    ).clamp_min(1e-6)
    cosine = (dot / denominator).clamp(-0.9999, 0.9999)
    return _region_mean(torch.acos(cosine), mask)


def multiscale_structure_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for kernel in (3, 7, 15):
        predicted_mean = F.avg_pool2d(prediction, kernel, stride=1, padding=kernel // 2)
        target_mean = F.avg_pool2d(target, kernel, stride=1, padding=kernel // 2)
        losses.append(_region_mean(torch.abs(predicted_mean - target_mean), mask))
    return torch.stack(losses).mean()


def gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    prediction_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    prediction_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    x_loss = _region_mean(torch.abs(prediction_x - target_x), mask[..., :, 1:])
    y_loss = _region_mean(torch.abs(prediction_y - target_y), mask[..., 1:, :])
    return (x_loss + y_loss) / 2.0


def _region_mean(values: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    expanded = region.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)

