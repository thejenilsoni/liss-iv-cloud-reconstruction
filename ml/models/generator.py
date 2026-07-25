import torch
from torch import nn
from torch.nn import functional as F


class GatedConv(nn.Module):
    """Learnable feature and validity gates for irregular cloud masks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        normalization: bool = True,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.feature = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.gate = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels) if normalization else nn.Identity()
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.norm(self.feature(inputs))
        gates = torch.sigmoid(self.gate(inputs))
        return self.activation(features) * gates


class ResidualGatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.first = GatedConv(channels, channels, dilation=dilation)
        self.second = GatedConv(channels, channels, dilation=dilation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.second(self.first(inputs))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fuse = GatedConv(in_channels + skip_channels, out_channels)
        self.refine = ResidualGatedBlock(out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(self.fuse(torch.cat([inputs, skip], dim=1)))


class MaskGuidedGenerator(nn.Module):
    """Gated encoder-decoder that predicts only masked multispectral pixels.

    The expected input concatenates normalized spectral bands and one cloud
    mask channel. The output preserves observed pixels by construction.
    """

    def __init__(self, bands: int = 3, base_channels: int = 48) -> None:
        super().__init__()
        self.bands = bands
        self.stem = GatedConv(bands + 1, base_channels, kernel_size=5)
        self.down_one = GatedConv(base_channels, base_channels * 2, stride=2)
        self.down_two = GatedConv(base_channels * 2, base_channels * 4, stride=2)
        self.down_three = GatedConv(base_channels * 4, base_channels * 8, stride=2)

        bottleneck_channels = base_channels * 8
        self.bottleneck = nn.Sequential(
            ResidualGatedBlock(bottleneck_channels, dilation=1),
            ResidualGatedBlock(bottleneck_channels, dilation=2),
            ResidualGatedBlock(bottleneck_channels, dilation=4),
            ResidualGatedBlock(bottleneck_channels, dilation=8),
            nn.Dropout2d(0.12),
        )

        self.up_three = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up_two = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up_one = DecoderBlock(base_channels * 2, base_channels, base_channels)
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, bands, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        image = inputs[:, : self.bands]
        mask = inputs[:, self.bands : self.bands + 1]

        level_one = self.stem(inputs)
        level_two = self.down_one(level_one)
        level_three = self.down_two(level_two)
        encoded = self.down_three(level_three)
        encoded = self.bottleneck(encoded)

        decoded = self.up_three(encoded, level_three)
        decoded = self.up_two(decoded, level_two)
        decoded = self.up_one(decoded, level_one)
        prediction = self.output(decoded)
        return image * (1.0 - mask) + prediction * mask


class PatchDiscriminator(nn.Module):
    """Patch-level critic conditioned on the cloudy observation and mask."""

    def __init__(self, bands: int = 3, base_channels: int = 48) -> None:
        super().__init__()
        channels = bands * 2 + 1
        blocks: list[nn.Module] = []
        for multiplier in (1, 2, 4, 8):
            output_channels = base_channels * multiplier
            blocks.extend(
                [
                    nn.utils.spectral_norm(
                        nn.Conv2d(channels, output_channels, 4, stride=2, padding=1)
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            channels = output_channels
        blocks.append(nn.utils.spectral_norm(nn.Conv2d(channels, 1, 3, padding=1)))
        self.network = nn.Sequential(*blocks)

    def forward(
        self,
        cloudy: torch.Tensor,
        candidate: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(torch.cat([cloudy, candidate, mask], dim=1))

