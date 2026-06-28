import torch
import torch.nn as nn


def weights_init(module: nn.Module) -> None:
    """
    Initialize the PatchGAN discriminator weights.

    This initialization follows the common GAN practice:
    - Convolution layers: Normal(mean=0, std=0.02)
    - BatchNorm layers: Normal(mean=1, std=0.02), bias=0
    """
    classname = module.__class__.__name__

    if "Conv" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)

    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator for ERA5-to-CERRA adversarial downscaling.

    The discriminator is fully convolutional and produces a spatial map of
    realism scores instead of a single scalar output. Each value in the output
    map corresponds to a local image patch.

    For unconditional PatchGAN:
        input_nc = 1
        input = CERRA temperature or predicted temperature

    For conditional PatchGAN:
        input_nc = 8
        input = concat([ERA5 predictors + static fields, CERRA/prediction], dim=1)

    No sigmoid is applied at the output because hinge loss uses raw logits.
    """

    def __init__(
        self,
        input_nc: int = 8,
        ndf: int = 64,
        n_layers: int = 3,
    ) -> None:
        super().__init__()

        kernel_size = 4
        padding = 1

        layers = []

        # First convolutional block:
        # No BatchNorm is used in the first layer, following standard PatchGAN practice.
        layers += [
            nn.Conv2d(
                in_channels=input_nc,
                out_channels=ndf,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        nf_mult_prev = 1

        # Intermediate downsampling blocks.
        # Each block doubles the number of feature channels up to a maximum factor of 8.
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)

            layers += [
                nn.Conv2d(
                    in_channels=ndf * nf_mult_prev,
                    out_channels=ndf * nf_mult,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Additional convolutional block with stride 1.
        # This keeps a denser patch-level output map.
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        layers += [
            nn.Conv2d(
                in_channels=ndf * nf_mult_prev,
                out_channels=ndf * nf_mult,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Final convolution:
        # Produces one raw realism score per patch.
        # No sigmoid is used.
        layers += [
            nn.Conv2d(
                in_channels=ndf * nf_mult,
                out_channels=1,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            )
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor with shape [B, input_nc, H, W].

        Returns:
            Patch-wise realism logits with shape [B, 1, H_out, W_out].
        """
        return self.model(x)
