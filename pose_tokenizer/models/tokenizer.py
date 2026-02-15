"""Compatibility tokenizer module for legacy imports."""

import torch
import torch.nn as nn

from .quantizer import ResidualVectorQuantizer


class RVQTokenizer(nn.Module):
    """Legacy RVQ tokenizer wrapper.

    Expects a config object with hidden_dim, num_quantizers, codebook_size, commitment_cost.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = getattr(config, "hidden_dim", 256)
        self.num_quantizers = getattr(config, "num_quantizers", 4)
        self.codebook_size = getattr(config, "codebook_size", 1024)
        self.commitment_cost = getattr(config, "commitment_cost", 0.25)
        self.dead_code_threshold = getattr(config, "dead_code_threshold", 1e-5)

        self.quantizer = ResidualVectorQuantizer(
            num_quantizers=self.num_quantizers,
            codebook_size=self.codebook_size,
            embedding_dim=self.hidden_dim,
            commitment_cost=self.commitment_cost,
            dead_code_threshold=self.dead_code_threshold,
        )

    def forward(self, x):
        quantized, indices, losses = self.quantizer(x)

        # Provide a legacy losses list for tests
        vq_loss = losses.get("vq_loss", torch.tensor(0.0, device=x.device))
        losses_list = [vq_loss for _ in range(self.num_quantizers)]

        return quantized, indices, losses_list

    def encode(self, x):
        return self.quantizer.encode(x)

    def decode(self, indices):
        return self.quantizer.decode(indices)
