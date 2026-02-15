"""Unified Pose Tokenizer model (encoder + tokenizer + decoder)."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from .encoder import PoseEncoder
from .decoder import PoseDecoder
from .quantizer import PoseTokenizer


class PoseTokenizerModel(nn.Module):
    """Pose tokenizer model combining encoder, RVQ tokenizer, and decoder."""

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.encoder = PoseEncoder(
            num_keypoints=config.num_keypoints,
            keypoint_dim=config.keypoint_dim,
            hidden_dim=config.hidden_dim,
            spatial_layers=getattr(config, "num_layers", 4),
            temporal_layers=getattr(config, "num_layers", 4),
            dropout=getattr(config, "stgcn_dropout", 0.1),
            use_temporal=True,
        )

        self.tokenizer = PoseTokenizer(
            encoder_dim=config.hidden_dim,
            num_quantizers=config.num_quantizers,
            codebook_size=config.codebook_size,
            commitment_cost=config.commitment_cost,
            shared_codebook=False,
            dead_code_threshold=getattr(config, "dead_code_threshold", 1e-5),
        )

        self.decoder = PoseDecoder(
            num_keypoints=config.num_keypoints,
            keypoint_dim=config.keypoint_dim,
            hidden_dim=config.hidden_dim,
            spatial_layers=getattr(config, "num_layers", 4),
            temporal_layers=getattr(config, "num_layers", 4),
            dropout=getattr(config, "decoder_dropout", 0.1),
            use_temporal=True,
        )

    def forward(
        self,
        poses: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with reconstruction and quantization losses."""
        encoded = self.encoder(poses, mask)
        quantized, indices, quant_losses = self.tokenizer(encoded)
        reconstructed = self.decoder(quantized, poses.shape[1:])

        return {
            "encoded": encoded,
            "quantized": quantized,
            "reconstructed": reconstructed,
            "indices": indices,
            "quant_losses": quant_losses,
        }

    @torch.no_grad()
    def encode(
        self,
        poses: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode poses to discrete tokens."""
        encoded = self.encoder(poses, mask)
        tokens = self.tokenizer.encode(encoded)

        if encoded.dim() == 3:
            lengths = mask.sum(dim=1) if mask is not None else torch.full(
                (encoded.size(0),), encoded.size(1), device=encoded.device, dtype=torch.long
            )
        else:
            lengths = torch.ones(encoded.size(0), device=encoded.device, dtype=torch.long)

        return tokens, lengths

    @torch.no_grad()
    def decode(
        self,
        tokens: torch.Tensor,
        target_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Decode discrete tokens to pose sequences."""
        quantized = self.tokenizer.decode(tokens)
        return self.decoder(quantized, target_shape)
