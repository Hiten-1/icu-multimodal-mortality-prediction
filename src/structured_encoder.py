"""Structured clinical data encoder (PyTorch + NumPy fallback)."""

from typing import List

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:

    class StructuredEncoder(nn.Module):
        """Feed-forward neural encoder for structured ICU vitals, labs, and demographics.

        Processes standardized structured inputs through hidden layers [64, 32] with
        BatchNorm, ReLU, and Dropout, and projects to a 24-dimensional latent vector.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int] | None = None,
            output_dim: int = 24,
            dropout: float = 0.25,
        ):
            super().__init__()
            dims = hidden_dims or [64, 32]
            layers = []
            prev_dim = input_dim
            for h_dim in dims:
                layers.extend(
                    [
                        nn.Linear(prev_dim, h_dim),
                        nn.BatchNorm1d(h_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
                prev_dim = h_dim
            layers.extend(
                [
                    nn.Linear(prev_dim, output_dim),
                    nn.ReLU(),
                ]
            )
            self.network = nn.Sequential(*layers)
            self.output_dim = output_dim

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.network(x)

else:

    class StructuredEncoder:  # type: ignore[no-redef]
        """Lightweight descriptor when PyTorch is not installed."""

        def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int] | None = None,
            output_dim: int = 24,
            dropout: float = 0.25,
        ):
            self.input_dim = input_dim
            self.hidden_dims = hidden_dims or [64, 32]
            self.dropout = dropout
            self.output_dim = output_dim
