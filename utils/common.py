from typing import Callable, List, Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron with configurable activation and dropout."""

    def __init__(
        self,
        dims: List[int],
        activation_fn: Callable[[], nn.Module] = nn.Mish,
        output_activation_fn: Optional[Callable[[], nn.Module]] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()

        if len(dims) < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))

        if output_activation_fn is not None:
            layers.append(output_activation_fn())

        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Squeeze(nn.Module):
    """Squeeze module for removing dimensions."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)
