from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

BITNET_SCALE_EPS = 1e-5


class BitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, scale_eps: float = BITNET_SCALE_EPS):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale_eps = scale_eps
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            bound = 1 / self.in_features**0.5 if self.in_features > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def quantized_weight(self) -> Tensor:
        weight = self.weight.float()
        scale = weight.abs().mean(dim=1, keepdim=True).clamp_min(self.scale_eps)
        signs = torch.where(weight >= 0, torch.ones_like(weight), -torch.ones_like(weight))
        quant = signs * scale
        return weight + (quant - weight).detach()

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.quantized_weight().to(x.dtype), bias)


def collect_bitlinear_weight_names(module: nn.Module) -> set[str]:
    names: set[str] = set()
    for name, child in module.named_modules():
        if isinstance(child, BitLinear):
            prefix = f"{name}." if name else ""
            names.add(f"{prefix}weight")
            if child.bias is not None:
                names.add(f"{prefix}bias")
    return names
