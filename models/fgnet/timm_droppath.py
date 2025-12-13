from typing import List, Union

import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


def calculate_drop_path_rates(
        drop_path_rate: float,
        depths: Union[int, List[int]],
        stagewise: bool = False,
) -> Union[List[float], List[List[float]]]:
    """Generate drop path rates for stochastic depth.

    This function handles two common patterns for drop path rate scheduling:
    1. Per-block: Linear increase from 0 to drop_path_rate across all blocks
    2. Stage-wise: Linear increase across stages, with same rate within each stage

    Args:
        drop_path_rate: Maximum drop path rate (at the end).
        depths: Either a single int for total depth (per-block mode) or
                list of ints for depths per stage (stage-wise mode).
        stagewise: If True, use stage-wise pattern. If False, use per-block pattern.
                   When depths is a list, stagewise defaults to True.

    Returns:
        For per-block mode: List of drop rates, one per block.
        For stage-wise mode: List of lists, drop rates per stage.
    """
    if isinstance(depths, int):
        # Single depth value - per-block pattern
        if stagewise:
            raise ValueError("stagewise=True requires depths to be a list of stage depths")
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths, device='cpu')]
        return dpr
    else:
        # List of depths - can be either pattern
        total_depth = sum(depths)
        if stagewise:
            # Stage-wise pattern: same drop rate within each stage
            dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, total_depth, device='cpu').split(depths)]
            return dpr
        else:
            # Per-block pattern across all stages
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth, device='cpu')]
            return dpr
