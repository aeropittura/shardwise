#!/usr/bin/env python3


import torch

import shardwise
from shardwise.core.grad_fns import AutogradContext as _AutogradContext


class AutogradContext(_AutogradContext):
    """
    DEPRECATED: Object used by AutogradFunctions for saving context information.
    """

    def __init__(self) -> None:
        raise DeprecationWarning(
            "shardwise.core.autograd.AutogradContext is deprecated. Please "
            "use shardwise.core.grad_fns.AutogradContext instead."
        )
        super().__init__(self)


def AutogradShardTensor(tensor, requires_grad: bool = True):
    """
    DEPRECATED: ShardTensor with support for autograd, akin to the `Variable`
    originally in PyTorch.
    """
    raise DeprecationWarning(
        "AutogradShardTensor is deprecated. Please set the "
        "requires_grad attribute on the ShardTensor instead."
    )
    if torch.is_tensor(tensor):
        tensor = shardwise.tensor(tensor)
    tensor.requires_grad = requires_grad
    return tensor
