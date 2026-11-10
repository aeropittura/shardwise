"""Dropout layers."""

import logging

from shardwise.nn.layers.base import Module


class Dropout(Module):
    r"""During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution. Furthermore, the outputs are scaled by a factor of
    :math:`\frac{1}{1-p}` during training. This means that during evaluation
    the module simply computes an identity function.

    Args:
        p: probability of an element to be zeroed. Default: 0.5

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input
    """

    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if inplace:
            logging.warning("Shardwise Dropout module does not support inplace computation.")
        self.p = p

    def forward(self, input):
        return input.dropout(p=self.p, training=self.training)

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        return Dropout(attributes["ratio"])


class DropoutNd(Module):
    """Randomly zero out entire channels (a channel is a nD feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a nD tensor :math:`\text{input}[i, j]`).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    Args:
        p (float, optional): probability of an element to be zero-ed.
    """

    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if inplace:
            logging.warning("Shardwise DropoutNd module does not support inplace computation.")
        self.p = p

    def forward(self, input):
        return input._feature_dropout(p=self.p, training=self.training)

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        return DropoutNd(attributes["ratio"])


class Dropout2d(DropoutNd):
    r"""
    Randomly zero out entire channels (a channel is a 2D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 2D tensor :math:`\text{input}[i, j]`).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    Usually the input comes from :class:`nn.Conv2d` modules.

    Args:
        p (float, optional): probability of an element to be zero-ed.
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    """

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        return Dropout2d(attributes["ratio"])


class Dropout3d(DropoutNd):
    r"""Randomly zero out entire channels (a channel is a 3D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 3D tensor :math:`\text{input}[i, j]`).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    Usually the input comes from :class:`nn.Conv3d` modules.

    Args:
        p (float, optional): probability of an element to be zeroed.

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)
    """

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        return Dropout3d(attributes["ratio"])
