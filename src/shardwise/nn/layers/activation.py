"""Activation functions."""

import logging
from shardwise.nn.layers.base import Module


class ReLU(Module):
    r"""
    Module that computes rectified linear unit (ReLU) activations element-wise.

    :math:`\text{ReLU}(x)= \max(0, x)`
    """

    def __init__(self, inplace=False):
        super().__init__()
        if inplace:
            logging.warning("Shardwise ReLU module does not support inplace computation.")

    def forward(self, x):
        return x.relu()

    @staticmethod
    def from_onnx(attributes=None):
        return ReLU()


class Hardtanh(Module):
    r"""Applies the Hardtanh function element-wise

    HardTtnh is defined as:

    .. math::
        \text{Hardtanh}(x) = \begin{cases}
            1 & \text{ if } x > 1 \\
            -1 & \text{ if } x < -1 \\
            x & \text{ otherwise } \\
        \end{cases}

    The range of the linear region :math:`[-1, 1]` can be adjusted using
    :attr:`min_val` and :attr:`max_val`.

    Args:
        min_val: minimum value of the linear region range. Default: -1
        max_val: maximum value of the linear region range. Default: 1
        inplace: can optionally do the operation in-place. Default: ``False``

    Keyword arguments :attr:`min_value` and :attr:`max_value`
    have been deprecated in favor of :attr:`min_val` and :attr:`max_val`.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: ../scripts/activation_images/Hardtanh.png

    Examples::

        >>> m = nn.Hardtanh(-2, 2)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, min_val=-1.0, max_val=1.0, inplace=False):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        if inplace:
            logging.warning(
                "Shardwise Hardtanh module does not support inplace computation."
            )

    def forward(self, input):
        if isinstance(input, list):
            input, min_val, max_val = input
            min_val, max_val = min_val.item(), max_val.item()
        else:
            min_val, max_val = self.min_val, self.max_val
        return input.hardtanh(min_val, max_val)

    @staticmethod
    def from_onnx(attributes=None):
        return Hardtanh(
            min_val=attributes.get("min", -1.0),
            max_val=attributes.get("max", 1.0),
        )


class ReLU6(Hardtanh):
    r"""Applies the element-wise function:

    .. math::
        \text{ReLU6}(x) = \min(\max(0,x), 6)

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: ../scripts/activation_images/ReLU6.png

    Examples::

        >>> m = nn.ReLU6()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, inplace=False):
        if inplace:
            logging.warning(
                "Shardwise ReLU6 module does not support inplace computation."
            )
        super(ReLU6, self).__init__(min_val=0, max_val=6, inplace=False)


class Sigmoid(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}
    """

    def forward(self, x):
        return x.sigmoid()

    @staticmethod
    def from_onnx(attributes=None):
        return Sigmoid()


class Softmax(Module):
    r"""Applies the Softmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range [0,1] and sum to 1.

    Softmax is defined as:

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Arguments:
        dim (int): A dimension along which Softmax will be computed (so every slice
            along dim will sum to 1).
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return input.softmax(self.dim)

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        assert "axis" in attributes, "axis attribute missing"
        return Softmax(attributes["axis"])


class LogSoftmax(Module):
    r"""Applies the :math:`\log(\text{Softmax}(x))` function to an n-dimensional
    input Tensor. The LogSoftmax formulation can be simplified as:

    .. math::
        \text{LogSoftmax}(x_{i}) =
        \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Arguments:
        dim (int): A dimension along which LogSoftmax will be computed.

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range \[-inf, 0\)
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return input.log_softmax(self.dim)

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        assert "axis" in attributes, "axis attribute missing"
        return LogSoftmax(attributes["axis"])
