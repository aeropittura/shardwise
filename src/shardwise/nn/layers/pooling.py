"""Pooling layers."""

from shardwise.nn.layers._onnx_attrs import _all_the_same
from shardwise.nn.layers.base import Module
from shardwise.ops.pooling import _adaptive_pool2d_helper


class _Pool2d(Module):
    """
    Module that performs 2D pooling.
    Applies a 2D max or average pooling over an input signal composed of several input
    planes.

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on
    both sides for :attr:`padding` number of points. :attr:`dilation` controls
    the spacing between the kernel points. It is harder to describe, but this
    `link`_ has a nice visualization of what :attr:`dilation` does.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`,
    :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the
        height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for
        the height dimension, and the second `int` for the width dimension

    Args:
        pool_type (str): specifies "average" or "max" pooling
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, pool_type, kernel_size, stride=None, padding=0, ceil_mode=False):
        super().__init__()
        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.ceil_mode = ceil_mode

    def forward(self, x):
        args = [self.kernel_size]
        kwargs = {
            "stride": self.stride,
            "padding": self.padding,
            "ceil_mode": self.ceil_mode,
        }
        if self.pool_type == "average":
            return x.avg_pool2d(*args, **kwargs)
        elif self.pool_type == "max":
            return x.max_pool2d(*args, **kwargs)
        else:
            raise ValueError(f"Unknown pooling type: {self.pool_type}")

    @staticmethod
    def from_onnx(pool_type, attributes=None):
        # check attributes:
        if attributes is None:
            attributes = {}
        if "pads" not in attributes:
            attributes["pads"] = [0]
        assert _all_the_same(["kernel_shape"]), "only square kernels are supported"
        assert _all_the_same(attributes["strides"]), "stride must be the same in each dimension"
        attributes["ceil_mode"] = attributes.get("ceil_mode", 0)
        attributes["ceil_mode"] = attributes["ceil_mode"] > 0

        # initialize module
        args = [attributes["kernel_shape"][0]]
        kwargs = {
            "stride": attributes["strides"][0],
            "padding": attributes["pads"][0],
            "ceil_mode": attributes["ceil_mode"],
        }
        if pool_type == "average":
            return AvgPool2d(*args, **kwargs)
        elif pool_type == "max":
            return MaxPool2d(*args, **kwargs)
        else:
            raise ValueError(f"Unknown pooling type: {pool_type}")


class AvgPool2d(_Pool2d):
    r"""
    Module that Applies a 2D average pooling
    over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C, H, W)`, output :math:`(N, C, H_{out}, W_{out})` and
    :attr:`kernel_size` :math:`(kH, kW)` can be precisely described as:

    .. math::

        out(N_i, C_j, h, w)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1}
        \sum_{n=0}^{kW-1} input(N_i, C_j, stride[0] \times h + m, stride[1]
        \times w + n)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on
    both sides for :attr:`padding` number of points.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can either be:

        - a single ``int`` -- in which case the same value is used for the
        height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for
        the height dimension, and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] -
                \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] -
                \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor
    """

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False):
        super().__init__(
            "average", kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode
        )

    @staticmethod
    def from_onnx(attributes=None):
        return super(AvgPool2d, AvgPool2d).from_onnx("average", attributes=attributes)


class MaxPool2d(_Pool2d):
    """
    Module that performs 2D max pooling (see :meth:`AvgPool2d`)
    """

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False):
        super().__init__("max", kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)

    @staticmethod
    def from_onnx(attributes=None):
        return super(MaxPool2d, MaxPool2d).from_onnx("max", attributes=attributes)


class AdaptiveAvgPool2d(Module):
    r"""Applies a 2D adaptive average pooling over an input signal composed of several input planes.

    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H.
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

    Examples:
        >>> # target output size of 5x7
        >>> m = nn.AdaptiveAvgPool2d((5, 7))
        >>> input = shardwise.randn(1, 64, 8, 9)
        >>> output = m(input)
        >>> # target output size of 7x7 (square)
        >>> m = nn.AdaptiveAvgPool2d(7)
        >>> input = shardwise.randn(1, 64, 10, 9)
        >>> output = m(input)
        >>> # target output size of 10x7
        >>> m = nn.AdaptiveAvgPool2d((None, 7))
        >>> input = shardwise.randn(1, 64, 10, 9)
        >>> output = m(input)
    """

    def __init__(self, output_size=None):
        super().__init__()
        self.output_size = output_size

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"

    def forward(self, input_tensor, output_size=None):
        if output_size is None:
            output_size = self.output_size
        assert output_size is not None, (
            "AdaptiveAvgPool2d requires an output_size in forward if not supplied in initialization"
        )
        resized_input, args, kwargs = _adaptive_pool2d_helper(
            input_tensor, output_size, reduction="mean"
        )
        return resized_input.avg_pool2d(*args, **kwargs)

    @staticmethod
    def from_onnx(attributes=None):
        if "shape" in attributes:
            return AdaptiveAvgPool2d(output_size=attributes["shape"])
        return AdaptiveAvgPool2d()


class AdaptiveMaxPool2d(Module):
    r"""Applies a 2D adaptive max pooling over an input signal composed of several input planes.

    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H.
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

    Examples:
        >>> # target output size of 5x7
        >>> m = nn.AdaptiveMaxPool2d((5, 7))
        >>> input = shardwise.randn(1, 64, 8, 9)
        >>> output = m(input)
        >>> # target output size of 7x7 (square)
        >>> m = nn.AdaptiveMaxPool2d(7)
        >>> input = shardwise.randn(1, 64, 10, 9)
        >>> output = m(input)
        >>> # target output size of 10x7
        >>> m = nn.AdaptiveMaxPool2d((None, 7))
        >>> input = shardwise.randn(1, 64, 10, 9)
        >>> output = m(input)

    """

    def __init__(self, output_size=None):
        super().__init__()
        self.output_size = output_size

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"

    def forward(self, input_tensor, output_size=None):
        if output_size is None:
            output_size = self.output_size
        assert output_size is not None, (
            "AdaptiveMaxPool2d requires an output_size in forward if not supplied in initialization"
        )
        resized_input, args, kwargs = _adaptive_pool2d_helper(
            input_tensor, output_size, reduction="max"
        )
        return resized_input.max_pool2d(*args, **kwargs)

    @staticmethod
    def from_onnx(attributes=None):
        if "shape" in attributes:
            return AdaptiveMaxPool2d(output_size=attributes["shape"])
        return AdaptiveMaxPool2d()


class GlobalAveragePool(Module):
    """
    GlobalAveragePool consumes an input tensor and applies average pooling
    across the values in the same channel. This is equivalent to AveragePool
    with kernel size equal to the spatial dimension of input tensor. This is an
    operation from the ONNX specification.
    """

    def forward(self, input):
        assert input.dim() > 2, "input needs to have more than two dimensions"

        # sum over all but batch dimension:
        result = input.shallow_copy()
        for dim in range(2, input.dim()):
            result = result.sum(dim, keepdim=True)

        # return average value:
        first_two_dims = input.size(0) * input.size(1)
        return result.div(input.nelement() / float(first_two_dims))

    @staticmethod
    def from_onnx(attributes=None):
        return GlobalAveragePool()
