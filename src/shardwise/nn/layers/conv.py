"""Convolution layers."""

import torch
from shardwise.nn.layers.base import Module


class Conv(Module):
    """
    Module that performs convolution, following the ONNX specification of that
    operation. Unlike other Conv modules, this module is stateless.
    """

    def __init__(self, stride, padding, dilation, groups=1):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        # unpack inputs:
        if len(x) == 2:
            x, weight = x
            bias = None
        elif len(x) == 3:
            x, weight, bias = x
        else:
            raise ValueError(f"Conv module must have 2 or 3 inputs, not {len(x)}")

        # prepare inputs into convolution function:
        dim = weight.dim() - 2
        if dim < 1 or dim > 2:
            raise ValueError(
                f"Convolution on {dim}-dimensional input is not supported."
            )
        args = [weight]
        kwargs = {
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "groups": self.groups,
        }

        # identify correct convolution function to use:
        if torch.is_tensor(x):
            func = getattr(torch.nn.functional, f"conv{dim}d", None)
            args = [x] + args + bias  # torch function takes different inputs
        else:
            func = getattr(x, f"conv{dim}d", None)

        # perform the convolution:
        x = func(*args, **kwargs)

        # add the bias term if it is specified, and wasn;t already added:
        if not torch.is_tensor(x) and bias is not None:
            bias = bias.unsqueeze(0)
            while bias.dim() < x.dim():
                bias = bias.unsqueeze(-1)
            x = x.add(bias)
        return x

    @staticmethod
    def from_onnx(attributes=None):
        # check attribute inputs:
        if attributes is None:
            attributes = {}
        for attr in ["strides", "pads", "dilations"]:
            assert attr in attributes, f"missing attribute {attr}"

        # Shardwise and Torch use a single padding number per dimension:
        padding = attributes["pads"]
        padding = [padding[idx] for idx in range(0, len(padding), 2)]

        # return module:
        return Conv(
            attributes["strides"],
            padding,
            attributes["dilations"],
            groups=attributes.get("group", 1),
        )


# TODO: Eliminate copy-pasta by implementing _Conv parent class
class Conv1d(Module):
    r"""
    Module that performs 1D convolution.

    Applies a 1D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, L)` and output :math:`(N, C_{\text{out}}, L_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k)
        \star \text{input}(N_i, k)


    where :math:`\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    and :math:`L` is a length of signal sequence.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a one-element tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters, of size:
          :math:`\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor`.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`,
    :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the
        height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for
        the height dimension, and the second `int` for the width dimension

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
        the input. Default: 0
        padding_mode (string, optional). Accepted values `zeros` and `circular`
        Default: `zeros`
        dilation (int or tuple, optional): Spacing between kernel elements.
        Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
        Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where

          .. math::
              L_{out} = \left\lfloor\frac{L_{in}  +
              2 \times \text{padding} - \text{dilation} \times
              (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """  # noqa: W605

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        # check inputs:
        super().__init__()
        assert isinstance(stride, int), "stride must be an integer"
        assert isinstance(padding, int), "padding must be an integer"

        # initialize model parameters:
        pytorch_module = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.register_parameter("weight", pytorch_module.weight)
        if bias:
            self.register_parameter("bias", pytorch_module.bias)

        # set other instance fields:
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        x = x.conv1d(
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if hasattr(self, "bias"):
            x = x.add(self.bias.unsqueeze(-1))
        return x


class Conv2d(Module):
    r"""
    Module that performs 2D convolution.

    Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}},
    H_{\text{out}}, W_{\text{out}})` can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k)
        \star \text{input}(N_i, k)


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters, of size:
          :math:`\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor`.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`,
    :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the
        height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for
        the height dimension, and the second `int` for the width dimension

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
        the input. Default: 0
        padding_mode (string, optional). Accepted values `zeros` and `circular`
        Default: `zeros`
        dilation (int or tuple, optional): Spacing between kernel elements.
        Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
        Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  +
              2 \times \text{padding}[0] - \text{dilation}[0] \times
              (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  +
              2 \times \text{padding}[1] - \text{dilation}[1] \times
              (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        # check inputs:
        super().__init__()

        # initialize model parameters:
        pytorch_module = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.register_parameter("weight", pytorch_module.weight)
        if bias:
            self.register_parameter("bias", pytorch_module.bias)
        else:
            self.bias = pytorch_module.bias

        # set other instance fields:
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        x = x.conv2d(
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if hasattr(self, "bias") and self.bias is not None:
            x = x.add(self.bias.unsqueeze(-1).unsqueeze(-1))
        return x
