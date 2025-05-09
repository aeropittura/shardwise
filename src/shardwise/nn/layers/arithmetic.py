"""Elementwise arithmetic, reductions and comparisons."""

import shardwise
import torch
import torch.onnx.symbolic_helper as sym_help
from shardwise.nn.layers._onnx_attrs import _identify_bool_attributes_with_defaults
from shardwise.nn.layers.base import Module


class Add(Module):
    """
    Module that sums two values.
    """

    def forward(self, input):
        assert isinstance(input, (list, tuple)), "input must be list or tuple"
        assert len(input) == 2, "input must contain two tensors"
        return input[0].add(input[1])

    @staticmethod
    def from_onnx(attributes=None):
        return Add()


class Sub(Module):
    """
    Module that subtracts two values.
    """

    def forward(self, input):
        assert isinstance(input, (list, tuple)), "input must be list or tuple"
        assert len(input) == 2, "input must contain two tensors"
        return input[0].sub(input[1])

    @staticmethod
    def from_onnx(attributes=None):
        return Sub()


class Mul(Module):
    """
    Module that multiplies two values.
    """

    def forward(self, input):
        assert isinstance(input, (list, tuple)), "input must be list or tuple"
        assert len(input) == 2, "input must contain two tensors"
        return input[0].mul(input[1])

    @staticmethod
    def from_onnx(attributes=None):
        return Mul()


class Div(Module):
    """
    Module that divides two values.
    """

    def forward(self, input):
        assert isinstance(input, (list, tuple)), "input must be list or tuple"
        assert len(input) == 2, "input must contain two tensors"
        return input[0].div(input[1])

    @staticmethod
    def from_onnx(attributes=None):
        return Div()


class Pow(Module):
    """
    Module that takes input to some power, where the power is an integer.
    """

    def forward(self, input):
        base, power = input
        if torch.is_tensor(power) and power.nelement() == 1:
            power = power.item()
            if int(power) == power:  # try to convert power to integer if possible
                power = int(power)
        return base.pow(power)

    @staticmethod
    def from_onnx(attributes=None):
        return Pow()


class Sqrt(Module):
    """
    Module that takes square-root of the input.
    """

    def forward(self, input):
        return input.sqrt()

    @staticmethod
    def from_onnx(attributes=None):
        return Sqrt()


class Exp(Module):
    """
    Module that calculates the exponential of the given input tensor, element-wise.
    """

    def forward(self, input):
        return input.exp()

    @staticmethod
    def from_onnx(attributes=None):
        return Exp()


class Erf(Module):
    """
    Module that calculates the error function of the given input tensor, element-wise.
    """

    def forward(self, input):
        return input.erf()

    @staticmethod
    def from_onnx(attributes=None):
        return Erf()


class _Reduce(Module):
    """
    Base class for the functionality of ONNX ReduceMean (defined here as Mean),
    and ONNX ReduceSum (defined here as Sum).
    """

    def __init__(self, dim, keepdim=False, reduction_fn="mean"):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.reduction_fn = reduction_fn

    def forward(self, input):
        return getattr(input, self.reduction_fn)(self.dim, keepdim=self.keepdim)


class Mean(_Reduce):
    """
    Module that computes the mean of the input tensor's element along the provided axes.
    If `keepdim` is True, the output tensor is of the same size as input
    except in the dimension(s) `dim` where it is of size 1.
    Otherwise, `dim` is squeezed, resulting in the output tensor having 1
    (or `len(dim)`) fewer dimension(s).
    """

    def __init__(self, dim, keepdim=False):
        super().__init__(dim, keepdim, "mean")

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        keepdim = _identify_bool_attributes_with_defaults(attributes, "keepdims", 1)
        return Mean(attributes["axes"], keepdim)


class Sum(_Reduce):
    """
    Module that computes the sum of the input tensor's element along the provided axes.
    If `keepdim` is True, the output tensor is of the same size as input
    except in the dimension(s) `dim` where it is of size 1.
    Otherwise, `dim` is squeezed, resulting in the output tensor having 1
    (or `len(dim)`) fewer dimension(s).
    """

    def __init__(self, dim, keepdim=False):
        super().__init__(dim, keepdim, "sum")

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        keepdim = _identify_bool_attributes_with_defaults(attributes, "keepdims", 1)
        return Sum(attributes["axes"], keepdim)


class Cast(Module):
    """
    Module that casts the input tensor to the specified type.
    """

    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        if torch.is_tensor(x):
            return x.to(dtype=self.dtype)
        return x  # this is a no-op as SecureTensors do not know their dtype

    @staticmethod
    def from_onnx(attributes=None):
        dtype = sym_help._get_const(attributes["to"], "i", "dtype")
        return Cast(dtype=sym_help.scalar_type_to_pytorch_type[dtype])


class Range(Module):
    """
    Module that returns a tensor with the specified range.
    """

    SUPPORTS_PLAINTEXT_INPUTS = True

    def forward(self, x):
        if len(x) == 2:
            start, end = tuple(x)
            step = 1
        elif len(x) == 3:
            start, end, step = tuple(x)
        else:
            raise ValueError(f"Expected 2 or 3 inputs, but received {len(x)}.")
        return torch.arange(start, end, step)

    @staticmethod
    def from_onnx(attributes=None):
        return Range()


class Equal(Module):
    """
    Module that compares two tensors to determine which elements are equal.
    """

    def forward(self, x):
        x1, x2 = tuple(x)
        if x1.size() != x2.size():
            return False
        return x1.eq(x2)

    @staticmethod
    def from_onnx(attributes=None):
        return Equal()


class Where(Module):
    """
    Module that returns elements from one tensor or the other depending on the
    value of the specified condition.
    """

    def forward(self, x):
        condition, x1, x2 = tuple(x)
        return shardwise.where(condition, x1, x2)

    @staticmethod
    def from_onnx(attributes=None):
        return Where()
