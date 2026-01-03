"""Layers that reshape, slice, gather, concatenate or pad tensors."""

import torch

import shardwise
from shardwise.nn.layers.base import Module


class Transpose(Module):
    """
    Module that transposes the input tensor similar to
    `numpy.transpose`. For example, when perm=(1, 0, 2), given an input
    tensor of shape (1, 2, 3), the output shape will be (2, 1, 3). Note
    that the signature of this module matches the ONNX specification
    and differs from `torch.transpose`

    Args:
        `perm`: list of ints
    """

    def __init__(self, perm) -> None:
        super().__init__()
        self.perm = perm

    def forward(self, input):
        # New Linear jit tracer causes Transpose module to have a weight
        if hasattr(self, "weight"):
            input = self.weight

        assert input.dim() == len(self.perm)
        return input.permute(self.perm)

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        # TODO: ONNX specification says the permutation should be
        # reversed if not provided in the attributes.  Because we
        # don't have the input size here, we need figure out a
        # different way of supporting this, if we want to do that.
        return Transpose(attributes["perm"])


class Squeeze(Module):
    r"""
    Returns a tensor with all the dimensions of :attr:`input` of size `1` removed.

    For example, if `input` is of shape:
    :math:`(A \times 1 \times B \times C \times 1 \times D)` then the `out` tensor
    will be of shape: :math:`(A \times B \times C \times D)`.

    When :attr:`dimension` is given, a squeeze operation is done only in the given
    dimension. If `input` is of shape: :math:`(A \times 1 \times B)`,
    ``squeeze(input, 0)`` leaves the tensor unchanged, but ``squeeze(input, 1)``
    will squeeze the tensor to the shape :math:`(A \times B)`.

    .. note:: The returned tensor shares the storage with the input tensor,
            so changing the contents of one will change the contents of the other.

    Args:
        dimension (int, optional): if given, the input will be squeezed only in
            this dimension
    """

    def __init__(self, dimension) -> None:
        super().__init__()
        if isinstance(dimension, (list, tuple)):
            assert len(dimension) == 1, "can only squeeze one dimension at a time"
            dimension = dimension[0]
        self.dimension = dimension

    def forward(self, input):
        return input.squeeze(self.dimension)

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        dimension = attributes["axes"]
        assert len(dimension) == 1, "can only squeeze one dimension at a time"
        return Squeeze(dimension[0])


class Unsqueeze(Module):
    """
    Module that unsqueezes a tensor.
    Returns a new tensor with a dimension of size one inserted at the
    specified position.

    The returned tensor shares the same underlying data with this tensor.
    A :attr:`dimension` value within the range ``[-input.dim() - 1, input.dim() + 1)``
    can be used. Negative :attr:`dimension` will correspond to :meth:`unsqueeze`
    applied at :attr:`dimension` = ``dim + input.dim() + 1``.

    Args:
        dimension (int): the index at which to insert the singleton dimension
    """

    SUPPORTS_PLAINTEXT_INPUTS = True

    def __init__(self, dimension) -> None:
        super().__init__()
        if isinstance(dimension, (list, tuple)):
            assert len(dimension) == 1, "can only squeeze one dimension at a time"
            dimension = dimension[0]
        self.dimension = dimension

    def forward(self, input):
        if isinstance(input, list):
            assert len(input) == 2, "list input must be [x, dimension]"
            input, dimension = input
            assert len(dimension) == 1, "can only unsqueeze one dimension at a time"
            dimension = int(dimension.item())
        else:
            dimension = self.dimension
        return input.unsqueeze(dimension)

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        dimension = attributes.get("axes", [None])
        assert len(dimension) == 1, "can only unsqueeze one dimension at a time"
        return Unsqueeze(dimension[0])


class Slice(Module):
    """
    Module that slices the input along the specified `axes` (list of `int`s) from
    the indices in `start`s to the indices in `end`s.

    This module definition matches ONNX opset version 11.
    """

    def __init__(self, starts, ends, axes=None) -> None:
        super().__init__()
        self.starts = starts
        self.ends = ends
        self.axes = axes

    def forward(self, x):
        # Process inputs:
        axes = None
        if isinstance(x, list):
            if len(x) == 3:
                x, starts, ends = x
                axes, steps = self.axes, 1
            elif len(x) == 4:
                x, starts, ends, axes = x
                steps = 1
            elif len(x) == 5:
                x, starts, ends, axes, steps = x
                if not torch.eq(steps.int(), 1).all():
                    raise ValueError("Only steps value of 1 currently supported.")
            else:
                raise ValueError("list input x must have 3, 4, or 5, values")
            starts, ends = starts.int().tolist(), ends.int().tolist()
        else:
            starts, ends, axes = self.starts, self.ends, self.axes
            steps = 1
        if axes is None:
            axes = list(range(len(starts)))

        # Perform slicing:
        output = x
        for idx, axis in enumerate(axes):
            start, end = int(starts[idx]), int(ends[idx])
            length = min(end, output.size(int(axis))) - start
            output = output.narrow(int(axis), start, length)
        return output

    @staticmethod
    def from_onnx(attributes=None):
        return Slice(
            attributes.get("starts", None),
            attributes.get("ends", None),
            axes=attributes.get("axes", None),
        )


class Expand(Module):
    """
    Module that expands a tensor to the specified size.
    """

    def forward(self, x):
        # unpack inputs:
        input, shape = tuple(x)
        if torch.is_tensor(shape):
            shape = shape.long().tolist()

        # broadcasting in ONNX is different from PyTorch when shape is 1:
        for idx in range(len(shape)):
            if shape[idx] == 1 and input.size(idx) > 1:
                shape[idx] = input.size(idx)

        # perform the expansion:
        return input.expand(shape)

    @staticmethod
    def from_onnx(attributes=None):
        return Expand()


class Flatten(Module):
    """
    Module that flattens the input tensor into a 2D matrix.

    Args:
        axis (int, optional): must not be larger than dimension
    """

    def __init__(self, axis: int = 1) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x):
        if self.axis == 0:
            return x.view(1, -1)
        else:
            assert self.axis <= x.dim(), "axis must not be larger than dimension"
            prod = 1
            for i in range(self.axis):
                prod *= x.size(i)
            return x.view(prod, -1)

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        # axis : int (default is 1)
        axis = 1
        if "axis" in attributes:
            axis = int(attributes["axis"])
            assert axis >= 0, "axis must not be negative"
        return Flatten(axis)


class Shape(Module):
    """
    Module that returns the shape of a tensor. If the input tensor is encrypted,
    the output size vector will be encrypted, too.
    """

    SUPPORTS_PLAINTEXT_INPUTS = True

    def __init__(self, dim=None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x, dim=None):
        dim = dim if dim is not None else self.dim
        if dim is None:
            size = torch.tensor(x.size())
        else:
            size = torch.tensor(x.size(dim))
        return size

    @staticmethod
    def from_onnx(attributes=None):
        return Shape()


class Concat(Module):
    """
    Module that concatenates tensors along a dimension.

    Args:
        dim (int, optional): the dimension over which to concatenate
    """

    def __init__(self, dimension) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, input):
        assert isinstance(input, (list, tuple)), "input needs to be a list or tuple"
        assert len(input) >= 1, "need at least one tensor to concatenate"
        return shardwise.cat(input, self.dimension)

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        dimension = attributes["axis"]
        return Concat(dimension)


class Reshape(Module):
    """
    Module that reshapes tensors to new dimensions.
    Returns a tensor with same data and number of elements as :attr:`self`,
    but with the specified shape.

    When possible, the returned tensor will be a view
    of :attr:`self`. Otherwise, it will be a copy. Contiguous inputs and inputs
    with compatible strides can be reshaped without copying, but you should not
    depend on the copying vs. viewing behavior.

    See :meth:`torch.Tensor.view` on when it is possible to return a view.
    A single dimension may be -1, in which case it's inferred from the remaining
    dimensions and the number of elements in :attr:`self`.

    Args:
        input (tuple): contains input tensor and shape (torch.Size)
    """

    def __init__(self, shape=None) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, tensor, shape=None):
        if isinstance(tensor, list) and len(tensor) == 2:
            tensor, shape = tensor
        shape = shape if shape is not None else self.shape
        assert shape is not None, (
            "Reshape requires a shape in forward if not supplied in initialization"
        )
        if torch.is_tensor(shape):
            shape = torch.Size(shape.long())
        return tensor.reshape(shape)

    @staticmethod
    def from_onnx(attributes=None):
        if "shape" in attributes:
            return Reshape(shape=attributes["shape"])
        return Reshape()


class Gather(Module):
    r"""
    Module that gathers elements from tensor according to indices. Given data tensor
    of rank :math:`r >= 1`, and indices tensor of rank :math:`q`, gather entries of
    the axis dimension of data (by default outer-most one as `axis = 0`)
    indexed by indices, and concatenates them in an output tensor of rank
    :math:`q + (r - 1)`. For example, for `axis = 0`: Let :math:`k =
    indices[i_{0}, ..., i_{q-1}]`. Then :math:`output[i_{0}, ..., i_{q-1}, j_{0},
    ..., j_{r-2}] = input[k, j_{0}, ..., j_{r-2}]`. This is an operation from the
    ONNX specification.

    Args:
        dimension (int): the axis along which to index
        index(tensor): the indices to select along the `dimension`
    """

    SUPPORTS_PLAINTEXT_INPUTS = True

    def __init__(self, dimension, indices=None) -> None:
        super().__init__()
        self.dimension = dimension
        self.indices = indices

    def forward(self, input):
        if not isinstance(input, (list, tuple)):
            tensor = input
            indices = self.indices
        elif len(input) == 1:
            tensor = input[0]
            indices = self.indices
        else:
            tensor, indices = input

        # indices need to be a PyTorch tensor:
        if shardwise.is_encrypted_tensor(indices):
            raise ValueError("Cannot perform Gather operation using encrypted indices.")
        elif isinstance(indices, (int, list, tuple)):
            indices = torch.tensor(indices)
        indices = indices.long()

        # ShardTensor input
        if shardwise.is_encrypted_tensor(tensor):
            result = tensor.take(indices, self.dimension)

        # Torch tensor input
        elif self.dimension is None or tensor.dim() == 0:
            result = torch.take(tensor, indices)
        else:
            all_indices = [slice(0, x) for x in tensor.size()]
            all_indices[self.dimension] = indices
            result = tensor[all_indices]
        return result

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        if "axis" not in attributes:
            attributes["axis"] = None
        if "shape" not in attributes:
            attributes["shape"] = None
        return Gather(attributes["axis"], indices=attributes["shape"])


class _ConstantPad(Module):
    """
    Module that pads a tensor.
    """

    def __init__(self, padding, value, ndims, mode: str = "constant") -> None:
        super().__init__()
        if isinstance(padding, (int)):
            padding = [padding, padding] * ndims
        self.padding = padding
        self.value = value
        self.mode = mode

    def forward(self, input):
        if isinstance(input, list):
            assert len(input) == 2, "input should be [tensor, pads] list"
            padding = tuple(input[1].int().tolist())
            input = input[0]
        else:
            padding = self.padding
        return input.pad(padding, value=self.value, mode=self.mode)

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        assert attributes["mode"] == b"constant", "only constant padding supported"
        return _ConstantPad(None, 0, 0, mode="constant")


class ConstantPad1d(_ConstantPad):
    """
    Module that pads a 1D tensor.
    """

    def __init__(self, padding, value, mode: str = "constant") -> None:
        super().__init__(padding, value, 1, mode=mode)


class ConstantPad2d(_ConstantPad):
    """
    Module that pads a 2D tensor.
    """

    def __init__(self, padding, value, mode: str = "constant") -> None:
        super().__init__(padding, value, 2, mode=mode)


class ConstantPad3d(_ConstantPad):
    """
    Module that pads a 3D tensor.
    """

    def __init__(self, padding, value, mode: str = "constant") -> None:
        super().__init__(padding, value, 3, mode=mode)
