"""Fully connected and matrix-multiplication layers."""

import torch

from shardwise.nn.layers.base import Module


class Gemm(Module):
    """
    Module that performs a general matrix multiplication.

    Unlike the `Linear` module, this module is stateless.
    """

    def __init__(
        self, alpha: float = 1.0, beta: float = 1.0, trans_a: bool = False, trans_b: bool = False
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.trans_a = trans_a
        self.trans_b = trans_b

    def forward(self, x):
        a, b, c = tuple(x)
        if self.trans_a:
            a = a.t()
        if self.trans_b:
            b = b.t()
        output = a.matmul(b).mul(self.alpha)
        output = output.add(c.mul(self.beta))
        return output

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        assert "alpha" in attributes, "attribute alpha missing"
        assert "beta" in attributes, "attribute beta missing"
        return Gemm(
            alpha=attributes["alpha"],
            beta=attributes["beta"],
            trans_a=attributes.get("transA", False),
            trans_b=attributes.get("transB", False),
        )


class Linear(Module):
    """
    Module that performs linear transformation.
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    """  # noqa: W605

    def __init__(self, in_features, out_features, bias: bool = True) -> None:
        super().__init__()

        # initialize model parameters:
        pytorch_module = torch.nn.Linear(in_features, out_features, bias=bias)
        self.register_parameter("weight", pytorch_module.weight)
        if bias:
            self.register_parameter("bias", pytorch_module.bias)

    def forward(self, x):
        output = x.matmul(self.weight.t())
        if hasattr(self, "bias"):
            output = output.add(self.bias)
        return output


class MatMul(Module):
    """
    Matrix product of two tensors.
    The behavior depends on the dimensionality of the tensors as followsf
        - If both tensors are 1-dimensional, the dot product (scalar) is returned.
        - If both arguments are 2-dimensional, the matrix-matrix product is returned.
        - If the first argument is 1-dimensional and the second argument is
        2-dimensional, a 1 is prepended to its dimension for the purpose of the
        matrix multiply. After the matrix multiply, the prepended dimension is removed.
        - If the first argument is 2-dimensional and the second argument is
        1-dimensional, the matrix-vector product is returned.
        - If both arguments are at least 1-dimensional and at least one argument is
        N-dimensional (where N > 2), then a batched matrix multiply is returned.
        If the first argument is 1-dimensional, a 1 is prepended to its dimension
        for the purpose of the batched matrix multiply and removed after. If the
        second argument is 1-dimensional, a 1 is appended to its dimension for the
        purpose of the batched matrix multiple and removed after.
        The non-matrix (i.e. batch) dimensions are broadcasted (and thus
        must be broadcastable).  For example, if :attr:`input` is a
        :math:`(j \times 1 \times n \times m)` tensor and :attr:`other` is
        a :math:`(k \times m \times p)` tensor, :attr:`out` will be an
        :math:`(j \times k \times n \times p)` tensor.

    Arguments:
        Option 1: [input1, input2]
            input1: first input matrix to be multiplied
            input2: second input matrix to be multiplied.
        Option 2: input1
            input1: first input matrix to be multiplied, if module
            is already initialized with the second (i.e. multiplier) matrix.
    """

    def __init__(self, weight=None) -> None:
        super().__init__()
        if weight is not None:
            self.register_parameter("weight", weight)

    def forward(self, x):
        if hasattr(self, "weight"):
            output = x.matmul(self.weight)
        else:
            assert isinstance(x, (list, tuple)), "input must be list or tuple"
            assert len(x) == 2, "input must contain two tensors"
            output = x[0].matmul(x[1])
        return output

    @staticmethod
    def from_onnx(attributes=None):
        return MatMul()
