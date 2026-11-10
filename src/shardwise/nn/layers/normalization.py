"""Normalization layers."""

import torch

import shardwise
from shardwise.nn.layers.base import Module


class BatchNormalization(Module):
    """
    Module that performs batch normalization following the ONNX specification.

    Unlike the `BatchNorm1d`, `BatchNorm2d`, and `BatchNorm3d` classes, this
    module is stateless. It takes `input`, `weight`, `bias`, `running_mean`, and
    `running_var` tensors as input into `forward()`.
    """

    def __init__(self, eps: float = 1e-05, momentum: float = 0.1) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.inv_var = None
        self._running_var_id = None

    def forward(self, x):
        assert len(x), f"BatchNormalization expects 5 inputs, not {len(x)}"
        input, weight, bias, running_mean, running_var = x

        # in inference mode, we may be able to re-use inverse variance:
        if not self.train:
            if id(running_var) != self._running_var_id:
                self.inv_var = self._compute_inv_var(running_var)
                self._running_var_id = id(running_var)
        else:
            self.inv_var = None
            self._running_var_id = None

        # perform batch normalization:
        output = input.batchnorm(
            weight,
            bias,
            running_mean=running_mean,
            running_var=running_var,
            training=self.training,
            eps=self.eps,
            momentum=self.momentum,
            inv_var=self.inv_var,
        )
        if self.training:  # NOTE: Training graph is different from evaluation graph.
            return output, running_mean, running_var, None, None
        else:
            return output

    def _compute_inv_var(self, running_var):
        """Computes inverse variance."""
        if isinstance(running_var, shardwise.ShardTensor):
            inv_var = running_var.add(self.eps).inv_sqrt()
        else:
            inv_var = running_var.add(self.eps).sqrt().reciprocal()
        return inv_var

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        return BatchNormalization(
            eps=attributes.get("epsilon", 1e-05),
            momentum=1.0 - attributes.get("momentum", 0.9),
        )  # NOTE: Role of momentum is reversed in ONNX specification.


class _BatchNorm(Module):
    """
    Module that performs batch normalization on 1D tensors. It is used as the
    base implementation for the `BatchNorm1d`, `BatchNorm2d`, and `BatchNorm3d`
    classes.

    Unlike the `BatchNormalization` class, this module is stateful.
    """

    def __init__(self, num_features, eps: float = 1e-05, momentum: float = 0.1) -> None:
        super().__init__()

        # initialize model parameters and buffers:
        pytorch_module = torch.nn.BatchNorm1d(num_features)
        for param in ["weight", "bias"]:
            self.register_parameter(param, getattr(pytorch_module, param))
        for buffer in ["running_mean", "running_var"]:
            self.register_buffer(buffer, getattr(pytorch_module, buffer))

        # set model attributes:
        self.eps = eps
        self.momentum = momentum

        # do not precompute inverse variance during training
        self.inv_var = None

    def forward(self, input):
        return input.batchnorm(
            self.weight,
            self.bias,
            running_mean=self.running_mean,
            running_var=self.running_var,
            training=self.training,
            eps=self.eps,
            momentum=self.momentum,
            inv_var=self.inv_var,
        )

    @staticmethod
    def from_onnx(parameters=None, attributes=None):
        # preprocess all attributes:
        if parameters is None:
            parameters = {}
        if attributes is None:
            attributes = {}

        num_features = len(parameters["running_mean"])

        # create module:
        if "momentum" not in attributes:
            attributes["momentum"] = 0.1
        kwargs = {"eps": attributes["epsilon"], "momentum": attributes["momentum"]}
        module = _BatchNorm(num_features, **kwargs)

        # set parameters:
        for key, value in parameters.items():
            if key in ["running_mean", "running_var"]:
                module.set_buffer(key, value)
            else:
                module.set_parameter(key, value)
        return module

    def train(self, mode: bool = True):
        """Freezes the inverse variance during inference to save computation"""
        super().train(mode=mode)
        if self.training:
            self.inv_var = None
        elif isinstance(self.running_var, shardwise.ShardTensor):
            self.inv_var = self.running_var.add(self.eps).inv_sqrt()
        else:
            self.inv_var = self.running_var.add(self.eps).sqrt().reciprocal()
        return self


class BatchNorm1d(_BatchNorm):
    """
    Module that performs batch normalization on 1D tensors.
    """

    pass


class BatchNorm2d(_BatchNorm):
    """
    Module that performs batch normalization on 2D tensors.
    """

    pass


class BatchNorm3d(_BatchNorm):
    """
    Module that performs batch normalization on 3D tensors.
    """

    pass


class GroupNorm(Module):
    """
    Module that performs group normalization on tensors.
    """

    def __init__(self, num_groups, num_channels, eps: float = 1e-5, affine: bool = True) -> None:
        raise NotImplementedError("GroupNorm is not implemented.")
