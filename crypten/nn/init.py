#!/usr/bin/env python3


import torch


# Makes nn.init functions
def make_crypten_compatible(initialization_function):
    def wrapper_func(tensor, *args, **kwargs):
        if not torch.is_tensor(tensor):
            result = torch.empty(tensor.size())
            result = initialization_function(result, *args, **kwargs)
            tensor.set(result)
            return tensor

        return initialization_function(tensor, *args, **kwargs)

    return wrapper_func


__all__ = [  # noqa: F822
    "constant_",
    "dirac_",
    "kaiming_normal_",
    "kaiming_uniform_",
    "normal_",
    "ones_",
    "orthogonal_",
    "sparse_",
    "trunc_normal_",
    "uniform_",
    "xavier_normal_",
    "xavier_uniform_",
    "zeros_",
]


for func_name in __all__:
    globals()[func_name] = make_crypten_compatible(getattr(torch.nn.init, func_name))
