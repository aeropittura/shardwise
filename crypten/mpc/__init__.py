#!/usr/bin/env python3


from crypten.config import cfg
from crypten.mpc import primitives, provider  # noqa: F401

from .context import run_multiprocess
from .mpc import MPCTensor
from .ptype import ptype


__all__ = [
    "MPCTensor",
    "primitives",
    "provider",
    "ptype",
    "run_multiprocess",
]

# the different private type attributes of an mpc encrypted tensor
arithmetic = ptype.arithmetic
binary = ptype.binary

# Set provider
__SUPPORTED_PROVIDERS = {
    "TFP": provider.TrustedFirstParty(),
    "TTP": provider.TrustedThirdParty(),
    "HE": provider.HomomorphicProvider(),
}


def get_default_provider():
    return __SUPPORTED_PROVIDERS[cfg.mpc.provider]


def ttp_required():
    return cfg.mpc.provider == "TTP"
