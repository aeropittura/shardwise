#!/usr/bin/env python3


from shardwise.settings import cfg
from shardwise import protocols, providers  # noqa: F401

from shardwise.sharing.launch import run_multiprocess
from shardwise.sharing.tensor import SecureTensor
from shardwise.sharing.ptype import ptype


__all__ = [
    "SecureTensor",
    "protocols",
    "providers",
    "ptype",
    "run_multiprocess",
]

# the different private type attributes of an mpc encrypted tensor
arithmetic = ptype.arithmetic
binary = ptype.binary

# Set provider
__SUPPORTED_PROVIDERS = {
    "TFP": providers.TrustedFirstParty(),
    "TTP": providers.TrustedThirdParty(),
    "HE": providers.HomomorphicProvider(),
}


def get_default_provider():
    return __SUPPORTED_PROVIDERS[cfg.mpc.provider]


def ttp_required():
    return cfg.mpc.provider == "TTP"
