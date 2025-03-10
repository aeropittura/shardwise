#!/usr/bin/env python3


from shardwise.comm.base import Communicator
from shardwise.comm.distributed import DistributedCommunicator
from shardwise.comm.in_process import InProcessCommunicator


__use_threads = False


def get():
    cls = InProcessCommunicator if __use_threads else DistributedCommunicator
    if not cls.is_initialized():
        raise RuntimeError("Shardwise not initialized. Please call shardwise.init() first.")

    return cls.get()


def _init(use_threads, rank=0, world_size=1, init_ttp=False):
    global __tls, __use_threads
    __use_threads = use_threads
    cls = InProcessCommunicator if __use_threads else DistributedCommunicator

    if cls.is_initialized():
        return

    cls.initialize(rank, world_size, init_ttp=init_ttp)


def uninit():
    global __use_threads
    cls = InProcessCommunicator if __use_threads else DistributedCommunicator
    cls.shutdown()
    __use_threads = False


def is_initialized():
    cls = InProcessCommunicator if __use_threads else DistributedCommunicator
    return cls.is_initialized()


# expose classes and functions in package:
__all__ = ["Communicator", "DistributedCommunicator", "get", "uninit", "is_initialized"]
