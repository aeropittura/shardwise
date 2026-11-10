#!/usr/bin/env python3


from enum import Enum

from shardwise.protocols import ArithmeticSharedTensor, BinarySharedTensor


class ptype(Enum):
    """Enumeration defining the private type attributes of encrypted tensors"""

    arithmetic = 0
    binary = 1

    def to_tensor(self):
        if self.value == 0:
            return ArithmeticSharedTensor
        elif self.value == 1:
            return BinarySharedTensor
        else:
            raise ValueError("Cannot convert %s to encrypted tensor" % (self.name))
