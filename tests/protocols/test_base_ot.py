#!/usr/bin/env python3


import unittest

import shardwise.protocols.ot.base_ot as baseOT
from tests.support.multiprocess import MultiProcessTestCase


class TestObliviousTransfer(MultiProcessTestCase):
    def test_BaseOT(self) -> None:
        ot = baseOT.BaseOT((self.rank + 1) % self.world_size)
        if self.rank == 0:
            # play the role of sender first
            msg0s = ["123", "abc"]
            msg1s = ["def", "123"]
            ot.send(msg0s, msg1s)

            # play the role of receiver later  with choice bit [0, 1]
            choices = [1, 0]
            msgcs = ot.receive(choices)
            self.assertEqual(msgcs, ["123", "123"])
        else:
            # play the role of receiver first with choice bit [1, 0]
            choices = [0, 1]
            msgcs = ot.receive(choices)

            # play the role of sender later
            msg0s = ["xyz", "123"]
            msg1s = ["123", "uvw"]
            ot.send(msg0s, msg1s)
            self.assertEqual(msgcs, ["123", "123"])


if __name__ == "__main__":
    unittest.main()
