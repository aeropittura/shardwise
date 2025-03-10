#!/usr/bin/env python3

# pyre-unsafe


import unittest

import shardwise
import shardwise.nn.tensorboard as tensorboard
from tests.support.multiprocess import MultiProcessTestCase


class TestTensorboard(MultiProcessTestCase):
    """This class tests the shardwise.nn.tensorboard package."""

    # pyre-fixme[14]: `setUp` overrides method defined in `MultiProcessTestCase`
    #  inconsistently.
    def setUp(self) -> None:
        super().setUp()
        if self.rank >= 0:
            shardwise.init()

    def test_tensorboard(self) -> None:
        # create small shardwise model:
        model = shardwise.nn.Graph("input", "output")
        model.add_module("intermediate1", shardwise.nn.ReLU(), ["input"])
        model.add_module("intermediate2", shardwise.nn.Constant(1), [])
        model.add_module("output", shardwise.nn.Add(), ["intermediate1", "intermediate2"])

        # create tensorboard graph:
        tensorboard.graph(model)
        self.assertTrue(True, "creation of tensorboard graph failed")


if __name__ == "__main__":
    unittest.main()
