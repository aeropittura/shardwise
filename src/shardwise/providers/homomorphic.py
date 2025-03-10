#!/usr/bin/env python3


from shardwise.providers.base import TupleProvider


class HomomorphicProvider(TupleProvider):
    NAME = "HE"

    def generate_additive_triple(self, size0, size1, op, *args, **kwargs):
        """Generate multiplicative triples of given sizes"""
        raise NotImplementedError("HomomorphicProvider not implemented")

    def square(self, size):
        """Generate square double of given size"""
        raise NotImplementedError("HomomorphicProvider not implemented")

    def generate_xor_triple(self, size0, size1):
        """Generate xor triples of given size"""
        raise NotImplementedError("HomomorphicProvider not implemented")

    def wrap_rng(self, size, num_parties):
        """Generate random shared tensor of given size and sharing of its wraps"""
        raise NotImplementedError("HomomorphicProvider not implemented")

    def B2A_rng(self, size):
        """Generate random bit tensor as arithmetic and binary shared tensors"""
        raise NotImplementedError("HomomorphicProvider not implemented")
