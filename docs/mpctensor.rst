SecureTensor
=========

An ``SecureTensor`` is a ``ShardTensor`` encrypted using the secure MPC protocol.
In order to support the mathematical operations required by the ``SecureTensor``,
Shardwise implements two kinds of secret-sharing protocols defined by ``ptype``:

* ``shardwise.sharing.arithmetic`` for arithmetic secret-sharing
* ``shardwise.sharing.binary`` for binary secret-sharing

Arithmetic secret sharing forms the basis for most of the mathematical
operations implemented by ``SecureTensor``.  Similarly, binary
secret-sharing allows for the evaluation of logical expressions.

We can use the ``ptype`` attribute to create a ``ShardTensor`` with the appropriate
secret-sharing protocol. For example:

.. code-block:: python

  # arithmetic secret-shared tensors
  x_enc = shardwise.tensor([1.0, 2.0, 3.0], ptype=shardwise.sharing.arithmetic)
  print("x_enc internal type:", x_enc.ptype)

  # binary secret-shared tensors
  y_enc = shardwise.tensor([1, 2, 1], ptype=shardwise.sharing.binary)
  print("y_enc internal type:", y_enc.ptype)


We also provide helpers to execute secure multi-party computations
in separate processes (see :ref:`communicator`).

For technical details see `Damgard et al. 2012`_ and `Beaver 1991`_ outlining the Beaver protocol
used in our implementation.

.. _Damgard et al. 2012: https://eprint.iacr.org/2011/535.pdf
.. _Beaver 1991: https://link.springer.com/chapter/10.1007/3-540-46766-1_34

For examples illustrating arithmetic and binary secret-sharing in
Shardwise, the ``ptype`` attribute, and the execution of secure
multi-party computations, please see `Tutorial 2
<https://github.com/facebookresearch/Shardwise/blob/master/tutorials/Tutorial_2_Inside_ShardTensors.ipynb>`_.

Tensor Operations
-----------------

.. automodule:: shardwise.sharing.tensor
    :members:

.. _communicator:

Communicator
------------

To execute multi-party computations locally, we provide
a ``@mpc.run_multiprocess`` function decorator,
which we developed to execute Shardwise code from a single script.
Shardwise follows the standard MPI programming model: it runs a separate
process for each party, but each process runs an identical (complete) program.
Each process has a ``rank`` variable to identify itself.

For example, two-party arithmetic secret-sharing:

.. code-block:: python

  import shardwise
  import shardwise.comm as comm

  @mpc.run_multiprocess(world_size=2)
  def examine_arithmetic_shares():
      x_enc = shardwise.tensor([1, 2, 3], ptype=shardwise.sharing.arithmetic)

      rank = comm.get().get_rank()
      print(f"Rank {rank}:\n {x_enc}")

  x = examine_arithmetic_shares()

.. autofunction:: shardwise.sharing.launch.run_multiprocess
.. autofunction:: shardwise.comm.Communicator.get_world_size
.. autofunction:: shardwise.comm.Communicator.get_rank
