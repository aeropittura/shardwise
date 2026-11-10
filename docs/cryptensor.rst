ShardTensor
==========

A ``ShardTensor`` is an encrypted ``torch`` tensor for secure computations.

Shardwise currently only supports secure MPC protocols
(though we intend to add support for other advanced encryption protocols).
Using the :doc:`mpctensor` backend, a ``ShardTensor`` acts as ``torch`` tensor
whose values are encrypted using the secure MPC protocol.

To create a shard_tensor,

.. code-block:: python

    # Create torch tensor
    x = torch.tensor([1.0, 2.0, 3.0])

    # Encrypt x
    x_enc = shardwise.tensor(x)

We can decrypt ``x_enc`` by calling ``x_enc.get_plain_text()``.

Tensor Operations
-----------------

``ShardTensor``s provide various operations similar to ``torch`` tensors.

.. automodule:: shardwise.tensor
    :members:

Autograd
--------
``ShardTensor``s support autograd similar to ``torch`` tensors. You can set
the ``requires_grad`` attribute on a ``ShardTensor`` to record forward
computations, and call ``backward()`` on a ``ShardTensor`` to perform
backpropagation.

.. code-block:: python

   # Create an ShardTensor that supports autograd
   x_enc = shardwise.tensor(x_enc, requires_grad=True)
   y_enc = x_enc.mul(2).sum()
   y_enc.backward()

For an example of backpropagation using ``ShardTensor``, please
see `Tutorial 7
<https://github.com/facebookresearch/Shardwise/blob/master/tutorials/Tutorial_7_Training_an_Encrypted_Neural_Network.ipynb>`_.


File I/O Utilities
------------------

Shardwise provides utilities for loading ``ShardTensors`` from files and
saving ``ShardTensors`` to files.

.. automodule:: shardwise
    :members: load, save
