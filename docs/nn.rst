nn
==

``shardwise.nn`` provides modules for defining and training neural
networks similar to ``torch.nn``.

From PyTorch to Shardwise
-----------------------

The simplest way to create a Shardwise network is to start with a
PyTorch network, and use the ``from_pytorch`` function to convert it
to a Shardwise network. This is particularly useful for pre-trained
PyTorch networks that need to be encrypted before use. 

.. automodule:: shardwise.nn
    :members: from_pytorch


.. note::

    In addition to the PyTorch network, the `from_pytorch` function
    also requires a dummy input of the shape of the model's input.
    The dummy input simply needs to be a `torch` tensor of the same
    shape; the values inside the tensor do not matter. For a complete
    example of how to use ``from_pytorch`` function, please see
    `Tutorial 4 <https://github.com/aeropittura/shardwise/blob/main/notebooks/tutorials/04_encrypted_classification.ipynb>`_.

Custom Shardwise Modules
----------------------

``shardwise.nn`` also provides several modules and containers
for directly building neural networks. Here is an example of how
to use these objects to build a Shardwise network: 

.. code-block:: python
       
       model = shardwise.nn.Sequential(
              shardwise.nn.Linear(num_inputs, num_intermediate),
              shardwise.nn.ReLU(),
              shardwise.nn.Linear(num_intermediate, num_outputs),
        )


Alternately, you can create a custom Shardwise network in much
the same way as you create a custom PyTorch network, i.e., you
can subclass ``shardwise.nn.Module`` and allow it contain other
``shardwise.nn.Module``, nesting them in a tree structure. You can assign
the submodules as regular attribute modules within them. for example:


.. code-block:: python

        class ShardwiseModel(shardwise.nn.Module):
            def __init__(self):
                super(ShardwiseModel, self).__init__()
                self.fc1 = shardwise.nn.Linear(20, 5) 
                self.fc2 = shardwise.nn.Linear(5, 2)

            def forward(self, x):
                x = self.fc1(x)
		x = self.fc2(x)
                return x


Generic ``Modules``
+++++++++++++++++++++++++++

.. autoclass:: shardwise.nn.layers.Module
    :members: train, eval, encrypt, decrypt, forward, zero_grad, modules, named_modules, named_parameters, update_parameters

.. autoclass:: shardwise.nn.layers.Container
    :members:

.. autoclass:: shardwise.nn.layers.Graph
    :members:

.. autoclass:: shardwise.nn.layers.Sequential
    :members:


``Modules`` for Encrypted Layers
++++++++++++++++++++++++++++++++++++++++

.. autoclass:: shardwise.nn.layers.Constant
   :members: forward

.. autoclass:: shardwise.nn.layers.Add
   :members: forward	

.. autoclass:: shardwise.nn.layers.Sub
   :members: forward	

.. autoclass:: shardwise.nn.layers.Squeeze
   :members: forward

.. autoclass:: shardwise.nn.layers.Unsqueeze
   :members: forward   

.. autoclass:: shardwise.nn.layers.Flatten
   :members: forward   

.. autoclass:: shardwise.nn.layers.Shape
   :members: forward

.. autoclass:: shardwise.nn.layers.Concat
   :members: forward

.. autoclass:: shardwise.nn.layers.Gather
   :members: forward   

.. autoclass:: shardwise.nn.layers.Reshape
   :members: forward

.. autoclass:: shardwise.nn.layers.ConstantPad1d
   :members: forward

.. autoclass:: shardwise.nn.layers.ConstantPad2d
   :members: forward	   

.. autoclass:: shardwise.nn.layers.ConstantPad3d
   :members: forward

.. autoclass:: shardwise.nn.layers.Linear
   :members: forward

.. autoclass:: shardwise.nn.layers.Conv2d
   :members: forward	

.. autoclass:: shardwise.nn.layers.ConstantPad1d
   :members: forward

.. autoclass:: shardwise.nn.layers.AvgPool2d
   :members: forward

.. autoclass:: shardwise.nn.layers.MaxPool2d
   :members: forward

.. autoclass:: shardwise.nn.layers.GlobalAveragePool
   :members: forward   

.. autoclass:: shardwise.nn.layers.BatchNorm1d
   :members: forward

.. autoclass:: shardwise.nn.layers.BatchNorm2d
   :members: forward

.. autoclass:: shardwise.nn.layers.BatchNorm3d
   :members: forward

.. autoclass:: shardwise.nn.layers.Dropout
   :members: forward

.. autoclass:: shardwise.nn.layers.Dropout2d
   :members: forward

.. autoclass:: shardwise.nn.layers.Dropout3d
   :members: forward

.. autoclass:: shardwise.nn.layers.DropoutNd
   :members: forward

.. autoclass:: shardwise.nn.layers.Softmax
   :members: forward

.. autoclass:: shardwise.nn.layers.LogSoftmax
   :members: forward

Loss Functions
---------------
Shardwise also provides a number of encrypted loss functions similar
to `torch.nn`. 

.. automodule:: shardwise.nn.loss
    :members:
