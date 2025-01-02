Shardwise Documentation
=======================
Shardwise is a Privacy Preserving Machine Learning framework written using `PyTorch
<http://pytorch.org/>`_ that allows researchers and developers to
train models using encrypted data. Shardwise currently supports
`Secure multi-party computation
<https://en.wikipedia.org/wiki/Secure_multi-party_computation>`_ as its encryption
mechanism.

.. toctree::
    :hidden:

    migration
    aws

Installation on Linux and Mac
=============================

Use Python 3.9 or later in a virtual environment:

.. code-block:: bash

    git clone https://github.com/aeropittura/shardwise.git
    cd shardwise
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -c constraints.txt -e ".[dev]"
    python -m unittest discover -s tests -t .

We do not support Windows yet.
For contributing to the latest development version, please see Contributing_.

.. _Contributing: https://github.com/aeropittura/shardwise/blob/main/CONTRIBUTING.md

Examples
========

To run the examples in the ``examples`` directory, you additionally need to do
the following

.. code-block:: bash

    pip install -c constraints.txt -e ".[examples]"

We have the following examples, covering a range of models

- The linear SVM example, ``secure_linear_svm``, generates random data and trains a
  SVM classifier on encrypted data.
- The LeNet example, ``secure_cifar``, trains an adaptation of LeNet on CIFAR in
  cleartext and encrypts the model and data for inference
- The TFE benchmark example, ``tfe_benchmarks``, trains three different network
  architectures on MNIST in cleartext, and encrypts the trained model and data
  for inference
- The bandits example, ``bandits``, trains a contextual bandits model on
  encrypted data (MNIST)
- The imagenet example, ``secure_imagenet``, does inference on pretrained model from
  ``torchvision``

For examples that train in the cleartext, we also provide pre-trained models in
cleartext in ``model`` subdirectory of each example.

You can check all example specific command line options by doing the following;
shown here for ``tfe_benchmarks``

.. code-block:: bash

    $ python3 -m examples.tfe_benchmarks.launcher --help

Some MPC specific options are

- ``--world_size`` Number of peers in MPC
- ``--multiprocess`` Run in multiprocess mode on one machine, where each peer is a
  separate process

Examples on AWS
---------------

Shardwise also provides a script ``aws_launcher`` to launch examples with
encrypted data on multiple AWS instances. See :doc:`aws`.

.. toctree::
    :maxdepth: 3
    :caption: Package Reference

    shardwise.ShardTensor <shard_tensor>
    shardwise.sharing.SecureTensor <secure_tensor>
    shardwise.nn <nn>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
