Migrating to Shardwise 0.4
==========================

Shardwise derives from CrypTen. The inherited implementation and attribution are
recorded in the repository's NOTICE and LICENSE files.

Public API
----------

Use ``shardwise.ShardTensor`` and ``shardwise.sharing.SecureTensor`` for encrypted
tensors, ``shardwise.tensor()`` to construct them, and
``ShardTensor.register_tensor_type()`` to register another backend.
The default tensor backend is now ``"secure"``. Update explicit calls to
``set_default_tensor_type()`` and backend arguments accordingly.

Configuration
-------------

Rename the top-level ``mpc`` YAML section to ``sharing`` and the
``nn.dpsmpc`` section to ``nn.private_training``. This is a breaking rename;
legacy keys are not aliases. Update custom YAML files, dotted-key overrides,
and direct attribute access together::

    from shardwise.settings import cfg

    with cfg.temp_override({"sharing.provider": "TFP"}):
        assert cfg.sharing.provider == "TFP"

Examples and benchmarks
-----------------------

Example directories and their Python modules now use ``secure_`` in place of
``mpc_``: ``secure_linear_svm``, ``secure_autograd_cnn``, ``secure_cifar``, and
``secure_imagenet``. Update local commands and AWS auxiliary file paths.

Historical benchmarks now take explicit Shardwise Git revisions rather than
monthly CrypTen snapshots. Run from the repository root::

    python benchmarks/run_historical_benchmarks.py --ref HEAD --device cpu

Only committed revisions with the Shardwise source layout are supported.
Dependencies use the compatibility versions in ``constraints.txt``; pass
``--constraints PATH`` to select another constraints file.
Results include a revision.json file recording the requested ref and resolved SHA.
