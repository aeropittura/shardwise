# Changelog

## Unreleased

This release reorganizes the package and includes breaking naming changes:

- The package is now `shardwise`, in a `src/` layout, with modules grouped into
  `core`, `ops`, `protocols`, `providers`, `sharing`, `comm`, `nn`, `optim`,
  `settings`, `gpu`, `utils` and `debug`.
- Renamed public API: `CrypTensor` → `ShardTensor`, `MPCTensor` → `SecureTensor`,
  `cryptensor()` → `shardwise.tensor()`, `register_cryptensor` →
  `register_tensor_type`, `*_cryptensor_type` → `*_tensor_type`.
- `nn.module` is split into the `nn.layers` package; `nn.onnx_converter` is now
  `nn.onnx_io`.
- Packaging moved to `pyproject.toml`; code is formatted and linted with Ruff
  and annotated with type hints.
- Tests moved to `tests/`, mirroring the package layout.

- The default backend is `secure`; configuration uses `sharing.*` and
  `nn.private_training.*`. See [the migration guide](docs/migration.rst).
- Example directories and modules use `secure_` names; notebook source and
  documentation follow the same API. Stale notebook outputs are cleared.
- Historical benchmarks run explicit Shardwise revisions in temporary,
  isolated environments and record the resolved commit.
- Upstream attribution is documented in `NOTICE`; the MIT notices are retained.
- Added a tested dependency constraints file for setup, CI, and documentation.
- The linear SVM example now honors its `--epochs` option in training.
