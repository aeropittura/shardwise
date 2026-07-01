# Changelog

## Unreleased

This release reorganizes the code without changing behavior:

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
