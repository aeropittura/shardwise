# Shardwise

Shardwise is a research toolkit for privacy-preserving machine learning. It uses
secure multiparty computation to perform tensor operations and model evaluation
on data shared across participating processes, without requiring those processes
to exchange their raw inputs.

The project combines encrypted tensor operations, automatic differentiation, and
neural network components with examples for training and inference.

## Capabilities

- Arithmetic and binary secret-sharing primitives.
- Tensor operations and automatic differentiation for encrypted computation.
- Neural network layers, optimizers, and model conversion utilities.
- Multiprocess communication and distributed execution helpers.
- CPU and GPU computation paths.
- Benchmarks, notebook tutorials, and runnable machine learning examples.

## Getting started

Shardwise needs Python 3.9 or later on Linux or macOS.

```bash
git clone https://github.com/aeropittura/shardwise.git
cd shardwise
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras: `.[examples]`, `.[benchmarks]`, `.[aws]`, `.[docs]` and `.[dev]`.

## A first encrypted computation

```python
import shardwise
import torch

shardwise.init()

x = shardwise.tensor(torch.tensor([1.0, 2.0, 3.0]))  # secret-shared ShardTensor
y = (x * 2 + 1).get_plain_text()  # tensor([3., 5., 7.])
```

To run code as several parties on one machine, decorate a function with
`shardwise.sharing.run_multiprocess(world_size=2)`.

## Run an example

```bash
pip install -e ".[examples]"
python examples/mpc_linear_svm/launcher.py --help
```

The [examples](examples) directory includes encrypted linear classification,
image classification, contextual bandits, and pretrained-model inference.

## Project layout

```
src/shardwise/
  core/        ShardTensor base class, autograd tensor, gradient functions, fixed-point encoding
  sharing/     SecureTensor (the default secret-shared tensor), ptype, multiprocess launcher
  protocols/   arithmetic and binary secret sharing, Beaver triples, circuits, conversions, OT
  providers/   sources of correlated randomness (trusted first/third party, homomorphic)
  ops/         functions shared by all tensor types (approximations, comparisons, pooling, ...)
  comm/        communicators (torch.distributed and in-process threads)
  nn/          layers/, losses, ONNX import (onnx_io), privacy-preserving training
  optim/       optimizers
  settings/    configuration object and default.yaml
  gpu/  utils/  debug/  zoo/
tests/         mirrors src/shardwise/
examples/  benchmarks/  notebooks/  docs/  scripts/
```

## Development

```bash
pip install -e ".[dev]"
pre-commit install
ruff check . && ruff format --check .
python -m unittest discover -s tests -t .
```

Most tests start several processes to simulate the parties, so the full suite
takes a while. Run a single module while iterating, e.g.
`python -m unittest tests.protocols.test_arithmetic`. GPU tests are skipped when
CUDA is unavailable. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Shardwise is released under the [MIT License](LICENSE).

## Research scope

Shardwise is intended for experimentation and research, rather than deployment
as a production security boundary. Privacy depends on the selected protocol,
its trust assumptions, and how inputs and outputs are handled. Explicit output
reconstruction reveals the reconstructed values to the receiving participants.
