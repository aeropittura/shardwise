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

Clone the repository and create an isolated Python environment:

```bash
git clone https://github.com/aeropittura/shardwise.git
cd shardwise
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install the runtime dependencies and the local package:

```bash
python -m pip install torch torchvision omegaconf onnx pandas pyyaml tensorboard future scipy scikit-learn
python -m pip install --no-deps -e .
```

The dependency list above uses `scikit-learn` in place of the legacy `sklearn`
package entry in `requirements.txt`. The package metadata specifies Python 3.7
or later; compatibility with every recent Python or dependency release is not
established. Consult [requirements.txt](requirements.txt) and [env.yml](env.yml)
when preparing a reproducible environment. Linux and macOS are the documented
platforms; Windows is not supported by the existing setup.

## Run an example

Install the additional example dependencies:

```bash
python -m pip install -r requirements.examples.txt
```

Inspect the options for the encrypted linear classifier example:

```bash
python examples/mpc_linear_svm/launcher.py --help
```

The [examples](examples) directory includes encrypted linear classification,
image classification, contextual bandits, and pretrained-model inference.
Each example has its own launcher and configuration options. Some examples
train in plaintext before evaluating an encrypted model; others demonstrate
training with encrypted data.

## Explore the repository

- [tutorials](tutorials): notebook walkthroughs of encrypted tensors, access
  control, model evaluation, and training.
- [examples](examples): end-to-end model demonstrations and launchers.
- [benchmarks](benchmarks): performance experiments and profiling tools.
- [configs](configs): runtime configuration defaults.
- [scripts](scripts): distributed execution and deployment helpers.
- [test](test): tests for tensors, protocols, communication, and model components.
- [docs](docs): documentation sources.

## Development

Install the test runner and run the tests relevant to a change:

```bash
python -m pip install pytest
python -m pytest test/test_common.py
```

Distributed and GPU tests may require additional processes, hardware, or runtime
configuration. Start with a small local example before running a larger workload.

## Research scope

Shardwise is intended for experimentation and research, rather than deployment
as a production security boundary. Privacy depends on the selected protocol,
its trust assumptions, and how inputs and outputs are handled. Explicit output
reconstruction reveals the reconstructed values to the receiving participants.
