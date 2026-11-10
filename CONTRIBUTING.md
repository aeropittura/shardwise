# Contributing

Thanks for helping improve Shardwise.

## Set up a development environment

Shardwise needs Python 3.9 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Before opening a pull request

1. Branch from `main`.
2. Add or update tests for any behavior you change; tests live in `tests/`,
   mirroring the layout of `src/shardwise/`.
3. Run the checks:

   ```bash
   ruff check .
   ruff format --check .
   python -m unittest discover -s tests -t .
   ```

4. Update the documentation in `docs/` if you change a public API.

Most tests start several processes to simulate the parties of a computation, so
the full suite takes a while. While iterating, run a single module, e.g.
`python -m unittest tests.protocols.test_arithmetic`.

## Reporting issues

Use GitHub issues. Include the Shardwise, PyTorch and Python versions, your
operating system, and the smallest snippet that reproduces the problem.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).
