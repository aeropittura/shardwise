#!/usr/bin/env python3
"""Benchmark committed Shardwise revisions in isolated temporary environments.

Example (from the repository root):
    python benchmarks/run_historical_benchmarks.py --ref HEAD --ref v0.4.0

Each revision must contain src/shardwise and benchmarks/benchmark.py. Dependencies
are installed using the selected Python interpreter; CPU is the default device.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ref", action="append", help="Git revision; repeat to compare (default: HEAD)"
    )
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("benchmarks/dash_app/data/history"))
    parser.add_argument(
        "--constraints", type=Path, default=Path(__file__).resolve().parents[1] / "constraints.txt"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--only-functions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.world_size < 1:
        parser.error("--world-size must be positive")
    return args


def run_revision(args, ref):
    repository = args.repository.resolve()
    sha = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"],
        cwd=repository,
        text=True,
    ).strip()
    output = args.output.resolve() / sha / args.device / f"{args.world_size}pc"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Results already exist: {output}; use --overwrite or a new --output")
    with tempfile.TemporaryDirectory(prefix="shardwise-benchmark-") as temporary:
        workspace = Path(temporary)
        checkout = workspace / "source"
        subprocess.run(
            ["git", "clone", "--quiet", "--no-hardlinks", str(repository), str(checkout)],
            check=True,
        )
        subprocess.run(["git", "checkout", "--quiet", "--detach", sha], cwd=checkout, check=True)
        if not (checkout / "src/shardwise").is_dir():
            raise ValueError(f"{ref} ({sha}) predates the Shardwise source layout")
        if not (checkout / "benchmarks/benchmark.py").is_file():
            raise ValueError(f"{ref} ({sha}) has no benchmark entry point")
        environment = workspace / "venv"
        venv.create(environment, with_pip=True)
        python = environment / "bin/python"
        install = [str(python), "-m", "pip", "install"]
        if args.constraints is not None:
            install.extend(["--constraint", str(args.constraints.resolve())])
        subprocess.run([*install, f"{checkout}[benchmarks]"], check=True)
        results = workspace / "results"
        results.mkdir()
        command = [
            str(python),
            "-m",
            "benchmarks.benchmark",
            "--path",
            str(results),
            "--device",
            args.device,
            "--world-size",
            str(args.world_size),
        ]
        if args.only_functions:
            command.append("--only-functions")
        subprocess.run(command, cwd=checkout, check=True)
        # Publish only successful runs. Failed installs/runs leave no partial results.
        output.mkdir(parents=True, exist_ok=True)
        for result in results.iterdir():
            (output / result.name).write_bytes(result.read_bytes())
        (output / "revision.json").write_text(
            json.dumps(
                {
                    "ref": ref,
                    "commit": sha,
                    "device": args.device,
                    "world_size": args.world_size,
                    "only_functions": args.only_functions,
                    "python": sys.version,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Saved {sha} benchmarks to {output}")


def main(argv=None):
    args = parse_args(argv)
    for ref in args.ref or ["HEAD"]:
        run_revision(args, ref)


if __name__ == "__main__":
    main()
