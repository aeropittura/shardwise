"""Regression checks for revision isolation and failed benchmark publication."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from benchmarks import run_historical_benchmarks as history


class TestHistoricalBenchmarks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repository = self.root / "repository"
        self.repository.mkdir()
        self.git("init", "--quiet")
        (self.repository / "README").write_text("legacy source")
        self.git("add", ".")
        self.git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "Legacy source",
        )
        self.legacy = self.git("rev-parse", "HEAD").strip()
        (self.repository / "src/shardwise").mkdir(parents=True)
        (self.repository / "src/shardwise/__init__.py").write_text("")
        (self.repository / "benchmarks").mkdir()
        (self.repository / "benchmarks/benchmark.py").write_text("")
        self.git("add", ".")
        self.git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "Shardwise layout",
        )
        self.sha = self.git("rev-parse", "HEAD").strip()
        self.args = history.parse_args(
            ["--repository", str(self.repository), "--output", str(self.root / "output")]
        )
        self.output = self.root / "output" / self.sha / "cpu/1pc"

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.repository, text=True)

    def test_legacy_revision_rejected_before_install(self):
        with patch.object(history.venv, "create") as create:
            with self.assertRaisesRegex(ValueError, "predates"):
                history.run_revision(self.args, self.legacy)
            create.assert_not_called()
        self.assertFalse((self.root / "output").exists())
        self.assertEqual(self.git("rev-parse", "HEAD").strip(), self.sha)

    def run_with_fake_environment(self, fail=False):
        real_run = subprocess.run

        def run(command, **kwargs):
            if command[0] == "git":
                return real_run(command, **kwargs)
            if "install" in command:
                self.assertIn("--constraint", command)
            else:
                # Confirm benchmark runs from the selected checkout, not the caller's tree.
                self.assertNotEqual(kwargs["cwd"], self.repository)
                if fail:
                    raise subprocess.CalledProcessError(1, command)
                results = Path(command[command.index("--path") + 1])
                (results / "func_benchmarks.csv").write_text("runtime\n1\n")
            return subprocess.CompletedProcess(command, 0)

        with (
            patch.object(history.venv, "create"),
            patch.object(history.subprocess, "run", side_effect=run),
        ):
            history.run_revision(self.args, "HEAD")

    def test_success_records_resolved_revision(self):
        self.run_with_fake_environment()
        metadata = json.loads((self.output / "revision.json").read_text())
        self.assertEqual(metadata["commit"], self.sha)
        self.assertEqual(metadata["ref"], "HEAD")
        self.assertEqual((self.output / "func_benchmarks.csv").read_text(), "runtime\n1\n")
        self.assertEqual(self.git("rev-parse", "HEAD").strip(), self.sha)
        with self.assertRaises(FileExistsError):
            history.run_revision(self.args, "HEAD")

    def test_failed_run_preserves_previous_results(self):
        self.output.mkdir(parents=True)
        existing = self.output / "func_benchmarks.csv"
        existing.write_text("previous run")
        self.args.overwrite = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_with_fake_environment(fail=True)
        self.assertEqual(existing.read_text(), "previous run")
