"""Tests for CLI entry point."""

import sys
import pytest


class TestCLICompare:
    def test_cli_compare_exits_0(self):
        """CLI invocation exits 0."""
        import importlib.util
        import sys
        import os

        # Add repo root to path
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from cli import main
        exit_code = main(["compare", "--dataset", "synthetic", "--k", "3"])
        assert exit_code == 0

    def test_cli_compare_with_output(self, tmp_path):
        """CLI with --output saves a valid JSON report."""
        import sys
        import os
        import json

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        output_file = str(tmp_path / "report.json")
        from cli import main
        exit_code = main(["compare", "--dataset", "synthetic", "--k", "3",
                          "--output", output_file])
        assert exit_code == 0
        with open(output_file) as f:
            data = json.load(f)
        assert "baseline" in data
        assert "candidate" in data
        assert "comparison" in data
