"""Launcher identity setup works without personal defaults or a live W&B call."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HELPER = Path(__file__).resolve().parents[2] / "mirl_ext/wandb_env.sh"


class WandbEnvironmentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / ".wandb_key").write_text("test-key\n")
        self.python = self.root / "verify-python"
        self.python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "assert 'configure_wandb_auth' in sys.stdin.read()\n"
            "assert os.environ['WANDB_API_KEY'] == 'test-key'\n"
            "print(json.dumps({key: os.environ.get(key) for key in "
            "['WANDB_EXPECTED_USERNAME', 'WANDB_ENTITY', 'WANDB_MODE', 'NETRC']}))\n"
        )
        self.python.chmod(0o700)
        self.env = {
            "PATH": os.defpath,
            "MIRL_CLUSTER_ROOT": str(self.root),
            "MIRL_WANDB_ENTITY": "test-team",
            "WANDB_EXPECTED_USERNAME": "test-user",
            "PYTHON": str(self.python),
            "NETRC": "/another-account/.netrc",
            "WANDB_API_KEY": "stale-key",
        }

    def run_helper(self):
        return subprocess.run(
            ["bash", "-eu", "-c", 'source "$1"', "test", str(HELPER)],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
        )

    def test_explicit_identity_replaces_stale_credentials(self):
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout),
            {
                "WANDB_EXPECTED_USERNAME": "test-user",
                "WANDB_ENTITY": "test-team",
                "WANDB_MODE": "online",
                "NETRC": None,
            },
        )

    def test_queued_job_can_load_identity_from_private_config(self):
        del self.env["WANDB_EXPECTED_USERNAME"]
        (self.root / "mirl.env").write_text("export WANDB_EXPECTED_USERNAME=test-user\n")
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["WANDB_EXPECTED_USERNAME"], "test-user")

    def test_explicit_identity_is_not_overridden_by_private_config(self):
        (self.root / "mirl.env").write_text("export WANDB_EXPECTED_USERNAME=other-user\n")
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["WANDB_EXPECTED_USERNAME"], "test-user")

    def test_missing_identity_fails_before_python(self):
        del self.env["WANDB_EXPECTED_USERNAME"]
        result = self.run_helper()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("WANDB_EXPECTED_USERNAME", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_missing_key_fails_before_python(self):
        self.env["WANDB_API_KEY_FILE"] = str(self.root / "missing-key")
        result = self.run_helper()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing W&B key", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_private_netrc_is_selected_when_present(self):
        private_netrc = self.root / ".netrc"
        private_netrc.touch()
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["NETRC"], str(private_netrc))


if __name__ == "__main__":
    unittest.main()
