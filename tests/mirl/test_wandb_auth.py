"""Verify file-based W&B credentials without serializing secrets into Ray."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from verl.utils.tracking import Tracking, configure_wandb_auth


@pytest.fixture
def file_auth(tmp_path, monkeypatch):
    key = "a" * 40  # Synthetic credential; never used for a network request.
    path = tmp_path / "key"
    path.write_text(key + "\n")
    path.chmod(0o600)
    monkeypatch.setenv("WANDB_API_KEY_FILE", str(path))
    monkeypatch.setenv("WANDB_EXPECTED_USERNAME", "test-user")
    monkeypatch.setenv("WANDB_API_KEY", "stale-inherited-key")
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    mock = MagicMock()
    mock.Api.return_value.viewer.username = "test-user"
    mock.Settings.side_effect = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "wandb", mock)
    return path, key, mock


def test_file_key_overrides_stale_actor_environment_without_login(file_auth, caplog):
    _, key, mock = file_auth
    previous_home = os.environ.get("HOME")
    assert configure_wandb_auth() == "test-user"
    assert os.environ["WANDB_API_KEY"] == key
    mock.Api.assert_called_once_with(api_key=key)
    mock.login.assert_not_called()
    assert os.environ.get("HOME") == previous_home
    assert key not in caplog.text


def test_wrong_identity_fails_before_run_creation(file_auth):
    _, _, mock = file_auth
    mock.Api.return_value.viewer.username = "wrong-account"
    with pytest.raises(RuntimeError, match="expected 'test-user'"):
        Tracking("test-project", "test-run", default_backend="wandb", config={"trainer": {}})
    mock.init.assert_not_called()
    mock.login.assert_not_called()


@pytest.mark.parametrize("expected_identity", [None, ""])
def test_file_auth_requires_expected_identity(file_auth, monkeypatch, expected_identity):
    _, _, mock = file_auth
    if expected_identity is None:
        monkeypatch.delenv("WANDB_EXPECTED_USERNAME")
    else:
        monkeypatch.setenv("WANDB_EXPECTED_USERNAME", expected_identity)
    with pytest.raises(ValueError, match="WANDB_EXPECTED_USERNAME"):
        configure_wandb_auth()
    mock.Api.assert_not_called()


def test_expected_identity_matching_is_case_insensitive(file_auth, monkeypatch):
    _, key, mock = file_auth
    monkeypatch.setenv("WANDB_EXPECTED_USERNAME", "TEST-USER")
    assert configure_wandb_auth() == "test-user"
    mock.Api.assert_called_once_with(api_key=key)


@pytest.mark.parametrize("missing", [False, True])
def test_bad_key_file_does_not_fall_back_to_shared_credentials(file_auth, missing):
    path, _, mock = file_auth
    if missing:
        path.unlink()
        error = FileNotFoundError
    else:
        path.write_text(" \n")
        error = ValueError
    with pytest.raises(error):
        configure_wandb_auth()
    mock.Api.assert_not_called()


def test_tracking_uses_explicit_verified_settings_and_preserves_proxy(file_auth):
    _, key, mock = file_auth
    config = {"trainer": {"wandb_proxy": "https://proxy.invalid"}}
    Tracking("test-project", "test-run", default_backend="wandb", config=config)
    settings = mock.init.call_args.kwargs["settings"]
    assert settings.api_key == key
    assert settings.https_proxy == "https://proxy.invalid"
    assert mock.init.call_args.kwargs["entity"] == "test-entity"
    assert "api_key" not in config["trainer"]
    mock.login.assert_not_called()


def test_existing_auth_is_unchanged_without_file_opt_in(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY_FILE", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "existing-environment-key")
    assert configure_wandb_auth() is None
    assert os.environ["WANDB_API_KEY"] == "existing-environment-key"


def test_ray_runtime_env_forwards_paths_and_identity_but_never_key(file_auth, monkeypatch):
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    path, key, _ = file_auth
    monkeypatch.setenv("WANDB_API_KEY", key)
    monkeypatch.setenv("WANDB_RUN_ID", "test-run")
    monkeypatch.setenv("WANDB_MODE", "online")
    env = get_ppo_ray_runtime_env()["env_vars"]
    assert env["WANDB_API_KEY_FILE"] == str(path)
    assert env["WANDB_EXPECTED_USERNAME"] == "test-user"
    assert env["WANDB_ENTITY"] == "test-entity"
    assert env["WANDB_RUN_ID"] == "test-run"
    assert env["WANDB_MODE"] == "online"
    assert "WANDB_API_KEY" not in env
    assert key not in repr(env)


def test_worker_auth_precedes_expensive_training_initialization():
    import ast

    source = (Path(__file__).resolve().parents[2] / "verl/trainer/main_ppo.py").read_text()
    tree = ast.parse(source)
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TaskRunnerV1")
    run = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
    auth_line = next(
        node.lineno for node in calls if isinstance(node.func, ast.Name) and node.func.id == "configure_wandb_auth"
    )
    trainer_line = next(
        node.lineno for node in calls if isinstance(node.func, ast.Name) and node.func.id == "trainer_cls"
    )
    assert auth_line < trainer_line
