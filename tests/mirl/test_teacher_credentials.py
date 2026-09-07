"""Teacher credentials are explicit and never borrowed from another checkout."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mirl_ext.sft.scripts.gen_sft_targets import load_api_key, make_client


def test_explicit_key_has_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("MIRL_OPENAI_KEY", " test-key \n")
    monkeypatch.setenv("MIRL_OPENAI_KEY_FILE", str(tmp_path / "missing-key"))
    assert load_api_key() == "test-key"


def test_key_file_is_supported(monkeypatch, tmp_path):
    key_file = tmp_path / "teacher-key"
    key_file.write_text(" test-key \n")
    monkeypatch.delenv("MIRL_OPENAI_KEY", raising=False)
    monkeypatch.setenv("MIRL_OPENAI_KEY_FILE", str(key_file))
    assert load_api_key() == "test-key"


@pytest.mark.parametrize("contents", [None, "", " \n"])
def test_missing_or_empty_key_fails_closed(monkeypatch, tmp_path, contents):
    key_file = tmp_path / "teacher-key"
    if contents is not None:
        key_file.write_text(contents)
    monkeypatch.setenv("MIRL_OPENAI_KEY", " ")
    monkeypatch.setenv("MIRL_OPENAI_KEY_FILE", str(key_file))
    with pytest.raises(SystemExit, match="No teacher API key"):
        load_api_key()


def test_teacher_endpoint_is_required_before_client_creation(monkeypatch):
    client = Mock()
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=client))
    monkeypatch.delenv("MIRL_OPENAI_BASE_URL", raising=False)
    with pytest.raises(SystemExit, match="MIRL_OPENAI_BASE_URL"):
        make_client(30)
    client.assert_not_called()


def test_teacher_client_preserves_timeout_and_retry_policy(monkeypatch):
    client = Mock()
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=client))
    monkeypatch.setenv("MIRL_OPENAI_BASE_URL", "https://teacher.example/v1")
    monkeypatch.setenv("MIRL_OPENAI_KEY", "test-key")
    assert make_client(45) is client.return_value
    client.assert_called_once_with(base_url="https://teacher.example/v1", api_key="test-key", timeout=45, max_retries=0)
