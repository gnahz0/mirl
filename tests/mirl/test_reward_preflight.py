"""Strict reward targets must not break the teacher's pre-billing route check."""

from types import SimpleNamespace

import pytest

from mirl_ext.sft.scripts.gen_sft_targets import _verify_reward_scorers


def test_preflight_uses_real_targets_for_every_reward_family():
    sources = {
        "ecg": "Normal",
        "mri": "Glioma Tumor",
        "initial_fingers": "A,B",
        "meld_emotion": "happy",
    }
    tasks = [SimpleNamespace(uid=source, data_source=source) for source in sources]
    _verify_reward_scorers(tasks, sources)


@pytest.mark.parametrize(
    ("source", "target"),
    [("ecg", "y"), ("mri", ""), ("unknown", "label"), ("intentqa", "answer")],
)
def test_invalid_targets_or_routes_fail_before_teacher_requests(source, target):
    with pytest.raises(SystemExit, match="reward preflight failed"):
        _verify_reward_scorers([SimpleNamespace(uid="example", data_source=source)], {"example": target})
