from __future__ import annotations

import pytest

from mirl_ext.rewards.combined import compute_score


@pytest.mark.parametrize(
    ("data_source", "prediction", "ground_truth"),
    [
        ("initial_fingers", "reasoning</think>\\boxed{A,B}", "A,B"),
        ("meld_emotion", "reasoning</think>\\boxed{happy}", "happy"),
        ("mri", "reasoning</think>\\boxed{Glioma Tumor}", "Glioma Tumor"),
        ("smellnet_base", "reasoning</think>\\boxed{allspice}", "allspice"),
        ("ecg", "reasoning</think>\\boxed{Normal}", "Normal"),
        (
            "haptic_tactile",
            "reasoning</think>\\boxed{lifting the board by its handle}",
            "lifting the board by its handle",
        ),
    ],
)
def test_combined_dispatch_and_qwen35_think_prefix(data_source, prediction, ground_truth):
    result = compute_score(data_source, prediction, ground_truth)
    assert result["score"] > 0
    assert result["acc"] == 1.0
    assert all(isinstance(value, float) for value in result.values())


def test_unknown_source_fails_closed():
    with pytest.raises(NotImplementedError, match="unknown"):
        compute_score("unknown", "x", "y")

