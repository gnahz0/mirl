from __future__ import annotations

import pytest

from mirl_ext.rewards.combined import compute_score


@pytest.mark.parametrize(
    ("data_source", "prediction", "ground_truth"),
    [
        ("initial_fingers", "reasoning</think>\\boxed{A,B}", "A,B"),
        ("meld_emotion", "reasoning</think>\\boxed{happy}", "happy"),
        ("mri", "reasoning</think>\\boxed{Glioma Tumor}", "Glioma Tumor"),
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


@pytest.mark.parametrize("source", ["intentqa", "mimeqa", "siq2"])
def test_hba_classification_reward_rejects_open_qa(source):
    with pytest.raises(NotImplementedError, match="open QA"):
        compute_score(source, "reasoning</think>\\boxed{answer}", "answer")


@pytest.mark.parametrize(
    ("source", "answer", "ground_truth", "expected"),
    [
        ("initial_fingers", "A,B", "A,B", 1.0),
        ("initial_fingers", "A", "A,B", 0.3),
        ("highest_pressure", "C", "A,B", 0.1),
        ("mri", "Glioma Tumor", "Glioma Tumor", 1.0),
        ("mri", "zzz", "Glioma Tumor", 0.0),
        ("meld_emotion", "happy", "happy", 1.0),
        ("meld_emotion", "zzz", "happy", 0.2),
        ("ecg", "Normal", "Normal", 1.0),
        ("ecg", "Other", "Normal", 0.0),
    ],
)
def test_modality_reward_weights(source, answer, ground_truth, expected):
    result = compute_score(source, "reasoning</think>\\boxed{" + answer + "}", ground_truth)
    assert result["score"] == pytest.approx(expected)


class _FixedResponseTokenizer:
    """Separate response length/masking checks from tokenizer vocabulary."""

    def __init__(self, response):
        self.response = response

    def decode(self, token_ids, skip_special_tokens=True):
        return self.response


@pytest.mark.parametrize(
    ("source", "answer", "ground_truth"),
    [
        ("ecg", "Normal", "Normal"),
        ("mri", "Glioma Tumor", "Glioma Tumor"),
        ("meld_emotion", "happy", "happy"),
        ("initial_fingers", "B,A", "A,B"),
        ("highest_pressure", "A", "A,B"),
    ],
)
@pytest.mark.parametrize("response_length", [32, 3584, 3585, 3840, 4096])
@pytest.mark.parametrize("log_overlong", [False, True])
def test_upstream_dapo_preserves_mirl_scores_and_applies_length_penalty(
    source, answer, ground_truth, response_length, log_overlong
):
    import asyncio
    from pathlib import Path

    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from verl import DataProto
    from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager
    from verl.trainer.ppo.reward import load_reward_manager

    response = "reasoning</think>\\boxed{" + answer + "}"
    expected = compute_score(source, response, ground_truth)
    config = OmegaConf.create(
        {
            "reward": {
                "reward_manager": {"source": "register", "name": "dapo"},
                "custom_reward_function": {
                    "path": str(Path(__file__).resolve().parents[2] / "mirl_ext/rewards/combined.py"),
                    "name": "compute_score",
                },
                "reward_kwargs": {
                    "max_resp_len": 4096,
                    "overlong_buffer_cfg": {"enable": True, "len": 512, "penalty_factor": 1.0, "log": log_overlong},
                },
            }
        }
    )
    attention_mask = torch.zeros(1, 2 + 4096, dtype=torch.long)
    attention_mask[:, : 2 + response_length] = 1
    data = DataProto.from_dict(
        tensors={
            "prompts": torch.ones(1, 2, dtype=torch.long),
            "responses": torch.ones(1, 4096, dtype=torch.long),
            "attention_mask": attention_mask,
        },
        non_tensors={
            "data_source": np.array([source], dtype=object),
            "reward_model": np.array([{"ground_truth": ground_truth}], dtype=object),
        },
    )

    async def score():
        manager = load_reward_manager(config, _FixedResponseTokenizer(response))
        assert isinstance(manager, DAPORewardManager)
        return manager, await manager.run_single(data)

    manager, result = asyncio.run(score())
    penalty = min((3584 - response_length) / 512, 0.0)
    assert float(result["reward_score"]) == pytest.approx(expected["score"] + penalty)
    # Accuracy/F1/format and the unshaped score must stay available for eval.
    extra = result["reward_extra_info"]
    assert {key: extra[key] for key in expected} == expected
    if log_overlong:
        assert float(extra["overlong_reward"]) == pytest.approx(penalty)
        assert bool(extra["overlong"]) == (penalty < 0)
    else:
        assert extra == expected
    rewards = manager.assemble_rm_scores(data, [result["reward_score"]])
    assert rewards[0, response_length - 1].item() == pytest.approx(expected["score"] + penalty)
    rewards[0, response_length - 1] = 0
    assert torch.count_nonzero(rewards).item() == 0


def test_grpo_normalizes_within_prompt_not_across_modalities():
    import numpy as np
    import torch

    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

    scores = torch.tensor([0.0, 0.2, 0.4, 0.6, 1.0])
    # Positive rescaling of a complete reward group should not change its
    # normalized advantages. This is a synthetic scale, not an HBA weight.
    rewards = torch.zeros(15, 3)
    rewards[:, -1] = torch.cat([scores, 1.7 * scores, torch.ones(5)])
    mask = torch.ones_like(rewards)
    mask[:, 0] = 0
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=np.repeat(["original_prompt", "rescaled_prompt", "constant_prompt"], 5),
        norm_adv_by_std_in_grpo=True,
    )
    torch.testing.assert_close(advantages[:5], advantages[5:10], atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(advantages[10:]).item() == 0
    assert torch.count_nonzero(advantages[:, 0]).item() == 0


def test_mixed_reward_metrics_have_the_uniform_numeric_schema_required_by_verl():
    import numpy as np

    samples = [
        ("initial_fingers", "A", "A,B"),
        ("meld_emotion", "happy", "sad"),
        ("mri", "Support Devices, Pleural Effusion", "Pleural Effusion, Support Devices"),
        ("ecg", "Normal", "Normal"),
    ]
    scores = [compute_score(src, "reasoning</think>\\boxed{" + pred + "}", gt) for src, pred, gt in samples]
    keys = {"score", "acc", "f1", "precision", "recall", "jaccard", "similarity", "format"}
    assert all(set(score) == keys for score in scores)
    # Match veRL's reward-loop/agent-loop collation: every sample must expose
    # every key from the first sample, otherwise mixed batches fail at runtime.
    collated = {key: np.array([score[key] for score in scores]) for key in scores[0]}
    assert all(array.shape == (4,) and np.isfinite(array).all() for array in collated.values())


@pytest.mark.parametrize(
    ("source", "prediction", "target", "expected_score", "expected_format"),
    [
        ("ecg", "Normal", "Normal", 1.0, 0.0),
        ("ecg", "\\boxed{Normal}", "Normal", 1.0, 0.0),
        ("ecg", "Abnormal", "Normal", 0.0, 0.0),
        ("ecg", "Not Normal.", "Normal", 0.0, 0.0),
        (
            "mri",
            "reasoning</think>\\boxed{Support Devices, Pleural Effusion}",
            "Pleural Effusion, Support Devices",
            1.0,
            1.0,
        ),
        ("meld_emotion", "[ HAPPY ]", "happy", 0.8, 0.0),
        ("ptsd_in_the_wild", "reasoning</think>\\boxed{ptsd}", "no ptsd", 0.2, 1.0),
    ],
)
def test_dispatch_preserves_each_selected_reward_recipe(source, prediction, target, expected_score, expected_format):
    result = compute_score(source, prediction, target)
    assert result["score"] == pytest.approx(expected_score)
    assert result["format"] == expected_format
