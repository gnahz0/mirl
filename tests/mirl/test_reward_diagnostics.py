"""Reward diagnostics: stdlib grouping tests and optional CPU tensor integration."""

import ast
import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

from mirl_ext.rewards.diagnostics import masked_reward_sums, reward_group_metrics


class RewardGroupDiagnosticsTests(unittest.TestCase):
    def test_uneven_unordered_groups_and_sources(self):
        # Three ECG prompts: one constant pair, one mixed triple, one singleton.
        scores = [1, 0, 0.5, 1, 1, 0.5, 0, 0.25]
        uids = ["a", "b", "c", "a", "b", "c", "b", "d"]
        sources = ["ecg", "ecg", "ct", "ecg", "ecg", "ct", "ecg", "ecg"]
        result = reward_group_metrics(scores, scores, uids, sources, [False] * 8)
        self.assertEqual(result["train_reward/ecg/response_count"], 6)
        self.assertEqual(result["train_reward/ecg/prompt_group_count"], 3)
        self.assertEqual(result["train_reward/ecg/singleton_group_count"], 1)
        self.assertEqual(result["train_reward/ecg/multi_response_group_count"], 2)
        self.assertEqual(result["train_reward/ecg/zero_variance_group_fraction"], 0.5)
        self.assertAlmostEqual(result["train_reward/ecg/outcome_reward_mean"], 3.25 / 6)
        self.assertEqual(result["train_reward/ct/zero_variance_group_fraction"], 1)
        self.assertFalse(any("tactile" in key for key in result))

    def test_distinguishes_outcome_from_advantage_reward_and_ignores_padding(self):
        result = reward_group_metrics(
            [1, 1, math.nan], [0.5, 0.25, math.nan],
            ["a", "a", "padding"], ["ct", "ct", "absent"], [False, False, True],
        )
        self.assertEqual(result["train_reward/ct/outcome_reward_mean"], 1)
        self.assertEqual(result["train_reward/ct/advantage_reward_mean"], 0.375)
        self.assertEqual(result["train_reward/ct/zero_variance_group_fraction"], 0)
        self.assertEqual(result["train_reward/ct/response_count"], 2)
        self.assertFalse(any("absent" in key for key in result))

    def test_singletons_do_not_get_fake_group_fractions(self):
        result = reward_group_metrics([0.7], [0.7], ["one"], ["ecg"], [False])
        self.assertEqual(result["train_reward/ecg/singleton_group_count"], 1)
        self.assertNotIn("train_reward/ecg/zero_variance_group_fraction", result)
        self.assertEqual(reward_group_metrics([], [], [], [], []), {})

    def test_does_not_mutate_inputs(self):
        rewards = [1.0, 7.0]
        reward_group_metrics(rewards, rewards, ["a", "a"], ["ecg"] * 2, [False] * 2)
        self.assertEqual(rewards, [1.0, 7.0])

    def test_invalid_inputs_fail_loudly(self):
        cases = [
            ([1], [1], [], ["ecg"], [False]),
            ([math.nan], [1], ["a"], ["ecg"], [False]),
            ([1], [math.inf], ["a"], ["ecg"], [False]),
            ([1], [1], [""], ["ecg"], [False]),
            ([1], [1], ["a"], [None], [False]),
            ([1, 1], [1, 1], ["a", "a"], ["ecg", "ct"], [False] * 2),
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                reward_group_metrics(*args)


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch is needed for tensor integration")
class MaskedRewardTensorTests(unittest.TestCase):
    def test_masking_detachment_and_empty_response(self):
        import torch

        rewards = torch.tensor([[99.0, 1.0, math.nan], [2.0, 3.0, 4.0]], requires_grad=True)
        mask = torch.tensor([[0, 1, 0], [0, 0, 0]])
        result = masked_reward_sums(rewards, mask)
        self.assertEqual(result, [1.0, 0.0])
        self.assertIsNone(rewards.grad)
        self.assertTrue(rewards.requires_grad)
        self.assertTrue(torch.isnan(rewards[0, 2]).item())

    def test_rejects_invalid_shapes_and_nonbinary_masks(self):
        import torch

        for rewards, mask in (
            (torch.ones(2, 3), torch.ones(2, 2)),
            (torch.ones(2), torch.ones(2)),
            (torch.ones(2, 3), torch.full((2, 3), 0.5)),
        ):
            with self.subTest(shape=rewards.shape), self.assertRaises(ValueError):
                masked_reward_sums(rewards, mask)

    def test_actual_v1_hook_is_opt_in_and_observes_post_kl_rewards(self):
        import numpy as np
        import torch

        # Exercise the real method without importing Ray/TransferQueue/GPU workers.
        path = Path(__file__).resolve().parents[2] / "verl/trainer/ppo/v1/trainer_base.py"
        tree = ast.parse(path.read_text())
        trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PPOTrainer")
        method = next(
            node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_compute_advantage"
        )

        class PaddedData(dict):
            def to_padded_tensor(self):
                return self

        class Batch:
            keys = ["a_0_0", "a_1_0"]
            tags = [{}, {}]
            partition_id = "train"

            def __len__(self):
                return len(self.keys)

        class Algo(SimpleNamespace):
            def get(self, key, default=None):
                return getattr(self, key, default)

        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                scores = torch.tensor([[99.0, 1.0], [99.0, 1.0]])
                fields_seen = []

                def get_batch(**kwargs):
                    fields_seen.extend(kwargs["select_fields"])
                    values = PaddedData(
                        uid=np.array(["a", "a"], dtype=object), rm_scores=scores,
                        response_mask=torch.tensor([[0, 1], [0, 1]]),
                    )
                    if "data_source" in kwargs["select_fields"]:
                        values["data_source"] = np.array(["ct", "ct"], dtype=object)
                    return values

                def apply_kl(data, **kwargs):
                    data.batch["token_level_rewards"] = scores - torch.tensor([[0, 0.5], [0, 0.75]])
                    return data, {}

                def advantage(data, **kwargs):
                    data.batch["advantages"] = torch.zeros_like(scores)
                    data.batch["returns"] = torch.zeros_like(scores)
                    return data

                scope = {
                    "KVBatchMeta": Batch, "np": np,
                    "tq": SimpleNamespace(kv_batch_get=get_batch, kv_batch_put=lambda **kwargs: kwargs["fields"]),
                    "DataProto": lambda batch: SimpleNamespace(batch=batch, non_tensor_batch={}),
                    "apply_kl_penalty": apply_kl,
                    "compute_advantage_for_multi_trajectories": advantage,
                    "response_to_nested": lambda value, mask: value,
                    "TensorDict": lambda value, batch_size: value,
                }
                exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
                self_obj = SimpleNamespace(
                    config=SimpleNamespace(
                        trainer={"log_reward_group_metrics": enabled},
                        algorithm=Algo(use_kl_in_reward=True, kl_penalty="kl", adv_estimator="grpo", gamma=1, lam=1),
                        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=2)),
                    ),
                    kl_ctrl_in_reward=object(),
                )
                metrics = {}
                output = scope["_compute_advantage"](self_obj, Batch(), metrics)
                self.assertEqual("data_source" in fields_seen, enabled)
                self.assertEqual(scores.tolist(), [[99.0, 1.0], [99.0, 1.0]])
                self.assertTrue(torch.equal(output["advantages"], torch.zeros_like(scores)))
                if enabled:
                    self.assertEqual(metrics["train_reward/ct/outcome_reward_mean"], 1)
                    self.assertEqual(metrics["train_reward/ct/advantage_reward_mean"], 0.375)
                    self.assertEqual(metrics["train_reward/ct/zero_variance_group_fraction"], 0)
                else:
                    self.assertEqual(metrics, {})


if __name__ == "__main__":
    unittest.main()
