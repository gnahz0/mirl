"""Exercise the actual validation/dump methods without importing GPU/Ray runtimes."""

import ast
import json
import os
import tempfile
import unittest
import uuid
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


def _validation_methods():
    # Load production methods, not a copied implementation, with queue services mocked below.
    path = Path(__file__).resolve().parents[4] / "verl/trainer/ppo/v1/trainer_base.py"
    tree = ast.parse(path.read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PPOTrainer")
    methods = [
        node
        for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_validate", "_write_generations"}
    ]
    cls = ast.ClassDef(name="ValidationMethods", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {"np": np, "json": json, "os": os, "uuid": uuid, "defaultdict": defaultdict}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["ValidationMethods"], namespace


class _PromptBatch(dict):
    def __len__(self):
        return len(self["raw_prompt"])


class _TokenRows(list):
    def to_padded_tensor(self, padding):
        return self


class _Scores(list):
    def sum(self, dim):
        return np.asarray(self).sum(axis=dim)


class ValidationDumpTests(unittest.TestCase):
    def _run_validation(self, dump_path, extras=None):
        methods, namespace = _validation_methods()
        records = {}
        pending = []
        submitted = []
        # Finish rows in reverse order, with intermediate outputs and two sessions for row 0.
        plans = [[(1, 0, 1), (0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 1, 1)], [(0, 0, 0)]]

        def generate(batch):
            batch_number = len(submitted)
            submitted.append(batch)
            keys = []
            for row, session, output in plans[batch_number]:
                uid = batch["uid"][row]
                source_index = row if batch_number == 0 else 2
                key = f"{uid}_{session}_{output}"
                keys.append(key)
                records[key] = {
                    "uid": uid,
                    "prompts": f"prompt-{source_index}",
                    "responses": f"output-{source_index}-{session}-{output}",
                    "rm_scores": [float(10 * (source_index + 1) + session + output)],
                    "num_turns": output + 1,
                    "reward_model": {"ground_truth": f"target-{source_index}"},
                    "data_source": batch["data_source"][row],
                    "extra_fields": {"reward_extra_info": {"acc": float(source_index == 0), "f1": 0.75}},
                }
            pending.append(SimpleNamespace(keys=keys, partition_id="val"))

        def get(keys, partition_id, select_fields):
            columns = {}
            for field in select_fields:
                values = [records[key][field] for key in keys]
                if field in {"prompts", "responses"}:
                    columns[field] = _TokenRows(values)
                elif field == "rm_scores":
                    columns[field] = _Scores(values)
                else:
                    columns[field] = np.array(values, dtype=object)
            return columns

        namespace["tq"] = SimpleNamespace(kv_batch_put=Mock(), kv_batch_get=get, kv_clear=Mock())
        namespace["tu"] = SimpleNamespace(
            get_tensordict=_PromptBatch, assign_non_tensor_data=lambda batch, key, value: batch.update({key: value})
        )
        if extras is None:
            extras = np.array(
                [
                    {"dataset": "inspect", "index": np.int64(144), "unrelated": "do-not-dump"},
                    json.dumps({"dataset": "mimic-cxr", "index": 235}),
                ],
                dtype=object,
            )
        stub = SimpleNamespace(
            config=SimpleNamespace(trainer={"validation_data_dir": dump_path}),
            global_steps=210,
            val_dataloader=[
                {"raw_prompt": ["a", "b"], "data_source": ["ct", "chest_xray"], "extra_info": extras},
                {"raw_prompt": ["c"], "data_source": ["initial_fingers"]},
            ],
            tokenizer=SimpleNamespace(pad_token_id=0, decode=lambda ids, **kwargs: ids),
            agent_loop_manager=SimpleNamespace(generate_sequences=generate),
            replay_buffer=SimpleNamespace(sample=lambda **kwargs: (pending.pop(0), None)),
            reward_loop_manager=SimpleNamespace(reward_loop_worker_handles=[object()]),
            _maybe_log_val_generations=Mock(),
            _val_metrics_update=Mock(return_value={"unchanged_metric": 1.0}),
        )
        stub._dump_generations = Mock(
            side_effect=lambda **kwargs: methods._write_generations(**kwargs, global_steps=stub.global_steps)
        )
        with patch.object(uuid, "uuid4", side_effect=["z-first", "a-second", "m-third"]):
            result = methods._validate(stub)
        return stub, result

    def test_metadata_follows_uid_through_replay_sort_and_final_session_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            stub, result = self._run_validation(directory)
            rows = [json.loads(line) for line in (Path(directory) / "210.jsonl").read_text().splitlines()]
        self.assertEqual(result, {"unchanged_metric": 1.0})
        self.assertEqual(
            [row["uid"] for row in rows],
            ["a-second_0_0", "a-second_0_1", "m-third_0_0", "z-first_0_0", "z-first_0_1", "z-first_1_1"],
        )
        expected_metadata = {
            0: ("ct", "inspect", 144),
            1: ("chest_xray", "mimic-cxr", 235),
            2: ("initial_fingers", None, None),
        }
        for row in rows:
            index = row["validation_index"]
            source, dataset, original_index = expected_metadata[index]
            self.assertEqual((row["data_source"], row["dataset"], row["index"]), (source, dataset, original_index))
            self.assertEqual(row["input"], f"prompt-{index}")
            self.assertTrue(row["output"].startswith(f"output-{index}-"))
            self.assertEqual(row["gts"], f"target-{index}")
            self.assertEqual(row["step"], 210)
            self.assertEqual(row["acc"], float(index == 0))
            self.assertEqual(row["f1"], 0.75)
            self.assertNotIn("unrelated", row)
        # Intermediate outputs inherit their session's final metrics, as before.
        self.assertEqual([row["score"] for row in rows], [21.0, 21.0, 30.0, 11.0, 11.0, 12.0])
        sources, uids, metrics, turns = stub._val_metrics_update.call_args.args
        self.assertEqual(sources, ["chest_xray", "ct", "ct", "initial_fingers"])
        self.assertEqual(uids, ["a-second", "z-first", "z-first", "m-third"])
        self.assertEqual(set(metrics), {"reward", "acc", "f1"})
        self.assertEqual(metrics["reward"], [21.0, 11.0, 12.0, 30.0])
        self.assertEqual(turns, [2, 2, 2, 1])

    def test_disabled_dump_does_not_inspect_metadata_or_change_metrics(self):
        class UnreadableMetadata:
            def __getitem__(self, index):
                raise AssertionError("Metadata must not be read when dumps are disabled")

        stub, result = self._run_validation(None, extras=UnreadableMetadata())
        stub._dump_generations.assert_not_called()
        self.assertEqual(result, {"unchanged_metric": 1.0})
        self.assertEqual(set(stub._val_metrics_update.call_args.args[2]), {"reward", "acc", "f1"})


if __name__ == "__main__":
    unittest.main()
