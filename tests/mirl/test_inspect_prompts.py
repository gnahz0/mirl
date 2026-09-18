"""INSPECT prompt corrections must not use or rewrite answer labels."""

import copy
import importlib.util
import json
import unittest

from mirl_ext.data.prompts import normalize_prompt_messages
from mirl_ext.data.schema import prompt_messages, prompt_text


PROMPT = (
    "<video>\nWhat type of Pulmonary Embolism (PE) is present? Answer with one of the following:\n"
    "No PE\nAcute Subsegmental- PE\nAcute PE\nSubsegmental- PE\nChronic PE"
)
CORRECTED = PROMPT.replace("Subsegmental- PE", "Subsegmental-only PE")


class InspectPromptTests(unittest.TestCase):
    def test_both_choice_lines_are_corrected_without_mutating_input(self):
        messages = [{"role": "user", "content": PROMPT}]
        before = copy.deepcopy(messages)
        result = normalize_prompt_messages(messages, data_source="ct", dataset="inspect")
        self.assertEqual(result[0]["content"], CORRECTED)
        self.assertEqual(messages, before)
        self.assertIsNot(result[0], messages[0])

    def test_normalization_is_idempotent(self):
        first = normalize_prompt_messages(
            [{"role": "user", "content": PROMPT}], data_source="ct", dataset="inspect"
        )
        self.assertEqual(first, normalize_prompt_messages(first, data_source="ct", dataset="inspect"))

    def test_unrelated_sources_and_datasets_are_unchanged(self):
        messages = [{"role": "user", "content": PROMPT}]
        for source, dataset in (("ct", "rspect"), ("chest_xray", "inspect"), ("ct", None)):
            with self.subTest(source=source, dataset=dataset):
                self.assertEqual(
                    normalize_prompt_messages(messages, data_source=source, dataset=dataset), messages
                )

    def test_only_complete_choice_lines_in_user_messages_change(self):
        untouched = (
            "Mention Subsegmental- PE in prose.\n"
            "Not Subsegmental- PE\nSubsegmental- PE syndrome\nSubsegmental-only PE\n"
            "Acute Subsegmental-only PE"
        )
        messages = [
            {"role": "system", "content": "Subsegmental- PE"},
            {"role": "user", "content": untouched},
            {"role": "assistant", "content": "Subsegmental- PE"},
        ]
        self.assertEqual(
            normalize_prompt_messages(messages, data_source="ct", dataset="inspect"), messages
        )

    def test_choice_line_whitespace_and_crlf_are_preserved(self):
        result = normalize_prompt_messages(
            [{"role": "user", "content": " \tAcute Subsegmental- PE \t\r\nSubsegmental- PE\r\n"}],
            data_source="ct",
            dataset="inspect",
        )
        self.assertEqual(result[0]["content"], " \tAcute Subsegmental-only PE \t\r\nSubsegmental-only PE\r\n")

    def test_shared_teacher_and_sft_prompt_reader_ignores_ground_truth(self):
        outputs = []
        for target in ("No PE", "Acute Subsegmental-only PE", "Subsegmental- PE"):
            for metadata in ({"dataset": "inspect"}, json.dumps({"dataset": "inspect"})):
                with self.subTest(target=target, metadata=metadata):
                    row = {
                        "data_source": "ct",
                        "extra_info": metadata,
                        "prompt": [{"role": "user", "content": PROMPT}],
                        "reward_model": {"ground_truth": target},
                    }
                    before = copy.deepcopy(row)
                    outputs.append(prompt_messages(row))
                    self.assertEqual(prompt_text(row), CORRECTED)
                    self.assertEqual(row, before)
        self.assertTrue(all(result == outputs[0] for result in outputs))

    def test_rl_and_validation_use_the_same_correction(self):
        if importlib.util.find_spec("datasets") is None:
            self.skipTest("RL integration dependency unavailable: datasets")
        from mirl_ext.data.dataset import MIRLDataset

        dataset = object.__new__(MIRLDataset)
        dataset.audio_key = "audios"
        dataset.image_key = "images"
        dataset.video_key = "videos"
        dataset.need_tools_kwargs = False
        dataset.processor = object()
        dataset.config = {"max_video_frames": 24}
        for split in ("train", "validation"):
            row = {
                "data_source": "ct",
                "extra_info": json.dumps({"dataset": "inspect", "split": split}),
                "prompt": [{"role": "user", "content": PROMPT}],
                "images": [],
                "videos": [{"video": "/unused/inspect.mp4"}],
                "reward_model": {"ground_truth": "No PE"},
            }
            before = copy.deepcopy(row)
            messages = dataset._build_messages(row, key="prompt")
            content = messages[0]["content"]
            self.assertEqual(content[-1]["text"], CORRECTED.removeprefix("<video>"))
            self.assertEqual(content[0]["nframes"], 8)
            self.assertEqual(row, before)


if __name__ == "__main__":
    unittest.main()
