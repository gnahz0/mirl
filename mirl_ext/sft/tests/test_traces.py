"""Pure-stdlib tests for append-only SFT trace reading."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mirl_ext.sft.traces import accepted_traces, last_records, read_status


class TraceReaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_jsonl(self, name: str, records: list[dict]) -> Path:
        path = self.root / name
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        return path

    def test_last_records_across_files_skips_bad_lines_and_last_file_wins(self) -> None:
        first = self.write_jsonl(
            "first.jsonl",
            [
                {"uid": "f#1", "status": "accepted", "response": "old"},
                {"uid": "f#2", "status": "accepted"},
            ],
        )
        first.write_text(
            first.read_text()
            + "not-json\n"
            + json.dumps({"status": "error"})
            + "\nnull\n[]\n"
            + json.dumps({"uid": None})
            + "\n"
        )
        second = self.write_jsonl(
            "second.jsonl",
            [
                {"uid": "f#1", "status": "exhausted", "response": "new"},
                {"uid": "f#3"},
            ],
        )

        records, skipped = last_records([first, second])

        self.assertEqual(skipped, 5)
        self.assertEqual(records["f#1"]["response"], "new")
        self.assertEqual(records["f#1"]["status"], "exhausted")

    def test_accepted_traces_filters_after_last_record_selection(self) -> None:
        traces = self.write_jsonl(
            "traces.jsonl",
            [
                {"uid": "f#1", "status": "accepted", "response": "old"},
                {"uid": "f#1", "status": "accepted", "response": "new"},
                {"uid": "f#2", "status": "accepted"},
                {"uid": "f#2", "status": "error"},
                {"uid": "f#3"},
            ],
        )

        accepted = accepted_traces([traces])

        self.assertEqual(set(accepted), {"f#1", "f#3"})
        self.assertEqual(accepted["f#1"]["response"], "new")

    def test_read_status_supports_resume_and_missing_file(self) -> None:
        traces = self.write_jsonl(
            "resume.jsonl",
            [
                {"uid": "a#1", "status": "accepted"},
                {"uid": "a#2", "status": "exhausted"},
                {"uid": "a#3", "status": "error"},
                {"uid": "a#3", "status": "accepted"},
                {"uid": "a#4"},
            ],
        )

        status = read_status(traces)
        todo = [
            uid
            for uid in ("a#1", "a#2", "a#3", "a#4", "a#5")
            if status.get(uid) in (None, "error")
        ]

        self.assertEqual(todo, ["a#5"])
        self.assertEqual(read_status(self.root / "missing.jsonl"), {})


if __name__ == "__main__":
    unittest.main()
