"""Focused tests for the answer-blind SFT pipeline. Pure-python: runnable with
pytest or directly (`python mirl_ext/sft/tests/test_sft_pipeline.py`).
"""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mirl_ext.rewards import combined  # noqa: E402
from mirl_ext.data.schema import OPEN_SOURCES, prompt_messages, prompt_text  # noqa: E402
from mirl_ext.sft.scripts.build_sft_parquet import (  # noqa: E402
    accepted_traces,
    build_record,
    check_record,
    sft_messages,
)
from mirl_ext.sft.scripts.gen_sft_targets import (  # noqa: E402
    SYSTEM_PROMPT,
    TeacherTask,
    build_request,
    read_status,
    validate,
)
from mirl_ext.sft.scripts.split_sft_rl import assign_groups  # noqa: E402
from mirl_ext.sft.scripts.stage_media import _stem  # noqa: E402

GOOD_THINK = (
    "<think>The upper zones show patchy consolidation with air bronchograms and "
    "the costophrenic angle is blunted, which favors effusion over simple "
    "atelectasis; cardiac silhouette is enlarged.</think>"
)


def _task(**kw) -> TeacherTask:
    base = dict(
        uid="climb_train#7",
        family="climb_train",
        data_source="ct",
        prompt="<image>\nWhat is shown? Answer with one of the following:\nNo PE\nAcute PE",
        image_paths=(),
        frame_paths=(),
        staging_version="v2",
    )
    base.update(kw)
    return TeacherTask(**base)


# ---- request construction stays answer-blind ----

def test_teacher_task_has_no_ground_truth_field():
    assert "ground_truth" not in {f.name for f in dataclasses.fields(TeacherTask)}
    assert "ground_truth" not in TeacherTask.from_row(
        {"uid": "x#1", "family": "climb_train", "prompt": "q", "ground_truth": "SECRET"}
    ).__dict__


def test_answer_key_sentinel_never_reaches_request():
    sentinel = "XKCD_SENTINEL_9d41f"
    task = TeacherTask.from_row(
        {"uid": "x#1", "family": "climb_train", "prompt": "Describe.", "ground_truth": sentinel}
    )
    messages, n_media = build_request(task, None)
    assert sentinel not in json.dumps(messages) and n_media == 0


def test_answer_conditioned_request_reveals_answer_and_swaps_system():
    from mirl_ext.sft.scripts.gen_sft_targets import RATIONALIZE_SYSTEM

    messages, _ = build_request(_task(), None, answer="No PE")
    assert messages[0]["content"] == RATIONALIZE_SYSTEM
    assert "VERIFIED ANSWER: No PE" in messages[1]["content"][0]["text"]
    # Default path stays answer-blind and uses the zero-shot system prompt.
    blind, _ = build_request(_task(), None)
    assert blind[0]["content"] == SYSTEM_PROMPT
    assert "VERIFIED ANSWER" not in blind[1]["content"][0]["text"]


def test_request_contains_no_demonstrations():
    messages, _ = build_request(_task(), None)
    assert [m["role"] for m in messages] == ["system", "user"]
    parts = messages[1]["content"]
    assert len(parts) == 1 and parts[0]["type"] == "text"  # context + question only
    assert messages[0]["content"] == SYSTEM_PROMPT


# ---- export helpers ----

def test_prompt_flattening_keeps_all_turns():
    row = {"prompt": [{"role": "system", "content": "SYS"},
                      {"role": "user", "content": "<image>\nQ?"}]}
    assert [m["role"] for m in prompt_messages(row)] == ["system", "user"]
    assert prompt_text(row) == "SYS\n\n<image>\nQ?"


def test_open_sources_never_marked_gradable():
    assert "haptic_tactile" in OPEN_SOURCES and "description" in OPEN_SOURCES
    assert "initial_fingers" not in OPEN_SOURCES and "ecg" not in OPEN_SOURCES


# ---- split ----

def test_split_groups_disjoint_and_locked_first():
    groups = {f"g{i}": list(range(i * 3, i * 3 + 3)) for i in range(8)}
    strata = {gid: "s" for gid in groups}
    assignment = assign_groups(groups, strata, seed=1)
    assert set(assignment) == set(groups)
    assert all(side in ("sft", "rl") for side in assignment.values())
    locked = {"g0": "rl", "g1": "rl"}
    relocked = assign_groups(groups, strata, seed=1, locked=locked)
    assert relocked["g0"] == relocked["g1"] == "rl"


def test_split_ratio_targets_20_80():
    groups = {f"g{i}": [i] for i in range(1000)}
    strata = {gid: "s" for gid in groups}
    assignment = assign_groups(groups, strata, seed=3, sft_frac=0.2)
    n_sft = sum(1 for side in assignment.values() if side == "sft")
    assert 180 <= n_sft <= 220, n_sft


# ---- validation ----

def _v(text, gt="No PE", task=None):
    return validate(text, gt, task or _task())


def test_validate_accepts_well_formed_correct_trace():
    ok, reason, predicted = _v(GOOD_THINK + " \\boxed{No PE}")
    assert (ok, reason, predicted) == (True, "ok", "no pe")


def test_boxed_must_be_anchored_and_unique():
    assert _v(GOOD_THINK + " \\boxed{No PE} trailing words")[1] == "text_after_boxed"
    assert _v(GOOD_THINK + " \\boxed{No PE} \\boxed{No PE}")[1] == "multiple_boxed"
    assert _v(GOOD_THINK + " the answer is No PE")[1] == "no_boxed"
    assert _v("no think tags at all \\boxed{No PE}")[1] == "format"
    assert _v("<think>  </think> \\boxed{No PE}")[1] == "empty_think"


def test_rationale_length_bounds():
    assert _v("<think>too short</think> \\boxed{No PE}")[1] == "think_too_short"
    assert _v(f"<think>{'x' * 4000}</think> \\boxed{{No PE}}")[1] == "think_too_long"


def test_leakage_phrases_rejected():
    for phrase in ("the correct answer is", "the provided answer", "given the answer",
                   "the verified answer", "ground truth", "I was told", "support set",
                   "examples above", "the description says", "the blue line"):
        text = f"<think>Reasoning mentioning {phrase} somewhere in the rationale, at length.</think> \\boxed{{No PE}}"
        assert _v(text)[1] == "leak", phrase


def test_wrong_answers_are_wrong_not_repaired():
    ok, reason, predicted = _v(GOOD_THINK + " \\boxed{Acute PE}")
    assert (ok, predicted) == (False, "acute pe") and reason == "wrong"


def test_ecg_gate_requires_verbatim_category():
    # rewards.ecg substring-matches category mentions ("Abnormal" contains
    # "Normal"); the SFT keep-gate must demand the exact label instead.
    t = _task(data_source="ecg")
    assert not _v(GOOD_THINK + " \\boxed{Abnormal}", gt="Normal", task=t)[0]
    assert not _v(GOOD_THINK + " \\boxed{no Myocardial Infarction}",
                  gt="Myocardial Infarction", task=t)[0]
    assert _v(GOOD_THINK + " \\boxed{Normal}", gt="Normal", task=t)[0]


def test_multilabel_and_letter_set_semantics_follow_rl_rewards():
    med = GOOD_THINK + " \\boxed{Support Devices, Pleural Effusion}"
    assert _v(med, gt="Pleural Effusion, Support Devices")[0]  # order-invariant set
    tac = GOOD_THINK + " \\boxed{B, A}"
    assert _v(tac, gt="A,B", task=_task(data_source="initial_fingers"))[0]
    hb = GOOD_THINK + " \\boxed{no ptsd}"
    assert _v(hb, gt="No PTSD", task=_task(data_source="ptsd_in_the_wild"))[0]
    assert combined.compute_score("ecg", GOOD_THINK + " \\boxed{Normal}", "Normal")["acc"] == 1.0


# ---- resume + parquet build ----

def _write_jsonl(records) -> Path:
    tmp = Path(tempfile.mkstemp(suffix=".jsonl")[1])
    tmp.write_text("".join(json.dumps(r) + "\n" for r in records))
    return tmp


def test_resume_skips_accepted_and_exhausted_retries_errors():
    tmp = _write_jsonl([
        {"uid": "a#1", "status": "accepted"},
        {"uid": "a#2", "status": "exhausted"},
        {"uid": "a#3", "status": "error"},
        {"uid": "a#3", "status": "accepted"},  # retried later and accepted
        {"uid": "a#4"},  # legacy record, no status
    ])
    status = read_status(tmp)
    todo = [uid for uid in ("a#1", "a#2", "a#3", "a#4", "a#5")
            if status.get(uid) in (None, "error")]
    assert todo == ["a#5"]


def test_accepted_traces_dedupe_and_filter():
    tmp = _write_jsonl([
        {"uid": "f#1", "status": "accepted", "response": "old"},
        {"uid": "f#1", "status": "accepted", "response": "new"},
        {"uid": "f#2", "status": "exhausted"},
        {"uid": "f#3", "status": "error"},
    ])
    traces = accepted_traces([tmp])
    assert set(traces) == {"f#1"} and traces["f#1"]["response"] == "new"


def _row():
    return {
        "data_source": "initial_fingers",
        "prompt": [{"role": "system", "content": "SYS"},
                   {"role": "user", "content": "<video>\nWhich fingers?"}],
        "images": [],
        "videos": [{"video": "/data/v1.mp4", "min_frames": None, "max_frames": 12}],
        "reward_model": {"style": "rule", "ground_truth": "A,B"},
        "extra_info": json.dumps({"question_type": "initial_fingers", "index": 3}),
    }


def _trace(**kw):
    base = {"uid": "tactile_train#0", "family": "tactile_train", "row_index": 0,
            "ground_truth": "A,B", "response": GOOD_THINK + " \\boxed{A, B}",
            "status": "accepted", "model": "m", "mode": "answer_blind_zero_shot",
            "prompt_version": "abz-v1", "accepted_attempt": 2}
    base.update(kw)
    return base


def test_join_preserves_media_and_merges_system_turn():
    record = build_record(_row(), _trace())
    # Same video, but max_frames rewritten to the video_frames config value so
    # the student samples exactly what the teacher saw, and None-valued keys
    # dropped (an explicit min_frames=None crashes qwen_vl_utils frame sampling).
    assert record["videos"][0]["video"] == _row()["videos"][0]["video"]
    assert record["videos"][0]["max_frames"] == 8 and record["images"] == []
    assert "min_frames" not in record["videos"][0]
    assert [m["role"] for m in record["messages"]] == ["user", "assistant"]
    assert record["messages"][0]["content"].startswith("SYS\n\n<video>")
    assert record["messages"][-1]["content"].endswith("\\boxed{A, B}")
    info = json.loads(record["extra_info"])
    assert info["question_type"] == "initial_fingers"  # original extra_info kept
    assert info["uid"] == "tactile_train#0" and info["accepted_attempt"] == 2
    check_record(record, "tactile_train#0")


def test_join_refuses_ground_truth_mismatch():
    try:
        build_record(_row(), _trace(ground_truth="C"))
        raise RuntimeError("should have refused")
    except AssertionError:
        pass


def test_placeholder_media_count_check():
    record = build_record(_row(), _trace())
    record["videos"] = []
    try:
        check_record(record, "tactile_train#0")
        raise RuntimeError("should have failed")
    except AssertionError:
        pass


def test_open_gt_rows_train_on_ground_truth_text():
    row = {
        "data_source": "haptic_tactile",
        "prompt": [{"role": "user", "content": "<image>\nDescribe the recording."}],
        "images": [{"image": "/x/plot.png"}],
        "videos": [],
        "reward_model": {"style": "open", "ground_truth": "A gloved hand squeezes a soft ball."},
        "extra_info": "{}",
    }
    gt = row["reward_model"]["ground_truth"]
    record = build_record(row, {"uid": "haptic_ts_train#0", "ground_truth": gt, "response": gt})
    check_record(record, "haptic_ts_train#0")
    assert record["messages"][-1] == {"role": "assistant", "content": gt}


def test_minted_haptic_mcq_row():
    from mirl_ext.sft.scripts.make_haptic_mcq import mint_row

    tactile_row = {
        "data_source": "initial_fingers",
        "prompt": [{"role": "system", "content": "You are an expert in videos."},
                   {"role": "user", "content": "<video>\nWhich fingers touch first?\nOptions:\nA. Thumb\nB. Palm"}],
        "reward_model": {"style": "rule", "ground_truth": "A,B"},
        "extra_info": json.dumps({"video_path": "reencoded/visual-tactile/rec_idx0.mp4"}),
    }
    row = mint_row(tactile_row, "/scratch/x/plot.png")
    assert row["prompt"][1]["content"].startswith("<image>\nWhich fingers touch first?")
    assert "<video>" not in row["prompt"][1]["content"]
    assert row["images"] == [{"image": "/scratch/x/plot.png"}] and row["videos"] == []
    assert row["reward_model"]["ground_truth"] == "A,B"
    assert json.loads(row["extra_info"])["stem"] == "rec_idx0"
    # Minted rows must survive the standard build path.
    record = build_record(row, {"uid": "haptic_mcq_train#0", "ground_truth": "A,B",
                                "response": GOOD_THINK + " \\boxed{A, B}"})
    check_record(record, "haptic_mcq_train#0")


def test_sft_messages_without_system_is_untouched():
    row = {"prompt": [{"role": "user", "content": "<image>\nQ"}]}
    assert sft_messages(row) == [{"role": "user", "content": "<image>\nQ"}]


# ---- misc invariants ----

def test_position_video_grid_expansion():
    try:
        import torch
    except ImportError:
        return  # cluster-only dependency
    from mirl_ext.sft.sft_dataset import position_video_grid

    grid = torch.tensor([[4, 16, 16], [1, 8, 8]])
    out = position_video_grid(grid)
    assert out.tolist() == [[1, 16, 16]] * 4 + [[1, 8, 8]]


def test_staging_stems_deterministic():
    assert _stem("/a/b.mp4") == _stem("/a/b.mp4") != _stem("/a/c.mp4")



if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                import traceback

                print(f"FAIL {name}: {exc}")
                traceback.print_exc()
    sys.exit(1 if failures else 0)
