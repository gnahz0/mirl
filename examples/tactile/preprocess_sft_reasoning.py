"""Convert annotation_verl_reasoning.json (JSONL) to parquet for verl SFT training.

Produces:
  - Parquet files with `messages` + `videos` columns for MultiTurnSFTDataset
  - RL-format JSONL test files (no reasoning) for generation-based evaluation

Split strategies (identical to convert_to_verl.py):
  - date:        group by date, random 80/20
  - participant:  group by participant, random 80/20
  - glove:        v1=train, v2=test
  - question:     Q17 and earlier=train, Q18+=test
  - task:         {actuation,push,pour,insert}=test, rest=train
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants (must match convert_to_verl.py for identical splits)
# ---------------------------------------------------------------------------
SEED = 42
TEST_RATIO = 0.2
QUESTION_SPLIT_CUTOFF = 17
TASK_TEST_SET = {"actuation", "push", "pour", "insert"}

QUESTION_NUMBER = {
    "initial_fingers": 1,
    "highest_pressure": 2,
    "more_deformable": 4,
    "deformation_type": 5,
    "objA_texture": 6,
    "objB_texture": 8,
    "grasp_location": 10,
    "contact_feature": 16,
    "local_shape": 17,
    "grip_stability": 18,
    "future_stability": 19,
    "force_level": 20,
    "shear_direction": 26,
    "object_motion": 27,
    "fail_reason": 30,
    "fail_improvement": 31,
}


# ---------------------------------------------------------------------------
# Split helpers (same logic as convert_to_verl.py)
# ---------------------------------------------------------------------------
def split_items_random(items, key_fn, test_ratio, rng):
    """Group items by key_fn, randomly assign groups to train/test."""
    groups = {}
    for item in items:
        k = key_fn(item)
        groups.setdefault(k, []).append(item)

    group_keys = sorted(groups.keys())
    rng.shuffle(group_keys)

    total = len(items)
    test_target = int(total * test_ratio)
    test_keys = set()
    test_count = 0
    for k in group_keys:
        if test_count >= test_target:
            break
        test_keys.add(k)
        test_count += len(groups[k])

    return {k: ("test" if k in test_keys else "train") for k in groups}


def first_word_task(task):
    if not task:
        return "unknown"
    return task.split("_")[0].lower()


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------
def read_reasoning_jsonl(path):
    """Read JSONL file, filter to reasoning_status == 'ok'."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("reasoning_status") != "ok":
                continue
            samples.append(d)
    return samples


def to_sft_record(sample):
    """Convert a reasoning sample to SFT parquet record (messages + videos)."""
    prompt_messages = sample["prompt"]  # [system, user]
    reasoning = sample["reasoning"]

    messages = list(prompt_messages) + [{"role": "assistant", "content": reasoning}]

    videos = sample.get("videos", [])

    return {"messages": messages, "videos": videos}


def to_rl_record(sample):
    """Convert a reasoning sample to RL-format JSONL record (no reasoning)."""
    return {
        "data_source": sample["data_source"],
        "prompt": sample["prompt"],
        "videos": sample.get("videos", []),
        "reward_model": sample["reward_model"],
        "extra_info": sample["extra_info"],
    }


def save_parquet(records, path):
    df = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(path), index=False)
    print(f"  {path.name}: {len(records)} samples")


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {path.name}: {len(records)} samples")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Preprocess reasoning JSONL → parquet for SFT + JSONL for eval")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "scripts" / "annotation_verl_reasoning.json"),
        help="Path to annotation_verl_reasoning.json (JSONL)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as input file directory)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent

    # Read and filter
    samples = read_reasoning_jsonl(input_path)
    print(f"Loaded {len(samples)} valid reasoning samples from {input_path}")

    rng = random.Random(SEED)

    # Extract extra_info for splitting
    def get_date(s):
        return s["extra_info"].get("date") or "unknown"

    def get_participant(s):
        return (s["extra_info"].get("participant") or "unknown").lower()

    def get_glove(s):
        return s["extra_info"].get("glove_type", "v1")

    def get_question_type(s):
        return s["extra_info"].get("question_type", "unknown")

    def get_task_first_word(s):
        return first_word_task(s["extra_info"].get("task"))

    # ---------- Compute splits ----------
    date_split = split_items_random(samples, get_date, TEST_RATIO, rng)
    participant_split = split_items_random(samples, get_participant, TEST_RATIO, rng)
    glove_split = {"v1": "train", "v2": "test"}

    # ---------- Write all-samples file ----------
    all_sft = [to_sft_record(s) for s in samples]
    save_parquet(all_sft, output_dir / "annotation_verl_reasoning_sft.parquet")

    # ---------- Helper to write a split ----------
    def write_split(name, split_fn):
        """Write train/test parquet + RL JSONL test + mini files for a split."""
        train_samples = [s for s in samples if split_fn(s) == "train"]
        test_samples = [s for s in samples if split_fn(s) == "test"]

        train_sft = [to_sft_record(s) for s in train_samples]
        test_sft = [to_sft_record(s) for s in test_samples]

        prefix = f"annotation_verl_reasoning_sft_split_{name}"
        save_parquet(train_sft, output_dir / f"{prefix}_train.parquet")
        save_parquet(test_sft, output_dir / f"{prefix}_test.parquet")

        # Mini test (10% of test)
        if test_samples:
            mini_samples = rng.sample(test_samples, max(1, len(test_samples) // 10))
            mini_sft = [to_sft_record(s) for s in mini_samples]
            save_parquet(mini_sft, output_dir / f"{prefix}_test_mini.parquet")
        else:
            mini_samples = []

        # RL-format JSONL for eval (test only)
        rl_prefix = f"annotation_verl_reasoning_split_{name}"
        test_rl = [to_rl_record(s) for s in test_samples]
        write_jsonl(test_rl, output_dir / f"{rl_prefix}_test.json")

        if mini_samples:
            mini_rl = [to_rl_record(s) for s in mini_samples]
            write_jsonl(mini_rl, output_dir / f"{rl_prefix}_test_mini.json")

        print(f"  Split '{name}': {len(train_samples)} train, {len(test_samples)} test, {len(mini_samples)} mini")

    # ---------- Date split ----------
    print("\nDate split:")
    write_split("date", lambda s: date_split.get(get_date(s), "train"))

    # ---------- Participant split ----------
    print("\nParticipant split:")
    write_split("participant", lambda s: participant_split.get(get_participant(s), "train"))

    # ---------- Glove split ----------
    print("\nGlove split:")
    write_split("glove", lambda s: glove_split.get(get_glove(s), "train"))

    # ---------- Question split ----------
    print("\nQuestion split:")

    def question_split_fn(s):
        q_type = get_question_type(s)
        q_num = QUESTION_NUMBER.get(q_type, 999)
        return "train" if q_num <= QUESTION_SPLIT_CUTOFF else "test"

    write_split("question", question_split_fn)

    # ---------- Task split ----------
    print("\nTask split:")

    def task_split_fn(s):
        fw = get_task_first_word(s)
        return "test" if fw in TASK_TEST_SET else "train"

    write_split("task", task_split_fn)

    print(f"\nDone. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
