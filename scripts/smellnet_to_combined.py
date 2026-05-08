"""
Convert SmellNet JSONL (from smellnet_build_json.py) to the combined dataset format
used by combined_qwen3_training_mini.sh.

Usage:
    python scripts/smellnet_to_combined.py \
        --input /home/alecz/scratch/alecz/SmellNet_subplot/all_train.jsonl \
        --output /home/alecz/mirl/data/smellnet_train.json

    python scripts/smellnet_to_combined.py \
        --input /home/alecz/scratch/alecz/SmellNet_subplot/all_test.jsonl \
        --output /home/alecz/mirl/data/smellnet_valid.json
"""

import argparse
import json
import os

BASE_SUBSTANCES = [
    "allspice", "almond", "angelica", "apple", "asparagus", "avocado",
    "banana", "brazil_nut", "broccoli", "brussel_sprouts", "cabbage",
    "cashew", "cauliflower", "chamomile", "chervil", "chestnuts", "chives",
    "cinnamon", "cloves", "coriander", "cumin", "dill", "garlic", "ginger",
    "hazelnut", "kiwi", "lemon", "mandarin_orange", "mango", "mint",
    "mugwort", "mustard", "nutmeg", "oregano", "peach", "peanuts", "pear",
    "pecans", "pili_nut", "pineapple", "pistachios", "potato", "radish",
    "saffron", "star_anise", "strawberry", "sweet_potato", "tomato",
    "turnip", "walnuts",
]

SYSTEM_PROMPT = (
    "You are an expert in analyzing sensor data from electronic noses (e-nose). "
    "The provided image shows time-series readings from multiple gas sensors "
    "(each subplot is a different sensor channel). "
    "Examine the sensor response patterns carefully and answer the question. "
    "You FIRST think about the reasoning process as an internal monologue "
    "and then provide the final answer. The reasoning process MUST BE "
    "enclosed within <think> </think> tags. The final answer MUST BE "
    "wrapped in \\boxed{}."
)


def make_base_entry(raw: dict, index: int) -> dict:
    label = raw["label"]
    options = ", ".join(BASE_SUBSTANCES)
    user_content = (
        f"<image>\n"
        f"The above image shows gas sensor readings from an electronic nose "
        f"exposed to a substance. Each subplot represents a different sensor channel "
        f"(NO2, C2H5OH, VOC, CO, Alcohol, LPG) over time.\n"
        f"What substance is the sensor detecting? "
        f"Answer with one word from the following: {options}"
    )
    return {
        "data_source": "smellnet_base",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "images": [{"image": raw["image"]}],
        "videos": [],
        "audios": [],
        "reward_model": {"style": "rule", "ground_truth": label},
        "extra_info": json.dumps({
            "index": index,
            "source_dataset": "smellnet",
            "dataset": "smellnet_base",
            "data_source": "smellnet_base",
        }),
    }


def make_mixture_entry(raw: dict, index: int) -> dict:
    label = raw["label"]
    user_content = (
        f"<image>\n"
        f"The above image shows gas sensor readings from an electronic nose "
        f"exposed to a mixture of substances. Each subplot represents a different "
        f"sensor channel (NO2, C2H5CH, VOC, CO) over time.\n"
        f"Identify the mixture composition. "
        f"Provide the substance names and their approximate percentages if visible "
        f"from the sensor response pattern (e.g. 'Almond20_Orange80')."
    )
    return {
        "data_source": "smellnet_mixture",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "images": [{"image": raw["image"]}],
        "videos": [],
        "audios": [],
        "reward_model": {"style": "rule", "ground_truth": label},
        "extra_info": json.dumps({
            "index": index,
            "source_dataset": "smellnet",
            "dataset": "smellnet_mixture",
            "data_source": "smellnet_mixture",
        }),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    entries = []
    with open(args.input) as f:
        for i, line in enumerate(f):
            raw = json.loads(line)
            if raw["task"] == "base":
                entries.append(make_base_entry(raw, i))
            else:
                entries.append(make_mixture_entry(raw, i))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Wrote {len(entries)} entries -> {args.output}")


if __name__ == "__main__":
    main()
