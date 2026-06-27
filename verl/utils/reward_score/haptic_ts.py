"""Reward scoring for haptic time-series open-ended description.

The prompt asks the model to "describe the video in a few sentences" and the
ground truth is a free-text description (style="open"). There is no closed label
to match, so the reward is token-overlap F1 between the prediction and the
reference (with a sequence-similarity blend for the logged "similarity" metric).

Training reward = f1_weight * token_f1 + sim_weight * similarity

Returns a dict matching the other reward_score modules.
"""

import re
from difflib import SequenceMatcher

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "for", "with", "and", "or", "by", "from",
    "as", "that", "this", "it", "its", "into", "over", "under",
}


def extract_boxed_answer(predict_str: str) -> str | None:
    idx = predict_str.rfind("\\boxed{")
    if idx < 0:
        return None
    depth = 0
    i = idx + len("\\boxed{") - 1
    while i < len(predict_str):
        if predict_str[i] == "{":
            depth += 1
        elif predict_str[i] == "}":
            depth -= 1
            if depth == 0:
                return predict_str[idx + len("\\boxed{"):i]
        i += 1
    return None


def _strip_think(text: str) -> str:
    """Drop a <think>...</think> block so we score the actual description."""
    return re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)


def _tokens(s: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", s.lower()) if w and w not in _STOP]


def _token_f1(pred: list[str], gt: list[str]) -> tuple[float, float, float]:
    if not pred and not gt:
        return 1.0, 1.0, 1.0
    if not pred or not gt:
        return 0.0, 0.0, 0.0
    pred_set, gt_set = set(pred), set(gt)
    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set)
    recall = tp / len(gt_set)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def _jaccard(pred: set[str], gt: set[str]) -> float:
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    return len(pred & gt) / len(pred | gt)


def compute_score(
    predict_str: str,
    ground_truth: str,
    f1_weight: float = 0.7,
    sim_weight: float = 0.3,
) -> dict:
    boxed = extract_boxed_answer(predict_str)
    pred_text = boxed if boxed is not None else _strip_think(predict_str)

    pred_tokens = _tokens(pred_text)
    gt_tokens = _tokens(ground_truth)

    precision, recall, f1 = _token_f1(pred_tokens, gt_tokens)
    jacc = _jaccard(set(pred_tokens), set(gt_tokens))
    seq = SequenceMatcher(None, " ".join(pred_tokens), " ".join(gt_tokens)).ratio()
    similarity = 0.5 * jacc + 0.5 * seq

    acc = 1.0 if pred_text.strip().lower() == ground_truth.strip().lower() else 0.0
    score = f1_weight * f1 + sim_weight * similarity

    return {
        "score": score,
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "jaccard": jacc,
        "similarity": similarity,
        "format": 0.0,
    }
