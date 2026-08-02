"""Reward scoring for haptic time-series open-ended description.

The prompt asks the model to describe the video; the ground truth is a free-text
description (style="open"). With no closed label, the reward is token-overlap F1
plus a sequence-similarity blend for the logged "similarity" metric.

Training reward = f1_weight * token_f1 + sim_weight * similarity
"""

import re
from difflib import SequenceMatcher

from ._common import extract_boxed_answer, jaccard, set_prf1

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "for", "with", "and", "or", "by", "from",
    "as", "that", "this", "it", "its", "into", "over", "under",
}


def _strip_think(text: str) -> str:
    """Drop a <think>...</think> block so we score the actual description."""
    return re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)


def _tokens(s: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", s.lower()) if w and w not in _STOP]


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

    precision, recall, f1 = set_prf1(set(pred_tokens), set(gt_tokens))
    jacc = jaccard(set(pred_tokens), set(gt_tokens))
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
