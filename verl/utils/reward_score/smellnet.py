"""Reward scoring for SmellNet olfactory classification tasks.

Ground truth and predictions are substance names or mixture compositions.
Uses exact-match accuracy + token similarity + format compliance,
following the same pattern as human_behaviour.py.

Training reward = acc + 0.2 * format + 0.5 * similarity
"""

import re
from difflib import SequenceMatcher


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


def _normalize_label(label: str) -> str:
    """Normalize substance labels for comparison (lowercase, strip, collapse whitespace)."""
    label = label.lower().strip()
    label = re.sub(r"[_\-]+", " ", label)
    label = re.sub(r"\s+", " ", label)
    return label


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, predict_str) else 0.0


def acc_reward(pred_label: str, gt_label: str) -> float:
    return 1.0 if _normalize_label(pred_label) == _normalize_label(gt_label) else 0.0


def token_metrics(pred_label: str, gt_label: str) -> dict:
    pred_tokens = set(_normalize_label(pred_label).split())
    gt_tokens = set(_normalize_label(gt_label).split())

    if not pred_tokens and not gt_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "jaccard": 1.0, "similarity": 1.0}
    if not pred_tokens or not gt_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0, "similarity": 0.0}

    tp = len(pred_tokens & gt_tokens)
    precision = tp / len(pred_tokens)
    recall = tp / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    jaccard = len(pred_tokens & gt_tokens) / len(pred_tokens | gt_tokens)
    seq_ratio = SequenceMatcher(None, _normalize_label(pred_label), _normalize_label(gt_label)).ratio()
    similarity = max(0.0, min(1.0, 0.5 * jaccard + 0.5 * seq_ratio))

    return {"precision": precision, "recall": recall, "f1": f1, "jaccard": jaccard, "similarity": similarity}


def compute_score(
    predict_str: str,
    ground_truth: str,
    format_weight: float = 0.2,
    sim_weight: float = 0.5,
) -> dict:
    """Compute reward and evaluation metrics for SmellNet classification.

    Training reward = acc + format_weight * format + sim_weight * similarity
    """
    boxed = extract_boxed_answer(predict_str)
    pred_label = boxed or ""
    gt_label = ground_truth

    fmt = format_reward(predict_str)
    acc = acc_reward(pred_label, gt_label)
    metrics = token_metrics(pred_label, gt_label)

    score = acc + format_weight * fmt + sim_weight * metrics["similarity"]

    return {
        "score": score,
        "acc": acc,
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "jaccard": metrics["jaccard"],
        "similarity": metrics["similarity"],
        "format": fmt,
    }
