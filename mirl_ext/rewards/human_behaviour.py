"""Reward scoring for human behaviour prediction tasks.

Ground truth and predictions are single words/phrases (e.g. "anger", "happy").
Exact-match accuracy + token similarity + format compliance.

Training reward = acc + format_weight * format + sim_weight * similarity
"""

from difflib import SequenceMatcher

from ._common import extract_boxed_answer, format_reward


def acc_reward(pred_label: str, gt_label: str) -> float:
    return 1.0 if pred_label == gt_label else 0.0


def token_metrics(pred_label: str, gt_label: str) -> dict:
    """Token-level precision, recall, F1, Jaccard, and blended similarity."""
    pred_tokens = set(pred_label.split())
    gt_tokens = set(gt_label.split())

    if not pred_tokens and not gt_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "jaccard": 1.0, "similarity": 1.0}
    if not pred_tokens or not gt_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0, "similarity": 0.0}

    tp = len(pred_tokens & gt_tokens)
    precision = tp / len(pred_tokens)
    recall = tp / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    jaccard = len(pred_tokens & gt_tokens) / len(pred_tokens | gt_tokens)
    seq_ratio = SequenceMatcher(None, pred_label, gt_label).ratio()
    similarity = max(0.0, min(1.0, 0.5 * jaccard + 0.5 * seq_ratio))

    return {"precision": precision, "recall": recall, "f1": f1, "jaccard": jaccard, "similarity": similarity}


def compute_score(
    predict_str: str,
    ground_truth: str,
    format_weight: float = 0.2,
    sim_weight: float = 0.5,
) -> dict:
    boxed = extract_boxed_answer(predict_str)
    pred_label = (boxed or "").lower()
    gt_label = ground_truth.lower()

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
