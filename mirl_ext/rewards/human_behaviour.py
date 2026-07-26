"""Reward scoring for human behaviour prediction tasks.

Ground truth and predictions are single words/phrases (e.g. "anger", "happy").
Uses exact-match accuracy + token similarity + format compliance.

Returns a dict so the reward manager can log evaluation-only metrics
alongside the training reward.

Training reward = acc + 0.2 * format + 0.5 * similarity
"""

import re
from difflib import SequenceMatcher


def extract_boxed_answer(predict_str: str) -> str | None:
    """Extract the content inside the last \\boxed{...} in the prediction."""
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


def format_reward(predict_str: str) -> float:
    """Check for <think>...</think>...\\boxed{...} format."""
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, predict_str) else 0.0


def acc_reward(pred_label: str, gt_label: str) -> float:
    """Exact-match accuracy on lowercased labels."""
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
    """Compute reward and evaluation metrics for human behaviour prediction.

    Training reward = acc + format_weight * format + sim_weight * similarity

    Returns:
        dict with keys:
            score: float      - the training reward
            acc: float        - exact-match accuracy (0 or 1)
            f1: float         - token-level F1
            precision: float  - token-level precision
            recall: float     - token-level recall
            jaccard: float    - token-level Jaccard similarity
            similarity: float - blended token Jaccard + sequence similarity
            format: float     - format compliance (0 or 1)
    """
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

