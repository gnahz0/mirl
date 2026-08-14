"""Reward scoring for tactile video QA (multi-choice, possibly multi-label).

Ground truth and predictions are comma-separated uppercase letters (e.g. "A", "B,C,D"),
compared order-agnostically via set operations.

Training reward = acc_weight * acc + sim_weight * jaccard + format_weight * format
"""

from ._common import extract_boxed_answer, format_reward, jaccard, set_prf1


def parse_labels(answer_str: str) -> set[str]:
    """Parse a comma-separated answer string into a set of normalized labels."""
    if not answer_str:
        return set()
    return {label.strip().upper() for label in answer_str.split(",") if label.strip()}


def acc_reward(pred_labels: set[str], gt_labels: set[str]) -> float:
    return 1.0 if pred_labels == gt_labels else 0.0


def compute_score(
    predict_str: str,
    ground_truth: str,
    acc_weight: float = 0.5,
    sim_weight: float = 0.4,
    format_weight: float = 0.1,
) -> dict:
    boxed = extract_boxed_answer(predict_str)
    pred_labels = parse_labels(boxed)
    gt_labels = parse_labels(ground_truth)

    fmt = format_reward(predict_str)
    acc = acc_reward(pred_labels, gt_labels)
    jacc = jaccard(pred_labels, gt_labels)
    precision, recall, f1 = set_prf1(pred_labels, gt_labels)

    score = acc_weight * acc + sim_weight * jacc + format_weight * fmt

    return {
        "score": score,
        "acc": acc,
        "jaccard": jacc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "similarity": jacc,
        "format": fmt,
    }
