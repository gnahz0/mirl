"""Exact seven-class ECG correctness for MIRL's CLIMB dataset adaptation.

CLIMB's seven unified classes, not original ECG-JEPA's five-class multilabel task:
https://github.com/DDVD233/CLIMB/blob/0f767b1ea168810b998981078dc27e7f9a4e4675/src/datasets/ecg/ptbxl.py#L35
This is a paper-metric-aligned RL fallback, not a published ECG-JEPA RL recipe.
Only exact correctness contributes reward; other metrics are diagnostic.
"""

from ._common import extract_boxed_answer, format_reward, jaccard, score_dict, set_prf1

CATEGORIES = [
    "Normal",
    "Conduction Disturbance",
    "Myocardial Infarction",
    "Ischemic ST-T Changes",
    "Other",
    "Atrial fibrillation/atrial flutter",
    "Hypertrophy",
]


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _predicted_category(text: str) -> str | None:
    """Accept exactly one canonical category, allowing only case/whitespace changes."""
    normalized = _norm(text)
    return next((category for category in CATEGORIES if _norm(category) == normalized), None)


def compute_score(predict_str: str, ground_truth: str) -> dict:
    """Score the last boxed answer, or an exact bare category when no box exists."""
    if not isinstance(ground_truth, str):
        raise ValueError("ECG ground truth must be one canonical category.")
    gt_category = _predicted_category(ground_truth)
    if gt_category is None:
        raise ValueError(f"Invalid ECG ground truth category: {ground_truth!r}")

    boxed = extract_boxed_answer(predict_str)
    pred_category = _predicted_category(boxed if boxed is not None else predict_str)
    pred_labels = {pred_category} if pred_category is not None else set()
    gt_labels = {gt_category}
    acc = float(pred_labels == gt_labels)
    precision, recall, f1 = set_prf1(pred_labels, gt_labels)
    jacc = jaccard(pred_labels, gt_labels)

    return score_dict(
        score=acc,
        acc=acc,
        precision=precision,
        recall=recall,
        f1=f1,
        jacc=jacc,
        similarity=jacc,
        fmt=format_reward(predict_str),
    )
