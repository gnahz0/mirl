"""Exact condition-set correctness for the MIRL adaptation of CLIMB-QA.

CLIMB Appendix B defines order-agnostic equality of predicted and target sets:
https://arxiv.org/html/2503.07667v1
This is a paper-metric-aligned RL fallback, not a published CLIMB RL recipe.
Only exact correctness contributes reward; other metrics are diagnostic.
"""

import re

from ._common import extract_boxed_answer, format_reward, jaccard, score_dict, set_prf1

_CONDITION_SEPARATOR = re.compile(r",|&|\band\b")


def parse_conditions(text: str) -> set[str]:
    """Normalize condition phrases across case, whitespace, and list separators."""
    return {
        normalized
        for condition in _CONDITION_SEPARATOR.split(text.lower())
        if (normalized := " ".join(condition.split()))
    }


def compute_score(predict_str: str, ground_truth: str) -> dict:
    """Reward an exact, nonempty condition set from the last boxed answer."""
    if not isinstance(ground_truth, str):
        raise ValueError("Medical ground truth must be a nonempty condition string.")
    gt_conditions = parse_conditions(ground_truth)
    if not gt_conditions:
        raise ValueError("Medical ground truth must contain at least one condition.")

    pred_conditions = parse_conditions(extract_boxed_answer(predict_str) or "")
    acc = float(pred_conditions == gt_conditions)
    precision, recall, f1 = set_prf1(pred_conditions, gt_conditions)
    jacc = jaccard(pred_conditions, gt_conditions)

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
