"""CLIMB-QA reward: upstream MIRL's ``0.5 * condition-set F1`` term.

Source: DDVD233/mirl, commit 16860b932c0300ba1ce9cee7c7b7ce975abdd96d,
examples/reward_function/medical.py::medical_compute_score_batch.
Keep exact-set accuracy diagnostic; omit the upstream bbox/JSON auxiliaries
because this pipeline requests boxed condition answers, not localization.
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
    """Reward condition-set overlap from the last boxed answer; keep exact accuracy."""
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
        score=0.5 * f1,
        acc=acc,
        precision=precision,
        recall=recall,
        f1=f1,
        jacc=jacc,
        similarity=jacc,
        fmt=format_reward(predict_str),
    )
