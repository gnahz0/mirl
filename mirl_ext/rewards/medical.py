"""Reward scoring for medical/CLIMB tasks (condition diagnosis).

Ground truth and predictions are comma-separated condition phrases
(e.g. "Pleural Effusion, Support Devices"). F1 over condition sets +
token similarity + format compliance.

Training reward = f1_weight * f1 + sim_weight * jaccard + format_weight * format
"""

from ._common import extract_boxed_answer, format_reward, jaccard, score_dict, set_prf1


def parse_conditions(text: str) -> set[str]:
    """Parse a multi-condition answer string into a normalized set."""
    text = text.strip().lower()
    for sep in [", ", " and ", " & ", ",", "&"]:
        if sep in text:
            return {c.strip() for c in text.split(sep) if c.strip()}
    return {text} if text else set()


def compute_score(
    predict_str: str,
    ground_truth: str,
    f1_weight: float = 0.5,
    sim_weight: float = 0.3,
    format_weight: float = 0.2,
) -> dict:
    boxed = extract_boxed_answer(predict_str)
    pred_text = (boxed or "").lower()
    gt_text = ground_truth.lower()

    pred_conditions = parse_conditions(pred_text)
    gt_conditions = parse_conditions(gt_text)

    fmt = format_reward(predict_str)
    acc = 1.0 if pred_conditions == gt_conditions else 0.0
    precision, recall, f1 = set_prf1(pred_conditions, gt_conditions)
    jacc = jaccard(set(pred_text.split()), set(gt_text.split()))

    score = f1_weight * f1 + sim_weight * jacc + format_weight * fmt

    return score_dict(
        score=score, acc=acc, precision=precision, recall=recall,
        f1=f1, jacc=jacc, similarity=jacc, fmt=fmt,
    )
