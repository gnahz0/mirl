"""Reward scoring for human behaviour prediction tasks.

Ground truth and predictions are single words/phrases (e.g. "anger", "happy").
Exact-match accuracy + token similarity + format compliance.

Training reward = acc + format_weight * format + sim_weight * similarity
"""

from difflib import SequenceMatcher

from ._common import extract_boxed_answer, format_reward, jaccard, score_dict, set_prf1


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
    acc = 1.0 if pred_label == gt_label else 0.0
    pred_tokens, gt_tokens = set(pred_label.split()), set(gt_label.split())
    precision, recall, f1 = set_prf1(pred_tokens, gt_tokens)
    jacc = jaccard(pred_tokens, gt_tokens)
    similarity = 0.5 * jacc + 0.5 * SequenceMatcher(None, pred_label, gt_label).ratio()

    score = acc + format_weight * fmt + sim_weight * similarity

    return score_dict(
        score=score, acc=acc, precision=precision, recall=recall,
        f1=f1, jacc=jacc, similarity=similarity, fmt=fmt,
    )
