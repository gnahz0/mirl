"""Reward scoring for SmellNet e-nose classification.

The prompt asks for a single substance label (e.g. "allspice", "brazil_nut")
chosen from a fixed list, with <think> reasoning and the answer in \\boxed{}.

Training reward = acc_weight * acc + sim_weight * jaccard + format_weight * format

sim_weight defaults to 0.0 by design: base labels have no meaningful partial
credit, and token overlap would reward shared substrings (brazil_nut vs pili_nut).
"""

import re

from ._common import extract_boxed_answer, format_reward, jaccard, score_dict, set_prf1


def _norm_label(s: str) -> str:
    """Lowercase and collapse spaces/hyphens to underscores (allspice, brazil_nut)."""
    return re.sub(r"[\s\-]+", "_", s.strip().lower()).strip("_")


def _word_set(s: str) -> set[str]:
    return {w for w in re.split(r"[\s_]+", s.strip().lower()) if w}


def compute_score(
    predict_str: str,
    ground_truth: str,
    acc_weight: float = 0.8,
    sim_weight: float = 0.0,
    format_weight: float = 0.2,
) -> dict:
    boxed = extract_boxed_answer(predict_str)
    pred_raw = boxed if boxed is not None else predict_str
    pred_label = _norm_label(pred_raw)
    gt_label = _norm_label(ground_truth)

    fmt = format_reward(predict_str)
    acc = 1.0 if pred_label == gt_label else 0.0
    pred_words, gt_words = _word_set(pred_raw), _word_set(ground_truth)
    precision, recall, f1 = set_prf1(pred_words, gt_words)
    jacc = jaccard(pred_words, gt_words)

    score = acc_weight * acc + sim_weight * jacc + format_weight * fmt

    return score_dict(
        score=score, acc=acc, precision=precision, recall=recall,
        f1=f1, jacc=jacc, similarity=jacc, fmt=fmt,
    )
