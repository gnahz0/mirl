"""Reward scoring for ECG pathology classification (PTB-XL superclasses).

The prompt asks the model to answer with one of a fixed set of categories
(e.g. "Normal", "Myocardial Infarction", "Atrial fibrillation/atrial flutter").

Training reward = acc_weight * acc + sim_weight * jaccard + format_weight * format

The format term is NOT cosmetic here: ECG was the only task run with
format_weight=0.0, and the only one that collapsed to a constant majority-class
answer ("Normal" for 100% of samples by step 150). Under GRPO, advantages are
standardized within each prompt's rollout group (A_i = (r_i - mean)/std), and
every rollout of one ECG prompt shares the same label -- so a reward that lets a
bare one-token "Normal" earn full credit lets the policy shrink to a single token
with zero intra-group diversity => std 0 => advantage 0 => no gradient to escape.
Forcing <think> reasoning via the format term restores rollout diversity. (Per-class
reward *weighting* would NOT help -- anything constant within a group cancels in
the std normalization.)
"""

import re

from ._common import extract_boxed_answer, format_reward, jaccard, set_prf1

CATEGORIES = [
    "Normal",
    "Conduction Disturbance",
    "Myocardial Infarction",
    "Ischemic ST-T Changes",
    "Other",
    "Atrial fibrillation/atrial flutter",
    "Hypertrophy",
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _word_set(s: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", s.strip().lower()) if w}


def _predicted_category(text: str) -> str | None:
    """Category mentioned in the response, preferring the last occurrence and the
    longest match (so "Atrial fibrillation/atrial flutter" wins over a stray "Other")."""
    norm = _norm(text)
    best, best_pos = None, -1
    for cat in sorted(CATEGORIES, key=len, reverse=True):
        pos = norm.rfind(_norm(cat))
        if pos >= 0 and (best is None or pos > best_pos):
            best, best_pos = cat, pos
    return best


def compute_score(
    predict_str: str,
    ground_truth: str,
    acc_weight: float = 0.7,
    sim_weight: float = 0.1,
    format_weight: float = 0.2,
) -> dict:
    boxed = extract_boxed_answer(predict_str)
    search_space = boxed if boxed is not None else predict_str
    pred_cat = _predicted_category(search_space)

    acc = 1.0 if pred_cat is not None and _norm(pred_cat) == _norm(ground_truth) else 0.0

    pred_words = _word_set(pred_cat) if pred_cat else _word_set(search_space)
    gt_words = _word_set(ground_truth)
    precision, recall, f1 = set_prf1(pred_words, gt_words)
    jacc = jaccard(pred_words, gt_words)
    fmt = format_reward(predict_str)

    score = acc_weight * acc + sim_weight * jacc + format_weight * fmt

    return {
        "score": score,
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "jaccard": jacc,
        "similarity": jacc,
        "format": fmt,
    }
