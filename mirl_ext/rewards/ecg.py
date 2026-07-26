"""Reward scoring for ECG pathology classification (PTB-XL superclasses).

The prompt asks the model to answer with one of a fixed set of categories
(e.g. "Normal", "Myocardial Infarction", "Atrial fibrillation/atrial flutter").

Training reward = acc_weight * acc + sim_weight * jaccard + format_weight * format

The format term (a <think>...</think>...\\boxed{...} requirement, identical to the
smellnet / medical / human_behaviour scorers) is NOT cosmetic here: ECG was the only
task in the suite with format_weight=0.0, and it was the only task that collapsed to a
constant majority-class answer ("Normal" for 100% of samples by step 150). The reason
is a GRPO interaction -- advantages are standardized within each prompt's rollout group
(A_i = (r_i - mean)/std), and every rollout of one ECG prompt shares the same label, so
a reward that lets a bare one-token "Normal" earn full credit lets the policy shrink its
output to a single token with ZERO intra-group diversity => std 0 => advantage 0 => no
gradient to ever escape. The other skewed tasks (meld/mosei emotion+sentiment, majority
~0.47-0.61) did NOT collapse precisely because their format term forces <think> reasoning,
which keeps rollouts diverse and the learning signal alive. Requiring the same format here
restores that diversity. (Note: per-class reward *weighting* would NOT help -- any
scalar that is constant within a group cancels in the std normalization.)

Returns a dict matching the other reward_score modules.
"""

import re

CATEGORIES = [
    "Normal",
    "Conduction Disturbance",
    "Myocardial Infarction",
    "Ischemic ST-T Changes",
    "Other",
    "Atrial fibrillation/atrial flutter",
    "Hypertrophy",
]


def extract_boxed_answer(predict_str: str) -> str | None:
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
    """Check for <think>...</think>...\\boxed{...} format (matches sibling scorers)."""
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, predict_str) else 0.0


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _word_set(s: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", s.strip().lower()) if w}


def _f1(pred: set[str], gt: set[str]) -> tuple[float, float, float]:
    if not pred and not gt:
        return 1.0, 1.0, 1.0
    if not pred or not gt:
        return 0.0, 0.0, 0.0
    tp = len(pred & gt)
    precision = tp / len(pred)
    recall = tp / len(gt)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def _jaccard(pred: set[str], gt: set[str]) -> float:
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    return len(pred & gt) / len(pred | gt)


def _predicted_category(text: str) -> str | None:
    """Pick the ground-truth-style category mentioned in the response, if any.

    Prefer the last occurrence (final answer) and the longest matching category
    so that e.g. "Atrial fibrillation/atrial flutter" wins over a stray "Other".
    """
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
    precision, recall, f1 = _f1(pred_words, gt_words)
    jacc = _jaccard(pred_words, gt_words)
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

