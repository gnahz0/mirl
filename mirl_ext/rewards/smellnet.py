"""Reward scoring for SmellNet e-nose classification.

The prompt asks for a single substance label (e.g. "allspice", "brazil_nut")
chosen from a fixed list, with reasoning in <think>...</think> and the final
answer wrapped in \\boxed{}.

Training reward = acc_weight * acc + sim_weight * jaccard + format_weight * format

Returns a dict matching the other reward_score modules so the reward manager can
aggregate the same keys across a mixed batch.
"""

import re


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


def _norm_label(s: str) -> str:
    """Lowercase and collapse spaces/hyphens to underscores (allspice, brazil_nut)."""
    return re.sub(r"[\s\-]+", "_", s.strip().lower()).strip("_")


def _word_set(s: str) -> set[str]:
    return {w for w in re.split(r"[\s_]+", s.strip().lower()) if w}


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


def compute_score(
    predict_str: str,
    ground_truth: str,
    acc_weight: float = 0.8,
    sim_weight: float = 0.0,
    format_weight: float = 0.2,
) -> dict:
    boxed = extract_boxed_answer(predict_str)
    # Fall back to the whole response if the model omitted \boxed{}.
    pred_raw = boxed if boxed is not None else predict_str
    pred_label = _norm_label(pred_raw)
    gt_label = _norm_label(ground_truth)

    fmt = format_reward(predict_str)
    acc = 1.0 if pred_label == gt_label else 0.0
    pred_words, gt_words = _word_set(pred_raw), _word_set(ground_truth)
    precision, recall, f1 = _f1(pred_words, gt_words)
    jacc = _jaccard(pred_words, gt_words)

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

