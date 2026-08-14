"""Shared scoring primitives for the mirl reward modules: boxed-answer
extraction, the ``<think>…</think>…\\boxed{}`` format check, and set-based
precision/recall/F1 + Jaccard."""

import re

_FORMAT_RE = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)


def extract_boxed_answer(predict_str: str) -> str | None:
    """Content inside the last ``\\boxed{…}`` in the prediction, or None."""
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
    """1.0 iff the output is a full ``<think>…</think>…\\boxed{…}`` string."""
    return 1.0 if re.fullmatch(_FORMAT_RE, predict_str) else 0.0


def set_prf1(pred: set, gt: set) -> tuple[float, float, float]:
    """Precision, recall, F1 over two sets. Both empty → 1s; one empty → 0s."""
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


def jaccard(pred: set, gt: set) -> float:
    """|∩| / |∪| over two sets. Both empty → 1.0; one empty → 0.0."""
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    return len(pred & gt) / len(pred | gt)
