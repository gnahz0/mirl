"""Shared scoring primitives for the mirl reward modules: boxed-answer
extraction, the ``<think>…</think>…\\boxed{}`` format check, and set-based
precision/recall/F1 + Jaccard."""

import re

_FORMAT_RE = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)


def extract_boxed_answer(predict_str: str) -> str | None:
    """Content inside the last ``\\boxed{…}`` in the prediction, or None.

    Delegates to verl's math_dapo helpers (NOT math_reward's variant, whose
    extra ``\\boxed `` space-form branch has different semantics). Lazy import:
    the rewards package stays stdlib-only at import time for the light SFT
    scripts; the first call pulls verl's full stack."""
    from verl.utils.reward_score.math_dapo import last_boxed_only_string, remove_boxed

    boxed = last_boxed_only_string(predict_str)
    return remove_boxed(boxed) if boxed is not None else None


def format_reward(predict_str: str) -> float:
    """1.0 iff the output is a full ``<think>…</think>…\\boxed{…}`` string."""
    return 1.0 if re.fullmatch(_FORMAT_RE, predict_str) else 0.0


def score_dict(score, acc, precision, recall, f1, jacc, similarity, fmt) -> dict:
    """The 8-key metrics dict every scorer returns (score formulas stay inline
    at the call sites; only the shared return literal lives here)."""
    return {
        "score": score,
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "jaccard": jacc,
        "similarity": similarity,
        "format": fmt,
    }


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
