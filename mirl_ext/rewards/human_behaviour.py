"""Human Behavior Atlas's closed-classification reward: 0.8 exact + 0.2 format.

Implements only the CLS scoring semantics from HBA's HARPO reward, not HARPO's
advantage estimator, QA embedding reward, or inner overlength penalty. veRL's
DAPO reward manager handles response-token length separately, exactly once.

Source: MIT-MI/human_behavior_atlas, commit 9cd5ced80243eb02d85a8aa49a3f373544d9d3a3
https://github.com/MIT-MI/human_behavior_atlas/blob/9cd5ced80243eb02d85a8aa49a3f373544d9d3a3/training/rl/reward_function/human_behaviour_harpo.py

Precision/recall/F1/Jaccard compare whole categorical labels, not their words.
They are per-response diagnostics, not dataset-level macro or weighted F1.
``similarity`` aliases label-set Jaccard for the shared metric interface; it
does not contribute an additional shaping term.
"""

import re

from ._common import format_reward, jaccard, score_dict, set_prf1


def compute_score(predict_str: str, ground_truth: str) -> dict:
    response = re.sub(r"\s*(<|>|/)\s*", r"\1", predict_str or "")
    # Match HBA's first-match priority, including its unboxed-text fallback.
    matches = (
        re.search(pattern, response) for pattern in (r"\\boxed{([^}]*)}", r"\[(.*?)\]", r"<answer>(.*?)</answer>")
    )
    pred_label = next((match.group(1) for match in matches if match), response).strip().lower()
    gt_label = (ground_truth or "").strip().lower()

    fmt = format_reward(response)
    acc = float(pred_label == gt_label)
    pred_labels = {pred_label} if pred_label else set()
    gt_labels = {gt_label} if gt_label else set()
    precision, recall, f1 = set_prf1(pred_labels, gt_labels)
    jacc = jaccard(pred_labels, gt_labels)

    return score_dict(
        score=0.8 * acc + 0.2 * fmt,
        acc=acc,
        precision=precision,
        recall=recall,
        f1=f1,
        jacc=jacc,
        similarity=jacc,
        fmt=fmt,
    )
