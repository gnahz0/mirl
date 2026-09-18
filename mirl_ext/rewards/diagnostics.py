"""Detached, per-source diagnostics for single-output GRPO reward groups."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence


def masked_reward_sums(rewards, response_mask) -> list[float]:
    """Reduce detached token tensors before transferring small row totals to CPU."""
    if rewards.ndim != 2 or rewards.shape != response_mask.shape:
        raise ValueError("reward and response-mask tensors must have the same two-dimensional shape")
    if not ((response_mask == 0) | (response_mask == 1)).all().item():
        raise ValueError("response masks must contain only zero or one")
    return rewards.detach().masked_fill(~response_mask.bool(), 0).sum(-1).cpu().tolist()


def reward_group_metrics(
    outcome_rewards: Sequence[float],
    advantage_rewards: Sequence[float],
    uids: Sequence[str],
    data_sources: Sequence[str],
    is_padding: Sequence[bool],
) -> dict[str, float]:
    """Summarize one complete controller batch, not a rank-local shard.

    Inputs are masked, detached per-response totals. ``outcome_rewards`` are reward-manager
    scores (including its length shaping); ``advantage_rewards`` also include
    any algorithm-side KL penalty. Padding is excluded. A real response with
    no active tokens contributes zero, as its masked reward total does.

    Means weight responses equally; the zero-variance fraction weights prompt
    groups equally. Singleton groups are counted separately because veRL's
    GRPO uses a different baseline for them, not a zero advantage. No metric
    is invented for an absent source or an unmeasured denominator.
    """
    size = len(uids)
    if any(len(values) != size for values in (outcome_rewards, advantage_rewards, data_sources, is_padding)):
        raise ValueError("reward diagnostic inputs must have the same row count")

    by_source = defaultdict(list)
    groups = defaultdict(list)
    source_by_uid = {}
    for outcome, advantage, uid, source, padding in zip(
        outcome_rewards, advantage_rewards, uids, data_sources, is_padding, strict=True
    ):
        if padding:
            continue
        if not isinstance(uid, str) or not uid or not isinstance(source, str) or not source:
            raise ValueError("reward diagnostics require nonempty string uids and data sources")
        if uid in source_by_uid and source_by_uid[uid] != source:
            raise ValueError(f"reward group {uid!r} contains multiple data sources")
        source_by_uid[uid] = source
        outcome, advantage = float(outcome), float(advantage)
        if not math.isfinite(outcome) or not math.isfinite(advantage):
            raise ValueError("masked reward totals must be finite")
        by_source[source].append((outcome, advantage))
        groups[(source, uid)].append(advantage)

    metrics = {}
    for source in sorted(by_source):
        prefix = f"train_reward/{source}"
        rows = by_source[source]
        source_groups = [values for (group_source, _), values in groups.items() if group_source == source]
        repeated_groups = [values for values in source_groups if len(values) > 1]
        metrics[f"{prefix}/response_count"] = len(rows)
        metrics[f"{prefix}/prompt_group_count"] = len(source_groups)
        metrics[f"{prefix}/singleton_group_count"] = len(source_groups) - len(repeated_groups)
        metrics[f"{prefix}/multi_response_group_count"] = len(repeated_groups)
        metrics[f"{prefix}/outcome_reward_mean"] = math.fsum(row[0] for row in rows) / len(rows)
        metrics[f"{prefix}/advantage_reward_mean"] = math.fsum(row[1] for row in rows) / len(rows)
        if repeated_groups:
            zero_count = sum(min(values) == max(values) for values in repeated_groups)
            metrics[f"{prefix}/zero_variance_group_count"] = zero_count
            metrics[f"{prefix}/zero_variance_group_fraction"] = zero_count / len(repeated_groups)
    return metrics
