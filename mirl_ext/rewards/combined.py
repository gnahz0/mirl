"""Dispatch the mixed MIRL dataset to its task-specific rule reward."""

from __future__ import annotations

from typing import Any

from mirl_ext.rewards import ecg, haptic_ts, human_behaviour, medical, smellnet, tactile
from mirl_ext.data.schema import HUMAN_BEHAVIOUR_SOURCES, MEDICAL_SOURCES, TACTILE_SOURCES


def _restore_qwen35_think_prefix(solution: str) -> str:
    """Qwen3.5's chat template ends the prompt with ``<think>``, so verl's decoded
    response carries only the closing tag; the format rewards expect both."""
    if "</think>" in solution and "<think>" not in solution:
        return "<think>" + solution
    return solution


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, float]:
    """Compute a MIRL reward using verl's custom reward function ABI."""
    del extra_info
    solution_str = _restore_qwen35_think_prefix(solution_str)

    if data_source in TACTILE_SOURCES:
        return tactile.compute_score(solution_str, ground_truth)
    if data_source in HUMAN_BEHAVIOUR_SOURCES:
        return human_behaviour.compute_score(solution_str, ground_truth)
    if data_source in MEDICAL_SOURCES:
        return medical.compute_score(solution_str, ground_truth)
    if data_source in {"smellnet_base", "smellnet_mixture"}:
        return smellnet.compute_score(solution_str, ground_truth)
    if data_source == "ecg":
        return ecg.compute_score(solution_str, ground_truth)
    if data_source == "haptic_tactile":
        return haptic_ts.compute_score(solution_str, ground_truth)
    raise NotImplementedError(f"MIRL reward is not implemented for data_source={data_source!r}")
