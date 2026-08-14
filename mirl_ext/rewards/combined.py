"""Dispatch the mixed MIRL dataset to its task-specific rule reward."""

from __future__ import annotations

from typing import Any

from mirl_ext.rewards import ecg, haptic_ts, human_behaviour, medical, smellnet, tactile


TACTILE_SOURCES = {
    "verify",
    "initial_fingers",
    "highest_pressure",
    "more_deformable",
    "deformation_type",
    "deformation_note",
    "objA_texture",
    "objB_texture",
    "objA_notes",
    "objB_notes",
    "grasp_location",
    "contact_feature",
    "local_shape",
    "grip_stability",
    "future_stability",
    "force_level",
    "shear_direction",
    "object_motion",
    "fail_reason",
    "fail_improvement",
    "description",
    "mat_description",
    "part_notes",
    "tactile_description",
}
HUMAN_BEHAVIOUR_SOURCES = {
    "cremad",
    "chsimsv2",
    "daicwoz",
    "intentqa",
    "meld_emotion",
    "meld_senti",
    "mimeqa",
    "mmpsy_anxiety",
    "mmpsy_depression",
    "mmsd",
    "mosei_emotion",
    "mosei_senti",
    "ptsd_in_the_wild",
    "siq2",
    "tess",
    "urfunny",
}
MEDICAL_SOURCES = {"chest_xray", "ct", "derm", "fundus", "mammo", "mri", "pathology", "ultrasound"}


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
