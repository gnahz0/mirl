# MIRL eval report: base vs pilot-SFT vs full-SFT (+ serving cross-check)

- Offline harness: `mirl_ext/eval/run_boxed_eval.py`, greedy, GRPO val parquets, RL scorer, RL 11264-token prompt filter.
- base = Qwen3.5-9B snapshot; pilot = SFT on 20% traces (p20-hom); full = SFT on all traces (sft-full-v1, step 3992).
- serving = same full-SFT model measured by the GRPO stack (`val_before_train`, job 601639, closed sources only, n<=40/source for video families).
- `acc` is exact-match on the boxed answer -- fine for single-label sources, punishing for multi-label (see chest_xray note).

## Family summary (offline)

| family | n | base | pilot | full | majority | full boxed% |
|---|---|---|---|---|---|---|
| smellnet_valid | 297 | 0.000 | 0.010 | 0.010 | 0.017 | 100% |
| ecg_valid | 9856 | 0.309 | 0.201 | 0.423 | 0.441 | 100% |
| haptic_ts_valid | 635 | 0.000 | 0.000 | 0.000 | 0.002 | 39% |
| climb_valid | 5615 | 0.155 | 0.204 | 0.211 | 0.017 | 100% |
| human_behaviour_valid_fast | 351 | 0.322 | 0.405 | 0.413 | 0.145 | 100% |
| tactile_valid_fast | 895 | 0.246 | 0.381 | 0.371 | 0.260 | 100% |

## Per-source breakdown

### smellnet_valid

| source | n | base | pilot | full | serving | full-base |
|---|---|---|---|---|---|---|
| smellnet_mixture | 247 | 0.000 | 0.008 | 0.008 | -- | +0.008 |
| smellnet_base | 50 | 0.000 | 0.020 | 0.020 | 0.020 | +0.020 |

### ecg_valid

| source | n | base | pilot | full | serving | full-base |
|---|---|---|---|---|---|---|
| ecg | 9856 | 0.309 | 0.201 | 0.423 | 0.421 | +0.114 |

### haptic_ts_valid

| source | n | base | pilot | full | serving | full-base |
|---|---|---|---|---|---|---|
| haptic_tactile | 635 | 0.000 | 0.000 | 0.000 | -- | +0.000 |

### climb_valid

| source | n | base | pilot | full | serving | full-base |
|---|---|---|---|---|---|---|
| chest_xray | 3966 | 0.042 | 0.069 | 0.062 | 0.062 | +0.019 |
| ct | 322 | 0.363 | 0.422 | 0.394 | 0.398 | +0.031 |
| fundus | 310 | 0.387 | 0.481 | 0.539 | 0.535 | +0.152 |
| derm | 274 | 0.453 | 0.540 | 0.544 | 0.547 | +0.091 |
| mammo | 245 | 0.135 | 0.367 | 0.494 | 0.490 | +0.359 |
| pathology | 216 | 0.657 | 0.796 | 0.778 | 0.778 | +0.120 |
| ultrasound | 145 | 0.400 | 0.414 | 0.628 | 0.607 | +0.228 |
| mri | 137 | 0.803 | 0.847 | 0.854 | 0.854 | +0.051 |

### human_behaviour_valid_fast

| source | n | base | pilot | full | serving | full-base |
|---|---|---|---|---|---|---|
| chsimsv2 | 40 | 0.400 | 0.600 | 0.600 | 0.550 | +0.200 |
| intentqa | 40 | 0.000 | 0.000 | 0.000 | -- | +0.000 |
| meld_emotion | 40 | 0.400 | 0.500 | 0.550 | 0.525 | +0.150 |
| meld_senti | 40 | 0.650 | 0.750 | 0.700 | 0.675 | +0.050 |
| mosei_emotion | 40 | 0.500 | 0.650 | 0.700 | 0.675 | +0.200 |
| mosei_senti | 40 | 0.225 | 0.275 | 0.300 | 0.300 | +0.075 |
| siq2 | 40 | 0.000 | 0.000 | 0.000 | -- | +0.000 |
| urfunny | 28 | 0.500 | 0.571 | 0.429 | 0.536 | -0.071 |
| mimeqa | 22 | 0.000 | 0.000 | 0.000 | -- | +0.000 |
| mmsd | 13 | 0.462 | 0.538 | 0.846 | 0.846 | +0.385 |
| ptsd_in_the_wild | 8 | 0.750 | 1.000 | 1.000 | 1.000 | +0.250 |

### tactile_valid_fast

| source | n | base | pilot | full | serving | full-base |
|---|---|---|---|---|---|---|
| contact_feature | 40 | 0.500 | 0.650 | 0.675 | 0.650 | +0.175 |
| deformation_note | 40 | 0.000 | 0.000 | 0.000 | -- | +0.000 |
| deformation_type | 40 | 0.500 | 0.625 | 0.675 | 0.675 | +0.175 |
| description | 40 | 0.000 | 0.000 | 0.000 | -- | +0.000 |
| fail_improvement | 40 | 0.025 | 0.125 | 0.225 | 0.175 | +0.200 |
| fail_reason | 40 | 0.075 | 0.350 | 0.200 | 0.225 | +0.125 |
| force_level | 40 | 0.125 | 0.475 | 0.425 | 0.450 | +0.300 |
| future_stability | 40 | 0.700 | 0.925 | 0.875 | 0.875 | +0.175 |
| grasp_location | 40 | 0.600 | 0.750 | 0.775 | 0.775 | +0.175 |
| grip_stability | 40 | 0.900 | 0.925 | 0.900 | 0.900 | +0.000 |
| highest_pressure | 40 | 0.025 | 0.050 | 0.100 | 0.100 | +0.075 |
| initial_fingers | 40 | 0.025 | 0.250 | 0.100 | 0.150 | +0.075 |
| local_shape | 40 | 0.500 | 0.725 | 0.725 | 0.775 | +0.225 |
| mat_description | 40 | 0.000 | 0.000 | 0.000 | -- | +0.000 |
| objA_notes | 40 | 0.000 | 0.000 | 0.000 | -- | +0.000 |
| objA_texture | 40 | 0.650 | 0.875 | 0.875 | 0.875 | +0.225 |
| objB_notes | 40 | 0.000 | 0.025 | 0.000 | -- | +0.000 |
| objB_texture | 40 | 0.350 | 0.775 | 0.825 | 0.775 | +0.475 |
| object_motion | 40 | 0.075 | 0.250 | 0.150 | 0.175 | +0.075 |
| part_notes | 40 | 0.000 | 0.025 | 0.000 | -- | +0.000 |
| shear_direction | 40 | 0.125 | 0.450 | 0.450 | 0.475 | +0.325 |
| tactile_description | 40 | 0.000 | 0.000 | 0.000 | -- | +0.000 |
| more_deformable | 15 | 0.867 | 0.733 | 0.867 | 0.867 | +0.000 |

## Notes

- **chest_xray** acc is mis-specified (exact SET match on multi-finding GTs, 82% multi-label). Honest per-finding mean set-F1: base 0.115 -> pilot 0.379 -> full 0.389 (a 3.4x gain the acc column hides). The RL reward already uses F1.
- **ecg** is single-label by upstream task definition on inherently multi-label recordings; majority class 0.441. Treat as proxy.
- **smellnet_base** stays at chance: matplotlib-plot representation + 121 SFT rows; fix planned via ts-native renders + few-shot episodes.
- **smellnet_mixture / open free-text sources** (descriptions, *_notes, haptic_tactile, intentqa, siq2, mimeqa): exact-match inapplicable by design; excluded from RL entirely. `--` in serving column = excluded there.
- serving vs full agree to noise on every large-n source (max |delta| 0.021 at n>=137); all bigger deltas are n<=40 sampling wiggle. Offline harness is validated as a serving proxy.
