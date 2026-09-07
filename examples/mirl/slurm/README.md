# MIRL Slurm launchers

These repository-owned launchers are retained for reproducibility and review.

`run_combined_b200.sbatch` is now the runnable two-B200 Qwen3.5 smoke launcher.
It defaults to `SMOKE=1`, creates the bounded eight-example fixture, and runs
one GRPO optimizer step. Submit it from the repository root:

```bash
sbatch examples/mirl/slurm/run_combined_b200.sbatch
```

All temporary files and runtime caches go below
`$MIRL_SCRATCH_ROOT`; the log goes to
`$MIRL_CLUSTER_ROOT/logs/mirl_qwen35_<job>.out`.

Stage-1 alignment uses the production launcher:

```bash
sbatch mirl_ext/alignment/run_stage1_b200.sbatch
```

The production launcher reads all experiment settings from
`mirl_ext/alignment/config/stage1_qwen35_siglip2_aicr.yaml` and writes checkpoints to
`$MIRL_SCRATCH_ROOT/checkpoints`.

`run_trainedve_raw_b200.sbatch` is retained as a provenance snapshot of the
historical Qwen3-VL raw-signal launcher. It targets a retired checkout and is
not runnable here; do not submit it (see
`docs/mirl/qwen35-migration-ledger.md`).

See [the MIRL overview](../../../docs/mirl/README.md) for the current workflow
and [the migration ledger](../../../docs/mirl/qwen35-migration-ledger.md)
for historical provenance and disposition.
