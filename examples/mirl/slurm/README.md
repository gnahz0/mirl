# MIRL Slurm launchers

These files began as snapshots of the cluster launchers that lived at
`/work/mit/ppliang_mit/alecz/` when the Qwen3.5 migration began on 2026-07-20.
They remain tracked here so launcher migrations are reviewable.

`run_combined_b200.sbatch` is now the runnable two-B200 Qwen3.5 smoke launcher.
It defaults to `SMOKE=1`, creates the bounded eight-example fixture, and runs
one GRPO optimizer step. Submit it from the repository root:

```bash
sbatch examples/mirl/slurm/run_combined_b200.sbatch
```

All temporary files and runtime caches go below
`/scratch/dvdai_mit/alecz`; the log goes to
`/work/mit/ppliang_mit/alecz/logs/mirl_qwen35_<job>.out`.

Stage-1 alignment uses the production launcher:

```bash
sbatch mirl_ext/alignment/run_stage1_b200.sbatch
```

The production launcher reads all experiment settings from
`mirl_ext/alignment/config/stage1_qwen35_siglip2_aicr.yaml` and writes checkpoints to
`/scratch/dvdai_mit/alecz/checkpoints`.

`run_trainedve_raw_b200.sbatch` is retained only for rebuilding the raw indexes.

See `docs/mirl/CONTINUATION.md` for the latest verified job and agent handoff,
`docs/mirl/README.md` for the current workflow, and
`docs/mirl/qwen35-migration-ledger.md` for provenance and disposition.
