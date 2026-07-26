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

`run_trainedve_raw_b200.sbatch` and `run_stage1_b200.sbatch` are still
historical Qwen3-VL snapshots for the deferred raw-signal and Stage-1 alignment
work. Do not submit those two files yet.

See `docs/mirl/CONTINUATION.md` for the latest verified job and agent handoff,
`docs/mirl/README.md` for the current workflow, and
`docs/mirl/qwen35-migration-ledger.md` for provenance and disposition.
