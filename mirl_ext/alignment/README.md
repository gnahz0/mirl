# Exporting the Stage-1 vision encoder

`export_stage1_vision.py` merges a Stage-1 alignment checkpoint into the
Qwen3.5 base model and saves a standalone HF checkpoint.

On aicr:

```bash
python -m mirl_ext.alignment.export_stage1_vision \
    --base-model $MIRL_CLUSTER_ROOT/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
    --vision-checkpoint $MIRL_CLUSTER_ROOT/stage1-current/alignment_state.pt \
    --output-dir $MIRL_SCRATCH_ROOT/models/qwen35-stage1-current
```
