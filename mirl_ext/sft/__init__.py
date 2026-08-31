"""SFT data-generation toolchain (v1: answer-blind zero-shot; see README.md).

export tasks (export_sft_tasks) -> stage teacher media (stage_media) ->
generate traces (gen_sft_targets; prompts live at its top) -> report
(report_traces) -> build veRL SFT parquet (build_sft_parquet). split_sft_rl
made the 50:50 split (already run); gen_sft_episodes generates few-shot
smellnet episodes and supplies the smellnet traces (zero-shot smellnet
measured 0%, episodes 98% yield; see README.md).

Modules are runnable directly (``python mirl_ext/sft/<name>.py``) and keep
sibling imports via a path insert, so they work both as scripts and as
``python -m mirl_ext.sft.<name>``.
"""
