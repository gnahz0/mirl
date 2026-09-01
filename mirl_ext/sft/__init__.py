"""SFT data-generation toolchain (v1: answer-blind zero-shot; see README.md).

export tasks (export_sft_tasks) -> stage teacher media (stage_media) ->
generate traces (gen_sft_targets; prompts live at its top) -> report
(report_traces) -> build veRL SFT parquet (build_sft_parquet). split_sft_rl
made the 50:50 split (already run). SmellNet is excluded from the project
(2026-08-31); its parquets remain on disk but no pipeline reads them.

Modules are runnable directly (``python mirl_ext/sft/<name>.py``) and keep
sibling imports via a path insert, so they work both as scripts and as
``python -m mirl_ext.sft.<name>``.
"""
