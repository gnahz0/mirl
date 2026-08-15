"""SFT data-generation toolchain (v1: answer-blind zero-shot; see README.md).

split (split_sft_rl) -> export tasks (export_sft_tasks) -> stage teacher media
(stage_media) -> generate traces (gen_sft_targets, prompts in teacher_context)
-> report (report_traces) -> build veRL SFT parquet (build_sft_parquet) ->
cluster smoke test (smoke_sft_load). audit_schema prints the parquet facts the
stages assume; gen_sft_episodes is the few-shot episode generator, not used in
v1.

Modules are runnable directly (``python mirl_ext/sft/<name>.py``) and keep
sibling imports via a path insert, so they work both as scripts and as
``python -m mirl_ext.sft.<name>``.
"""
