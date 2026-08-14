"""SFT data-generation toolchain: split (split_sft_rl) -> export tasks
(export_sft_tasks) -> generate traces (gen_sft_episodes; gen_sft_targets is the
older answer-conditioned generator, kept as the shared API-client/helper module
and for --blind readability evals) -> build veRL SFT parquet (build_sft_parquet).

Modules are runnable directly (``python mirl_ext/sft/<name>.py``) and keep
sibling imports via a path insert, so they work both as scripts and as
``python -m mirl_ext.sft.<name>``.
"""
