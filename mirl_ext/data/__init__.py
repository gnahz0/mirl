"""Data facts and IO shared across stages: schema.py (row shape, family
taxonomies, stems, source sets), signals.py (raw sensor loaders + pseudo-video
render), dataset.py (GRPO's MIRLDataset, loaded by verl via data.custom_cls
file path), plus one-off index tools.

No eager submodule imports here: schema must stay importable on machines
without torch (laptop-side generation and tests).
"""
