"""Narrow, answer-independent corrections to source prompt vocabularies."""

from __future__ import annotations

import copy
import re


_INSPECT_CHOICE = re.compile(r"^([ \t]*)(Acute )?Subsegmental- PE([ \t]*)(?=\r?$)", re.MULTILINE)


def normalize_prompt_messages(
    messages: list[dict], *, data_source: str | None, dataset: str | None
) -> list[dict]:
    """Copy prompt messages, repairing only INSPECT's two malformed choices.

    INSPECT targets use ``Subsegmental-only PE`` but older exports omitted
    ``only`` in their choice lines. Apply the same vocabulary fix to every
    INSPECT row, without consulting its answer or modifying assistant targets.
    """
    normalized = copy.deepcopy(messages)
    if data_source != "ct" or dataset != "inspect":
        return normalized

    for message in normalized:
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, str):
            message["content"] = _INSPECT_CHOICE.sub(
                lambda match: f"{match[1]}{match[2] or ''}Subsegmental-only PE{match[3]}", content
            )
    return normalized
