from __future__ import annotations

import re


_TYPO_FIXES = {
    "maitain": "maintain",
    "Maitain": "Maintain",
}


def normalize_text(text: str) -> str:
    result = text.strip()
    for old, new in _TYPO_FIXES.items():
        result = result.replace(old, new)
    result = re.sub(r"\s+", " ", result)
    return result


def normalize_instruction(text: str) -> str:
    result = normalize_text(text)
    return result.rstrip(".")


def subtask_text(segment: dict, *, prefer_english: bool = True) -> str:
    primary_key = "action_english" if prefer_english else "action"
    fallback_key = "action" if prefer_english else "action_english"
    text = segment.get(primary_key) or segment.get(fallback_key) or ""
    return normalize_instruction(str(text))


def overview_subtask_prompt(global_prompt: str, subtask: str) -> str:
    global_text = normalize_instruction(global_prompt)
    subtask_text_normalized = normalize_instruction(subtask)
    return f"{global_text}. Current step: {subtask_text_normalized}."
