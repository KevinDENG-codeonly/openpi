from __future__ import annotations

from collections import Counter

from utils import instructions


def extract_subtask_segments(episode: dict) -> list[dict]:
    annotation = episode.get("annotation") or {}
    annotation_data = annotation.get("annotation_data") or {}
    segments = annotation_data.get("data")
    return segments if isinstance(segments, list) else []


def usable_subtask_segments(episode: dict, *, prefer_english: bool = True) -> list[dict]:
    result = []
    for segment in extract_subtask_segments(episode):
        try:
            start_time = float(segment["start_time"])
            end_time = float(segment["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if end_time <= start_time:
            continue
        text = instructions.subtask_text(segment, prefer_english=prefer_english)
        if not text:
            continue
        result.append({**segment, "start_time": start_time, "end_time": end_time, "_prompt_text": text})
    return result


def summarize_subtask_labeling(episodes: list[dict], *, prefer_english: bool = True) -> dict:
    segment_counts = []
    action_counts = Counter()
    action_english_counts = Counter()
    normalized_counts = Counter()
    version_counts = Counter()

    for episode in episodes:
        annotation = episode.get("annotation") or {}
        annotation_data = annotation.get("annotation_data") or {}
        if annotation_data:
            version_counts[annotation_data.get("version", "unknown")] += 1

        segments = extract_subtask_segments(episode)
        segment_counts.append(len(segments))
        for segment in segments:
            action = segment.get("action")
            action_english = segment.get("action_english")
            if action:
                action_counts[action] += 1
            if action_english:
                action_english_counts[action_english] += 1
            text = instructions.subtask_text(segment, prefer_english=prefer_english)
            if text:
                normalized_counts[text] += 1

    labeled_counts = [count for count in segment_counts if count > 0]
    return {
        "total_episodes": len(episodes),
        "labeled_episodes": len(labeled_counts),
        "segment_count_min": min(labeled_counts) if labeled_counts else 0,
        "segment_count_max": max(labeled_counts) if labeled_counts else 0,
        "segment_count_distribution": dict(Counter(labeled_counts)),
        "versions": dict(version_counts),
        "actions": action_counts.most_common(),
        "actions_english": action_english_counts.most_common(),
        "normalized_actions": normalized_counts.most_common(),
        "present": bool(labeled_counts),
    }
