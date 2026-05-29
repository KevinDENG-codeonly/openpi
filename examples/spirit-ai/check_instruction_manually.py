"""
Manual utility script for checking and fixing task instructions in Spirit AI LeRobot datasets.

Spirit AI humanoid robot datasets are collected in LeRobot v2.1 format. This script checks whether
each episode has a valid task instruction and, if missing or incorrect, creates a repaired copy of the
dataset with the specified default prompt applied to all episodes.

Usage (dry-run / check only):
    uv run examples/spirit-ai/check_instruction_manually.py \
        --default_prompt "Fold the cardboard sheet along the creases to form a box" \
        --dataset_dir /path/to/source_dataset

Usage (apply fix, write repaired copy):
    uv run examples/spirit-ai/check_instruction_manually.py \
        --default_prompt "Fold the cardboard sheet along the creases to form a box" \
        --dataset_dir /path/to/source_dataset \
        --output_dir /path/to/output_dataset \
        --apply
"""

from collections import Counter
import json
from pathlib import Path
import shutil

import tyro


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file and return a list of dicts."""
    items = []
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _write_jsonl(path: Path, items: list[dict]) -> None:
    """Write a list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _check_tasks(tasks: list[dict], default_prompt: str) -> dict:
    """Check task entries against the expected default prompt.

    Returns a summary dict with check results.
    """
    total = len(tasks)
    matching = [t for t in tasks if t.get("task", "").strip() == default_prompt.strip()]
    missing = [t for t in tasks if not t.get("task", "").strip()]
    mismatched = [t for t in tasks if t.get("task", "").strip() and t.get("task", "").strip() != default_prompt.strip()]

    return {
        "total_tasks": total,
        "matching": len(matching),
        "missing": len(missing),
        "mismatched": len(mismatched),
        "mismatched_details": [
            {"task_index": t.get("task_index", "<missing>"), "current": t.get("task", "")} for t in mismatched
        ],
        "needs_fix": len(missing) > 0 or len(mismatched) > 0,
    }


def _extract_subtask_segments(episode: dict) -> list[dict]:
    """Return subtask labeling segments from an episode, if present."""
    annotation = episode.get("annotation") or {}
    annotation_data = annotation.get("annotation_data") or {}
    segments = annotation_data.get("data")
    return segments if isinstance(segments, list) else []


def _check_subtask_labeling(episodes: list[dict]) -> dict:
    """Summarize optional per-episode subtask labeling metadata."""
    total = len(episodes)
    segment_counts = []
    action_counts = Counter()
    action_english_counts = Counter()
    version_counts = Counter()

    for episode in episodes:
        annotation = episode.get("annotation") or {}
        annotation_data = annotation.get("annotation_data") or {}
        if annotation_data:
            version_counts[annotation_data.get("version", "unknown")] += 1

        segments = _extract_subtask_segments(episode)
        segment_counts.append(len(segments))
        for segment in segments:
            action = segment.get("action")
            action_english = segment.get("action_english")
            if action:
                action_counts[action] += 1
            if action_english:
                action_english_counts[action_english] += 1

    labeled_counts = [count for count in segment_counts if count > 0]
    return {
        "total_episodes": total,
        "labeled_episodes": len(labeled_counts),
        "segment_count_min": min(labeled_counts) if labeled_counts else 0,
        "segment_count_max": max(labeled_counts) if labeled_counts else 0,
        "segment_count_distribution": dict(Counter(labeled_counts)),
        "versions": dict(version_counts),
        "actions": action_counts.most_common(),
        "actions_english": action_english_counts.most_common(),
        "present": bool(labeled_counts),
    }


def _print_subtask_labeling_summary(summary: dict) -> None:
    """Print optional subtask labeling information without making it a pass/fail condition."""
    print("Subtask labeling:")
    if summary["total_episodes"] == 0:
        print("  Present:         unknown (episodes metadata not found)")
        print()
        return

    print(f"  Present:         {'yes' if summary['present'] else 'no'}")
    print(f"  Episodes:        {summary['labeled_episodes']}/{summary['total_episodes']}")
    if not summary["present"]:
        print("  Note:            optional, but useful for downstream analysis/training.")
        print()
        return

    print(f"  Segments/episode: min={summary['segment_count_min']} max={summary['segment_count_max']}")
    print(f"  Distribution:    {summary['segment_count_distribution']}")
    if summary["versions"]:
        print(f"  Versions:        {summary['versions']}")

    actions_english = summary["actions_english"]
    actions = summary["actions"]
    if actions_english:
        print(f"  Unique subtasks: {len(actions_english)}")
        for action, count in actions_english:
            print(f"    - {action!r} ({count} episodes)")
    elif actions:
        print(f"  Unique subtasks: {len(actions)}")
        for action, count in actions:
            print(f"    - {action!r} ({count} episodes)")
    print()


def _fix_tasks(tasks: list[dict], default_prompt: str) -> list[dict]:
    """Return a new tasks list with missing/wrong task text replaced by default_prompt."""
    fixed = []
    for t in tasks:
        new_t = dict(t)
        if not new_t.get("task", "").strip() or new_t["task"].strip() != default_prompt.strip():
            new_t["task"] = default_prompt
        fixed.append(new_t)
    return fixed


def _fix_episodes(episodes: list[dict], default_prompt: str) -> list[dict]:
    """Return a new episodes list with task references updated to default_prompt."""
    fixed = []
    for ep in episodes:
        new_ep = dict(ep)
        # Fix the tasks list field
        if "tasks" in new_ep:
            new_ep["tasks"] = [default_prompt]
        # Fix the task_id_map field
        if "task_id_map" in new_ep:
            new_ep["task_id_map"] = dict.fromkeys(new_ep["task_id_map"], default_prompt)
        fixed.append(new_ep)
    return fixed


def _fix_dataset_json(dataset_json: dict, default_prompt: str) -> dict:
    """Return an updated top-level dataset.json with task references fixed."""
    result = dict(dataset_json)
    # Fix tasks array
    if "tasks" in result:
        result["tasks"] = _fix_tasks(result["tasks"], default_prompt)
    # Fix episodes array
    if "episodes" in result:
        result["episodes"] = _fix_episodes(result["episodes"], default_prompt)
    return result


def main(
    dataset_dir: str,
    default_prompt: str,
    output_dir: str = "",
    *,
    apply: bool = False,
    overwrite: bool = False,
):
    """Check and optionally fix task instructions in a Spirit AI LeRobot dataset.

    Args:
        dataset_dir: Path to the source LeRobot dataset directory.
        default_prompt: The expected task instruction text. Missing/wrong tasks will be set to this value.
        output_dir: Path for the repaired output dataset. Required when --apply is set.
        apply: If set, write a repaired copy to output_dir. Otherwise only report (dry-run).
        overwrite: If set, allow overwriting an existing output_dir.
    """
    src = Path(dataset_dir).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {src}")

    # --- Read metadata ---
    meta_dir = src / "meta"
    tasks_path = meta_dir / "tasks.jsonl"
    episodes_path = meta_dir / "episodes.jsonl"
    info_path = meta_dir / "info.json"
    dataset_json_path = src / "dataset.json"

    if not tasks_path.exists():
        raise FileNotFoundError(f"tasks.jsonl not found at {tasks_path}")

    tasks = _read_jsonl(tasks_path)
    episodes = _read_jsonl(episodes_path) if episodes_path.exists() else []
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    dataset_json = json.loads(dataset_json_path.read_text()) if dataset_json_path.exists() else {}

    # --- Print dataset info ---
    print(f"Dataset:          {src}")
    print(f"Robot type:       {info.get('robot_type', 'unknown')}")
    print(f"FPS:              {info.get('fps', 'unknown')}")
    print(f"Total episodes:   {info.get('total_episodes', len(episodes))}")
    print(f"Total frames:     {info.get('total_frames', 'unknown')}")
    print(f"Total tasks:      {info.get('total_tasks', len(tasks))}")
    print(f"Default prompt:   {default_prompt!r}")
    print()

    # --- Check tasks ---
    check = _check_tasks(tasks, default_prompt)
    print("Task check results:")
    print(f"  Matching:       {check['matching']}/{check['total_tasks']}")
    print(f"  Missing text:   {check['missing']}/{check['total_tasks']}")
    print(f"  Mismatched:     {check['mismatched']}/{check['total_tasks']}")
    for d in check["mismatched_details"]:
        print(f"    task_index={d['task_index']}: {d['current']!r}")
    print()

    # --- Check optional subtask labeling metadata ---
    subtask_check = _check_subtask_labeling(episodes)
    _print_subtask_labeling_summary(subtask_check)

    if not check["needs_fix"]:
        print("PASS - All tasks already have the expected instruction. No fix needed.")
        return

    print("FIX NEEDED - Some tasks are missing or have mismatched instruction text.")

    if not apply:
        print("Dry-run mode. Use --apply and --output_dir to write a repaired copy.")
        return

    # --- Apply fix ---
    if not output_dir:
        raise ValueError("--output_dir is required when --apply is set.")
    dst = Path(output_dir).resolve()
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {dst}. Use --overwrite to replace it.")
    if dst.exists() and overwrite:
        shutil.rmtree(dst)

    print(f"Writing repaired dataset to: {dst}")

    # Copy the full dataset tree (data + videos + meta)
    shutil.copytree(src, dst)

    # Overwrite metadata files with fixed task instructions
    fixed_tasks = _fix_tasks(tasks, default_prompt)
    _write_jsonl(dst / "meta" / "tasks.jsonl", fixed_tasks)
    print(f"  Fixed meta/tasks.jsonl ({len(fixed_tasks)} tasks)")

    if episodes:
        fixed_episodes = _fix_episodes(episodes, default_prompt)
        _write_jsonl(dst / "meta" / "episodes.jsonl", fixed_episodes)
        print(f"  Fixed meta/episodes.jsonl ({len(fixed_episodes)} episodes)")

    if dataset_json:
        fixed_dataset_json = _fix_dataset_json(dataset_json, default_prompt)
        (dst / "dataset.json").write_text(json.dumps(fixed_dataset_json, indent=2, ensure_ascii=False))
        print("  Fixed dataset.json")

    print()
    print("Done. You can verify the result by re-running this script on the output directory:")
    print(f"  python {__file__} --default_prompt {default_prompt!r} --dataset_dir {dst}")


if __name__ == "__main__":
    tyro.cli(main)
