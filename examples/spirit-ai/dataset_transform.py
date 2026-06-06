"""
CLI for checking, repairing, and transforming Spirit AI LeRobot datasets.

Examples:
    uv run python examples/spirit-ai/dataset_transform.py check \
        --dataset_dir /path/to/dataset \
        --default_prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps."

    uv run python examples/spirit-ai/dataset_transform.py repair-instruction \
        --dataset_dir /path/to/dataset \
        --default_prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps." \
        --output_dir /path/to/repaired_dataset \
        --apply

    uv run python examples/spirit-ai/dataset_transform.py build-multiscale \
        --dataset_dir /path/to/source_dataset \
        --output_dir /path/to/output_dataset \
        --global_prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps." \
        --slice-episodes
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from utils import annotations
from utils import instructions
from utils import lerobot_io


def check_tasks(tasks: list[dict], default_prompt: str, *, allow_derived_prompts: bool = False) -> dict:
    expected = default_prompt.strip()
    expected_base = instructions.normalize_instruction(expected)

    def is_matching(task_text: str) -> bool:
        text = task_text.strip()
        if text == expected:
            return True
        if not allow_derived_prompts:
            return False
        return text.startswith(f"{expected_base}. Current step: ")

    matching = [task for task in tasks if is_matching(task.get("task", ""))]
    missing = [task for task in tasks if not task.get("task", "").strip()]
    mismatched = [task for task in tasks if task.get("task", "").strip() and not is_matching(task.get("task", ""))]
    return {
        "total_tasks": len(tasks),
        "matching": len(matching),
        "missing": len(missing),
        "mismatched": len(mismatched),
        "mismatched_details": [
            {"task_index": task.get("task_index", "<missing>"), "current": task.get("task", "")}
            for task in mismatched
        ],
        "needs_fix": bool(missing or mismatched),
    }


def print_subtask_labeling_summary(summary: dict) -> None:
    print("Subtask labeling:")
    if summary["total_episodes"] == 0:
        print("  Present:         unknown (episodes metadata not found)")
        print()
        return

    print(f"  Present:         {'yes' if summary['present'] else 'no'}")
    print(f"  Episodes:        {summary['labeled_episodes']}/{summary['total_episodes']}")
    if not summary["present"]:
        print("  Note:            optional. build-multiscale only uses it when --slice-episodes is set.")
        print()
        return

    print(f"  Segments/episode: min={summary['segment_count_min']} max={summary['segment_count_max']}")
    print(f"  Distribution:    {summary['segment_count_distribution']}")
    if summary["versions"]:
        print(f"  Versions:        {summary['versions']}")

    normalized_actions = summary["normalized_actions"]
    if normalized_actions:
        print(f"  Unique normalized subtasks: {len(normalized_actions)}")
        for action, count in normalized_actions:
            print(f"    - {action!r} ({count} segments)")
    print()


def load_dataset_metadata(dataset_dir: Path) -> tuple[list[dict], list[dict], dict, dict]:
    meta_dir = dataset_dir / "meta"
    tasks_path = meta_dir / "tasks.jsonl"
    episodes_path = meta_dir / "episodes.jsonl"
    if not tasks_path.exists():
        raise FileNotFoundError(f"tasks.jsonl not found at {tasks_path}")
    tasks = lerobot_io.read_jsonl(tasks_path)
    episodes = lerobot_io.read_jsonl(episodes_path) if episodes_path.exists() else []
    info = lerobot_io.read_json(meta_dir / "info.json")
    dataset_json = lerobot_io.read_json(dataset_dir / "dataset.json")
    return tasks, episodes, info, dataset_json


def print_dataset_overview(dataset_dir: Path, info: dict, episodes: list[dict], tasks: list[dict], prompt: str) -> None:
    print(f"Dataset:          {dataset_dir}")
    print(f"Robot type:       {info.get('robot_type', 'unknown')}")
    print(f"FPS:              {info.get('fps', 'unknown')}")
    print(f"Total episodes:   {info.get('total_episodes', len(episodes))}")
    print(f"Total frames:     {info.get('total_frames', 'unknown')}")
    print(f"Total tasks:      {info.get('total_tasks', len(tasks))}")
    print(f"Default prompt:   {prompt!r}")
    print()


def print_task_check_results(check: dict) -> None:
    print("Task check results:")
    print(f"  Matching:       {check['matching']}/{check['total_tasks']}")
    print(f"  Missing text:   {check['missing']}/{check['total_tasks']}")
    print(f"  Mismatched:     {check['mismatched']}/{check['total_tasks']}")
    for detail in check["mismatched_details"]:
        print(f"    task_index={detail['task_index']}: {detail['current']!r}")
    print()


def run_check(dataset_dir: str, default_prompt: str, *, allow_derived_prompts: bool = False) -> dict:
    src = Path(dataset_dir).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {src}")
    tasks, episodes, info, _ = load_dataset_metadata(src)
    print_dataset_overview(src, info, episodes, tasks, default_prompt)

    check = check_tasks(tasks, default_prompt, allow_derived_prompts=allow_derived_prompts)
    print_task_check_results(check)
    print_subtask_labeling_summary(annotations.summarize_subtask_labeling(episodes))
    if check["needs_fix"]:
        print("FIX NEEDED - Some tasks are missing or have mismatched instruction text.")
    else:
        print("PASS - All tasks already have the expected instruction. No fix needed.")
    return check


def fix_tasks(tasks: list[dict], default_prompt: str) -> list[dict]:
    fixed = []
    for task in tasks:
        new_task = dict(task)
        if not new_task.get("task", "").strip() or new_task["task"].strip() != default_prompt.strip():
            new_task["task"] = default_prompt
        fixed.append(new_task)
    return fixed


def fix_episodes(episodes: list[dict], default_prompt: str) -> list[dict]:
    fixed = []
    for episode in episodes:
        new_episode = dict(episode)
        if "tasks" in new_episode:
            new_episode["tasks"] = [default_prompt]
        if "task_id_map" in new_episode:
            new_episode["task_id_map"] = dict.fromkeys(new_episode["task_id_map"], default_prompt)
        fixed.append(new_episode)
    return fixed


def fix_dataset_json(dataset_json: dict, default_prompt: str) -> dict:
    result = dict(dataset_json)
    if "tasks" in result:
        result["tasks"] = fix_tasks(result["tasks"], default_prompt)
    if "episodes" in result:
        result["episodes"] = fix_episodes(result["episodes"], default_prompt)
    return result


def run_repair_instruction(
    *,
    dataset_dir: str,
    default_prompt: str,
    output_dir: str,
    apply: bool,
    overwrite: bool,
) -> None:
    src = Path(dataset_dir).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {src}")
    tasks, episodes, info, dataset_json = load_dataset_metadata(src)
    print_dataset_overview(src, info, episodes, tasks, default_prompt)
    check = check_tasks(tasks, default_prompt)
    print_task_check_results(check)
    print_subtask_labeling_summary(annotations.summarize_subtask_labeling(episodes))

    if not check["needs_fix"]:
        print("PASS - All tasks already have the expected instruction. No fix needed.")
        return
    print("FIX NEEDED - Some tasks are missing or have mismatched instruction text.")
    if not apply:
        print("Dry-run mode. Use --apply and --output_dir to write a repaired copy.")
        return
    if not output_dir:
        raise ValueError("--output_dir is required when --apply is set.")

    dst = Path(output_dir).resolve()
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {dst}. Use --overwrite to replace it.")
    if dst.exists():
        shutil.rmtree(dst)

    print(f"Writing repaired dataset to: {dst}")
    shutil.copytree(src, dst)

    fixed_tasks = fix_tasks(tasks, default_prompt)
    lerobot_io.write_jsonl(dst / "meta" / "tasks.jsonl", fixed_tasks)
    print(f"  Fixed meta/tasks.jsonl ({len(fixed_tasks)} tasks)")

    if episodes:
        fixed_episodes = fix_episodes(episodes, default_prompt)
        lerobot_io.write_jsonl(dst / "meta" / "episodes.jsonl", fixed_episodes)
        print(f"  Fixed meta/episodes.jsonl ({len(fixed_episodes)} episodes)")

    if dataset_json:
        fixed_dataset_json = fix_dataset_json(dataset_json, default_prompt)
        lerobot_io.write_json(dst / "dataset.json", fixed_dataset_json)
        print("  Fixed dataset.json")

    print("Done.")


def run_build_multiscale(args: argparse.Namespace) -> None:
    from utils import dataset_builder

    if not args.slice_episodes:
        print("Note: --slice-episodes is not set. Building global/full episodes only.")
    summary = dataset_builder.build_multiscale_dataset(
        dataset_dir=Path(args.dataset_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        global_prompt=args.global_prompt,
        overwrite=args.overwrite,
        slice_episodes=args.slice_episodes,
        global_repeat=args.global_repeat,
        subtask_repeat=args.subtask_repeat,
        prefer_english=not args.prefer_chinese_subtasks,
        video_mode=args.video_mode,
        min_slice_frames=args.min_slice_frames,
        progress=not args.quiet,
        video_workers=args.video_workers,
    )
    print(f"Output dataset:    {summary.output_dir}")
    print(f"Total episodes:    {summary.total_episodes}")
    print(f"Global episodes:   {summary.global_episodes}")
    print(f"Subtask episodes:  {summary.subtask_episodes}")
    print(f"Skipped subtasks:  {summary.skipped_subtasks}")
    print(f"Total frames:      {summary.total_frames}")
    print(f"Total tasks:       {summary.total_tasks}")
    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Check task instructions and optional subtask annotations.")
    check.add_argument("--dataset-dir", "--dataset_dir", dest="dataset_dir", required=True)
    check.add_argument("--default-prompt", "--default_prompt", dest="default_prompt", required=True)
    check.add_argument("--allow-derived-prompts", "--allow_derived_prompts", action="store_true")

    repair = subparsers.add_parser("repair-instruction", help="Repair dataset task instructions.")
    repair.add_argument("--dataset-dir", "--dataset_dir", dest="dataset_dir", required=True)
    repair.add_argument("--default-prompt", "--default_prompt", dest="default_prompt", required=True)
    repair.add_argument("--output-dir", "--output_dir", dest="output_dir", default="")
    repair.add_argument("--apply", action="store_true")
    repair.add_argument("--overwrite", action="store_true")

    build = subparsers.add_parser("build-multiscale", help="Build global/full plus optional subtask-sliced data.")
    build.add_argument("--dataset-dir", "--dataset_dir", dest="dataset_dir", required=True)
    build.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    build.add_argument("--global-prompt", "--global_prompt", dest="global_prompt", required=True)
    build.add_argument("--slice-episodes", "--slice_episodes", dest="slice_episodes", action="store_true")
    build.add_argument("--global-repeat", "--global_repeat", dest="global_repeat", type=int, default=1)
    build.add_argument("--subtask-repeat", "--subtask_repeat", dest="subtask_repeat", type=int, default=1)
    build.add_argument("--min-slice-frames", "--min_slice_frames", dest="min_slice_frames", type=int, default=2)
    build.add_argument("--prefer-chinese-subtasks", "--prefer_chinese_subtasks", action="store_true")
    build.add_argument("--video-mode", "--video_mode", choices=["link-full", "copy-full", "slice"], default="link-full")
    build.add_argument("--video-workers", "--video_workers", dest="video_workers", type=int, default=1)
    build.add_argument("--quiet", action="store_true", help="Disable build progress messages.")
    build.add_argument("--overwrite", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "check":
        run_check(args.dataset_dir, args.default_prompt, allow_derived_prompts=args.allow_derived_prompts)
    elif args.command == "repair-instruction":
        run_repair_instruction(
            dataset_dir=args.dataset_dir,
            default_prompt=args.default_prompt,
            output_dir=args.output_dir,
            apply=args.apply,
            overwrite=args.overwrite,
        )
    elif args.command == "build-multiscale":
        run_build_multiscale(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
