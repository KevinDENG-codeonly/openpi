"""Plan and build physical global-prompt augmentations for LeRobot datasets."""

from __future__ import annotations

from collections import Counter
import copy
import dataclasses
from datetime import UTC
from datetime import datetime
import hashlib
import math
from pathlib import Path
import random
import shutil
import time
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from utils import lerobot_io

_RECOVERY_LABELS = ("none", "left", "right", "both")
_LEFT_RECOVERY_TASK_INDEX = 34
_RIGHT_RECOVERY_TASK_INDICES = frozenset({35, 36, 37, 38})
_MANIFEST_VERSION = 1


@dataclasses.dataclass(frozen=True)
class CandidateSummary:
    trial: int
    episode_indices: tuple[int, ...]
    duplicate_frames: int
    duplicate_frame_ratio: float
    final_global_prompt_ratio: float
    recovery_episode_counts: dict[str, int]
    recovery_l1_error: float


@dataclasses.dataclass(frozen=True)
class GlobalPromptAugmentationPlan:
    source_dir: Path
    source_fingerprints: dict[str, str]
    source_episode_count: int
    source_frame_count: int
    global_task_index: int
    global_task_text: str
    duplicate_episode_fraction: float
    duplicate_episode_count: int
    candidate_count: int
    seed: int
    candidates: tuple[CandidateSummary, ...]
    selected_trial: int
    selected_episode_indices: tuple[int, ...]
    selected_duplicate_frames: int
    selected_duplicate_frame_ratio: float
    selected_final_global_prompt_ratio: float


@dataclasses.dataclass(frozen=True)
class GlobalPromptAugmentationSummary:
    output_dir: Path
    source_episodes: int
    duplicate_episodes: int
    total_episodes: int
    source_frames: int
    duplicate_frames: int
    total_frames: int
    global_task_index: int
    total_tasks: int
    final_global_prompt_ratio: float


@dataclasses.dataclass(frozen=True)
class GlobalPromptAugmentationValidation:
    total_episodes: int
    total_frames: int
    duplicate_episodes: int
    duplicate_frames: int
    global_prompt_frame_ratio: float
    task_index_counts: dict[int, int]
    action_horizon_crossing_frames: int
    action_horizon_checked_frames: int


def plan_global_prompt_augmentation(
    *,
    dataset_dir: Path,
    global_task_index: int,
    duplicate_episode_fraction: float,
    candidate_count: int,
    seed: int,
) -> GlobalPromptAugmentationPlan:
    dataset_dir = dataset_dir.resolve()
    info, tasks, episodes = _load_metadata(dataset_dir)
    _validate_plan_arguments(duplicate_episode_fraction, candidate_count)
    task_map = _validate_task_map(tasks, global_task_index)
    source_episode_count, source_frame_count, length_by_episode = _validate_episode_metadata(info, episodes)

    recovery_class_by_episode: dict[int, str] = {}
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        table = _read_episode_table(dataset_dir, info, episode_index, columns=["episode_index", "task_index"])
        if table.num_rows != length_by_episode[episode_index]:
            raise ValueError(
                f"episode={episode_index} parquet rows ({table.num_rows}) do not match metadata length "
                f"({length_by_episode[episode_index]})"
            )
        episode_ids = {int(value) for value in table["episode_index"].to_pylist()}
        if episode_ids != {episode_index}:
            raise ValueError(f"episode={episode_index} has unexpected parquet episode_index values: {episode_ids}")
        task_indices = {int(value) for value in table["task_index"].to_pylist()}
        unknown = task_indices.difference(task_map)
        if unknown:
            raise ValueError(f"episode={episode_index} contains task IDs missing from tasks.jsonl: {sorted(unknown)}")
        if global_task_index in task_indices:
            raise ValueError(
                f"source episode={episode_index} already contains global_task_index={global_task_index}; "
                "the augmentation source must contain only the original prompt distribution"
            )
        recovery_class_by_episode[episode_index] = recovery_class(task_indices)

    source_recovery_counts = count_recovery_classes(tuple(length_by_episode), recovery_class_by_episode)
    duplicate_episode_count = round(source_episode_count * duplicate_episode_fraction)
    episode_indices = tuple(sorted(length_by_episode))
    rng = random.Random(seed)
    candidates: list[CandidateSummary] = []
    for trial in range(candidate_count):
        selected = tuple(sorted(rng.sample(episode_indices, duplicate_episode_count)))
        duplicate_frames = sum(length_by_episode[index] for index in selected)
        duplicate_frame_ratio = duplicate_frames / source_frame_count
        candidate_recovery_counts = count_recovery_classes(selected, recovery_class_by_episode)
        recovery_l1_error = sum(
            abs(
                candidate_recovery_counts[label] / duplicate_episode_count
                - source_recovery_counts[label] / source_episode_count
            )
            for label in _RECOVERY_LABELS
        )
        candidates.append(
            CandidateSummary(
                trial=trial,
                episode_indices=selected,
                duplicate_frames=duplicate_frames,
                duplicate_frame_ratio=duplicate_frame_ratio,
                final_global_prompt_ratio=duplicate_frames / (source_frame_count + duplicate_frames),
                recovery_episode_counts=candidate_recovery_counts,
                recovery_l1_error=recovery_l1_error,
            )
        )

    acceptable = [candidate for candidate in candidates if 0.24 <= candidate.duplicate_frame_ratio <= 0.26]
    if not acceptable:
        raise ValueError(
            "No random candidate has a duplicate-frame ratio in [0.24, 0.26]; "
            "increase candidate_count or broaden the accepted range"
        )
    selected = min(
        acceptable,
        key=lambda candidate: (
            abs(candidate.duplicate_frame_ratio - duplicate_episode_fraction),
            candidate.recovery_l1_error,
            candidate.trial,
        ),
    )
    selected_candidates = tuple(
        dataclasses.replace(candidate, episode_indices=()) for candidate in candidates if candidate.trial != selected.trial
    )
    selected_candidates += (selected,)
    selected_candidates = tuple(sorted(selected_candidates, key=lambda candidate: candidate.trial))

    return GlobalPromptAugmentationPlan(
        source_dir=dataset_dir,
        source_fingerprints=_source_fingerprints(dataset_dir),
        source_episode_count=source_episode_count,
        source_frame_count=source_frame_count,
        global_task_index=global_task_index,
        global_task_text=task_map[global_task_index],
        duplicate_episode_fraction=duplicate_episode_fraction,
        duplicate_episode_count=duplicate_episode_count,
        candidate_count=candidate_count,
        seed=seed,
        candidates=selected_candidates,
        selected_trial=selected.trial,
        selected_episode_indices=selected.episode_indices,
        selected_duplicate_frames=selected.duplicate_frames,
        selected_duplicate_frame_ratio=selected.duplicate_frame_ratio,
        selected_final_global_prompt_ratio=selected.final_global_prompt_ratio,
    )


def count_recovery_classes(episode_indices: tuple[int, ...], recovery_class_by_episode: dict[int, str]) -> dict[str, int]:
    counts = dict.fromkeys(_RECOVERY_LABELS, 0)
    for episode_index in episode_indices:
        counts[recovery_class_by_episode[episode_index]] += 1
    return counts


def recovery_class(task_indices: set[int]) -> str:
    has_left = _LEFT_RECOVERY_TASK_INDEX in task_indices
    has_right = bool(task_indices.intersection(_RIGHT_RECOVERY_TASK_INDICES))
    if has_left and has_right:
        return "both"
    if has_left:
        return "left"
    if has_right:
        return "right"
    return "none"


def write_plan_manifest(path: Path, plan: GlobalPromptAugmentationPlan) -> None:
    lerobot_io.write_json(path, _plan_payload(plan))


def _plan_payload(plan: GlobalPromptAugmentationPlan) -> dict:
    selected_trial = plan.selected_trial
    return {
        "manifest_version": _MANIFEST_VERSION,
        "source_dir": str(plan.source_dir),
        "source_fingerprints": dict(plan.source_fingerprints),
        "source_episode_count": plan.source_episode_count,
        "source_frame_count": plan.source_frame_count,
        "global_task_index": plan.global_task_index,
        "global_task_text": plan.global_task_text,
        "duplicate_episode_fraction": plan.duplicate_episode_fraction,
        "duplicate_episode_count": plan.duplicate_episode_count,
        "candidate_count": plan.candidate_count,
        "seed": plan.seed,
        "selected_trial": selected_trial,
        "selected_episode_indices": list(plan.selected_episode_indices),
        "selected_duplicate_frames": plan.selected_duplicate_frames,
        "selected_duplicate_frame_ratio": plan.selected_duplicate_frame_ratio,
        "selected_final_global_prompt_ratio": plan.selected_final_global_prompt_ratio,
        "candidates": [
            {
                "trial": candidate.trial,
                "episode_indices": list(candidate.episode_indices) if candidate.trial == selected_trial else [],
                "duplicate_frames": candidate.duplicate_frames,
                "duplicate_frame_ratio": candidate.duplicate_frame_ratio,
                "final_global_prompt_ratio": candidate.final_global_prompt_ratio,
                "recovery_episode_counts": candidate.recovery_episode_counts,
                "recovery_l1_error": candidate.recovery_l1_error,
            }
            for candidate in plan.candidates
        ],
    }


def read_plan_manifest(path: Path) -> GlobalPromptAugmentationPlan:
    payload = lerobot_io.read_json(path)
    if payload.get("manifest_version") != _MANIFEST_VERSION:
        raise ValueError(f"Unsupported manifest_version: {payload.get('manifest_version')!r}")
    candidates = tuple(
        CandidateSummary(
            trial=int(candidate["trial"]),
            episode_indices=tuple(int(index) for index in candidate.get("episode_indices", [])),
            duplicate_frames=int(candidate["duplicate_frames"]),
            duplicate_frame_ratio=float(candidate["duplicate_frame_ratio"]),
            final_global_prompt_ratio=float(candidate["final_global_prompt_ratio"]),
            recovery_episode_counts={str(key): int(value) for key, value in candidate["recovery_episode_counts"].items()},
            recovery_l1_error=float(candidate["recovery_l1_error"]),
        )
        for candidate in payload["candidates"]
    )
    selected_trial = int(payload["selected_trial"])
    selected_candidates = [candidate for candidate in candidates if candidate.trial == selected_trial]
    if len(selected_candidates) != 1:
        raise ValueError(f"Manifest must contain exactly one selected trial={selected_trial}")
    if tuple(payload["selected_episode_indices"]) != selected_candidates[0].episode_indices:
        raise ValueError("Manifest selected_episode_indices does not match the selected candidate")
    return GlobalPromptAugmentationPlan(
        source_dir=Path(payload["source_dir"]).resolve(),
        source_fingerprints={str(key): str(value) for key, value in payload["source_fingerprints"].items()},
        source_episode_count=int(payload["source_episode_count"]),
        source_frame_count=int(payload["source_frame_count"]),
        global_task_index=int(payload["global_task_index"]),
        global_task_text=str(payload["global_task_text"]),
        duplicate_episode_fraction=float(payload["duplicate_episode_fraction"]),
        duplicate_episode_count=int(payload["duplicate_episode_count"]),
        candidate_count=int(payload["candidate_count"]),
        seed=int(payload["seed"]),
        candidates=candidates,
        selected_trial=selected_trial,
        selected_episode_indices=tuple(int(index) for index in payload["selected_episode_indices"]),
        selected_duplicate_frames=int(payload["selected_duplicate_frames"]),
        selected_duplicate_frame_ratio=float(payload["selected_duplicate_frame_ratio"]),
        selected_final_global_prompt_ratio=float(payload["selected_final_global_prompt_ratio"]),
    )


def _validate_plan_arguments(duplicate_episode_fraction: float, candidate_count: int) -> None:
    if not 0 < duplicate_episode_fraction <= 1:
        raise ValueError("duplicate_episode_fraction must be in (0, 1]")
    if candidate_count < 1:
        raise ValueError("candidate_count must be >= 1")


def _load_metadata(dataset_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    meta_dir = dataset_dir / "meta"
    info = lerobot_io.read_json(meta_dir / "info.json")
    tasks = lerobot_io.read_jsonl(meta_dir / "tasks.jsonl")
    episodes = lerobot_io.read_jsonl(meta_dir / "episodes.jsonl")
    return info, tasks, episodes


def _validate_task_map(tasks: list[dict], global_task_index: int) -> dict[int, str]:
    task_map = {int(task["task_index"]): str(task["task"]) for task in tasks}
    expected = set(range(len(task_map)))
    if set(task_map) != expected:
        raise ValueError(f"tasks.jsonl task_index values must be contiguous from 0: {sorted(task_map)}")
    if global_task_index not in task_map:
        raise ValueError(f"global_task_index={global_task_index} not found in tasks.jsonl")
    if not task_map[global_task_index].strip():
        raise ValueError(f"global_task_index={global_task_index} has empty task text")
    if "Current step:" in task_map[global_task_index]:
        raise ValueError(f"global_task_index={global_task_index} must be a pure global prompt")
    if len(set(task_map.values())) != 14:
        raise ValueError(f"expected 14 unique instructions in tasks.jsonl, got {len(set(task_map.values()))}")
    return task_map


def _validate_episode_metadata(info: dict, episodes: list[dict]) -> tuple[int, int, dict[int, int]]:
    expected_indices = list(range(len(episodes)))
    actual_indices = [int(episode["episode_index"]) for episode in episodes]
    if actual_indices != expected_indices:
        raise ValueError("episodes.jsonl episode_index values must be contiguous from 0")
    length_by_episode = {int(episode["episode_index"]): int(episode["length"]) for episode in episodes}
    source_frame_count = sum(length_by_episode.values())
    if source_frame_count <= 0:
        raise ValueError("source dataset contains no frames")
    if int(info.get("total_episodes", len(episodes))) != len(episodes):
        raise ValueError("meta/info.json total_episodes does not match episodes.jsonl")
    if int(info.get("total_frames", source_frame_count)) != source_frame_count:
        raise ValueError("meta/info.json total_frames does not match episode lengths")
    return len(episodes), source_frame_count, length_by_episode


def _read_episode_table(dataset_dir: Path, info: dict, episode_index: int, columns: list[str] | None = None) -> pa.Table:
    path = dataset_dir / lerobot_io.format_data_path(info, episode_index)
    if not path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {path}")
    return pq.read_table(path, columns=columns)


def _source_fingerprints(dataset_dir: Path) -> dict[str, str]:
    return {
        relative: _sha256(dataset_dir / relative)
        for relative in ("meta/tasks.jsonl", "meta/episodes.jsonl", "meta/info.json")
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_column(table: pa.Table, name: str, values: Any) -> pa.Table:
    arrow_type = table.schema.field(name).type if name in table.column_names else None
    array = pa.array(list(values), type=arrow_type)
    if name in table.column_names:
        return table.set_column(table.column_names.index(name), name, array)
    return table.append_column(name, array)


def build_global_prompt_augmented_dataset(
    *,
    dataset_dir: Path,
    output_dir: Path,
    plan: GlobalPromptAugmentationPlan,
    overwrite: bool = False,
    progress: bool = True,
) -> GlobalPromptAugmentationSummary:
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    if dataset_dir == output_dir:
        raise ValueError("output_dir must be different from dataset_dir")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(f"Output parent directory not found: {output_dir.parent}")

    info, tasks, episodes = _load_metadata(dataset_dir)
    task_map = _validate_task_map(tasks, plan.global_task_index)
    source_episode_count, source_frame_count, length_by_episode = _validate_episode_metadata(info, episodes)
    _validate_plan_against_source(
        dataset_dir=dataset_dir,
        plan=plan,
        task_map=task_map,
        source_episode_count=source_episode_count,
        source_frame_count=source_frame_count,
        length_by_episode=length_by_episode,
    )

    source_stats = lerobot_io.read_jsonl(dataset_dir / "meta" / "episodes_stats.jsonl")
    stats_by_episode = {int(item["episode_index"]): item for item in source_stats}
    if stats_by_episode and set(stats_by_episode) != set(length_by_episode):
        raise ValueError("episodes_stats.jsonl episode indices do not match episodes.jsonl")
    source_dataset_json = lerobot_io.read_json(dataset_dir / "dataset.json")

    source_bytes = _tree_size(dataset_dir)
    duplicate_bytes = sum(
        _episode_storage_bytes(dataset_dir, info, source_episode_index)
        for source_episode_index in plan.selected_episode_indices
    )
    required_bytes = math.ceil((source_bytes + duplicate_bytes) * 1.10)
    available_bytes = shutil.disk_usage(output_dir.parent).free
    if available_bytes < required_bytes:
        raise RuntimeError(
            "Insufficient free space for physical dataset copy: "
            f"required_at_least={required_bytes} bytes, available={available_bytes} bytes"
        )

    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Output path exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()) and not overwrite:
            raise FileExistsError(f"Output directory is non-empty: {output_dir}. Use overwrite=True to replace it.")
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            output_dir.rmdir()

    if progress:
        print(
            f"Copying source dataset ({source_bytes} bytes) and preparing {len(plan.selected_episode_indices)} "
            f"physical duplicate episodes ({duplicate_bytes} bytes).",
            flush=True,
        )
    shutil.copytree(dataset_dir, output_dir, copy_function=shutil.copy2)

    new_episodes = copy.deepcopy(episodes)
    new_episode_stats = copy.deepcopy(source_stats)
    next_global_index = source_frame_count
    next_episode_index = source_episode_count
    started_at = time.monotonic()
    for duplicate_number, source_episode_index in enumerate(plan.selected_episode_indices, start=1):
        source_episode = episodes[source_episode_index]
        source_table = _read_episode_table(dataset_dir, info, source_episode_index)
        length = source_table.num_rows
        if length != int(source_episode["length"]):
            raise ValueError(
                f"episode={source_episode_index} parquet rows ({length}) do not match metadata "
                f"length ({source_episode['length']})"
            )

        new_episode_index = next_episode_index
        rewritten = source_table
        rewritten = _set_column(rewritten, "episode_index", [new_episode_index] * length)
        rewritten = _set_column(rewritten, "frame_index", range(length))
        rewritten = _set_column(rewritten, "index", range(next_global_index, next_global_index + length))
        rewritten = _set_column(rewritten, "task_index", [plan.global_task_index] * length)
        target_data_path = output_dir / lerobot_io.format_data_path(info, new_episode_index)
        target_data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(rewritten, target_data_path)

        for video_key in lerobot_io.video_keys(info):
            source_video_path = dataset_dir / lerobot_io.format_video_path(info, video_key, source_episode_index)
            target_video_path = output_dir / lerobot_io.format_video_path(info, video_key, new_episode_index)
            if not source_video_path.exists():
                raise FileNotFoundError(f"Source video not found: {source_video_path}")
            target_video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_video_path, target_video_path)

        new_episode = copy.deepcopy(source_episode)
        new_episode.update(
            {
                "episode_index": new_episode_index,
                "length": length,
                "tasks": [plan.global_task_text],
                "task_id_map": {str(plan.global_task_index): plan.global_task_text},
                "source_episode_index": source_episode_index,
                "transform_kind": "global_prompt_duplicate",
                "global_prompt_selection_trial": plan.selected_trial,
            }
        )
        new_episodes.append(new_episode)

        if stats_by_episode:
            new_stats = copy.deepcopy(stats_by_episode[source_episode_index])
        else:
            new_stats = {"episode_index": source_episode_index, "stats": {}}
        new_stats["episode_index"] = new_episode_index
        new_episode_stats.append(new_stats)

        next_global_index += length
        next_episode_index += 1
        if progress and (
            duplicate_number == 1
            or duplicate_number == len(plan.selected_episode_indices)
            or duplicate_number % 25 == 0
        ):
            print(
                f"Appended duplicate {duplicate_number}/{len(plan.selected_episode_indices)} "
                f"(source_episode={source_episode_index}, target_episode={new_episode_index}, "
                f"elapsed={time.monotonic() - started_at:.1f}s).",
                flush=True,
            )

    output_info = copy.deepcopy(info)
    output_info["total_episodes"] = len(new_episodes)
    output_info["total_frames"] = next_global_index
    output_info["total_tasks"] = len(tasks)
    output_info["total_chunks"] = math.ceil(len(new_episodes) / int(output_info["chunks_size"]))
    output_info["total_videos"] = len(new_episodes) * len(lerobot_io.video_keys(output_info))
    output_info["splits"] = {"train": f"0:{len(new_episodes)}"}

    output_meta = output_dir / "meta"
    lerobot_io.write_json(output_meta / "info.json", output_info)
    lerobot_io.write_jsonl(output_meta / "tasks.jsonl", tasks)
    lerobot_io.write_jsonl(output_meta / "episodes.jsonl", new_episodes)
    lerobot_io.write_jsonl(output_meta / "episodes_stats.jsonl", new_episode_stats)

    final_global_prompt_ratio = plan.selected_duplicate_frames / next_global_index
    audit_manifest = _plan_payload(plan)
    audit_manifest["build_summary"] = {
        "created_at": datetime.now(UTC).isoformat(),
        "copy_mode": "copy",
        "source_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "source_bytes": source_bytes,
        "selected_duplicate_storage_bytes": duplicate_bytes,
        "estimated_required_bytes_with_margin": required_bytes,
        "selected_episode_indices": list(plan.selected_episode_indices),
        "source_episodes": source_episode_count,
        "duplicate_episodes": len(plan.selected_episode_indices),
        "total_episodes": len(new_episodes),
        "source_frames": source_frame_count,
        "duplicate_frames": plan.selected_duplicate_frames,
        "total_frames": next_global_index,
        "global_task_index": plan.global_task_index,
        "total_tasks": len(tasks),
        "final_global_prompt_ratio": final_global_prompt_ratio,
    }
    lerobot_io.write_json(output_meta / "global_prompt_augmentation_manifest.json", audit_manifest)

    output_dataset_json = copy.deepcopy(source_dataset_json)
    task_name = str(output_dataset_json.get("task_name", "dataset"))
    if not task_name.endswith("_global_prompt_augmented"):
        task_name += "_global_prompt_augmented"
    output_dataset_json.update(
        {
            "task_name": task_name,
            "info": output_info,
            "tasks": copy.deepcopy(tasks),
            "episodes": copy.deepcopy(new_episodes),
            "episode_stats": copy.deepcopy(new_episode_stats),
            "files": _local_file_manifest(output_dir),
        }
    )
    lerobot_io.write_json(output_dir / "dataset.json", output_dataset_json)

    return GlobalPromptAugmentationSummary(
        output_dir=output_dir,
        source_episodes=source_episode_count,
        duplicate_episodes=len(plan.selected_episode_indices),
        total_episodes=len(new_episodes),
        source_frames=source_frame_count,
        duplicate_frames=plan.selected_duplicate_frames,
        total_frames=next_global_index,
        global_task_index=plan.global_task_index,
        total_tasks=len(tasks),
        final_global_prompt_ratio=final_global_prompt_ratio,
    )


def _validate_plan_against_source(
    *,
    dataset_dir: Path,
    plan: GlobalPromptAugmentationPlan,
    task_map: dict[int, str],
    source_episode_count: int,
    source_frame_count: int,
    length_by_episode: dict[int, int],
) -> None:
    if plan.source_dir.resolve() != dataset_dir.resolve():
        raise ValueError(f"manifest source_dir does not match dataset_dir: {plan.source_dir} != {dataset_dir}")
    current_fingerprints = _source_fingerprints(dataset_dir)
    if current_fingerprints != plan.source_fingerprints:
        raise ValueError(
            "manifest fingerprint mismatch: source metadata changed after planning; "
            f"planned={plan.source_fingerprints}, current={current_fingerprints}"
        )
    if plan.global_task_text != task_map[plan.global_task_index]:
        raise ValueError("manifest global task text does not match tasks.jsonl")
    if plan.source_episode_count != source_episode_count or plan.source_frame_count != source_frame_count:
        raise ValueError("manifest source episode/frame totals do not match the current dataset")
    selected = plan.selected_episode_indices
    if len(selected) != plan.duplicate_episode_count:
        raise ValueError("manifest selected episode count does not match duplicate_episode_count")
    if selected != tuple(sorted(set(selected))):
        raise ValueError("manifest selected episode indices must be sorted and unique")
    if any(index not in length_by_episode for index in selected):
        raise ValueError("manifest selected episode index is outside the source episode range")
    duplicate_frames = sum(length_by_episode[index] for index in selected)
    if duplicate_frames != plan.selected_duplicate_frames:
        raise ValueError("manifest selected_duplicate_frames does not match source episode lengths")
    if not math.isclose(
        duplicate_frames / source_frame_count,
        plan.selected_duplicate_frame_ratio,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("manifest selected_duplicate_frame_ratio does not match source episode lengths")


def _tree_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _episode_storage_bytes(dataset_dir: Path, info: dict, episode_index: int) -> int:
    paths = [dataset_dir / lerobot_io.format_data_path(info, episode_index)]
    paths.extend(
        dataset_dir / lerobot_io.format_video_path(info, video_key, episode_index)
        for video_key in lerobot_io.video_keys(info)
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Selected episode files are missing: {missing}")
    return sum(path.stat().st_size for path in paths)


def _local_file_manifest(root: Path) -> list[dict[str, str]]:
    files = []
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = path.relative_to(root)
        suffix = relative.suffix.lstrip(".") or "file"
        files.append({"path": str(relative), "type": suffix})
    return files


def validate_global_prompt_augmented_dataset(
    *,
    dataset_dir: Path,
    plan: GlobalPromptAugmentationPlan,
    action_horizon: int,
) -> GlobalPromptAugmentationValidation:
    if action_horizon < 1:
        raise ValueError("action_horizon must be >= 1")
    dataset_dir = dataset_dir.resolve()
    source_dir = plan.source_dir.resolve()
    source_info, source_tasks, source_episodes = _load_metadata(source_dir)
    source_task_map = _validate_task_map(source_tasks, plan.global_task_index)
    source_episode_count, source_frame_count, source_lengths = _validate_episode_metadata(source_info, source_episodes)
    _validate_plan_against_source(
        dataset_dir=source_dir,
        plan=plan,
        task_map=source_task_map,
        source_episode_count=source_episode_count,
        source_frame_count=source_frame_count,
        length_by_episode=source_lengths,
    )

    info, tasks, episodes = _load_metadata(dataset_dir)
    failures: list[str] = []
    try:
        task_map = _validate_task_map(tasks, plan.global_task_index)
    except ValueError as exc:
        failures.append(f"task_records: {exc}")
        task_map = {int(task["task_index"]): str(task.get("task", "")) for task in tasks}
    if task_map != source_task_map:
        failures.append("task_records: output task mapping differs from source task mapping")
    if len(tasks) != 40:
        failures.append(f"task_records: expected 40 rows, got {len(tasks)}")
    if len(set(task_map.values())) != 14:
        failures.append(f"task_records: expected 14 unique instruction strings, got {len(set(task_map.values()))}")

    expected_total_episodes = source_episode_count + plan.duplicate_episode_count
    expected_total_frames = source_frame_count + plan.selected_duplicate_frames
    actual_episode_indices = [int(episode["episode_index"]) for episode in episodes]
    expected_episode_indices = list(range(expected_total_episodes))
    if actual_episode_indices != expected_episode_indices:
        failures.append("episode_records: episode_index values are not contiguous")
    if len(episodes) != expected_total_episodes:
        failures.append(f"episode_records: expected {expected_total_episodes} episodes, got {len(episodes)}")

    lengths_by_output_episode: dict[int, int] = {}
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        lengths_by_output_episode[episode_index] = int(episode.get("length", -1))
    measured_total_frames = sum(length for length in lengths_by_output_episode.values() if length >= 0)
    if int(info.get("total_episodes", -1)) != expected_total_episodes:
        failures.append("info_totals: total_episodes is incorrect")
    if int(info.get("total_frames", -1)) != measured_total_frames:
        failures.append("info_totals: total_frames does not equal episode lengths")
    if measured_total_frames != expected_total_frames:
        failures.append("info_totals: output frame total does not match source plus selected duplicates")
    if int(info.get("total_tasks", -1)) != 40:
        failures.append("info_totals: total_tasks is not 40")
    expected_split = f"0:{expected_total_episodes}"
    if (info.get("splits") or {}).get("train") != expected_split:
        failures.append(f"info_totals: splits.train is not {expected_split!r}")

    source_task_sequences: dict[int, list[int]] = {}
    task_index_counts: Counter[int] = Counter()
    measured_duplicate_frames = 0
    next_global_index = 0
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        table = _read_episode_table(
            dataset_dir,
            info,
            episode_index,
            columns=["episode_index", "frame_index", "index", "task_index"],
        )
        length = table.num_rows
        expected_length = lengths_by_output_episode.get(episode_index)
        if expected_length != length:
            failures.append(f"episode_rows: episode={episode_index} parquet length differs from metadata")
        episode_values = {int(value) for value in table["episode_index"].to_pylist()}
        if episode_values != {episode_index}:
            failures.append(f"episode_rows: episode={episode_index} has unexpected episode_index values")
        frame_indices = table["frame_index"].to_pylist()
        if frame_indices != list(range(length)):
            failures.append(f"episode_rows: episode={episode_index} frame_index is not locally contiguous")
        global_indices = [int(value) for value in table["index"].to_pylist()]
        expected_indices = list(range(next_global_index, next_global_index + length))
        if global_indices != expected_indices:
            failures.append(f"global_index: episode={episode_index} is not continuous at {next_global_index}")
        next_global_index += length

        task_indices = [int(value) for value in table["task_index"].to_pylist()]
        task_index_counts.update(task_indices)
        unknown_task_ids = set(task_indices).difference(task_map)
        if unknown_task_ids:
            failures.append(f"task_ids: episode={episode_index} has unknown IDs {sorted(unknown_task_ids)}")
        if episode_index < source_episode_count:
            source_table = _read_episode_table(source_dir, source_info, episode_index, columns=["task_index"])
            source_task_indices = [int(value) for value in source_table["task_index"].to_pylist()]
            source_task_sequences[episode_index] = source_task_indices
            if task_indices != source_task_indices:
                failures.append(f"original_task_sequences: episode={episode_index} changed")
        else:
            measured_duplicate_frames += length
            if set(task_indices) != {plan.global_task_index}:
                failures.append(
                    f"duplicate_task_index: episode={episode_index} expected only task_index={plan.global_task_index}, "
                    f"got={sorted(set(task_indices))}"
                )
            if episode.get("transform_kind") != "global_prompt_duplicate":
                failures.append(f"duplicate_metadata: episode={episode_index} has wrong transform_kind")

    if next_global_index != int(info.get("total_frames", -1)):
        failures.append("global_index: total frame count does not match info.total_frames")
    if measured_duplicate_frames != plan.selected_duplicate_frames:
        failures.append("duplicate_frames: measured duplicate frame count differs from manifest")
    if set(task_index_counts).difference(task_map):
        failures.append("task_ids: output contains IDs absent from tasks.jsonl")

    for duplicate_offset, source_episode_index in enumerate(plan.selected_episode_indices):
        target_episode_index = source_episode_count + duplicate_offset
        for video_key in lerobot_io.video_keys(info):
            source_video = source_dir / lerobot_io.format_video_path(source_info, video_key, source_episode_index)
            target_video = dataset_dir / lerobot_io.format_video_path(info, video_key, target_episode_index)
            if not target_video.is_file():
                failures.append(f"duplicate_videos: missing episode={target_episode_index} key={video_key}")
            elif not source_video.is_file() or target_video.stat().st_ino == source_video.stat().st_ino:
                failures.append(f"duplicate_videos: episode={target_episode_index} key={video_key} is not a physical copy")

    try:
        from openpi.transforms import PromptFromLeRobotTask

        prompt_transform = PromptFromLeRobotTask(task_map)
        global_prompt = prompt_transform({"task_index": plan.global_task_index})["prompt"]
        if global_prompt != plan.global_task_text:
            failures.append("prompt_mapping: global task prompt differs from manifest")
        non_global_prompts = [text for index, text in task_map.items() if index != plan.global_task_index]
        if not any("Current step:" in text for text in non_global_prompts):
            failures.append("prompt_mapping: original hierarchical prompt lacks Current step:")
    except (ImportError, KeyError, ValueError) as exc:
        failures.append(f"prompt_mapping: {exc}")

    manifest_path = dataset_dir / "meta" / "global_prompt_augmentation_manifest.json"
    output_manifest = lerobot_io.read_json(manifest_path)
    if output_manifest.get("source_fingerprints") != _source_fingerprints(source_dir):
        failures.append("manifest_fingerprint: output audit manifest does not match source")
    if output_manifest.get("selected_episode_indices") != list(plan.selected_episode_indices):
        failures.append("manifest_fingerprint: output audit manifest has a different selection")

    action_horizon_crossing_frames = 0
    action_horizon_checked_frames = 0
    for episode_index in range(source_episode_count):
        task_indices = source_task_sequences.get(episode_index)
        if task_indices is None:
            source_table = _read_episode_table(source_dir, source_info, episode_index, columns=["task_index"])
            task_indices = [int(value) for value in source_table["task_index"].to_pylist()]
        checked = max(0, len(task_indices) - action_horizon + 1)
        action_horizon_checked_frames += checked
        action_horizon_crossing_frames += sum(
            len(set(task_indices[start : start + action_horizon])) != 1 for start in range(checked)
        )

    if failures:
        raise ValueError("Global prompt augmentation validation failed: " + "; ".join(failures))

    return GlobalPromptAugmentationValidation(
        total_episodes=len(episodes),
        total_frames=next_global_index,
        duplicate_episodes=plan.duplicate_episode_count,
        duplicate_frames=measured_duplicate_frames,
        global_prompt_frame_ratio=measured_duplicate_frames / next_global_index,
        task_index_counts=dict(sorted(task_index_counts.items())),
        action_horizon_crossing_frames=action_horizon_crossing_frames,
        action_horizon_checked_frames=action_horizon_checked_frames,
    )
