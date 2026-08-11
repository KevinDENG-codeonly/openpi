"""Plan and build physical global-prompt augmentations for LeRobot datasets."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
import random
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
    selected_trial = plan.selected_trial
    payload = {
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
    lerobot_io.write_json(path, payload)


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
