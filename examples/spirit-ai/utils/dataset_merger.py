from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
import shutil
import time

import pyarrow as pa
import pyarrow.parquet as pq

from utils import lerobot_io


@dataclasses.dataclass(frozen=True)
class MergeSummary:
    output_dir: Path
    total_episodes: int
    total_frames: int
    total_tasks: int
    base_episodes: int
    base_frames: int
    supplement_episodes: int
    supplement_frames: int
    supplement_repeat: int


@dataclasses.dataclass
class _MergeState:
    episodes: list[dict]
    episode_stats: list[dict]
    tasks: list[dict]
    task_to_index: dict[str, int]
    next_episode_index: int
    next_global_frame_index: int


@dataclasses.dataclass
class _ProgressTracker:
    label: str
    total: int
    enabled: bool
    unit: str = "items"
    completed: int = 0
    started_at: float = dataclasses.field(default_factory=time.monotonic)
    last_report_at: float = dataclasses.field(default_factory=time.monotonic)

    def start(self) -> None:
        if not self.enabled:
            return
        print(f"{self.label}: start total={self.total} {self.unit}", flush=True)

    def advance(self, count: int = 1, *, force: bool = False, details: str = "") -> None:
        self.completed += count
        if not self.enabled:
            return
        now = time.monotonic()
        should_report = (
            force
            or self.completed == 1
            or self.completed == self.total
            or self.completed % 25 == 0
            or now - self.last_report_at >= 5.0
        )
        if not should_report:
            return
        self.last_report_at = now
        percent = 100.0 if self.total == 0 else min(100.0, self.completed / self.total * 100.0)
        suffix = f" {details}" if details else ""
        print(
            (
                f"{self.label}: {percent:5.1f}% "
                f"({self.completed}/{self.total} {self.unit}), "
                f"elapsed={now - self.started_at:.1f}s{suffix}"
            ),
            flush=True,
        )

    def finish(self) -> None:
        if not self.enabled:
            return
        elapsed = time.monotonic() - self.started_at
        print(f"{self.label}: done {self.completed}/{self.total} {self.unit}, elapsed={elapsed:.1f}s", flush=True)


def merge_datasets(
    *,
    base_dir: Path,
    supplement_dir: Path,
    output_dir: Path,
    supplement_repeat: int = 1,
    overwrite: bool = False,
    progress: bool = True,
) -> MergeSummary:
    if supplement_repeat < 1:
        raise ValueError("supplement_repeat must be >= 1")
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Base dataset directory not found: {base_dir}")
    if not supplement_dir.is_dir():
        raise FileNotFoundError(f"Supplement dataset directory not found: {supplement_dir}")
    if output_dir == base_dir or output_dir == supplement_dir:
        raise ValueError("output_dir must be different from all input dataset directories")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    base = _load_dataset(base_dir)
    supplement = _load_dataset(supplement_dir)
    _validate_compatible(base, supplement)

    _progress("Step 1/5: Copy base dataset", enabled=progress)
    copy_started_at = time.monotonic()
    _progress(f"  source: {base_dir}", enabled=progress)
    _progress(f"  target: {output_dir}", enabled=progress)
    shutil.copytree(base_dir, output_dir)
    _progress(f"  done in {time.monotonic() - copy_started_at:.1f}s", enabled=progress)

    _progress("Step 2/5: Initialize merged metadata state", enabled=progress)
    state = _init_state(base)
    supplement_frames_per_repeat = _sum_episode_lengths(supplement["episodes"])
    total_supplement_outputs = len(supplement["episodes"]) * supplement_repeat
    total_video_copies = total_supplement_outputs * len(lerobot_io.video_keys(base["info"]))
    _progress(
        (
            f"Appending supplement: episodes={len(supplement['episodes'])}, repeat={supplement_repeat}, "
            f"frames_per_repeat={supplement_frames_per_repeat}, video_copies={total_video_copies}"
        ),
        enabled=progress,
    )
    _progress(f"  next_episode_index={state.next_episode_index}", enabled=progress)
    _progress(f"  next_global_frame_index={state.next_global_frame_index}", enabled=progress)

    _progress("Step 3/5: Append supplement episodes and copy videos", enabled=progress)
    append_progress = _ProgressTracker(
        label="Appending supplement episodes",
        total=total_supplement_outputs,
        enabled=progress,
        unit="episodes",
    )
    append_progress.start()
    appended = 0
    appended_frames = 0
    for repeat_index in range(supplement_repeat):
        for source_episode in supplement["episodes"]:
            appended += 1
            length = _append_episode(
                supplement_dir=supplement_dir,
                output_dir=output_dir,
                output_info=base["info"],
                supplement_info=supplement["info"],
                source_episode=source_episode,
                source_episode_stats=supplement["episode_stats_by_index"].get(int(source_episode["episode_index"])),
                source_dataset_name=supplement_dir.name,
                repeat_index=repeat_index,
                state=state,
            )
            appended_frames += length
            append_progress.advance(
                details=(
                    f"frames={appended_frames}/{supplement_frames_per_repeat * supplement_repeat} "
                    f"source_episode={source_episode['episode_index']} "
                    f"repeat={repeat_index + 1}/{supplement_repeat}"
                )
            )
    append_progress.finish()

    _progress("Step 4/5: Write merged metadata", enabled=progress)
    _write_metadata(
        output_dir=output_dir,
        base_dataset_json=base["dataset_json"],
        base_info=base["info"],
        state=state,
    )
    _progress("  metadata written", enabled=progress)

    supplement_frames = supplement_frames_per_repeat * supplement_repeat
    _progress("Step 5/5: Merge summary", enabled=progress)
    _progress(f"  total_episodes={len(state.episodes)}", enabled=progress)
    _progress(f"  total_frames={state.next_global_frame_index}", enabled=progress)
    if state.next_global_frame_index:
        _progress(f"  supplement_fraction={supplement_frames / state.next_global_frame_index:.2%}", enabled=progress)
    return MergeSummary(
        output_dir=output_dir,
        total_episodes=len(state.episodes),
        total_frames=state.next_global_frame_index,
        total_tasks=len(state.tasks),
        base_episodes=len(base["episodes"]),
        base_frames=_sum_episode_lengths(base["episodes"]),
        supplement_episodes=len(supplement["episodes"]) * supplement_repeat,
        supplement_frames=supplement_frames,
        supplement_repeat=supplement_repeat,
    )


def _progress(message: str, *, enabled: bool) -> None:
    if enabled:
        print(message, flush=True)


def _load_dataset(dataset_dir: Path) -> dict:
    meta_dir = dataset_dir / "meta"
    tasks = lerobot_io.read_jsonl(meta_dir / "tasks.jsonl")
    episodes = lerobot_io.read_jsonl(meta_dir / "episodes.jsonl")
    episode_stats_path = meta_dir / "episodes_stats.jsonl"
    episode_stats = lerobot_io.read_jsonl(episode_stats_path) if episode_stats_path.exists() else []
    return {
        "dir": dataset_dir,
        "info": lerobot_io.read_json(meta_dir / "info.json"),
        "tasks": tasks,
        "episodes": episodes,
        "episode_stats": episode_stats,
        "episode_stats_by_index": {
            int(item["episode_index"]): item for item in episode_stats if "episode_index" in item
        },
        "dataset_json": lerobot_io.read_json(dataset_dir / "dataset.json"),
    }


def _validate_compatible(base: dict, supplement: dict) -> None:
    base_info = base["info"]
    supplement_info = supplement["info"]
    for key in ("robot_type", "fps", "data_path", "video_path", "chunks_size"):
        if base_info.get(key) != supplement_info.get(key):
            raise ValueError(
                f"Incompatible dataset info field {key!r}: "
                f"base={base_info.get(key)!r}, supplement={supplement_info.get(key)!r}"
            )
    if base_info.get("features") != supplement_info.get("features"):
        raise ValueError("Input datasets have different feature schemas.")
    if lerobot_io.video_keys(base_info) != lerobot_io.video_keys(supplement_info):
        raise ValueError("Input datasets have different video keys.")
    _validate_episode_indices(base["episodes"], label="base")
    _validate_episode_indices(supplement["episodes"], label="supplement")


def _validate_episode_indices(episodes: list[dict], *, label: str) -> None:
    expected = list(range(len(episodes)))
    actual = [int(episode["episode_index"]) for episode in episodes]
    if actual != expected:
        raise ValueError(f"{label} episode_index values must be contiguous from 0; got first values {actual[:10]}")


def _init_state(base: dict) -> _MergeState:
    tasks = [copy.deepcopy(task) for task in base["tasks"]]
    task_to_index = {}
    for task in tasks:
        task_to_index.setdefault(str(task["task"]), int(task["task_index"]))
    episodes = [copy.deepcopy(episode) for episode in base["episodes"]]
    episode_stats = [copy.deepcopy(item) for item in base["episode_stats"]]
    if not episode_stats:
        episode_stats = [{"episode_index": int(episode["episode_index"]), "stats": {}} for episode in episodes]
    return _MergeState(
        episodes=episodes,
        episode_stats=episode_stats,
        tasks=tasks,
        task_to_index=task_to_index,
        next_episode_index=len(episodes),
        next_global_frame_index=_sum_episode_lengths(episodes),
    )


def _sum_episode_lengths(episodes: list[dict]) -> int:
    return sum(int(episode["length"]) for episode in episodes)


def _append_episode(
    *,
    supplement_dir: Path,
    output_dir: Path,
    output_info: dict,
    supplement_info: dict,
    source_episode: dict,
    source_episode_stats: dict | None,
    source_dataset_name: str,
    repeat_index: int,
    state: _MergeState,
) -> int:
    source_episode_index = int(source_episode["episode_index"])
    new_episode_index = state.next_episode_index
    task_text = _episode_task_text(source_episode)
    task_index = _task_index_for_text(state, task_text)

    source_table = _read_episode_table(supplement_dir, supplement_info, source_episode_index)
    table = _rewrite_episode_columns(
        source_table,
        episode_index=new_episode_index,
        task_index=task_index,
        global_frame_start=state.next_global_frame_index,
    )
    _write_episode_table(output_dir, output_info, new_episode_index, table)
    _copy_episode_videos(
        supplement_dir=supplement_dir,
        output_dir=output_dir,
        supplement_info=supplement_info,
        output_info=output_info,
        source_episode_index=source_episode_index,
        new_episode_index=new_episode_index,
    )
    _record_episode(
        source_episode=source_episode,
        source_episode_stats=source_episode_stats,
        source_dataset_name=source_dataset_name,
        repeat_index=repeat_index,
        new_episode_index=new_episode_index,
        length=table.num_rows,
        task_index=task_index,
        task_text=task_text,
        state=state,
    )
    return table.num_rows


def _episode_task_text(episode: dict) -> str:
    tasks = episode.get("tasks") or []
    if not tasks:
        task_id_map = episode.get("task_id_map") or {}
        tasks = list(task_id_map.values())
    if len(tasks) != 1:
        raise ValueError(f"Expected each supplement episode to have exactly one task, got {tasks!r}")
    return str(tasks[0])


def _task_index_for_text(state: _MergeState, task_text: str) -> int:
    if task_text in state.task_to_index:
        return state.task_to_index[task_text]
    task_index = len(state.tasks)
    state.task_to_index[task_text] = task_index
    state.tasks.append({"task_index": task_index, "task": task_text})
    return task_index


def _read_episode_table(dataset_dir: Path, info: dict, episode_index: int) -> pa.Table:
    path = dataset_dir / lerobot_io.format_data_path(info, episode_index)
    if not path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {path}")
    return pq.read_table(path)


def _write_episode_table(output_dir: Path, info: dict, episode_index: int, table: pa.Table) -> None:
    path = output_dir / lerobot_io.format_data_path(info, episode_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _rewrite_episode_columns(
    table: pa.Table,
    *,
    episode_index: int,
    task_index: int,
    global_frame_start: int,
) -> pa.Table:
    length = table.num_rows
    result = table
    result = _set_column(result, "episode_index", [episode_index] * length)
    result = _set_column(result, "frame_index", range(length))
    result = _set_column(result, "index", range(global_frame_start, global_frame_start + length))
    result = _set_column(result, "task_index", [task_index] * length)
    return result


def _set_column(table: pa.Table, name: str, values) -> pa.Table:
    arrow_type = table.schema.field(name).type if name in table.column_names else None
    array = pa.array(list(values), type=arrow_type)
    if name in table.column_names:
        return table.set_column(table.column_names.index(name), name, array)
    return table.append_column(name, array)


def _copy_episode_videos(
    *,
    supplement_dir: Path,
    output_dir: Path,
    supplement_info: dict,
    output_info: dict,
    source_episode_index: int,
    new_episode_index: int,
) -> None:
    for video_key in lerobot_io.video_keys(supplement_info):
        source = supplement_dir / lerobot_io.format_video_path(supplement_info, video_key, source_episode_index)
        if not source.exists():
            raise FileNotFoundError(f"Missing source video: episode={source_episode_index} key={video_key} path={source}")
        target = output_dir / lerobot_io.format_video_path(output_info, video_key, new_episode_index)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _record_episode(
    *,
    source_episode: dict,
    source_episode_stats: dict | None,
    source_dataset_name: str,
    repeat_index: int,
    new_episode_index: int,
    length: int,
    task_index: int,
    task_text: str,
    state: _MergeState,
) -> None:
    episode = copy.deepcopy(source_episode)
    episode["episode_index"] = new_episode_index
    episode["length"] = length
    episode["tasks"] = [task_text]
    episode["task_id_map"] = {str(task_index): task_text}
    episode["source_dataset"] = source_dataset_name
    episode["source_episode_index"] = int(source_episode["episode_index"])
    episode["source_repeat_index"] = repeat_index
    episode["transform_kind"] = episode.get("transform_kind", "merged_supplement")

    if source_episode_stats is not None:
        episode_stats = copy.deepcopy(source_episode_stats)
        episode_stats["episode_index"] = new_episode_index
    else:
        episode_stats = {"episode_index": new_episode_index, "stats": {}}

    state.episodes.append(episode)
    state.episode_stats.append(episode_stats)
    state.next_episode_index += 1
    state.next_global_frame_index += length


def _write_metadata(
    *,
    output_dir: Path,
    base_dataset_json: dict,
    base_info: dict,
    state: _MergeState,
) -> None:
    new_info = dict(base_info)
    new_info["total_episodes"] = len(state.episodes)
    new_info["total_frames"] = state.next_global_frame_index
    new_info["total_tasks"] = len(state.tasks)

    lerobot_io.write_json(output_dir / "meta" / "info.json", new_info)
    lerobot_io.write_jsonl(output_dir / "meta" / "tasks.jsonl", state.tasks)
    lerobot_io.write_jsonl(output_dir / "meta" / "episodes.jsonl", state.episodes)
    lerobot_io.write_jsonl(output_dir / "meta" / "episodes_stats.jsonl", state.episode_stats)

    dataset_json = dict(base_dataset_json)
    dataset_json["task_name"] = f"{base_dataset_json.get('task_name', 'dataset')}_merged"
    dataset_json["info"] = new_info
    dataset_json["tasks"] = state.tasks
    dataset_json["episodes"] = state.episodes
    dataset_json["episode_stats"] = state.episode_stats
    files = lerobot_io.relative_files(output_dir)
    if "dataset.json" not in files:
        files.append("dataset.json")
    dataset_json["files"] = sorted(files)
    lerobot_io.write_json(output_dir / "dataset.json", dataset_json)
