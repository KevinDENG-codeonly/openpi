from __future__ import annotations

import concurrent.futures
import dataclasses
import copy
import shutil
import subprocess
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from utils import annotations
from utils import instructions
from utils import lerobot_io


VideoMode = Literal["link-full", "copy-full", "slice"]


@dataclasses.dataclass(frozen=True)
class BuildSummary:
    output_dir: Path
    total_episodes: int
    total_frames: int
    total_tasks: int
    global_episodes: int
    subtask_episodes: int
    skipped_subtasks: int


@dataclasses.dataclass
class _BuildState:
    next_episode_index: int = 0
    next_global_frame_index: int = 0
    episodes: list[dict] = dataclasses.field(default_factory=list)
    episode_stats: list[dict] = dataclasses.field(default_factory=list)
    total_frames: int = 0
    global_episodes: int = 0
    subtask_episodes: int = 0
    skipped_subtasks: int = 0


def _progress(message: str, *, enabled: bool) -> None:
    if enabled:
        print(message, flush=True)


def build_multiscale_dataset(
    *,
    dataset_dir: Path,
    output_dir: Path,
    global_prompt: str,
    overwrite: bool = False,
    slice_episodes: bool = False,
    global_repeat: int = 1,
    subtask_repeat: int = 1,
    prefer_english: bool = True,
    video_mode: VideoMode = "link-full",
    min_slice_frames: int = 2,
    progress: bool = True,
    video_workers: int = 1,
) -> BuildSummary:
    if global_repeat < 0 or subtask_repeat < 0:
        raise ValueError("global_repeat and subtask_repeat must be >= 0")
    if video_mode not in {"link-full", "copy-full", "slice"}:
        raise ValueError(f"Unsupported video_mode: {video_mode}")
    if video_workers < 1:
        raise ValueError("video_workers must be >= 1")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if slice_episodes and video_mode == "slice" and shutil.which("ffmpeg") is None:
        raise RuntimeError("--video-mode slice requires ffmpeg to be available on PATH")

    meta_dir = dataset_dir / "meta"
    info = lerobot_io.read_json(meta_dir / "info.json")
    episodes = lerobot_io.read_jsonl(meta_dir / "episodes.jsonl")
    source_dataset_json = lerobot_io.read_json(dataset_dir / "dataset.json")
    total_global = len(episodes) * global_repeat
    total_subtasks = 0
    if slice_episodes:
        total_subtasks = sum(
            len(annotations.usable_subtask_segments(episode, prefer_english=prefer_english)) for episode in episodes
        ) * subtask_repeat

    _progress(
        (
            f"Building multiscale dataset: {len(episodes)} source episodes, "
            f"{total_global} global outputs, {total_subtasks} subtask outputs, "
            f"video_mode={video_mode}, video_workers={video_workers}"
        ),
        enabled=progress,
    )

    lerobot_io.prepare_output_dir(dataset_dir, output_dir, overwrite=overwrite)
    lerobot_io.copy_non_episode_files(dataset_dir, output_dir)

    global_task = instructions.normalize_text(global_prompt)
    task_to_index = {global_task: 0}
    tasks = [{"task_index": 0, "task": global_task}]
    if slice_episodes:
        for episode in episodes:
            for segment in annotations.usable_subtask_segments(episode, prefer_english=prefer_english):
                prompt = instructions.overview_subtask_prompt(global_task, segment["_prompt_text"])
                if prompt not in task_to_index:
                    task_to_index[prompt] = len(tasks)
                    tasks.append({"task_index": task_to_index[prompt], "task": prompt})

    state = _BuildState()

    global_counter = 0
    for repeat_index in range(global_repeat):
        for episode in episodes:
            global_counter += 1
            source_episode_index = int(episode["episode_index"])
            if global_counter == 1 or global_counter % 25 == 0 or global_counter == total_global:
                _progress(
                    (
                        f"[global {global_counter}/{total_global}] "
                        f"source_episode={source_episode_index} repeat={repeat_index + 1}/{global_repeat}"
                    ),
                    enabled=progress,
                )
            _append_full_episode(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                info=info,
                source_episode=episode,
                task_index=0,
                task_text=global_task,
                state=state,
                video_mode="copy-full" if video_mode == "copy-full" else "link-full",
                progress=progress,
                video_workers=video_workers,
            )

    if slice_episodes:
        subtask_counter = 0
        for repeat_index in range(subtask_repeat):
            for episode in episodes:
                source_episode_index = int(episode["episode_index"])
                source_table = _read_source_table(dataset_dir, info, source_episode_index)
                segments = annotations.usable_subtask_segments(episode, prefer_english=prefer_english)
                for segment_index, segment in enumerate(segments, start=1):
                    subtask_counter += 1
                    prompt = instructions.overview_subtask_prompt(global_task, segment["_prompt_text"])
                    if subtask_counter == 1 or subtask_counter % 25 == 0 or subtask_counter == total_subtasks:
                        _progress(
                            (
                                f"[subtask {subtask_counter}/{total_subtasks}] "
                                f"source_episode={source_episode_index} segment={segment_index}/{len(segments)} "
                                f"time={segment['start_time']:.3f}-{segment['end_time']:.3f} "
                                f"repeat={repeat_index + 1}/{subtask_repeat}"
                            ),
                            enabled=progress,
                        )
                    _append_sliced_episode(
                        dataset_dir=dataset_dir,
                        output_dir=output_dir,
                        info=info,
                        source_episode=episode,
                        source_table=source_table,
                        segment=segment,
                        task_index=task_to_index[prompt],
                        task_text=prompt,
                        state=state,
                        video_mode=video_mode,
                        min_slice_frames=min_slice_frames,
                        progress=progress,
                        video_workers=video_workers,
                    )

    _write_metadata(
        output_dir=output_dir,
        source_dataset_json=source_dataset_json,
        info=info,
        tasks=tasks,
        state=state,
    )
    return BuildSummary(
        output_dir=output_dir,
        total_episodes=len(state.episodes),
        total_frames=state.total_frames,
        total_tasks=len(tasks),
        global_episodes=state.global_episodes,
        subtask_episodes=state.subtask_episodes,
        skipped_subtasks=state.skipped_subtasks,
    )


def _append_full_episode(
    *,
    dataset_dir: Path,
    output_dir: Path,
    info: dict,
    source_episode: dict,
    task_index: int,
    task_text: str,
    state: _BuildState,
    video_mode: VideoMode,
    progress: bool,
    video_workers: int,
) -> None:
    source_episode_index = int(source_episode["episode_index"])
    source_table = _read_source_table(dataset_dir, info, source_episode_index)
    new_episode_index = state.next_episode_index
    table = _rewrite_episode_columns(
        source_table,
        episode_index=new_episode_index,
        task_index=task_index,
        global_frame_start=state.next_global_frame_index,
        reset_timestamp=True,
    )
    _write_episode_table(output_dir, info, new_episode_index, table)
    _copy_or_slice_videos(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        info=info,
        source_episode_index=source_episode_index,
        new_episode_index=new_episode_index,
        video_mode=video_mode,
        start_time=None,
        end_time=None,
        progress=progress,
        video_workers=video_workers,
    )
    _record_episode(
        source_episode=source_episode,
        state=state,
        new_episode_index=new_episode_index,
        length=table.num_rows,
        task_index=task_index,
        task_text=task_text,
        kind="global",
    )
    state.global_episodes += 1


def _append_sliced_episode(
    *,
    dataset_dir: Path,
    output_dir: Path,
    info: dict,
    source_episode: dict,
    source_table: pa.Table,
    segment: dict,
    task_index: int,
    task_text: str,
    state: _BuildState,
    video_mode: VideoMode,
    min_slice_frames: int,
    progress: bool,
    video_workers: int,
) -> None:
    timestamp = source_table["timestamp"]
    mask = pc.and_(
        pc.greater_equal(timestamp, pa.scalar(segment["start_time"], type=timestamp.type)),
        pc.less(timestamp, pa.scalar(segment["end_time"], type=timestamp.type)),
    )
    table = source_table.filter(mask)
    if table.num_rows < min_slice_frames:
        state.skipped_subtasks += 1
        return

    new_episode_index = state.next_episode_index
    reset_timestamp = video_mode == "slice"
    table = _rewrite_episode_columns(
        table,
        episode_index=new_episode_index,
        task_index=task_index,
        global_frame_start=state.next_global_frame_index,
        reset_timestamp=reset_timestamp,
    )
    _write_episode_table(output_dir, info, new_episode_index, table)
    _copy_or_slice_videos(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        info=info,
        source_episode_index=int(source_episode["episode_index"]),
        new_episode_index=new_episode_index,
        video_mode=video_mode,
        start_time=segment["start_time"],
        end_time=segment["end_time"],
        progress=progress,
        video_workers=video_workers,
    )
    _record_episode(
        source_episode=source_episode,
        state=state,
        new_episode_index=new_episode_index,
        length=table.num_rows,
        task_index=task_index,
        task_text=task_text,
        kind="subtask",
        segment=segment,
    )
    state.subtask_episodes += 1


def _read_source_table(dataset_dir: Path, info: dict, episode_index: int) -> pa.Table:
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
    reset_timestamp: bool,
) -> pa.Table:
    length = table.num_rows
    result = table
    result = _set_column(result, "episode_index", [episode_index] * length)
    result = _set_column(result, "frame_index", range(length))
    result = _set_column(result, "index", range(global_frame_start, global_frame_start + length))
    result = _set_column(result, "task_index", [task_index] * length)
    if reset_timestamp and "timestamp" in result.column_names and length:
        result = _zero_base_numeric_column(result, "timestamp")
    if reset_timestamp and "timestamp_perf" in result.column_names and length:
        result = _zero_base_numeric_column(result, "timestamp_perf")
    return result


def _set_column(table: pa.Table, name: str, values) -> pa.Table:
    arrow_type = table.schema.field(name).type if name in table.column_names else None
    array = pa.array(list(values), type=arrow_type)
    if name in table.column_names:
        return table.set_column(table.column_names.index(name), name, array)
    return table.append_column(name, array)


def _zero_base_numeric_column(table: pa.Table, name: str) -> pa.Table:
    column = table[name]
    first_value = column[0].as_py()
    values = [value.as_py() - first_value for value in column]
    return _set_column(table, name, values)


def _copy_or_slice_videos(
    *,
    dataset_dir: Path,
    output_dir: Path,
    info: dict,
    source_episode_index: int,
    new_episode_index: int,
    video_mode: VideoMode,
    start_time: float | None,
    end_time: float | None,
    progress: bool,
    video_workers: int,
) -> None:
    keys = lerobot_io.video_keys(info)
    slice_jobs = []
    for video_number, video_key in enumerate(keys, start=1):
        source = dataset_dir / lerobot_io.format_video_path(info, video_key, source_episode_index)
        if not source.exists():
            continue
        target = output_dir / lerobot_io.format_video_path(info, video_key, new_episode_index)
        if video_mode == "slice" and start_time is not None and end_time is not None:
            _progress(
                (
                    f"  slicing video {video_number}/{len(keys)} key={video_key} "
                    f"source_episode={source_episode_index} -> episode={new_episode_index} "
                    f"time={start_time:.3f}-{end_time:.3f}"
                ),
                enabled=progress,
            )
            slice_jobs.append(
                {
                    "source": source,
                    "target": target,
                    "start_time": start_time,
                    "end_time": end_time,
                    "video_key": video_key,
                    "source_episode_index": source_episode_index,
                    "new_episode_index": new_episode_index,
                }
            )
        else:
            mode = "copy" if video_mode == "copy-full" else "hardlink"
            lerobot_io.link_or_copy_file(source, target, mode=mode)

    if slice_jobs:
        _run_slice_jobs(slice_jobs, video_workers=video_workers)


def _run_slice_jobs(slice_jobs: list[dict], *, video_workers: int) -> None:
    if video_workers == 1 or len(slice_jobs) == 1:
        for job in slice_jobs:
            _slice_video(**job)
        return

    max_workers = min(video_workers, len(slice_jobs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_slice_video, **job) for job in slice_jobs]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def _slice_video(
    source: Path,
    target: Path,
    *,
    start_time: float,
    end_time: float,
    video_key: str,
    source_episode_index: int,
    new_episode_index: int,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_time:.6f}",
        "-to",
        f"{end_time:.6f}",
        "-i",
        str(source),
        "-c",
        "copy",
        str(target),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to slice video "
            f"key={video_key} source_episode={source_episode_index} new_episode={new_episode_index} "
            f"time={start_time:.6f}-{end_time:.6f} source={source} target={target}"
        ) from exc


def _record_episode(
    *,
    source_episode: dict,
    state: _BuildState,
    new_episode_index: int,
    length: int,
    task_index: int,
    task_text: str,
    kind: str,
    segment: dict | None = None,
) -> None:
    episode = copy.deepcopy(source_episode)
    episode["episode_index"] = new_episode_index
    episode["length"] = length
    episode["tasks"] = [task_text]
    episode["task_id_map"] = {str(task_index): task_text}
    episode["source_episode_index"] = int(source_episode["episode_index"])
    episode["transform_kind"] = kind
    if segment is not None:
        episode["subtask"] = {
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "action": segment.get("action"),
            "action_english": segment.get("action_english"),
            "prompt_text": segment["_prompt_text"],
        }
        annotation = episode.get("annotation") or {}
        annotation_data = dict(annotation.get("annotation_data") or {})
        annotation_data["data"] = [dict(segment)]
        annotation["annotation_data"] = annotation_data
        episode["annotation"] = annotation

    state.episodes.append(episode)
    state.episode_stats.append({"episode_index": new_episode_index, "stats": {}})
    state.next_episode_index += 1
    state.next_global_frame_index += length
    state.total_frames += length


def _write_metadata(
    *,
    output_dir: Path,
    source_dataset_json: dict,
    info: dict,
    tasks: list[dict],
    state: _BuildState,
) -> None:
    new_info = dict(info)
    new_info["total_episodes"] = len(state.episodes)
    new_info["total_frames"] = state.total_frames
    new_info["total_tasks"] = len(tasks)

    lerobot_io.write_json(output_dir / "meta" / "info.json", new_info)
    lerobot_io.write_jsonl(output_dir / "meta" / "tasks.jsonl", tasks)
    lerobot_io.write_jsonl(output_dir / "meta" / "episodes.jsonl", state.episodes)
    lerobot_io.write_jsonl(output_dir / "meta" / "episodes_stats.jsonl", state.episode_stats)

    dataset_json = dict(source_dataset_json)
    dataset_json["task_name"] = f"{source_dataset_json.get('task_name', 'dataset')}_multiscale"
    dataset_json["info"] = new_info
    dataset_json["tasks"] = tasks
    dataset_json["episodes"] = state.episodes
    dataset_json["episode_stats"] = state.episode_stats
    files = lerobot_io.relative_files(output_dir)
    if "dataset.json" not in files:
        files.append("dataset.json")
    dataset_json["files"] = sorted(files)
    lerobot_io.write_json(output_dir / "dataset.json", dataset_json)
