from __future__ import annotations

import dataclasses
import math
from pathlib import Path
import re
import subprocess
import time

import pyarrow.parquet as pq

from utils import lerobot_io


@dataclasses.dataclass(frozen=True)
class VideoSyncIssue:
    episode_index: int
    video_key: str
    message: str
    parquet_rows: int | None = None
    decoded_frames: int | None = None
    parquet_timestamp: float | None = None
    decoded_timestamp: float | None = None


@dataclasses.dataclass(frozen=True)
class DatasetVideoSyncSummary:
    checked_episodes: int
    checked_videos: int
    issues: list[VideoSyncIssue]


def validate_source_dataset_structure(
    dataset_dir: Path,
    info: dict,
    episodes: list[dict],
    *,
    allow_missing_videos: bool,
    check_video_tail: bool = True,
    progress: bool = True,
) -> None:
    """Cheap source-dataset validation before generating derived episodes."""
    if not episodes:
        raise ValueError("meta/episodes.jsonl is empty or missing")

    video_keys = lerobot_io.video_keys(info)
    if not video_keys:
        raise ValueError("No video features found in meta/info.json")

    started = time.monotonic()
    for checked, episode in enumerate(episodes, start=1):
        episode_index = int(episode["episode_index"])
        table = pq.read_table(dataset_dir / lerobot_io.format_data_path(info, episode_index), columns=["timestamp"])
        expected_length = int(episode.get("length", table.num_rows))
        if table.num_rows != expected_length:
            raise ValueError(
                f"episode={episode_index} parquet rows ({table.num_rows}) do not match metadata length "
                f"({expected_length})"
            )
        _validate_timestamps(table["timestamp"].to_pylist(), episode_index=episode_index)
        last_ts = float(table["timestamp"][table.num_rows - 1].as_py())

        for video_key in video_keys:
            video_path = dataset_dir / lerobot_io.format_video_path(info, video_key, episode_index)
            if not video_path.exists() and not allow_missing_videos:
                raise FileNotFoundError(f"Missing source video: episode={episode_index} key={video_key} path={video_path}")
            if video_path.exists() and check_video_tail:
                try:
                    decode_timestamp_with_torchcodec(video_path, last_ts, tolerance_s=0.0001)
                except Exception as exc:  # noqa: BLE001 - keep the decoder error visible.
                    raise RuntimeError(
                        f"Source video tail is not decodable: episode={episode_index} key={video_key} "
                        f"timestamp={last_ts} path={video_path} error={exc!r}"
                    ) from exc

        if progress and (checked == 1 or checked % 25 == 0 or checked == len(episodes)):
            print(
                f"Source preflight: {checked}/{len(episodes)} episodes, "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )


def validate_episode_videos(
    *,
    dataset_dir: Path,
    info: dict,
    episode_index: int,
    timestamps: list[float],
    strict_frame_count: bool,
    expected_frame_count: int | None = None,
    tolerance_s: float = 0.0001,
) -> list[VideoSyncIssue]:
    issues: list[VideoSyncIssue] = []
    if not timestamps:
        issues.append(VideoSyncIssue(episode_index, "<all>", "episode has no timestamps"))
        return issues

    _validate_timestamps(timestamps, episode_index=episode_index)
    parquet_rows = len(timestamps)
    last_ts = float(timestamps[-1])

    for video_key in lerobot_io.video_keys(info):
        video_path = dataset_dir / lerobot_io.format_video_path(info, video_key, episode_index)
        if not video_path.exists():
            issues.append(VideoSyncIssue(episode_index, video_key, f"missing video: {video_path}", parquet_rows))
            continue

        decoded_timestamps = ffprobe_frame_timestamps(video_path)
        decoded_frames = len(decoded_timestamps)
        if decoded_frames == 0:
            issues.append(VideoSyncIssue(episode_index, video_key, "video has no decodable frames", parquet_rows, 0))
            continue

        expected = expected_frame_count if expected_frame_count is not None else parquet_rows
        if strict_frame_count and decoded_frames != expected:
            issues.append(
                VideoSyncIssue(
                    episode_index=episode_index,
                    video_key=video_key,
                    message=f"decoded frame count mismatch: expected {expected}, got {decoded_frames}",
                    parquet_rows=parquet_rows,
                    decoded_frames=decoded_frames,
                    parquet_timestamp=last_ts,
                    decoded_timestamp=decoded_timestamps[-1],
                )
            )

        try:
            decode_timestamp_with_torchcodec(video_path, last_ts, tolerance_s=tolerance_s)
        except Exception as exc:  # noqa: BLE001 - report the original decoder failure.
            issues.append(
                VideoSyncIssue(
                    episode_index=episode_index,
                    video_key=video_key,
                    message=f"torchcodec cannot decode last parquet timestamp: {exc!r}",
                    parquet_rows=parquet_rows,
                    decoded_frames=decoded_frames,
                    parquet_timestamp=last_ts,
                    decoded_timestamp=decoded_timestamps[-1],
                )
            )

    return issues


def validate_dataset_video_sync(
    dataset_dir: Path,
    *,
    strict_frame_count: bool = False,
    episode_indices: set[int] | None = None,
    progress: bool = True,
    max_issues: int | None = None,
) -> DatasetVideoSyncSummary:
    info = lerobot_io.read_json(dataset_dir / "meta" / "info.json")
    episodes = lerobot_io.read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    video_keys = lerobot_io.video_keys(info)
    issues: list[VideoSyncIssue] = []
    checked_episodes = 0
    checked_videos = 0
    started = time.monotonic()

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        if episode_indices is not None and episode_index not in episode_indices:
            continue
        table = pq.read_table(dataset_dir / lerobot_io.format_data_path(info, episode_index), columns=["timestamp"])
        timestamps = [float(value) for value in table["timestamp"].to_pylist()]
        expected_length = int(episode.get("length", table.num_rows))
        if table.num_rows != expected_length:
            issues.append(
                VideoSyncIssue(
                    episode_index=episode_index,
                    video_key="<parquet>",
                    message=f"parquet rows ({table.num_rows}) do not match metadata length ({expected_length})",
                    parquet_rows=table.num_rows,
                )
            )

        issues.extend(
            validate_episode_videos(
                dataset_dir=dataset_dir,
                info=info,
                episode_index=episode_index,
                timestamps=timestamps,
                strict_frame_count=strict_frame_count,
                expected_frame_count=table.num_rows,
            )
        )
        checked_episodes += 1
        checked_videos += len(video_keys)

        if progress and (checked_episodes == 1 or checked_episodes % 25 == 0):
            print(
                f"Verifying video sync: {checked_episodes} episodes, {checked_videos} videos, "
                f"issues={len(issues)}, elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
        if max_issues is not None and len(issues) >= max_issues:
            break

    return DatasetVideoSyncSummary(checked_episodes=checked_episodes, checked_videos=checked_videos, issues=issues)


def ffprobe_frame_timestamps(video_path: Path) -> list[float]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    timestamps = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # ffprobe may append side-data fields to the timestamp line; keep the leading number only.
        match = re.match(r"^[-+]?\d+(?:\.\d+)?", line)
        if match is None:
            continue
        timestamps.append(float(match.group(0)))
    return timestamps


def decode_timestamp_with_torchcodec(video_path: Path, timestamp: float, *, tolerance_s: float) -> None:
    from lerobot.common.datasets.video_utils import decode_video_frames

    decode_video_frames(video_path, [float(timestamp)], tolerance_s, "torchcodec")


def _validate_timestamps(timestamps: list[float], *, episode_index: int) -> None:
    if not timestamps:
        raise ValueError(f"episode={episode_index} has no timestamps")
    previous = None
    for timestamp in timestamps:
        value = float(timestamp)
        if not math.isfinite(value):
            raise ValueError(f"episode={episode_index} has non-finite timestamp: {value}")
        if previous is not None and value < previous:
            raise ValueError(f"episode={episode_index} timestamps are not monotonic: {previous} -> {value}")
        previous = value
