"""Tests for global-prompt episode augmentation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

EXAMPLE_DIR = Path(__file__).parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from utils import global_prompt_augmenter as augmenter  # noqa: E402, I001
from utils import lerobot_io  # noqa: E402


GLOBAL_PROMPT = (
    "Use both grippers to erect carton blank from partially opened box, fold and press both side "
    "walls inward to form open box body, and then place formed box in right side placement area on table."
)
DATASET_TRANSFORM_PATH = EXAMPLE_DIR / "dataset_transform.py"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_episode(root: Path, info: dict, episode_index: int, task_indices: list[int]) -> None:
    length = len(task_indices)
    table = pa.table(
        {
            "episode_index": pa.array([episode_index] * length, type=pa.int64()),
            "frame_index": pa.array(range(length), type=pa.int64()),
            "index": pa.array(range(episode_index * 3, episode_index * 3 + length), type=pa.int64()),
            "task_index": pa.array(task_indices, type=pa.int64()),
            "timestamp": pa.array([step / 30 for step in range(length)], type=pa.float64()),
        }
    )
    path = root / lerobot_io.format_data_path(info, episode_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    video_path = root / lerobot_io.format_video_path(info, "cam", episode_index)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(f"fake-video-{episode_index}".encode())


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    info = {
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {"cam": {"dtype": "video"}},
        "total_episodes": 8,
        "total_frames": 24,
        "total_tasks": 40,
        "total_chunks": 1,
        "total_videos": 8,
        "splits": {"train": "0:8"},
    }
    tasks = [
        {
            "task_index": index,
            "task": "unknown" if index == 0 else f"{GLOBAL_PROMPT} Current step: stage {(index - 1) % 12}.",
        }
        for index in range(40)
    ]
    tasks[39]["task"] = GLOBAL_PROMPT
    write_jsonl(root / "meta" / "tasks.jsonl", tasks)

    episode_task_indices = [
        [1, 1, 1],
        [34, 34, 34],
        [35, 35, 35],
        [34, 35, 35],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    for episode_index, task_indices in enumerate(episode_task_indices):
        write_episode(root, info, episode_index, task_indices)

    episodes = [
        {"episode_index": index, "length": 3, "tasks": ["legacy"], "task_id_map": {"0": "legacy"}}
        for index in range(8)
    ]
    write_jsonl(root / "meta" / "episodes.jsonl", episodes)
    write_jsonl(root / "meta" / "episodes_stats.jsonl", [{"episode_index": index, "stats": {}} for index in range(8)])
    lerobot_io.write_json(root / "meta" / "info.json", info)
    lerobot_io.write_json(
        root / "dataset.json",
        {"task_name": "synthetic", "info": info, "tasks": tasks, "episodes": episodes, "episode_stats": [], "files": []},
    )
    return root


def remove_task(path: Path, task_index: int) -> None:
    write_jsonl(path, [row for row in lerobot_io.read_jsonl(path) if row["task_index"] != task_index])


def rewrite_task_index(dataset_dir: Path, episode_index: int, task_index: int) -> None:
    info = lerobot_io.read_json(dataset_dir / "meta" / "info.json")
    path = dataset_dir / lerobot_io.format_data_path(info, episode_index)
    table = pq.read_table(path)
    values = pa.array([task_index] * table.num_rows, type=table.schema.field("task_index").type)
    rewritten = table.set_column(table.column_names.index("task_index"), "task_index", values)
    pq.write_table(rewritten, path)


def load_dataset_transform_module():
    module_name = f"dataset_transform_test_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, DATASET_TRANSFORM_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def build_fixture_dataset(synthetic_dataset: Path, tmp_path: Path) -> tuple[Path, augmenter.GlobalPromptAugmentationPlan]:
    plan = augmenter.plan_global_prompt_augmentation(
        dataset_dir=synthetic_dataset,
        global_task_index=39,
        duplicate_episode_fraction=0.25,
        candidate_count=50,
        seed=20260811,
    )
    output_dir = tmp_path / "augmented"
    augmenter.build_global_prompt_augmented_dataset(
        dataset_dir=synthetic_dataset,
        output_dir=output_dir,
        plan=plan,
        overwrite=False,
    )
    return output_dir, plan


def test_plan_selects_exact_episode_count_and_is_reproducible(synthetic_dataset: Path):
    first = augmenter.plan_global_prompt_augmentation(
        dataset_dir=synthetic_dataset,
        global_task_index=39,
        duplicate_episode_fraction=0.25,
        candidate_count=50,
        seed=20260811,
    )
    second = augmenter.plan_global_prompt_augmentation(
        dataset_dir=synthetic_dataset,
        global_task_index=39,
        duplicate_episode_fraction=0.25,
        candidate_count=50,
        seed=20260811,
    )

    assert first.selected_episode_indices == second.selected_episode_indices
    assert len(first.selected_episode_indices) == 2
    assert first.selected_duplicate_frame_ratio >= 0.0
    assert len(first.candidates) == 50
    assert first.global_task_text == GLOBAL_PROMPT


def test_plan_rejects_missing_global_task(synthetic_dataset: Path):
    remove_task(synthetic_dataset / "meta" / "tasks.jsonl", task_index=39)

    with pytest.raises(ValueError, match="global_task_index=39"):
        augmenter.plan_global_prompt_augmentation(
            dataset_dir=synthetic_dataset,
            global_task_index=39,
            duplicate_episode_fraction=0.25,
            candidate_count=10,
            seed=1,
        )


def test_build_copies_source_and_appends_global_prompt_duplicates(synthetic_dataset: Path, tmp_path: Path):
    output_dir, plan = build_fixture_dataset(synthetic_dataset, tmp_path)

    assert len(lerobot_io.read_jsonl(output_dir / "meta" / "episodes.jsonl")) == 10
    assert lerobot_io.read_json(output_dir / "meta" / "info.json")["total_tasks"] == 40

    source_path = synthetic_dataset / "data/chunk-000/episode_000000.parquet"
    output_path = output_dir / "data/chunk-000/episode_000000.parquet"
    assert output_path.read_bytes() == source_path.read_bytes()
    assert output_path.stat().st_ino != source_path.stat().st_ino

    duplicate_episode_index = 8
    duplicate_path = output_dir / lerobot_io.format_data_path(
        lerobot_io.read_json(output_dir / "meta" / "info.json"), duplicate_episode_index
    )
    duplicate = pq.read_table(duplicate_path)
    assert set(duplicate["task_index"].to_pylist()) == {39}
    assert set(duplicate["episode_index"].to_pylist()) == {duplicate_episode_index}
    assert plan.selected_duplicate_frames == duplicate.num_rows * len(plan.selected_episode_indices)


def test_build_rejects_stale_plan(synthetic_dataset: Path, tmp_path: Path):
    plan = augmenter.plan_global_prompt_augmentation(
        dataset_dir=synthetic_dataset,
        global_task_index=39,
        duplicate_episode_fraction=0.25,
        candidate_count=10,
        seed=1,
    )
    tasks_path = synthetic_dataset / "meta" / "tasks.jsonl"
    tasks = lerobot_io.read_jsonl(tasks_path)
    tasks[39]["task"] = "changed global prompt"
    write_jsonl(tasks_path, tasks)

    with pytest.raises(ValueError, match="fingerprint"):
        augmenter.build_global_prompt_augmented_dataset(
            dataset_dir=synthetic_dataset,
            output_dir=tmp_path / "augmented",
            plan=plan,
            overwrite=False,
        )


def test_validate_rejects_duplicate_episode_with_non_global_task(synthetic_dataset: Path, tmp_path: Path):
    output_dir, plan = build_fixture_dataset(synthetic_dataset, tmp_path)
    rewrite_task_index(output_dir, episode_index=8, task_index=1)

    with pytest.raises(ValueError, match="expected only task_index=39"):
        augmenter.validate_global_prompt_augmented_dataset(
            dataset_dir=output_dir,
            plan=plan,
            action_horizon=10,
        )


def test_validate_accepts_physical_copy_and_reports_prompt_distribution(synthetic_dataset: Path, tmp_path: Path):
    output_dir, plan = build_fixture_dataset(synthetic_dataset, tmp_path)

    report = augmenter.validate_global_prompt_augmented_dataset(
        dataset_dir=output_dir,
        plan=plan,
        action_horizon=2,
    )

    assert report.total_episodes == 10
    assert report.total_frames == 30
    assert report.duplicate_episodes == 2
    assert report.duplicate_frames == 6
    assert report.global_prompt_frame_ratio == pytest.approx(0.2)
    assert report.task_index_counts[39] == 6
    assert report.action_horizon_checked_frames == 16
    assert report.action_horizon_crossing_frames == 1


def test_cli_exposes_plan_build_and_validate_commands():
    module = load_dataset_transform_module()
    parser = module.build_parser()

    assert (
        parser.parse_args(
            [
                "plan-global-prompt-augmentation",
                "--dataset-dir",
                "/tmp/source",
                "--manifest-path",
                "/tmp/plan.json",
            ]
        ).command
        == "plan-global-prompt-augmentation"
    )
    assert (
        parser.parse_args(
            [
                "augment-global-prompts",
                "--dataset-dir",
                "/tmp/source",
                "--output-dir",
                "/tmp/output",
                "--selection-manifest",
                "/tmp/plan.json",
            ]
        ).command
        == "augment-global-prompts"
    )
    assert (
        parser.parse_args(
            [
                "validate-global-prompt-augmentation",
                "--dataset-dir",
                "/tmp/output",
                "--selection-manifest",
                "/tmp/plan.json",
            ]
        ).command
        == "validate-global-prompt-augmentation"
    )
