# Global Prompt Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, physically copied LeRobot dataset at `/home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_14Annotations` that retains all hierarchical `global + Current step` samples and appends a random 25% episode duplicate set using pure global prompt task index `39`.

**Architecture:** Add a two-phase dataset-transform workflow. The planning phase draws many uniform random 178-episode candidates from the 712 source episodes, records their frame ratios and recovery coverage, and writes a human-reviewable manifest. The build phase consumes the approved manifest, physically copies the full source dataset, appends duplicate full episodes with `task_index=39`, regenerates LeRobot metadata, and runs structural, prompt, video, and distribution validation.

**Tech Stack:** Python 3.11, `pyarrow`, existing SpiritAI `lerobot_io` and `video_sync` utilities, `pytest`, OpenPI `PromptFromLeRobotTask`.

---

## Fixed dataset contract

| Item | Value |
|---|---|
| Source dataset | `/home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_10Annotations` |
| Output dataset | `/home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_14Annotations` |
| Source population | 712 episodes, 1,148,747 frames |
| Duplicate episode count | `round(712 * 0.25) = 178` |
| Global task mapping | `task_index=39`, already present in source `meta/tasks.jsonl` |
| Global text | `Use both grippers to erect carton blank from partially opened box, fold and press both side walls inward to form open box body, and then place formed box in right side placement area on table.` |
| File mode | Physical copy only; no hard links, symlinks, or video re-encoding |
| Default seed | `20260811` |
| Candidate count | 1,000 independent uniform samples without replacement |

The source has a known metadata mismatch: `meta/tasks.jsonl` now contains 40 task IDs (`0..39`), while source `meta/info.json` still says `total_tasks=39`. The source must remain untouched. The output must correct both `meta/info.json.total_tasks` and `dataset.json.info.total_tasks` to `40`.

The selection deliberately does **not** require an exact 25% frame ratio. Every candidate contains exactly 178 episodes; its duplicated-frame ratio is measured after sampling. A candidate is acceptable when its duplicated-frame ratio is in `[0.24, 0.26]`, and the selected candidate is the acceptable one closest to `0.25`. Recovery coverage is a secondary ranking signal and an audit field, not a fixed quota. The final global-prompt training share is `duplicate_frames / (source_frames + duplicate_frames)`, approximately 20% when the duplicate-frame ratio is approximately 25%.

For a candidate with `n` selected episodes, define `recovery_l1_error` exactly as:

```python
sum(
    abs(candidate_recovery_counts[label] / n - source_recovery_counts[label] / source_episode_count)
    for label in ("none", "left", "right", "both")
)
```

The manifest format is JSON object version `1`. Store `Path` values as strings and tuples as JSON arrays. Store all candidate summaries, but store full `episode_indices` only for the selected candidate; non-selected candidates contain `trial`, ratios, frame count, recovery counts, and `recovery_l1_error`. This keeps the audit file compact while preserving enough evidence to explain the selection.

## File structure

| File | Change | Responsibility |
|---|---|---|
| `examples/spirit-ai/utils/global_prompt_augmenter.py` | Create | Plan candidates, validate manifests, physically copy/append episodes, regenerate metadata, and validate the completed output. |
| `examples/spirit-ai/global_prompt_augmenter_test.py` | Create | Synthetic LeRobot fixtures and regression tests for candidate planning, physical copying, metadata, prompts, and failure paths. |
| `examples/spirit-ai/dataset_transform.py` | Modify | Add `plan-global-prompt-augmentation`, `augment-global-prompts`, and `validate-global-prompt-augmentation` commands. |
| `examples/spirit-ai/README.md` | Modify | Document the intended global/subtask training distribution and the two-command workflow. |
| `examples/spirit-ai/docs/dataset_transform_cli.md` | Modify | Document command arguments, manifest review, physical-copy semantics, and validation commands. |
| `/home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_14Annotations.global_prompt_sampling.json` | Create at execution time | Human-review manifest generated before any dataset copy. |
| `/home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_14Annotations/meta/global_prompt_augmentation_manifest.json` | Create at execution time | Immutable copy of the approved selection plus build/validation facts. |

## Data flow

```text
source tasks.jsonl (0..39, task 39 = pure global prompt)
                 │
                 ├── plan command: 1,000 random candidates of 178 episodes
                 │       └── reviewed selection manifest
                 │
                 └── build command
                         ├── physical copy of original 712 episodes (task_index 1..38 unchanged)
                         └── append 178 copied episodes (new indices 712..889; all task_index=39)
                                  │
                                  └── regenerate info/tasks/episodes/stats/dataset.json and validate
```

### Task 1: Add manifest planning and deterministic candidate selection

**Files:**

- Create: `examples/spirit-ai/utils/global_prompt_augmenter.py`
- Create: `examples/spirit-ai/global_prompt_augmenter_test.py`

- [ ] **Step 1: Write failing tests for source inspection and repeated random selection.**

Create a minimal synthetic LeRobot fixture with eight one-file episodes, a contiguous task map `0..39`, one global task at `39`, and Parquet columns `episode_index`, `frame_index`, `index`, `task_index`, and `timestamp`. Use two recovery categories in the fixture so the manifest has meaningful coverage statistics.

Define the fixture helpers in the test file before the tests that use them:

```python
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

from utils import global_prompt_augmenter as augmenter
from utils import lerobot_io


GLOBAL_PROMPT = (
    "Use both grippers to erect carton blank from partially opened box, fold and press both side "
    "walls inward to form open box body, and then place formed box in right side placement area on table."
)
DATASET_TRANSFORM_PATH = Path(__file__).with_name("dataset_transform.py")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\\n" for row in rows), encoding="utf-8")


def write_episode(root: Path, info: dict, episode_index: int, task_indices: list[int]) -> None:
    length = len(task_indices)
    table = pa.table(
        {
            "episode_index": pa.array([episode_index] * length, type=pa.int64()),
            "frame_index": pa.array(range(length), type=pa.int64()),
            "index": pa.array(range(episode_index * 10, episode_index * 10 + length), type=pa.int64()),
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
    episode_task_indices = [[1, 1, 1], [34, 34, 34], [35, 35, 35], [34, 35, 35], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    for episode_index, task_indices in enumerate(episode_task_indices):
        write_episode(root, info, episode_index, task_indices)
    episodes = [{"episode_index": index, "length": 3, "tasks": ["legacy"], "task_id_map": {"0": "legacy"}} for index in range(8)]
    write_jsonl(root / "meta" / "episodes.jsonl", episodes)
    write_jsonl(root / "meta" / "episodes_stats.jsonl", [{"episode_index": index, "stats": {}} for index in range(8)])
    lerobot_io.write_json(root / "meta" / "info.json", info)
    lerobot_io.write_json(root / "dataset.json", {"task_name": "synthetic", "info": info, "tasks": tasks, "episodes": episodes, "episode_stats": [], "files": []})
    return root


def remove_task(path: Path, task_index: int) -> None:
    write_jsonl(path, [row for row in lerobot_io.read_jsonl(path) if row["task_index"] != task_index])


def rewrite_task_index(dataset_dir: Path, episode_index: int, task_index: int) -> None:
    info = lerobot_io.read_json(dataset_dir / "meta" / "info.json")
    path = dataset_dir / lerobot_io.format_data_path(info, episode_index)
    table = pq.read_table(path)
    values = pa.array([task_index] * table.num_rows, type=table.schema.field("task_index").type)
    pq.write_table(table.set_column(table.column_names.index("task_index"), "task_index", values), path)


def build_fixture_dataset(synthetic_dataset: Path, tmp_path: Path) -> tuple[Path, augmenter.GlobalPromptAugmentationPlan]:
    plan = augmenter.plan_global_prompt_augmentation(dataset_dir=synthetic_dataset, global_task_index=39, duplicate_episode_fraction=0.25, candidate_count=50, seed=20260811)
    output_dir = tmp_path / "augmented"
    augmenter.build_global_prompt_augmented_dataset(dataset_dir=synthetic_dataset, output_dir=output_dir, plan=plan, overwrite=False)
    return output_dir, plan


def load_dataset_transform_module():
    spec = importlib.util.spec_from_file_location(f"dataset_transform_test_{uuid4().hex}", DATASET_TRANSFORM_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
```

```python
def test_plan_selects_exact_episode_count_and_is_reproducible(synthetic_dataset, tmp_path):
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


def test_plan_rejects_missing_or_non_global_task_39(synthetic_dataset):
    remove_task(synthetic_dataset / "meta" / "tasks.jsonl", task_index=39)

    with pytest.raises(ValueError, match="global_task_index=39"):
        augmenter.plan_global_prompt_augmentation(
            dataset_dir=synthetic_dataset,
            global_task_index=39,
            duplicate_episode_fraction=0.25,
            candidate_count=10,
            seed=1,
        )
```

- [ ] **Step 2: Run the new tests to verify they fail before implementation.**

Run:

```bash
uv run pytest examples/spirit-ai/global_prompt_augmenter_test.py -k 'plan_' -v
```

Expected: FAIL because `utils.global_prompt_augmenter` does not exist.

- [ ] **Step 3: Implement planning types, source fingerprinting, and candidate ranking.**

Add these public immutable types and functions to `global_prompt_augmenter.py`:

```python
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


`plan_global_prompt_augmentation(dataset_dir: Path, global_task_index: int, duplicate_episode_fraction: float, candidate_count: int, seed: int) -> GlobalPromptAugmentationPlan` loads source metadata and Parquet task IDs, evaluates candidates, and returns the selected plan without writing dataset files.

`write_plan_manifest(path: Path, plan: GlobalPromptAugmentationPlan) -> None` serializes manifest version `1` using the JSON rules in the fixed contract.

`read_plan_manifest(path: Path) -> GlobalPromptAugmentationPlan` validates manifest version `1`, converts strings/lists back to `Path`/tuples, and rejects malformed or missing selection fields.
```

Implement the selection as follows:

```python
rng = random.Random(seed)
sample_size = round(len(episodes) * duplicate_episode_fraction)
for trial in range(candidate_count):
    selected = tuple(sorted(rng.sample(episode_indices, sample_size)))
    duplicate_frames = sum(length_by_episode[index] for index in selected)
    ratio = duplicate_frames / source_frame_count
    candidate_recovery_counts = count_recovery_classes(selected, recovery_class_by_episode)
    recovery_l1_error = sum(
        abs(candidate_recovery_counts[label] / sample_size - source_recovery_counts[label] / len(episodes))
        for label in ("none", "left", "right", "both")
    )
    candidates.append(
        CandidateSummary(
            trial=trial,
            episode_indices=selected,
            duplicate_frames=duplicate_frames,
            duplicate_frame_ratio=ratio,
            final_global_prompt_ratio=duplicate_frames / (source_frame_count + duplicate_frames),
            recovery_episode_counts=candidate_recovery_counts,
            recovery_l1_error=recovery_l1_error,
        )
    )

acceptable = [item for item in candidates if 0.24 <= item.duplicate_frame_ratio <= 0.26]
if not acceptable:
    raise ValueError("No random candidate has a duplicate-frame ratio in [0.24, 0.26]")
selected = min(
    acceptable,
    key=lambda item: (
        abs(item.duplicate_frame_ratio - duplicate_episode_fraction),
        item.recovery_l1_error,
        item.trial,
    ),
)
```

Use this helper so every candidate has all four recovery keys, including zeros:

```python
def count_recovery_classes(
    episode_indices: tuple[int, ...], recovery_class_by_episode: dict[int, str]
) -> dict[str, int]:
    counts = {"none": 0, "left": 0, "right": 0, "both": 0}
    for episode_index in episode_indices:
        counts[recovery_class_by_episode[episode_index]] += 1
    return counts
```

Classify recovery from the actual Parquet `task_index` values, never from the noisy legacy `episodes.jsonl.tasks` text: `34` means left recovery; any of `35, 36, 37, 38` means right recovery; report `none`, `left`, `right`, or `both`. Fingerprint `meta/tasks.jsonl`, `meta/episodes.jsonl`, and `meta/info.json` with SHA-256 so a stale plan cannot be applied after labels change. Require a contiguous task map containing `0..39`, require that task `39` has no `Current step:`, and report rather than mutate the known source `total_tasks` mismatch.

- [ ] **Step 4: Run planning tests and static checks.**

Run:

```bash
uv run pytest examples/spirit-ai/global_prompt_augmenter_test.py -k 'plan_' -v
uv run ruff check examples/spirit-ai/utils/global_prompt_augmenter.py examples/spirit-ai/global_prompt_augmenter_test.py
```

Expected: all selected tests PASS and Ruff reports no diagnostics.

- [ ] **Step 5: Commit the planning functionality.**

```bash
git add examples/spirit-ai/utils/global_prompt_augmenter.py examples/spirit-ai/global_prompt_augmenter_test.py
git commit -m "feat: plan global prompt episode augmentation"
```

### Task 2: Build a standalone physical-copy augmented dataset

**Files:**

- Modify: `examples/spirit-ai/utils/global_prompt_augmenter.py`
- Modify: `examples/spirit-ai/global_prompt_augmenter_test.py`

- [ ] **Step 1: Write failing tests for physical copy, duplicate rows, and regenerated metadata.**

Add a test that builds from the synthetic source using an approved manifest. It must prove that the output owns separate files and that its appended episodes receive only task `39`.

```python
def test_build_copies_source_and_appends_global_prompt_duplicates(synthetic_dataset, tmp_path):
    plan = augmenter.plan_global_prompt_augmentation(
        dataset_dir=synthetic_dataset,
        global_task_index=39,
        duplicate_episode_fraction=0.25,
        candidate_count=50,
        seed=20260811,
    )
    output_dir = tmp_path / "augmented"
    summary = augmenter.build_global_prompt_augmented_dataset(
        dataset_dir=synthetic_dataset,
        output_dir=output_dir,
        plan=plan,
        overwrite=False,
    )

    assert summary.total_episodes == 10
    assert summary.total_tasks == 40
    assert summary.global_task_index == 39
    assert (output_dir / "data/chunk-000/episode_000000.parquet").read_bytes() == (
        synthetic_dataset / "data/chunk-000/episode_000000.parquet"
    ).read_bytes()
    assert (output_dir / "data/chunk-000/episode_000000.parquet").stat().st_ino != (
        synthetic_dataset / "data/chunk-000/episode_000000.parquet"
    ).stat().st_ino

    duplicate = pq.read_table(output_dir / "data/chunk-000/episode_000008.parquet")
    assert set(duplicate["task_index"].to_pylist()) == {39}
    assert set(duplicate["episode_index"].to_pylist()) == {8}
```

Add tests that reject a manifest whose source fingerprints no longer match and reject a non-empty output directory unless `overwrite=True` was explicitly passed.

- [ ] **Step 2: Run the build tests and verify they fail.**

Run:

```bash
uv run pytest examples/spirit-ai/global_prompt_augmenter_test.py -k 'build_' -v
```

Expected: FAIL because `build_global_prompt_augmented_dataset` does not exist.

- [ ] **Step 3: Implement physical copy and append logic.**

Add these public API definitions:

```python
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


`build_global_prompt_augmented_dataset(dataset_dir: Path, output_dir: Path, plan: GlobalPromptAugmentationPlan, overwrite: bool, progress: bool = True) -> GlobalPromptAugmentationSummary` verifies the reviewed plan, creates the self-contained output, and returns measured output totals.
```

Implementation requirements:

1. Verify the manifest fingerprints and task-39 text before touching `output_dir`.
2. Refuse `output_dir == dataset_dir`; refuse a non-empty output directory unless `overwrite=True`; when overwrite is set, delete only the resolved `output_dir` after confirming it is exactly the user-supplied target.
3. Estimate required free space as the byte size of the source tree plus the selected Parquet/video bytes plus a 10% margin. Compare it with `shutil.disk_usage(output_dir.parent).free` before copying.
4. Use `shutil.copytree(dataset_dir, output_dir, copy_function=shutil.copy2)` for the original 712 episodes. Do not call `os.link`, `os.symlink`, or `lerobot_io.link_or_copy_file(mode="hardlink")`.
5. Assign each selected source episode a new `episode_index` in sorted selection order, starting at 712. Read the source Parquet with `pq.read_table`, then replace only these columns:

```python
rewritten = table
rewritten = _set_column(rewritten, "episode_index", [new_episode_index] * length)
rewritten = _set_column(rewritten, "frame_index", range(length))
rewritten = _set_column(rewritten, "index", range(next_global_index, next_global_index + length))
rewritten = _set_column(rewritten, "task_index", [plan.global_task_index] * length)
```

6. Write each rewritten table with `pq.write_table` at the path from `lerobot_io.format_data_path(info, new_episode_index)`. Copy each source video to the matching new episode path with `shutil.copy2`; do not re-encode it.
7. Deep-copy each selected source episode record, set `episode_index`, `length`, `tasks=[global_task_text]`, `task_id_map={"39": global_task_text}`, `source_episode_index`, `transform_kind="global_prompt_duplicate"`, and `global_prompt_selection_trial=plan.selected_trial`. Deep-copy its matching `episodes_stats.jsonl` item and set only `episode_index`.
8. Regenerate metadata from the complete target state, not by editing source JSON in place:

```python
new_info["total_episodes"] = len(all_episodes)
new_info["total_frames"] = next_global_index
new_info["total_tasks"] = len(tasks)  # 40
new_info["total_chunks"] = math.ceil(len(all_episodes) / int(new_info["chunks_size"]))
new_info["total_videos"] = len(all_episodes) * len(lerobot_io.video_keys(new_info))
new_info["splits"] = {"train": f"0:{len(all_episodes)}"}
```

Write `meta/info.json`, `meta/tasks.jsonl`, `meta/episodes.jsonl`, and `meta/episodes_stats.jsonl`. Rebuild `dataset.json` with updated `info`, `tasks`, `episodes`, and `episode_stats`; generate a local file manifest from actual target files with objects of the form `{"path": relative_path, "type": suffix_without_dot}` and do not retain stale source `download_url` values for duplicate files.

9. Write the approved manifest to `meta/global_prompt_augmentation_manifest.json` plus a `build_summary` section containing totals, exact selected IDs, output creation time, and byte-copy mode `"copy"`.

- [ ] **Step 4: Run the build tests and inspect the synthetic output.**

Run:

```bash
uv run pytest examples/spirit-ai/global_prompt_augmenter_test.py -k 'build_' -v
uv run python -m py_compile examples/spirit-ai/utils/global_prompt_augmenter.py
```

Expected: all build tests PASS; the output original Parquet inode differs from its source inode; appended Parquet rows use only task `39`.

- [ ] **Step 5: Commit the builder.**

```bash
git add examples/spirit-ai/utils/global_prompt_augmenter.py examples/spirit-ai/global_prompt_augmenter_test.py
git commit -m "feat: build standalone global prompt augmented datasets"
```

### Task 3: Add semantic validation and CLI entry points

**Files:**

- Modify: `examples/spirit-ai/utils/global_prompt_augmenter.py`
- Modify: `examples/spirit-ai/global_prompt_augmenter_test.py`
- Modify: `examples/spirit-ai/dataset_transform.py`

- [ ] **Step 1: Write failing tests for validator failures and CLI parsing.**

```python
def test_validate_rejects_duplicate_episode_with_non_global_task(synthetic_dataset, tmp_path):
    output_dir, plan = build_fixture_dataset(synthetic_dataset, tmp_path)
    rewrite_task_index(output_dir, episode_index=8, task_index=1)

    with pytest.raises(ValueError, match="expected only task_index=39"):
        augmenter.validate_global_prompt_augmented_dataset(
            dataset_dir=output_dir,
            plan=plan,
            action_horizon=10,
        )


def test_cli_exposes_plan_build_and_validate_commands(monkeypatch):
    module = load_dataset_transform_module()
    parser = module.build_parser()
    assert parser.parse_args(["plan-global-prompt-augmentation", "--dataset-dir", "/tmp/source", "--manifest-path", "/tmp/plan.json"]).command == "plan-global-prompt-augmentation"
    assert parser.parse_args(["augment-global-prompts", "--dataset-dir", "/tmp/source", "--output-dir", "/tmp/output", "--selection-manifest", "/tmp/plan.json"]).command == "augment-global-prompts"
    assert parser.parse_args(["validate-global-prompt-augmentation", "--dataset-dir", "/tmp/output", "--selection-manifest", "/tmp/plan.json"]).command == "validate-global-prompt-augmentation"
```

- [ ] **Step 2: Run semantic and CLI tests to verify they fail.**

Run:

```bash
uv run pytest examples/spirit-ai/global_prompt_augmenter_test.py -k 'validate_ or cli_' -v
```

Expected: FAIL because the validation API and CLI commands do not exist.

- [ ] **Step 3: Implement validation and CLI wiring.**

Add this API:

```python
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


`validate_global_prompt_augmented_dataset(dataset_dir: Path, plan: GlobalPromptAugmentationPlan, action_horizon: int) -> GlobalPromptAugmentationValidation` scans the complete output and either returns its measured validation report or raises a `ValueError` containing every failed invariant name.
```

The validator must fail if any of these invariants is false:

- task records are contiguous `0..39`, have 40 rows, and have exactly 14 unique instruction strings;
- `info.total_tasks == 40`, `info.total_episodes == 890`, `info.total_frames` equals the sum of episode lengths, and `info.splits.train == "0:890"` for this target build;
- every original target episode `0..711` preserves its source `task_index` sequence;
- every duplicate target episode `712..889` has a contiguous local `frame_index`, one `episode_index`, and only `task_index=39`;
- global frame `index` is continuous and unique across `0..total_frames-1`;
- every task ID found in Parquet exists in `tasks.jsonl`;
- all duplicate videos exist and have a different inode from their source video, proving physical copy;
- `PromptFromLeRobotTask(tasks)({"task_index": 39})["prompt"]` equals the pure global text, while an original non-39 task contains `Current step:`;
- the manifest fingerprint matches the source information copied into the output manifest.

For the action-horizon audit, inspect original hierarchical episode task-index sequences and count start frames whose next `action_horizon` task IDs are not all equal. Report this value; do not delete or relabel those frames.

Wire these CLI commands in `dataset_transform.py`:

```text
plan-global-prompt-augmentation
  --dataset-dir PATH --manifest-path PATH
  --global-task-index 39 --duplicate-episode-fraction 0.25
  --candidate-count 1000 --seed 20260811

augment-global-prompts
  --dataset-dir PATH --output-dir PATH --selection-manifest PATH
  --overwrite --quiet

validate-global-prompt-augmentation
  --dataset-dir PATH --selection-manifest PATH --action-horizon 10
```

`plan-global-prompt-augmentation` must write only the manifest. `augment-global-prompts` must require a reviewed manifest and must never select a new random sample. `validate-global-prompt-augmentation` must exit non-zero after printing all failed invariant names.

- [ ] **Step 4: Run the full focused test suite.**

Run:

```bash
uv run pytest examples/spirit-ai/global_prompt_augmenter_test.py -v
uv run ruff check examples/spirit-ai/utils/global_prompt_augmenter.py examples/spirit-ai/global_prompt_augmenter_test.py examples/spirit-ai/dataset_transform.py
```

Expected: all tests PASS and no lint diagnostics.

- [ ] **Step 5: Commit the validation and CLI layer.**

```bash
git add examples/spirit-ai/utils/global_prompt_augmenter.py examples/spirit-ai/global_prompt_augmenter_test.py examples/spirit-ai/dataset_transform.py
git commit -m "feat: validate global prompt dataset augmentation"
```

### Task 4: Document the workflow and create the reviewable sampling manifest

**Files:**

- Modify: `examples/spirit-ai/README.md`
- Modify: `examples/spirit-ai/docs/dataset_transform_cli.md`

- [ ] **Step 1: Add the physical-copy workflow to both documentation files.**

Document that the original frame-level hierarchical prompts stay unchanged, while only the selected duplicate episodes receive task index `39`. Include this exact plan command:

```bash
uv run python examples/spirit-ai/dataset_transform.py plan-global-prompt-augmentation \
  --dataset-dir /home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_10Annotations \
  --manifest-path /home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_14Annotations.global_prompt_sampling.json \
  --global-task-index 39 \
  --duplicate-episode-fraction 0.25 \
  --candidate-count 1000 \
  --seed 20260811
```

State that the manifest must be manually reviewed before the copy command is run, and that physical copies make the output independently movable and deletable.

- [ ] **Step 2: Generate the real planning manifest only.**

Run:

```bash
uv run python examples/spirit-ai/dataset_transform.py plan-global-prompt-augmentation \
  --dataset-dir /home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_10Annotations \
  --manifest-path /home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_14Annotations.global_prompt_sampling.json \
  --global-task-index 39 \
  --duplicate-episode-fraction 0.25 \
  --candidate-count 1000 \
  --seed 20260811
```

Expected: one JSON manifest, no dataset directory contents written or changed. Review its selected episode IDs, duplicate-frame ratio, final global-prompt ratio, recovery coverage, source fingerprints, and top candidate summaries with the user.

- [ ] **Step 3: Commit documentation and the non-dataset manifest policy.**

```bash
git add examples/spirit-ai/README.md examples/spirit-ai/docs/dataset_transform_cli.md
git commit -m "docs: document global prompt augmentation workflow"
```

### Task 5: Build the approved dataset and run full verification

**Files:**

- Create at execution time: `/home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_14Annotations/**`

- [ ] **Step 1: Recheck the exact output target before destructive overwrite.**

Run:

```bash
find /home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_14Annotations -mindepth 1 -maxdepth 1 -print
df -h /home/deng/Documents/dataset
```

Expected: the target is still empty or the user explicitly confirms `--overwrite`; free space exceeds the preflight estimate emitted by the build command.

- [ ] **Step 2: Build from the approved manifest using physical copies.**

Run:

```bash
uv run python examples/spirit-ai/dataset_transform.py augment-global-prompts \
  --dataset-dir /home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_10Annotations \
  --output-dir /home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_14Annotations \
  --selection-manifest /home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_14Annotations.global_prompt_sampling.json \
  --overwrite
```

Expected: a standalone dataset with 890 episodes, 40 task IDs, 14 unique prompt strings, original task indices unchanged for episodes `0..711`, and index `39` only for `712..889`.

- [ ] **Step 3: Run structural, prompt, and video validation.**

Run:

```bash
uv run python examples/spirit-ai/dataset_transform.py validate-global-prompt-augmentation \
  --dataset-dir /home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_14Annotations \
  --selection-manifest /home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_14Annotations.global_prompt_sampling.json \
  --action-horizon 10

uv run python examples/spirit-ai/dataset_transform.py verify-video-sync \
  --dataset-dir /home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_14Annotations \
  --strict-frame-count
```

Expected: both commands exit `0`; the first prints the actual approximate global-prompt share and H10 stage-boundary audit; the second reports zero video issues.

- [ ] **Step 4: Run a training-loader smoke check with task-derived prompts.**

Run a short `uv run python` check that loads one original and one duplicate Parquet row, reads target `tasks.jsonl`, applies `PromptFromLeRobotTask`, and asserts:

```python
assert "Current step:" in original_prompt
assert duplicate_prompt == GLOBAL_PROMPT
```

Expected: both assertions pass, confirming the `prompt_from_task=True` path will expose the intended two prompt forms.

- [ ] **Step 5: Deliver the verification report for user sign-off.**

Report the selected seed and trial, 178 source episode IDs, duplicate-frame ratio, final global-prompt ratio, task-index frame histogram, action-horizon boundary rate, physical-copy inode checks, video-sync result, and any deviations from the target distribution. Do not begin model training as part of this plan.

## Self-review checklist

- Source data remains immutable; all output writes are under the explicitly named target directory or its sibling manifest path.
- Random sampling is repeated and reviewable, but not made artificially exact at the frame level.
- Task `39` is verified before use, and output metadata resolves the source's `total_tasks` mismatch to `40`.
- The output uses physical copies for both original and duplicate data/videos; no hard links or symlinks are allowed.
- The plan validates metadata, Parquet columns, prompts, video integrity, file independence, and residual action-horizon boundary risk.
