# Spirit AI Dataset Transform CLI

`examples/spirit-ai/dataset_transform.py` is the dataset preparation entry point for Spirit AI LeRobot datasets. It replaces the old `check_instruction_manually.py` workflow while keeping the old script as a compatibility wrapper.

## Check Instructions

Check whether `meta/tasks.jsonl` matches the expected task instruction and print optional subtask annotation statistics:

```bash
uv run python examples/spirit-ai/dataset_transform.py check \
    --dataset_dir /home/deng/Documents/dataset/20260512_FoldPaperBox_Moz1WB_MixedTask5+7_Slice \
    --default_prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps."
```

## Repair Instructions

Write a repaired copy whose task text is the desired global instruction:

```bash
uv run python examples/spirit-ai/dataset_transform.py repair-instruction \
    --dataset_dir /path/to/source_dataset \
    --default_prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps." \
    --output_dir /path/to/repaired_dataset \
    --apply
```

Without `--apply`, the command is a dry run.

## Build Global + Subtask Data

By default, `build-multiscale` does not use subtask annotations. This is intentional because not every dataset has subtask metadata.

Global-only build:

```bash
uv run python examples/spirit-ai/dataset_transform.py build-multiscale \
    --dataset_dir /home/deng/Documents/dataset/20260512_FoldPaperBox_Moz1WB_MixedTask5+7_Slice \
    --output_dir /home/deng/Documents/dataset/20260512_FoldPaperBox_Moz1WB_MixedTask5+7_Multiscale \
    --global_prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps."
```

To explicitly generate subtask-sliced episodes, add `--slice-episodes`:

```bash
uv run python examples/spirit-ai/dataset_transform.py build-multiscale \
    --dataset_dir /home/deng/Documents/dataset/20260512_FoldPaperBox_Moz1WB_MixedTask5+7_Slice \
    --output_dir /home/deng/Documents/dataset/20260512_FoldPaperBox_Moz1WB_MixedTask5+7_Multiscale \
    --global_prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps." \
    --slice-episodes \
    --global_repeat 1 \
    --subtask_repeat 1
```

With `global_repeat=1` and `subtask_repeat=1`, the full/global trajectory frames and the sum of the subtask-sliced frames are approximately `1:1`.

## Prompt Format

Full/global episodes use:

```text
Assemble the cardboard box by erecting the flat sheet and folding the side flaps.
```

Subtask-sliced episodes use:

```text
Assemble the cardboard box by erecting the flat sheet and folding the side flaps. Current step: Fold the right side flap of the upright box using the right arm.
```

This keeps deployment aligned with a global prompt such as `fold the box`, while giving training extra local supervision from subtask annotations.

## Add a Reviewed Global-Prompt Episode Sample

For the corrected FoldBox dataset, the original episodes already contain the hierarchical
`global prompt + Current step:` task IDs. The augmentation workflow keeps those rows unchanged
and appends a random sample of complete episodes whose Parquet rows use the pure global prompt at
task index `39`. It does not slice or re-encode videos. The output is a standalone physical copy;
files are copied with `shutil.copy2`, so the source and output can be moved or deleted independently.

First generate a reproducible review manifest. The planner draws many candidates, each containing
about 25% of the source episodes, and selects a candidate whose duplicated-frame ratio is close to
25%. Recovery coverage is recorded for audit only; it is not a hard quota.

```bash
uv run python examples/spirit-ai/dataset_transform.py plan-global-prompt-augmentation \
    --dataset-dir /home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_10Annotations \
    --manifest-path /home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_14Annotations.global_prompt_sampling.json \
    --global-task-index 39 \
    --duplicate-episode-fraction 0.25 \
    --candidate-count 1000 \
    --seed 20260811
```

Review the selected episode IDs, source fingerprints, duplicate-frame ratio, final global-prompt
share, and recovery counts in the manifest before copying any data. Then build from that exact
manifest; this command never samples again:

```bash
uv run python examples/spirit-ai/dataset_transform.py augment-global-prompts \
    --dataset-dir /home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_10Annotations \
    --output-dir /home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_14Annotations \
    --selection-manifest /home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_14Annotations.global_prompt_sampling.json \
    --overwrite
```

The build corrects `meta/info.json.total_tasks` to `40`, appends duplicate episodes after the
original episode range, regenerates the local metadata and file manifest, and writes an audit copy
to `meta/global_prompt_augmentation_manifest.json`. Validate the result before training:

```bash
uv run python examples/spirit-ai/dataset_transform.py validate-global-prompt-augmentation \
    --dataset-dir /home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_14Annotations \
    --selection-manifest /home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_14Annotations.global_prompt_sampling.json \
    --action-horizon 10

uv run python examples/spirit-ai/dataset_transform.py verify-video-sync \
    --dataset-dir /home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_14Annotations \
    --strict-frame-count
```

## Video Modes

`--video_mode link-full` is the default. For subtask episodes, it links the original full video and keeps source timestamps, so video frame lookup still points to the original recording time.

Other modes:

```text
link-full   Hardlink original videos. Fast and usually enough for training.
copy-full   Copy original videos instead of hardlinking.
slice       Use ffmpeg to write true sliced videos and reset timestamps.
```

Use `--video_mode slice` only when `ffmpeg` is available and you need each generated episode video to be physically trimmed.

When `--video_mode slice` is used, the default `--video_slice_codec reencode` path re-encodes each clip at 30 fps and then verifies that each camera video has the same real decodable frame count as the generated parquet episode. This is slower than stream copy, but it avoids tail-frame decode failures during training.

`--video_slice_codec copy` keeps the old fast `ffmpeg -c copy` behavior. It is not recommended for training datasets because stream-copy trimming can leave MP4 header metadata inconsistent with the real decodable frame sequence.

The CLI prints the planned video operation, periodic percentage progress, and a final completion summary. It no longer prints one line per camera video. Use `--quiet` to suppress these progress messages.

Video slicing is serial by default. Use `--video_workers` to run multiple camera video slice jobs in parallel for each generated subtask episode:

```bash
uv run python examples/spirit-ai/dataset_transform.py build-multiscale \
    --dataset_dir /path/to/source_dataset \
    --output_dir /path/to/output_dataset \
    --global_prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps." \
    --slice-episodes \
    --video_mode slice \
    --video_slice_codec reencode \
    --video_workers 6
```

The dataset metadata and parquet files are still generated sequentially; only the per-camera `ffmpeg` slice jobs are parallelized.

Missing source videos fail the build by default. Use `--allow_missing_videos` only for intentionally partial datasets.

## Verify Video Sync

Before training on a generated dataset, verify that parquet timestamps can be decoded from the videos:

```bash
uv run python examples/spirit-ai/dataset_transform.py verify-video-sync \
    --dataset_dir /path/to/output_dataset
```

For physically sliced datasets, also require the real decoded frame count to equal the episode parquet row count:

```bash
uv run python examples/spirit-ai/dataset_transform.py verify-video-sync \
    --dataset_dir /path/to/output_dataset \
    --strict_frame_count
```

To inspect one episode:

```bash
uv run python examples/spirit-ai/dataset_transform.py verify-video-sync \
    --dataset_dir /path/to/output_dataset \
    --episode_index 1519 \
    --strict_frame_count
```

## After Building

1. Run `check` on the output dataset. For multiscale data, include `--allow-derived-prompts` because subtask prompts intentionally differ from the global prompt:

```bash
uv run python examples/spirit-ai/dataset_transform.py check \
    --dataset_dir /path/to/output_dataset \
    --default_prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps." \
    --allow-derived-prompts
```

2. Run `verify-video-sync` on the output dataset. Add `--strict_frame_count` when `--video_mode slice` was used.
3. Create or update the LeRobot cache symlink for the new dataset.
4. Recompute normalization stats for the training config.
5. Train with `DataConfig(prompt_from_task=True)` so openpi reads the generated task prompts from `task_index`.
