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
