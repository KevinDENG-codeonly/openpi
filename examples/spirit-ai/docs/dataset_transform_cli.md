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

When `--video_mode slice` is used, the CLI prints progress for source episodes, subtask segments, and each camera video being sliced. Use `--quiet` to suppress these progress messages.

Video slicing is serial by default. Use `--video_workers` to run multiple camera video slice jobs in parallel for each generated subtask episode:

```bash
uv run python examples/spirit-ai/dataset_transform.py build-multiscale \
    --dataset_dir /path/to/source_dataset \
    --output_dir /path/to/output_dataset \
    --global_prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps." \
    --slice-episodes \
    --video_mode slice \
    --video_workers 6
```

The dataset metadata and parquet files are still generated sequentially; only the per-camera `ffmpeg` slice jobs are parallelized.

## After Building

1. Run `check` on the output dataset. For multiscale data, include `--allow-derived-prompts` because subtask prompts intentionally differ from the global prompt:

```bash
uv run python examples/spirit-ai/dataset_transform.py check \
    --dataset_dir /path/to/output_dataset \
    --default_prompt "Assemble the cardboard box by erecting the flat sheet and folding the side flaps." \
    --allow-derived-prompts
```

2. Create or update the LeRobot cache symlink for the new dataset.
3. Recompute normalization stats for the training config.
4. Train with `DataConfig(prompt_from_task=True)` so openpi reads the generated task prompts from `task_index`.
