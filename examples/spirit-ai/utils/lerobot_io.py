from __future__ import annotations

import json
import os
from pathlib import Path
import shutil


def read_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(item, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare_output_dir(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst == src:
        raise ValueError("output_dir must be different from dataset_dir")
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {dst}. Use --overwrite to replace it.")
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True)


def episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def format_data_path(info: dict, episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    return Path(template.format(episode_chunk=episode_chunk(episode_index, chunks_size), episode_index=episode_index))


def format_video_path(info: dict, video_key: str, episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    template = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    return Path(
        template.format(
            episode_chunk=episode_chunk(episode_index, chunks_size),
            episode_index=episode_index,
            video_key=video_key,
        )
    )


def video_keys(info: dict) -> list[str]:
    features = info.get("features") or {}
    return [key for key, value in features.items() if isinstance(value, dict) and value.get("dtype") == "video"]


def link_or_copy_file(src: Path, dst: Path, *, mode: str = "hardlink") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        os.symlink(src, dst)
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode != "hardlink":
        raise ValueError(f"Unsupported file mode: {mode}")
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_non_episode_files(src: Path, dst: Path) -> None:
    for child in src.iterdir():
        if child.name in {"data", "videos", "meta", "dataset.json"}:
            continue
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def relative_files(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())
