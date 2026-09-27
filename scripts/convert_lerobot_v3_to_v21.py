#!/usr/bin/env python
"""Convert a local LeRobot v3.x dataset layout into v2.1 so openpi's pinned lerobot 0.1.0 can load it.

Keeps original v3 files under `_v3_original/` and writes v2.1 parquet/video/meta next to them.

v3 may pack several episodes into one `file-XXX.mp4`. This script reads
`meta/episodes/**/*.parquet` (`file_index`, `from_timestamp`, `to_timestamp`)
and splits each episode into its own v2.1 mp4.

Usage:
    uv run python scripts/convert_lerobot_v3_to_v21.py --dataset-root dataset/a10_vr_9_21_4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


V21_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def _move_aside(src: Path, backup_root: Path, dataset_root: Path) -> None:
    rel = src.relative_to(dataset_root)
    dst = backup_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))


def _load_episode_video_map(meta_dir: Path, video_keys: list[str]) -> dict[int, dict]:
    """episode_index -> {key: {chunk, file, from_ts, to_ts, length}}"""
    ep_files = sorted((meta_dir / "episodes").rglob("*.parquet"))
    if not ep_files:
        return {}
    eps = pd.concat([pd.read_parquet(p) for p in ep_files], ignore_index=True)
    out: dict[int, dict] = {}
    for _, row in eps.iterrows():
        ep_idx = int(row["episode_index"])
        rec: dict = {"length": int(row["length"]) if "length" in row else None}
        for key in video_keys:
            prefix = f"videos/{key}"
            chunk_col = f"{prefix}/chunk_index"
            file_col = f"{prefix}/file_index"
            from_col = f"{prefix}/from_timestamp"
            to_col = f"{prefix}/to_timestamp"
            if file_col not in eps.columns:
                continue
            rec[key] = {
                "chunk": int(row[chunk_col]) if chunk_col in row and pd.notna(row[chunk_col]) else 0,
                "file": int(row[file_col]),
                "from_ts": float(row[from_col]) if from_col in row and pd.notna(row[from_col]) else 0.0,
                "to_ts": float(row[to_col]) if to_col in row and pd.notna(row[to_col]) else None,
            }
        out[ep_idx] = rec
    return out


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ff = shutil.which("ffmpeg")
        if not ff:
            raise RuntimeError("需要 ffmpeg 或 imageio-ffmpeg 才能按时间戳切开 v3 合并视频")
        return ff


def _extract_video_segment(src: Path, dst: Path, from_ts: float, to_ts: float | None, fps: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    ff = _ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_f:
        tmp = Path(tmp_f.name)
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error"]
    if from_ts > 1e-6:
        cmd += ["-ss", f"{from_ts:.6f}"]
    cmd += ["-i", str(src)]
    if to_ts is not None and to_ts > from_ts:
        cmd += ["-t", f"{max(to_ts - from_ts, 1.0 / fps):.6f}"]
    cmd += [
        "-vf",
        "setpts=PTS-STARTPTS",
        "-r",
        str(int(round(fps))),
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"切视频失败 {src} [{from_ts}, {to_ts}):\n{r.stderr}")
    shutil.move(str(tmp), str(dst))


def _write_tasks_jsonl(meta_dir: Path) -> int:
    tasks: list[tuple[int, str]] = []
    tasks_parquet = meta_dir / "tasks.parquet"
    if tasks_parquet.exists():
        tasks_df = pd.read_parquet(tasks_parquet)
        if "task_index" in tasks_df.columns:
            name_col = next((c for c in tasks_df.columns if c != "task_index"), None)
            if name_col is not None:
                for _, row in tasks_df.iterrows():
                    tasks.append((int(row["task_index"]), str(row[name_col])))
            elif not isinstance(tasks_df.index, pd.RangeIndex):
                for name, row in tasks_df.iterrows():
                    tasks.append((int(row["task_index"]), str(name)))
        else:
            for i, name in enumerate(tasks_df.index.astype(str)):
                tasks.append((i, name))
    if not tasks:
        tasks = [(0, "pick_and_place")]
    tasks = sorted(set(tasks), key=lambda x: x[0])
    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for idx, name in tasks:
            f.write(json.dumps({"task_index": idx, "task": name}, ensure_ascii=False) + "\n")
    return len(tasks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--rewrite-timestamps",
        action="store_true",
        help="Rewrite timestamps to i/fps so LeRobot v2.1 sync check (1e-4s) passes.",
    )
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    meta_dir = root / "meta"
    info_path = meta_dir / "info.json"
    if not info_path.exists():
        print(f"缺少 {info_path}", file=sys.stderr)
        sys.exit(1)

    info = json.loads(info_path.read_text())
    version = str(info.get("codebase_version", ""))
    backup_root = root / "_v3_original"
    if version.startswith("v2.") and not backup_root.exists():
        print(f"{root} 已经是 {version}，跳过转换。")
        return
    if version.startswith("v2.") and backup_root.exists():
        v3_info_path = meta_dir / "info.v3.json"
        if v3_info_path.exists():
            info = json.loads(v3_info_path.read_text())
            version = str(info.get("codebase_version", ""))
    if not version.startswith("v3"):
        print(f"未识别的 codebase_version={version}", file=sys.stderr)
        sys.exit(1)

    fps = float(info.get("fps", 30))
    video_keys = [k for k, spec in info["features"].items() if spec.get("dtype") == "video"]
    video_src_tmpl = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")

    if backup_root.exists():
        print(f"发现 {backup_root}，从备份续跑（先清掉不完整的 v2.1 输出）。")
        for p in (root / "data").rglob("episode_*.parquet"):
            p.unlink()
        v21_videos = root / "videos" / "chunk-000"
        if v21_videos.exists():
            shutil.rmtree(v21_videos)
        source_root = backup_root
    else:
        parquet_files = sorted((root / "data").rglob("*.parquet"))
        if not parquet_files:
            print(f"未找到 parquet: {root / 'data'}", file=sys.stderr)
            sys.exit(1)
        video_files = sorted((root / "videos").rglob("*.mp4")) if (root / "videos").exists() else []
        (meta_dir / "info.v3.json").write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n")
        backup_root.mkdir(parents=True)
        for src in parquet_files:
            _move_aside(src, backup_root, root)
        for mp4 in video_files:
            _move_aside(mp4, backup_root, root)
        source_root = backup_root

    parquet_files = sorted((source_root / "data").rglob("*.parquet"))
    if not parquet_files:
        print(f"备份中没有 parquet: {source_root / 'data'}", file=sys.stderr)
        sys.exit(1)
    frames = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    if "episode_index" not in frames.columns:
        print("parquet 缺少 episode_index", file=sys.stderr)
        sys.exit(1)

    episode_ids = sorted(int(x) for x in frames["episode_index"].unique())
    if episode_ids != list(range(len(episode_ids))):
        print(f"episode_index 不连续: {episode_ids[:20]}", file=sys.stderr)
        sys.exit(1)

    ep_video_map = _load_episode_video_map(meta_dir, video_keys)

    rewritten = 0
    for ep_idx in episode_ids:
        ep = frames[frames["episode_index"] == ep_idx].sort_values(
            "frame_index" if "frame_index" in frames.columns else "index"
        )
        ep = ep.reset_index(drop=True)
        if args.rewrite_timestamps or "timestamp" not in ep.columns:
            ep = ep.copy()
            ep["timestamp"] = (np.arange(len(ep), dtype=np.float32) / fps).astype(np.float32)
            rewritten += 1
        else:
            ts = np.asarray(ep["timestamp"].to_numpy(), dtype=np.float64).reshape(-1)
            if ts.size >= 2 and np.any(np.abs(np.diff(ts) - (1.0 / fps)) > 1e-4):
                ep = ep.copy()
                ep["timestamp"] = (np.arange(len(ep), dtype=np.float32) / fps).astype(np.float32)
                rewritten += 1

        chunk = ep_idx // int(info.get("chunks_size", 1000))
        dst_parquet = root / V21_DATA_PATH.format(episode_chunk=chunk, episode_index=ep_idx)
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)
        ep.to_parquet(dst_parquet, index=False)

        for key in video_keys:
            mapping = (ep_video_map.get(ep_idx) or {}).get(key)
            if mapping is not None:
                src_video = source_root / video_src_tmpl.format(
                    video_key=key, chunk_index=mapping["chunk"], file_index=mapping["file"]
                )
                from_ts = float(mapping["from_ts"])
                to_ts = mapping["to_ts"]
            else:
                src_video = source_root / video_src_tmpl.format(
                    video_key=key, chunk_index=chunk, file_index=ep_idx
                )
                from_ts, to_ts = 0.0, None
                if not src_video.exists():
                    candidates = sorted((source_root / "videos" / key).rglob("*.mp4"))
                    if len(candidates) == 1:
                        src_video = candidates[0]
            if not src_video.exists():
                print(f"找不到视频 {key} episode {ep_idx}，试过 {src_video}", file=sys.stderr)
                sys.exit(1)

            dst_video = root / V21_VIDEO_PATH.format(
                episode_chunk=chunk, video_key=key, episode_index=ep_idx
            )
            _extract_video_segment(src_video, dst_video, from_ts, to_ts, fps)
            print(
                f"episode {ep_idx} {key}: {src_video.name} [{from_ts:.3f}, {to_ts}] -> {dst_video.relative_to(root)}"
            )

    n_tasks = _write_tasks_jsonl(meta_dir)

    info["codebase_version"] = "v2.1"
    info["data_path"] = V21_DATA_PATH
    info["video_path"] = V21_VIDEO_PATH
    info["total_episodes"] = len(episode_ids)
    info["total_frames"] = int(len(frames))
    info["total_tasks"] = n_tasks
    info["total_chunks"] = 1 + (max(episode_ids) // int(info.get("chunks_size", 1000)))
    info["total_videos"] = len(video_keys) * len(episode_ids)
    info["splits"] = {"train": f"0:{len(episode_ids)}"}
    info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n")

    print(f"已转换为 LeRobot v2.1：{root}")
    print(f"episodes={len(episode_ids)} frames={len(frames)} tasks={n_tasks} rewritten_timestamps={rewritten}")
    print(f"v3 原文件在 {backup_root}；原始 info 备份为 {meta_dir / 'info.v3.json'}")
    print("接下来请运行: uv run python scripts/repair_lerobot_dataset_meta.py --dataset-root", root)


if __name__ == "__main__":
    main()
