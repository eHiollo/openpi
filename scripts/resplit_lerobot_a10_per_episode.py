#!/usr/bin/env python3
"""
将「多块合并」的 A10 数据集重排为 LeRobot 元数据里声明的布局：
  data/chunk-000/file-{episode_index:03d}.parquet
  videos/<video_key>/chunk-000/file-{episode_index:03d}.mp4

适用：data 与 video 各为 7 个合并文件、parquet 内已有 0..N-1 的 episode_index，
     但 info.json 的 data_path / video_path 按「每集一文件」命名（否则 load 会断言语句失败）。

用法：
  uv run python scripts/resplit_lerobot_a10_per_episode.py \\
    --dataset-root dataset/a10_dataset_4_22

会先把现有 data/、videos/ 移入 <dataset-root>/_backup_merged_layout/，再生成新结构。
视频切分使用 imageio_ffmpeg 自带的 ffmpeg 可执行文件（与 uv 环境一同安装，无需系统 PATH）。

"""

from __future__ import annotations

import json
import argparse
import shutil
import sys
from pathlib import Path

import av
import pandas as pd


def _load_episode_lengths(meta_dir: Path) -> list[int]:
    p = meta_dir / "episodes.jsonl"
    if not p.exists():
        print(f"缺少 {p}", file=sys.stderr)
        sys.exit(1)
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    out: list[int] = []
    for i, line in enumerate(lines):
        row = json.loads(line)
        if int(row["episode_index"]) != i:
            print(f"episodes.jsonl 行顺序异常: 期望 episode_index=={i}", file=sys.stderr)
            sys.exit(1)
        out.append(int(row["length"]))
    return out


def _episodes_in_each_merged_pq(data_chunk: Path) -> list[tuple[str, list[int]]]:
    """每个合并 parquet 文件 -> 该文件内 (有序) episode_index 列表。"""
    files = sorted(data_chunk.glob("file-*.parquet"))
    blocks: list[tuple[str, list[int]]] = []
    for f in files:
        df = pd.read_parquet(f)
        if "episode_index" not in df.columns:
            print(f"{f} 无 episode_index", file=sys.stderr)
            sys.exit(1)
        idxs = sorted(df["episode_index"].unique().tolist())
        blocks.append((f.name, idxs))
    return blocks


def _split_video_pyav(src: Path, dst: Path, start_frame: int, n_frames: int) -> None:
    """与 parquet 行数严格一致；输出文件时间戳从 0 起，与每集内 timestamp 列一致（LeRobot VideoReader 用 pts）。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    inp = av.open(str(src))
    in_v = inp.streams.video[0]
    in_v.thread_type = "AUTO"
    out = av.open(str(dst), "w")
    rate = in_v.average_rate or 15
    out_stream = out.add_stream("libx264", rate=rate)
    out_stream.width = in_v.width
    out_stream.height = in_v.height
    out_stream.pix_fmt = "yuv420p"
    out_stream.options = {"crf": "23", "preset": "medium"}
    fps = float(in_v.average_rate) if in_v.average_rate is not None else 15.0
    # add_stream 后 time_base 常为 None，libx264 输出多使用 1/15360、15fps 下每帧 +1024
    from fractions import Fraction

    tb: Fraction = Fraction(1, 15_360)
    sec_per_frame = 1.0 / fps
    pts_step = int(round(sec_per_frame / float(tb)))
    dec_idx = 0
    written = 0
    try:
        for frame in inp.decode(in_v):
            if dec_idx < start_frame:
                dec_idx += 1
                continue
            if written >= n_frames:
                break
            f = frame.reformat(width=frame.width, height=frame.height, format="yuv420p")
            # 禁止沿用源长视频的绝对 pts，否则每集独立 mp4 仍带合并前时间轴（~100s），与 parquet 内 0~ 秒不一致
            f.pts = written * pts_step
            f.time_base = tb
            for packet in out_stream.encode(f):
                out.mux(packet)
            written += 1
            dec_idx += 1
        for packet in out_stream.encode(None):
            out.mux(packet)
    finally:
        out.close()
        inp.close()
    if written != n_frames:
        print(
            f"切分后帧数不符: 期望 {n_frames} 实际 {written} (源 {src} start={start_frame})",
            file=sys.stderr,
        )
        sys.exit(1)


def _build_episode_video_offsets(
    merged_data_chunk: Path, lengths: list[int], total_eps: int
) -> tuple[dict[int, int], dict[tuple[int, int], int]]:
    blocks = _episodes_in_each_merged_pq(merged_data_chunk)
    if not blocks:
        print(f"未找到合并 parquet: {merged_data_chunk}", file=sys.stderr)
        sys.exit(1)
    ep_to_merged: dict[int, int] = {}
    offset_in_merged: dict[tuple[int, int], int] = {}
    for bi, (_fn, ep_idxs) in enumerate(blocks):
        start_f = 0
        for ep in ep_idxs:
            ep_to_merged[ep] = bi
            offset_in_merged[(bi, ep)] = start_f
            start_f += lengths[ep]
    if set(ep_to_merged.keys()) != set(range(total_eps)):
        print("parquet 中 episode 与 episodes.jsonl 覆盖不一致", file=sys.stderr)
        sys.exit(1)
    return ep_to_merged, offset_in_merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument(
        "--videos-only",
        action="store_true",
        help="不移动/重写 parquet，仅从 _backup_merged_layout 用 PyAV 重切 50 段视频（修正少帧等）。",
    )
    args = ap.parse_args()
    root: Path = args.dataset_root.resolve()
    meta = root / "meta"
    info_path = meta / "info.json"
    if not info_path.exists():
        print(f"缺少 {info_path}", file=sys.stderr)
        sys.exit(1)
    info = json.loads(info_path.read_text())
    total_eps = int(info["total_episodes"])
    if total_eps < 1:
        sys.exit(1)

    lengths = _load_episode_lengths(meta)
    if len(lengths) != total_eps:
        print("episodes.jsonl 条数与 info.total_episodes 不一致", file=sys.stderr)
        sys.exit(1)

    video_key = "observation.images.right"
    backup = root / "_backup_merged_layout"

    if args.videos_only:
        merged_data = backup / "data" / "chunk-000"
        if not merged_data.is_dir():
            print(f"--videos-only 需要合并版 parquet: {merged_data}", file=sys.stderr)
            sys.exit(1)
        ep_to_merged, offset_in_merged = _build_episode_video_offsets(merged_data, lengths, total_eps)
        merged_vdir = backup / "videos" / video_key / "chunk-000"
        if not merged_vdir.is_dir():
            print(f"未找到 {merged_vdir}", file=sys.stderr)
            sys.exit(1)
        new_vdir = root / "videos" / video_key / "chunk-000"
        new_vdir.mkdir(parents=True, exist_ok=True)
        for ep in range(total_eps):
            bi = ep_to_merged[ep]
            src = merged_vdir / f"file-{bi:03d}.mp4"
            if not src.is_file():
                print(f"缺少源视频 {src}", file=sys.stderr)
                sys.exit(1)
            st = offset_in_merged[(bi, ep)]
            n = lengths[ep]
            dst = new_vdir / f"file-{ep:03d}.mp4"
            _split_video_pyav(src, dst, st, n)
    else:
        data_chunk = root / "data" / "chunk-000"
        if not data_chunk.is_dir():
            print(f"缺少 {data_chunk}", file=sys.stderr)
            sys.exit(1)
        ep_to_merged, offset_in_merged = _build_episode_video_offsets(data_chunk, lengths, total_eps)
        if backup.exists():
            print(f"已存在 {backup}，请先删除或移走再跑全量，或改 --videos-only。", file=sys.stderr)
            sys.exit(1)
        backup.mkdir(parents=True)
        for name in ("data", "videos"):
            p = root / name
            if p.exists():
                shutil.move(str(p), str(backup / name))
        new_data = root / "data" / "chunk-000"
        new_data.mkdir(parents=True, exist_ok=True)
        merged_root = backup / "data" / "chunk-000"
        all_dfs: list[pd.DataFrame] = []
        for f in sorted(merged_root.glob("file-*.parquet")):
            all_dfs.append(pd.read_parquet(f))
        df = pd.concat(all_dfs, ignore_index=True).sort_values(
            ["episode_index", "frame_index"], kind="mergesort"
        )
        for ep in range(total_eps):
            sub = df[df["episode_index"] == ep].copy()
            out_pq = new_data / f"file-{ep:03d}.parquet"
            sub.to_parquet(out_pq, index=False)
        merged_vdir = backup / "videos" / video_key / "chunk-000"
        if not merged_vdir.is_dir():
            print(
                f"未找到 {merged_vdir}，无法切分视频。已写好 parquet。",
                file=sys.stderr,
            )
        else:
            new_vdir = root / "videos" / video_key / "chunk-000"
            new_vdir.mkdir(parents=True, exist_ok=True)
            for ep in range(total_eps):
                bi = ep_to_merged[ep]
                src = merged_vdir / f"file-{bi:03d}.mp4"
                if not src.is_file():
                    print(f"缺少源视频 {src}", file=sys.stderr)
                    sys.exit(1)
                st = offset_in_merged[(bi, ep)]
                n = lengths[ep]
                dst = new_vdir / f"file-{ep:03d}.mp4"
                _split_video_pyav(src, dst, st, n)

    # 与视频编码一致，更新 info 中 codec（H.264）
    if (root / "videos" / video_key / "chunk-000" / "file-000.mp4").is_file() and "features" in info:
        vfeat = info["features"].get(video_key)
        if vfeat and isinstance(vfeat, dict) and "info" in vfeat:
            vfeat["info"] = vfeat.get("info", {})
            vfeat["info"]["video.codec"] = "h264"
            vfeat["info"]["video.pix_fmt"] = "yuv420p"
    info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")

    print("完成。")
    if not args.videos_only:
        print(f"  合并前数据已备份: {backup}")
        print(f"  新 data: {root / 'data' / 'chunk-000'}（{total_eps} 个 parquet）")
    else:
        print("  （仅重切视频，data/ 与 parquet 未改动）")
    print(f"  视频目录: {root / 'videos' / video_key / 'chunk-000'}")
    if (root / "videos" / video_key / "chunk-000" / "file-000.mp4").is_file():
        print("  分集视频为 H.264，已回写 info.json 中的 video.codec / pix_fmt。")
    print("建议随后运行:")
    print(f"  uv run python scripts/repair_lerobot_dataset_meta.py --dataset-root {root}")


if __name__ == "__main__":
    main()
