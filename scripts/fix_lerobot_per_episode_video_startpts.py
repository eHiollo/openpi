#!/usr/bin/env python3
"""
分集 mp4 若保留「合并长视频」的绝对时间戳，LeRobot + torchvision 读出的 frame['pts'] 会仍是 ~100s，
与 parquet 里本集内 timestamp（从 0 开始）无法对齐，触发 tolerance 断言。

本脚本对每个 file-XXX.mp4 执行：
  setpts=PTS-STARTPTS  且  -r <fps>  与 info.json 的 fps 一致
使时间轴从 0 起、帧率与采集一致（默认不重采样内容，但需重编码以应用 setpts）。

用法：
  uv run python scripts/fix_lerobot_per_episode_video_startpts.py \\
    --dataset-root dataset/a10_dataset_4_22 \\
    --video-key observation.images.right
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--video-key", type=str, default="observation.images.right")
    args = ap.parse_args()
    root: Path = args.dataset_root.resolve()
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = int(info.get("fps", 15))
    vdir = root / "videos" / args.video_key / "chunk-000"
    if not vdir.is_dir():
        print(f"缺少 {vdir}", file=sys.stderr)
        sys.exit(1)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    files = sorted(vdir.glob("file-*.mp4"))
    if not files:
        print(f"{vdir} 下无 mp4", file=sys.stderr)
        sys.exit(1)
    for f in files:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as t:
            tmp = Path(t.name)
        cmd = [
            ff,
            "-y",
            "-i",
            str(f),
            "-vf",
            "setpts=PTS-STARTPTS",
            "-r",
            str(fps),
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
        if r.returncode != 0:
            print(f"失败 {f.name}:\n{r.stderr}", file=sys.stderr)
            tmp.unlink(missing_ok=True)
            sys.exit(1)
        shutil.move(str(tmp), str(f))
        print(f"ok {f.name}")
    print(f"完成 {len(files)} 个文件 (fps={fps})。")


if __name__ == "__main__":
    main()
