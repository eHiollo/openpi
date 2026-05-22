#!/usr/bin/env python
"""从 parquet 重建 LeRobot v3 的 meta/episodes.jsonl 与 meta/episodes_stats.jsonl（不猜 length）。

- episodes.jsonl：按 parquet 里 episode_index 分组计数，length = 行数。
- episodes_stats.jsonl：每段内从真实列计算统计；数值列补全 q01/q10/q50/q90/q99；
  视频列：parquet 有路径列时用 compute_episode_stats；否则用 meta/stats.json 全局统计占位（stderr 提示一次）。

用法（数据集根下须有 data/**/*.parquet）：

  uv run python scripts/repair_lerobot_dataset_meta.py \\
    --dataset-root /path/to/dataset/a10_dataset_4_22

tasks 列表文案来自 meta/tasks.jsonl 首行 task 的小写形式（与 Dataset_A10 的 episodes.jsonl 风格一致）。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

from lerobot.common.datasets.compute_stats import compute_episode_stats, estimate_num_samples, sample_images
from lerobot.common.datasets.utils import serialize_dict


def _find_parquet_files(dataset_root: pathlib.Path) -> list[pathlib.Path]:
    data_dir = dataset_root / "data"
    if not data_dir.is_dir():
        return []
    return sorted(data_dir.rglob("*.parquet"))


def _task_label_from_meta(meta_dir: pathlib.Path) -> str:
    tasks_path = meta_dir / "tasks.jsonl"
    if not tasks_path.exists():
        return "reach the yellow lemon"
    with open(tasks_path, encoding="utf-8") as f:
        line = f.readline().strip()
    if not line:
        return "reach the yellow lemon"
    row = json.loads(line)
    t = row.get("task", "")
    return str(t).strip().lower() if isinstance(t, str) else "reach the yellow lemon"


def _stack_vector_column(series: pd.Series, *, target_dim: int) -> np.ndarray:
    raw = series.to_numpy()
    if len(raw) == 0:
        return np.zeros((0, target_dim), dtype=np.float32)
    first = np.asarray(raw[0], dtype=np.float32).reshape(-1)
    if first.size != target_dim:
        raise ValueError(f"Expected dim {target_dim}, got {first.size} for first row")
    out = np.empty((len(raw), target_dim), dtype=np.float32)
    out[0] = first
    for i in range(1, len(raw)):
        v = np.asarray(raw[i], dtype=np.float32).reshape(-1)
        if v.size != target_dim:
            raise ValueError(f"Row {i}: expected dim {target_dim}, got {v.size}")
        out[i] = v
    return out


def _add_scalar_quantiles(stats: dict[str, np.ndarray], x: np.ndarray) -> None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    for name, q in (("q01", 1), ("q10", 10), ("q50", 50), ("q90", 90), ("q99", 99)):
        stats[name] = np.array([np.nanpercentile(x, q)])


def _add_vector_quantiles(stats: dict[str, np.ndarray], x: np.ndarray) -> None:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(x.shape)
    for name, q in (("q01", 1), ("q10", 10), ("q50", 50), ("q90", 90), ("q99", 99)):
        stats[name] = np.nanpercentile(x, q, axis=0)


def _synthetic_video_image_stats(num_frame_rows: int) -> dict[str, np.ndarray]:
    """当 parquet 无图像路径且无法从 mp4 逐帧采样时，生成与 LeRobot v2.1 一致的占位统计（CHW 归一化到 0~1）。"""
    n = estimate_num_samples(num_frame_rows)
    chw = (3, 1, 1)
    zeros = np.zeros(chw, dtype=np.float64)
    ones = np.ones(chw, dtype=np.float64)
    half = np.full(chw, 0.5, dtype=np.float64)
    small = np.full(chw, 0.01, dtype=np.float64)
    return {
        "min": zeros.copy(),
        "max": ones.copy(),
        "mean": half.copy(),
        "std": small.copy(),
        "count": np.array([n], dtype=np.int64),
        "q01": np.full(chw, 0.01, dtype=np.float64),
        "q10": np.full(chw, 0.1, dtype=np.float64),
        "q50": half.copy(),
        "q90": np.full(chw, 0.9, dtype=np.float64),
        "q99": np.full(chw, 0.99, dtype=np.float64),
    }


def _add_video_quantiles(stats: dict[str, np.ndarray], paths: list[str]) -> None:
    """分位数在 0~1 域上算，与 compute_episode_stats 里对 min/max/mean/std 的 /255 尺度一致。"""
    imgs_u8 = sample_images(paths)
    ep_ft = imgs_u8.astype(np.float64) / 255.0
    axes = (0, 2, 3)
    keepdims = True
    for name, q in (("q01", 1), ("q10", 10), ("q50", 50), ("q90", 90), ("q99", 99)):
        qv = np.nanpercentile(ep_ft, q, axis=axes, keepdims=keepdims)
        stats[name] = np.squeeze(qv, axis=0)


def _json_leaf_to_numpy(obj: dict) -> dict[str, np.ndarray]:
    """将 stats.json 里单个 feature 的 {min: [...], ...} 转为 ndarray。"""
    return {k: np.asarray(v) for k, v in obj.items()}


def _paths_from_video_column(series: pd.Series, dataset_root: pathlib.Path) -> list[str]:
    out: list[str] = []
    for v in series.to_numpy():
        if v is None:
            continue
        p: str | None = None
        if isinstance(v, str) and v:
            p = v
        elif isinstance(v, dict) and "path" in v and isinstance(v["path"], str):
            p = v["path"]
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], str):
            p = v[0]
        if not p:
            continue
        pp = pathlib.Path(p)
        if not pp.is_absolute():
            pp = dataset_root / pp
        out.append(str(pp))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=pathlib.Path, required=True)
    args = p.parse_args()

    root = args.dataset_root.resolve()
    meta_dir = root / "meta"
    info_path = meta_dir / "info.json"
    if not info_path.exists():
        print(f"缺少 {info_path}", file=sys.stderr)
        sys.exit(1)

    info = json.loads(info_path.read_text())
    features: dict = info["features"]
    total_episodes = int(info["total_episodes"])
    total_frames_meta = int(info.get("total_frames", -1))

    parquet_files = _find_parquet_files(root)
    if not parquet_files:
        print(
            f"未找到任何 parquet：{root / 'data'}/**/*.parquet\n"
            "请先把完整数据（含 data 目录）放到该路径下，再运行本脚本。",
            file=sys.stderr,
        )
        sys.exit(1)

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    if "episode_index" not in df.columns:
        print("parquet 中缺少 episode_index 列", file=sys.stderr)
        sys.exit(1)

    optional_in_parquet = {"image", "video"}
    required_cols = [
        k
        for k, spec in features.items()
        if spec.get("dtype") != "string" and spec.get("dtype") not in optional_in_parquet
    ]
    missing = [k for k in required_cols if k not in df.columns]
    if missing:
        print(f"parquet 缺少与 info.features 对应的列: {missing}", file=sys.stderr)
        sys.exit(1)

    stats_json_path = meta_dir / "stats.json"
    global_stats: dict = {}
    if stats_json_path.exists():
        global_stats = json.loads(stats_json_path.read_text())
    video_fallback_warned = False
    synthetic_video_warned = False

    counts = df.groupby("episode_index", sort=True).size()
    lengths = [int(counts.loc[i]) for i in range(total_episodes)]
    if len(counts) != total_episodes or list(counts.index) != list(range(total_episodes)):
        print(
            f"episode_index 不连续或与 info.total_episodes={total_episodes} 不一致：\n"
            f"实际 index: {list(counts.index)[:20]}{'...' if len(counts) > 20 else ''}",
            file=sys.stderr,
        )
        sys.exit(1)

    n_rows = len(df)
    sum_len = sum(lengths)
    if sum_len != n_rows:
        print(f"episode length 之和 {sum_len} 与总行数 {n_rows} 不一致", file=sys.stderr)
        sys.exit(1)
    if total_frames_meta >= 0 and n_rows != total_frames_meta:
        print(
            f"警告：info.json total_frames={total_frames_meta} 与 parquet 总行数={n_rows} 不一致，"
            "将以 parquet 为准。若需一致请更新 info.json。",
            file=sys.stderr,
        )

    task_label = _task_label_from_meta(meta_dir)
    episodes_path = meta_dir / "episodes.jsonl"
    with open(episodes_path, "w", encoding="utf-8") as f:
        for ep_idx, length in enumerate(lengths):
            f.write(
                json.dumps(
                    {"episode_index": ep_idx, "tasks": [task_label], "length": length},
                    ensure_ascii=False,
                )
                + "\n"
            )

    episodes_stats_path = meta_dir / "episodes_stats.jsonl"
    with open(episodes_stats_path, "w", encoding="utf-8") as f:
        for ep_idx in range(total_episodes):
            sub = df[df["episode_index"] == ep_idx]
            episode_data: dict = {}
            for key, spec in features.items():
                dtype = spec.get("dtype")
                if dtype == "string":
                    continue
                if dtype in ("image", "video"):
                    if key not in df.columns:
                        continue
                    paths = _paths_from_video_column(sub[key], root)
                    if paths:
                        episode_data[key] = paths
                    continue
                shape = tuple(spec.get("shape", (1,)))
                if len(shape) == 1 and shape[0] == 1:
                    episode_data[key] = np.asarray(sub[key].to_numpy(), dtype=np.float32).reshape(-1)
                elif len(shape) == 1 and shape[0] > 1:
                    episode_data[key] = _stack_vector_column(sub[key], target_dim=int(shape[0]))
                else:
                    print(f"警告：未处理特征 {key} shape={shape}，跳过", file=sys.stderr)

            ep_stats = compute_episode_stats(episode_data, features)

            for key, spec in features.items():
                if spec.get("dtype") == "string":
                    continue
                if spec["dtype"] in ("image", "video") and key not in ep_stats:
                    if key in global_stats:
                        if not video_fallback_warned:
                            print(
                                f"提示：parquet 无列 {key}，已用 meta/stats.json 的全局统计为每段 episode 占位（非逐段视频统计）。",
                                file=sys.stderr,
                            )
                            video_fallback_warned = True
                        ep_stats[key] = _json_leaf_to_numpy(global_stats[key])
                        ep_stats[key]["count"] = np.array([estimate_num_samples(len(sub))], dtype=np.int64)
                    else:
                        if not synthetic_video_warned:
                            print(
                                f"提示：parquet 与 meta/stats.json 均无 {key}；"
                                f"已写入合成图像/视频统计占位（仅满足 LeRobot 元数据加载；训练仍从 mp4 解码）。",
                                file=sys.stderr,
                            )
                            synthetic_video_warned = True
                        ep_stats[key] = _synthetic_video_image_stats(len(sub))
                    continue

                if key not in ep_stats:
                    continue
                st = ep_stats[key]
                if spec["dtype"] in ("image", "video"):
                    paths = episode_data.get(key)
                    if not isinstance(paths, list) or not paths:
                        if key in global_stats:
                            ep_stats[key] = _json_leaf_to_numpy(global_stats[key])
                            ep_stats[key]["count"] = np.array([estimate_num_samples(len(sub))], dtype=np.int64)
                        else:
                            if not synthetic_video_warned:
                                print(
                                    f"提示：无法解析 {key} 的帧图像路径且 meta/stats.json 无该键；"
                                    f"已用合成统计占位（LeRobot 加载用；视频解码路径需与 info.json 一致）。",
                                    file=sys.stderr,
                                )
                                synthetic_video_warned = True
                            ep_stats[key] = _synthetic_video_image_stats(len(sub))
                        continue
                    _add_video_quantiles(st, paths)
                else:
                    shape = tuple(spec.get("shape", (1,)))
                    arr = episode_data.get(key)
                    if isinstance(arr, np.ndarray) and arr.ndim == 2:
                        _add_vector_quantiles(st, arr)
                    elif isinstance(arr, np.ndarray) and arr.ndim == 1:
                        _add_scalar_quantiles(st, arr)

            serialized = serialize_dict(ep_stats)
            f.write(json.dumps({"episode_index": ep_idx, "stats": serialized}, ensure_ascii=False) + "\n")

    print(f"已写入 {episodes_path}（{total_episodes} 条，总帧数 {n_rows}）")
    print(f"已写入 {episodes_stats_path}（{total_episodes} 条）")


if __name__ == "__main__":
    main()
