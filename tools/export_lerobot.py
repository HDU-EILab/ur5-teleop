"""把自研 episode 格式导出为 LeRobot v2.1 数据集（Pi0.5 训练用）。

设计：
- 主时钟用 global 相机逐帧时间戳；每帧从 robot.csv 取最近一行，构造 state/action。
- state(13) = 实测 TCP 6 + 关节 6 + 夹爪实测位置(0-1)
- action(7) = 目标 TCP 6 + 夹爪命令(0-1，从 trigger 按 binary 模式推导)
- B 类（含人工扰动窗口）**截断**：把扰动窗口（左右各留 margin）剪掉，
  窗口前后切成独立 segment；过短的 segment 丢弃。A 类整条保留。
- 视频按 segment 用 ffmpeg 精确截取并重编 h264（30fps CFR）。
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

FPS = 30.0
CHUNKS_SIZE = 1000

# 录制时两路相机装反的源 episode（导出时交换回来）
CAMERA_SWAP_SOURCES = {
    "20261111_142247_ep001",
}
TASK_EN = ("Put the red cylinder and the blue cube into the storage box. "
           "Leave all other objects outside the box.")

STATE_NAMES = ["robot_x", "robot_y", "robot_z", "robot_rx", "robot_ry", "robot_rz",
               "q1", "q2", "q3", "q4", "q5", "q6", "gripper"]
ACTION_NAMES = ["target_x", "target_y", "target_z", "target_rx", "target_ry", "target_rz",
                "gripper"]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_timestamps(path: Path) -> list[float]:
    rows = read_csv_rows(path)
    return [float(r["t_mono"]) for r in rows]


def gripper_command(trigger: float) -> float:
    """binary 模式：扳机过半 → 闭合(0)，否则全开(1000)；归一化到 0-1。"""
    trigger = max(0.0, min(1.0, float(trigger)))
    return 0.0 if trigger >= 0.5 else 1.0


def segment_ranges(n_frames: int, windows: list[list[float]],
                   windows_origin: float, frame_times: list[float],
                   margin_s: float, min_frames: int) -> list[tuple[int, int]]:
    """返回保留帧的闭区间 [(start, end), ...]（按扰动窗口切分）。"""
    keep = [True] * n_frames
    for start, end in windows:
        lo = start - windows_origin - margin_s
        hi = end - windows_origin + margin_s
        for idx in range(n_frames):
            t = frame_times[idx] - windows_origin
            if lo <= t <= hi:
                keep[idx] = False
    segments: list[tuple[int, int]] = []
    idx = 0
    while idx < n_frames:
        if not keep[idx]:
            idx += 1
            continue
        j = idx
        while j + 1 < n_frames and keep[j + 1]:
            j += 1
        if j - idx + 1 >= min_frames:
            segments.append((idx, j))
        idx = j + 1
    return segments


def cut_video(src: Path, dst: Path, a: int, b: int, fps: float) -> None:
    """精确截取源视频第 a..b 帧（闭区间），输出 30fps CFR h264，帧数严格为 b-a+1。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    filter_chain = (f"select='between(n\\,{a}\\,{b})',"
                    f"setpts=N/({fps:g}*TB)")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           "-vf", filter_chain, "-vsync", "cfr", "-r", f"{fps:g}",
           "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24",
           "-pix_fmt", "yuv420p", str(dst)]
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-dir", default="data/episodes")
    ap.add_argument("--out", default="data/lerobot_ur5_pickplace")
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--cut-margin", type=float, default=0.25)
    ap.add_argument("--min-segment-s", type=float, default=1.5)
    args = ap.parse_args()

    root = Path(args.episodes_dir)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)
    (out / "data" / "chunk-000").mkdir(parents=True)
    for cam in ("global", "wrist"):
        (out / "videos" / "chunk-000" / f"observation.images.{cam}").mkdir(parents=True)

    episode_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("2026")])
    min_frames = int(args.min_segment_s * args.fps)
    stats_sum: dict[str, list[np.ndarray]] = {"state": [], "action": []}
    total_frames = 0
    episodes_meta: list[dict[str, Any]] = []
    cut_info: list[dict[str, Any]] = []
    ep_index = 0

    for src in episode_dirs:
        meta_path = src / "meta.json"
        robot_path = src / "robot.csv"
        g_ts_path = src / "videos" / "global_timestamps.csv"
        w_ts_path = src / "videos" / "wrist_timestamps.csv"
        if not (meta_path.exists() and robot_path.exists() and g_ts_path.exists()):
            print(f"[skip] {src.name}: 缺文件")
            continue
        meta = json.loads(meta_path.read_text())
        rows = read_csv_rows(robot_path)
        frame_times = read_timestamps(g_ts_path)
        wrist_times = read_timestamps(w_ts_path) if w_ts_path.exists() else frame_times
        # 两路相机帧数略有差异（wrist 晚起 0.04~0.12 s）：统一截断到共同帧数，保证视频帧数==数据行数
        n_frames = min(len(frame_times), len(wrist_times))
        if n_frames < min_frames:
            print(f"[skip] {src.name}: 帧数太少 {n_frames}")
            continue
        windows = meta.get("disturbance_windows") or []
        origin = meta.get("started_at", 0.0)
        segments = segment_ranges(n_frames, windows, origin, frame_times,
                                  args.cut_margin, min_frames)
        if not segments:
            print(f"[skip] {src.name}: 截断后无有效 segment")
            continue
        robot_times = [float(r["t_mono"]) for r in rows]

        def nearest_row(t: float) -> dict[str, str]:
            i = bisect.bisect_left(robot_times, t)
            if i == 0:
                return rows[0]
            if i >= len(rows):
                return rows[-1]
            if abs(robot_times[i] - t) < abs(t - robot_times[i - 1]):
                return rows[i]
            return rows[i - 1]

        for a, b in segments:
            seg_len = b - a + 1
            states: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            for fi in range(a, b + 1):
                t = frame_times[fi]
                r = nearest_row(t)
                state = np.array([float(r[k]) for k in
                                  ("robot_x", "robot_y", "robot_z", "robot_rx", "robot_ry",
                                   "robot_rz", "q1", "q2", "q3", "q4", "q5", "q6")],
                                 dtype=np.float32)
                grip = float(r["gripper_position"] or 1000) / 1000.0
                state = np.concatenate([state, [grip]]).astype(np.float32)
                action6 = np.array([float(r[k]) for k in
                                    ("target_x", "target_y", "target_z", "target_rx",
                                     "target_ry", "target_rz")], dtype=np.float32)
                grip_cmd = gripper_command(float(r["trigger_right"] or 0.0))
                action = np.concatenate([action6, [grip_cmd]]).astype(np.float32)
                states.append(state)
                actions.append(action)

            # 写视频（两路相机用同一帧区间；已截断到共同帧数，帧数严格等于 seg_len）
            swapped = src.name in CAMERA_SWAP_SOURCES
            for cam in ("global", "wrist"):
                src_cam = ("wrist" if cam == "global" else "global") if swapped else cam
                src_video = src / "videos" / f"{src_cam}.mp4"
                dst_video = (out / "videos" / "chunk-000" /
                             f"observation.images.{cam}" / f"episode_{ep_index:06d}.mp4")
                cut_video(src_video, dst_video, a, b, args.fps)

            # parquet
            df = pd.DataFrame({
                "index": np.arange(total_frames, total_frames + seg_len, dtype=np.int64),
                "episode_index": np.full(seg_len, ep_index, dtype=np.int64),
                "frame_index": np.arange(seg_len, dtype=np.int64),
                "timestamp": (np.arange(seg_len, dtype=np.float32) / np.float32(args.fps)),
                "task_index": np.zeros(seg_len, dtype=np.int64),
                "observation.state": [s for s in states],
                "action": [a_ for a_ in actions],
            })
            df.to_parquet(out / "data" / "chunk-000" / f"episode_{ep_index:06d}.parquet",
                          index=False)

            # 统计量
            st = np.stack(states)
            ac = np.stack(actions)
            stats_sum["state"].append(st)
            stats_sum["action"].append(ac)

            episodes_meta.append({"episode_index": ep_index,
                                  "tasks": [TASK_EN],
                                  "length": seg_len})
            is_pre = len(windows) > 0 and (frame_times[a] - origin) < windows[0][0] - origin
            cut_info.append({"episode_index": ep_index,
                             "source": src.name,
                             "frames": [a, b],
                             "source_started_at": origin,
                             "disturbance_windows": windows,
                             "category": meta.get("disturbance_stage", "A" if not windows else "?"),
                             "part": ("full" if not windows else ("pre" if is_pre else "post"))})
            total_frames += seg_len
            ep_index += 1
            print(f"[ok] {src.name} frames {a}-{b} ({seg_len}) -> episode {ep_index-1}")

    # meta 文件
    n_eps = ep_index
    info = {
        "codebase_version": "v2.1",
        "robot_type": "ur5",
        "total_episodes": n_eps,
        "total_frames": int(total_frames),
        "total_tasks": 1,
        "total_videos": n_eps * 2,
        "total_chunks": max(1, math.ceil(n_eps / CHUNKS_SIZE)),
        "chunks_size": CHUNKS_SIZE,
        "fps": args.fps,
        "splits": {"train": f"0:{n_eps}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [13], "names": STATE_NAMES},
            "action": {"dtype": "float32", "shape": [7], "names": ACTION_NAMES},
            "observation.images.global": {
                "dtype": "video", "shape": [720, 1280, 3],
                "names": ["height", "width", "channels"],
                "info": {"video.height": 720, "video.width": 1280,
                         "video.codec": "h264", "video.pix_fmt": "yuv420p",
                         "video.is_depth_map": False, "video.fps": args.fps,
                         "has_audio": False}},
            "observation.images.wrist": {
                "dtype": "video", "shape": [720, 1280, 3],
                "names": ["height", "width", "channels"],
                "info": {"video.height": 720, "video.width": 1280,
                         "video.codec": "h264", "video.pix_fmt": "yuv420p",
                         "video.is_depth_map": False, "video.fps": args.fps,
                         "has_audio": False}},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    with (out / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"task_index": 0, "task": TASK_EN}, ensure_ascii=False) + "\n")
    with (out / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        for row in episodes_meta:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "meta" / "cut_info.json").open("w", encoding="utf-8") as fh:
        json.dump(cut_info, fh, ensure_ascii=False, indent=2)
    with (out / "meta" / "segment_tags.json").open("w", encoding="utf-8") as fh:
        json.dump([{"episode_index": c["episode_index"], "category": c["category"],
                    "part": c["part"], "source": c["source"]} for c in cut_info],
                  fh, ensure_ascii=False, indent=2)

    def feature_stats(arr: np.ndarray) -> dict[str, list[float]]:
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }

    stats = {
        "observation.state": feature_stats(np.concatenate(stats_sum["state"], axis=0)),
        "action": feature_stats(np.concatenate(stats_sum["action"], axis=0)),
    }
    # 图像特征统计不参与训练归一化（state/action 才有），给占位值避免缺失
    for cam in ("global", "wrist"):
        stats[f"observation.images.{cam}"] = {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}
    (out / "meta" / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    print(f"\n完成: episodes={n_eps}, frames={total_frames}, out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
