"""把自研 episode 格式导出为 LeRobot v2.1 数据集（Pi0.5 训练用）。

设计：
- 在双相机和机器人共同时间范围建立均匀时间轴，各源独立取最近样本。
- 各源偏差默认限制为 1.5/FPS 秒，剔除采样间隙并切成连续片段。
- state(13) = 实测 TCP 6 + 关节 6 + 夹爪实测位置(0-1)
- action(7) = 目标 TCP 6 + 夹爪命令(0-1，从 trigger 按 binary 模式推导)
- B 类（含人工扰动窗口）**截断**：把扰动窗口（左右各留 margin）剪掉，
  窗口前后切成独立 segment；过短的 segment 丢弃。A 类整条保留。
- 视频按对齐后的帧索引重采样并重编 h264（30fps CFR）。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ur5_teleop.frames import policy_rotvec
from ur5_teleop.lerobot_recording import validate_camera_recording, write_batch_statistics

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


def nearest_indices(times, samples):
    times = np.asarray(times, dtype=np.float64)
    if len(times) == 0 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Source timestamps must be finite and strictly increasing")
    right = np.clip(np.searchsorted(times, samples), 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(np.abs(times[left] - samples) <= np.abs(times[right] - samples), left, right)


def synchronized_frames(global_times, wrist_times, robot_times, fps):
    sources = [np.asarray(times, dtype=np.float64) for times in (global_times, wrist_times, robot_times)]
    for times in sources:
        nearest_indices(times, [])
    start = max(times[0] for times in sources)
    end = min(times[-1] for times in sources)
    if end < start:
        raise ValueError("Camera and robot recordings have no common time range")
    timeline = start + np.arange(int(np.floor((end - start) * fps)) + 1) / fps
    indices = [nearest_indices(times, timeline) for times in sources]
    return timeline, indices


def gripper_command(trigger: float) -> float:
    """binary 模式：扳机过半 → 闭合(0)，否则全开(1000)；归一化到 0-1。"""
    trigger = max(0.0, min(1.0, float(trigger)))
    return 0.0 if trigger >= 0.5 else 1.0


def segment_ranges(n_frames: int, windows: list[list[float]],
                   windows_origin: float, frame_times: list[float],
                   margin_s: float, min_frames: int,
                   valid_samples=None) -> list[tuple[int, int]]:
    """按有效采样掩码及扰动窗口返回连续保留帧的闭区间。"""
    keep = [True] * n_frames if valid_samples is None else list(valid_samples)
    if len(keep) != n_frames:
        raise ValueError("Validity mask must match the aligned timeline")
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


def cut_video(src: Path, dst: Path, indices, fps: float) -> tuple[int, int]:
    """Decode monotonic source indices, including repeated frames, into a CFR video."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "pipe:0",
           "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24",
           "-threads", "2", "-pix_fmt", "yuv420p", str(dst)]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0]))
        decoded = int(indices[0]) - 1
        frame = None
        for index in indices:
            while decoded < int(index):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode {src} frame {decoded + 1}")
                decoded += 1
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        error = process.stderr.read().decode()
        if process.wait() != 0:
            raise RuntimeError(f"Video encode failed: {error}")
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        cap.release()
        process.stderr.close()
    return height, width


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-dir", default="data/episodes")
    ap.add_argument("--out", default="data/lerobot_ur5_pickplace")
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--cut-margin", type=float, default=0.25)
    ap.add_argument("--min-segment-s", type=float, default=1.5)
    ap.add_argument("--max-source-offset-s", type=float,
                    help="Maximum nearest-sample offset for each camera and robot source, default 1.5/fps")
    ap.add_argument("--exclude-source", action="append", default=[])
    ap.add_argument("--include-source", action="append", default=[])
    args = ap.parse_args(argv)

    root = Path(args.episodes_dir)
    destination = Path(args.out)
    out = destination.with_name(destination.name + '.building')
    if destination.exists() or out.exists():
        raise FileExistsError(f"Output or staging directory already exists: {destination}")
    if not np.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("fps must be positive and finite")
    max_offset = args.max_source_offset_s if args.max_source_offset_s is not None else 1.5 / args.fps
    if not np.isfinite(max_offset) or max_offset <= 0:
        raise ValueError("max-source-offset-s must be positive and finite")
    (out / "meta").mkdir(parents=True)
    (out / "data" / "chunk-000").mkdir(parents=True)
    for cam in ("global", "wrist"):
        (out / "videos" / "chunk-000" / f"observation.images.{cam}").mkdir(parents=True)

    episode_dirs = sorted([d for d in root.iterdir() if d.is_dir() and (d / "meta.json").is_file()
                           and d.name not in args.exclude_source
                           and (not args.include_source or d.name in args.include_source)])
    missing = set(args.include_source) - {d.name for d in episode_dirs}
    if missing:
        raise FileNotFoundError(f"Requested recordings missing: {sorted(missing)}")
    if not np.isfinite(args.cut_margin) or args.cut_margin < 0:
        raise ValueError("cut-margin must be finite and non-negative")
    if not np.isfinite(args.min_segment_s) or args.min_segment_s <= 0:
        raise ValueError("min-segment-s must be positive and finite")
    min_frames = math.ceil(args.min_segment_s * args.fps)
    tasks = []
    total_frames = 0
    episodes_meta: list[dict[str, Any]] = []
    cut_info: list[dict[str, Any]] = []
    ep_index = 0
    alignment = []
    video_shapes = {}

    for src in episode_dirs:
        meta_path = src / "meta.json"
        robot_path = src / "robot.csv"
        g_ts_path = src / "videos" / "global_timestamps.csv"
        w_ts_path = src / "videos" / "wrist_timestamps.csv"
        if not all(p.exists() for p in (meta_path, robot_path, g_ts_path, w_ts_path)):
            raise FileNotFoundError(f"Missing source metadata, state or camera timestamps: {src}")
        meta = json.loads(meta_path.read_text())
        validate_camera_recording(meta)
        rows = read_csv_rows(robot_path)
        swapped = src.name in CAMERA_SWAP_SOURCES
        camera_sources = {'global': 'wrist' if swapped else 'global', 'wrist': 'global' if swapped else 'wrist'}
        global_times = read_timestamps(src / 'videos' / f'{camera_sources["global"]}_timestamps.csv')
        wrist_times = read_timestamps(src / 'videos' / f'{camera_sources["wrist"]}_timestamps.csv')
        robot_times = [float(r['t_mono']) for r in rows]
        frame_times, (global_indices, wrist_indices, robot_indices) = synchronized_frames(
            global_times, wrist_times, robot_times, args.fps)
        n_frames = len(frame_times)
        offsets = np.stack([np.abs(np.asarray(times)[indices] - frame_times)
                            for times, indices in ((global_times, global_indices),
                                                   (wrist_times, wrist_indices),
                                                   (robot_times, robot_indices))])
        valid_samples = np.all(offsets <= max_offset, axis=0)
        camera_skew = np.abs(np.asarray(global_times)[global_indices] - np.asarray(wrist_times)[wrist_indices]) * 1000
        alignment.append({'source': src.name, 'frames': n_frames,
                          'median_camera_skew_ms': float(np.median(camera_skew)),
                          'max_camera_skew_ms': float(np.max(np.abs(np.asarray(global_times)[global_indices] - np.asarray(wrist_times)[wrist_indices])) * 1000),
                          'max_robot_offset_ms': float(np.max(offsets[2]) * 1000),
                          'invalid_timing_frames': int(np.count_nonzero(~valid_samples))})
        if n_frames < min_frames:
            print(f"[skip] {src.name}: 帧数太少 {n_frames}")
            continue
        windows = meta.get("disturbance_windows") or []
        origin = meta.get("started_at", 0.0)
        segments = segment_ranges(n_frames, windows, origin, frame_times,
                                  args.cut_margin, min_frames, valid_samples)
        retained = np.concatenate([np.arange(a, b + 1) for a, b in segments]) if segments else np.array([], dtype=int)
        alignment[-1]['retained_frames'] = int(len(retained))
        if len(retained):
            alignment[-1]['retained_max_camera_skew_ms'] = float(camera_skew[retained].max())
            alignment[-1]['retained_max_source_offset_ms'] = float(offsets[:, retained].max() * 1000)
        if not segments:
            print(f"[skip] {src.name}: 截断后无有效 segment")
            continue
        task = meta.get("task", TASK_EN)
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"Invalid task text in {src}")
        if task not in tasks:
            tasks.append(task)
        task_index = tasks.index(task)
        for a, b in segments:
            seg_len = b - a + 1
            states: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            for fi in range(a, b + 1):
                r = rows[robot_indices[fi]]
                state = np.array([float(r[k]) for k in
                                  ("robot_x", "robot_y", "robot_z", "robot_rx", "robot_ry",
                                   "robot_rz", "q1", "q2", "q3", "q4", "q5", "q6")],
                                 dtype=np.float32)
                grip = float(r["gripper_position"]) / 1000.0
                state = np.concatenate([state, [grip]]).astype(np.float32)
                state[3:6] = policy_rotvec(state[3:6])
                action6 = np.array([float(r[k]) for k in
                                    ("target_x", "target_y", "target_z", "target_rx",
                                     "target_ry", "target_rz")], dtype=np.float32)
                action6[3:6] = policy_rotvec(action6[3:6])
                grip_cmd = gripper_command(float(r["trigger_right"] or 0.0))
                action = np.concatenate([action6, [grip_cmd]]).astype(np.float32)
                if not np.isfinite(state).all() or not np.isfinite(action).all():
                    raise ValueError(f"Non-finite state/action in {src} at aligned frame {fi}")
                states.append(state)
                actions.append(action)

            for cam, indices in (('global', global_indices), ('wrist', wrist_indices)):
                src_cam = camera_sources[cam]
                src_video = src / "videos" / f"{src_cam}.mp4"
                dst_video = (out / "videos" / "chunk-000" /
                             f"observation.images.{cam}" / f"episode_{ep_index:06d}.mp4")
                shape = cut_video(src_video, dst_video, indices[a:b + 1], args.fps)
                if cam in video_shapes and video_shapes[cam] != shape:
                    raise ValueError(f"Video dimensions changed for camera {cam}: {shape}")
                video_shapes[cam] = shape

            # parquet
            df = pd.DataFrame({
                "index": np.arange(total_frames, total_frames + seg_len, dtype=np.int64),
                "episode_index": np.full(seg_len, ep_index, dtype=np.int64),
                "frame_index": np.arange(seg_len, dtype=np.int64),
                "timestamp": (np.arange(seg_len, dtype=np.float32) / np.float32(args.fps)),
                "task_index": np.full(seg_len, task_index, dtype=np.int64),
                "observation.state": [s for s in states],
                "action": [a_ for a_ in actions],
            })
            df.to_parquet(out / "data" / "chunk-000" / f"episode_{ep_index:06d}.parquet",
                          index=False)

            episodes_meta.append({"episode_index": ep_index,
                                  "tasks": [task],
                                  "length": seg_len})
            is_pre = len(windows) > 0 and (frame_times[a] - origin) < windows[0][0] - origin
            cut_info.append({"episode_index": ep_index,
                             "source": src.name,
                             "frames": [a, b],
                             "source_started_at": origin,
                             "disturbance_windows": windows,
                             "category": meta.get("disturbance_stage") or ("B" if windows else "A"),
                             "part": ("full" if not windows else ("pre" if is_pre else "post"))})
            total_frames += seg_len
            ep_index += 1
            print(f"[ok] {src.name} frames {a}-{b} ({seg_len}) -> episode {ep_index-1}")

    # meta 文件
    n_eps = ep_index
    if n_eps == 0:
        raise ValueError("No exportable episodes")
    info = {
        "codebase_version": "v2.1",
        "robot_type": "ur5",
        "total_episodes": n_eps,
        "total_frames": int(total_frames),
        "total_tasks": len(tasks),
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
                "dtype": "video", "shape": [*video_shapes['global'], 3],
                "names": ["height", "width", "channels"],
                "info": {"video.height": video_shapes['global'][0], "video.width": video_shapes['global'][1],
                         "video.codec": "h264", "video.pix_fmt": "yuv420p",
                         "video.is_depth_map": False, "video.fps": args.fps,
                         "has_audio": False}},
            "observation.images.wrist": {
                "dtype": "video", "shape": [*video_shapes['wrist'], 3],
                "names": ["height", "width", "channels"],
                "info": {"video.height": video_shapes['wrist'][0], "video.width": video_shapes['wrist'][1],
                         "video.codec": "h264", "video.pix_fmt": "yuv420p",
                         "video.is_depth_map": False, "video.fps": args.fps,
                         "has_audio": False}},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    with (out / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as fh:
        for task_index, task in enumerate(tasks):
            fh.write(json.dumps({"task_index": task_index, "task": task}, ensure_ascii=False) + "\n")
    with (out / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        for row in episodes_meta:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "meta" / "cut_info.json").open("w", encoding="utf-8") as fh:
        json.dump(cut_info, fh, ensure_ascii=False, indent=2)
    with (out / "meta" / "segment_tags.json").open("w", encoding="utf-8") as fh:
        json.dump([{"episode_index": c["episode_index"], "category": c["category"],
                    "part": c["part"], "source": c["source"]} for c in cut_info],
                  fh, ensure_ascii=False, indent=2)

    write_batch_statistics(out)
    (out / 'meta/alignment.json').write_text(json.dumps({'sources': alignment, 'excluded_sources': args.exclude_source,
        'rotation_convention': 'equivalent positive-Y axis-angle', 'timeline': 'uniform common camera/robot interval',
        'max_source_offset_s': max_offset, 'cut_margin_s': args.cut_margin,
        'min_segment_s': args.min_segment_s}, indent=2))
    out.rename(destination)
    print(f"\n完成: episodes={n_eps}, frames={total_frames}, out={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
