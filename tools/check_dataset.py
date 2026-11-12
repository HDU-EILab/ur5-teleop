"""校验导出的 LeRobot v2.1 数据集：episode 数、视频帧数==parquet 行数、无 NaN、fps=30。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/lerobot_ur5_pickplace")
    args = ap.parse_args()
    root = Path(args.root)
    m = json.loads((root / "meta" / "info.json").read_text())
    bad: list[tuple] = []
    total = 0
    lens: list[int] = []
    for ep in range(m["total_episodes"]):
        df = pd.read_parquet(root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
        total += len(df)
        lens.append(len(df))
        for cam in ("global", "wrist"):
            path = root / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{ep:06d}.mp4"
            cap = cv2.VideoCapture(str(path))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if n != len(df) or abs(fps - 30.0) > 0.1:
                bad.append((ep, cam, n, len(df), round(fps, 2)))
        s = np.stack(df["observation.state"].values)
        a = np.stack(df["action"].values)
        if s.shape[1] != 13 or a.shape[1] != 7:
            bad.append((ep, "shape", s.shape, a.shape))
        if df.isna().any().any():
            bad.append((ep, "nan"))
    print(f"episodes={m['total_episodes']} frames={total} fps={m['fps']}")
    print(f"episode 长度 min/median/max = {min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}")
    print("异常:", bad if bad else "无 ✅")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
