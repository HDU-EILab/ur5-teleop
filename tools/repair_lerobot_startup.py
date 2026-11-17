"""Trim verified green-camera startup prefixes synchronously in a LeRobot v2.1 dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CAMERAS = ("global", "wrist")


def green_fraction(frame):
    small = cv2.resize(frame, (160, 90)).astype(np.int16)
    blue, green, red = small.transpose(2, 0, 1)
    return float(((green > red + 12) & (green > blue + 12) & (green > 50)).mean())


def startup_prefix(video, *, window=90, threshold=0.05):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    bad = []
    ratios = []
    try:
        for index in range(window):
            ok, frame = capture.read()
            if not ok:
                break
            fraction = green_fraction(frame)
            ratios.append(fraction)
            if fraction > threshold:
                bad.append(index)
    finally:
        capture.release()
    cut = max(bad, default=-1) + 1
    if cut and len(ratios) - cut < 5:
        raise ValueError(
            f"No stable startup tail found within {window} frames: {video}"
        )
    return cut, ratios


def audit_video(video, expected_frames, fps, threshold):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot decode {video}")
    if abs(capture.get(cv2.CAP_PROP_FPS) - fps) > 0.01:
        raise ValueError(f"Unexpected FPS: {video}")
    count = 0
    sums = np.zeros(3, np.float64)
    squares = np.zeros(3, np.float64)
    minimum = np.ones(3)
    maximum = np.zeros(3)
    max_green = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (720, 1280, 3):
                raise ValueError(f"Unexpected resolution: {video}: {frame.shape}")
            fraction = green_fraction(frame)
            max_green = max(max_green, fraction)
            if fraction > threshold:
                raise ValueError(
                    f"Residual green artifact: {video} frame {count} fraction {fraction:.3f}"
                )
            # Deterministic spatial sampling for RGB normalization, across every decoded frame.
            pixels = cv2.resize(frame, (160, 90))[:, :, ::-1].astype(np.float64) / 255
            sums += pixels.mean((0, 1))
            squares += np.square(pixels).mean((0, 1))
            minimum = np.minimum(minimum, pixels.min((0, 1)))
            maximum = np.maximum(maximum, pixels.max((0, 1)))
            count += 1
    finally:
        capture.release()
    if count != expected_frames:
        raise ValueError(
            f"Decode count {count} differs from {expected_frames}: {video}"
        )
    return dict(
        frames=count,
        sum=sums.tolist(),
        square_sum=squares.tolist(),
        min=minimum.tolist(),
        max=maximum.tolist(),
        max_green_fraction=max_green,
    )


def repair(
    source, destination, *, ffmpeg, previous_revision=None, timing=None, workers=2
):
    source, destination = Path(source), Path(destination)
    building = destination.with_name(destination.name + ".building")
    if destination.exists() or building.exists():
        raise FileExistsError("Output or staging dataset already exists")
    info = json.loads((source / "meta/info.json").read_text())
    if info["codebase_version"] != "v2.1":
        raise ValueError("Expected LeRobot v2.1")
    episodes = [
        json.loads(line)
        for line in (source / "meta/episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    cuts = json.loads((source / "meta/cut_info.json").read_text())
    tags = json.loads((source / "meta/segment_tags.json").read_text())
    shutil.copytree(
        source, building, ignore=shutil.ignore_patterns("data", "videos", ".cache")
    )
    rows = []
    total = 0
    encodings = []
    for index in range(info["total_episodes"]):
        frame = pd.read_parquet(source / f"data/chunk-000/episode_{index:06d}.parquet")
        prefixes = {}
        for camera in CAMERAS:
            video = (
                source
                / f"videos/chunk-000/observation.images.{camera}/episode_{index:06d}.mp4"
            )
            prefixes[camera] = startup_prefix(video)[0]
        trim = max(prefixes.values())
        remaining = frame.iloc[trim:].copy().reset_index(drop=True)
        if len(remaining) < round(1.5 * info["fps"]):
            raise ValueError(f"Episode {index} would be shorter than 1.5 seconds")
        remaining["index"] = np.arange(total, total + len(remaining), dtype=np.int64)
        remaining["frame_index"] = np.arange(len(remaining), dtype=np.int64)
        remaining["timestamp"] = (np.arange(len(remaining)) / info["fps"]).astype(
            np.float32
        )
        remaining["episode_index"] = np.full(len(remaining), index, dtype=np.int64)
        path = building / f"data/chunk-000/episode_{index:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        remaining.to_parquet(path, index=False)
        episodes[index]["length"] = len(remaining)
        cuts[index]["frames"][0] += trim
        cuts[index]["startup_trim_frames"] = trim
        rows.append(
            dict(
                episode_index=index,
                input_frames=len(frame),
                trimmed_prefix=trim,
                camera_prefixes=prefixes,
                output_frames=len(remaining),
            )
        )
        for camera in CAMERAS:
            relative = (
                f"videos/chunk-000/observation.images.{camera}/episode_{index:06d}.mp4"
            )
            out = building / relative
            out.parent.mkdir(parents=True, exist_ok=True)
            if trim:
                encodings.append((source / relative, out, trim, len(remaining)))
            else:
                shutil.copy2(source / relative, out)
        total += len(remaining)

    def encode(item):
        video, out, trim, length = item
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-an",
                "-vf",
                f"trim=start_frame={trim},setpts=PTS-STARTPTS",
                "-frames:v",
                str(length),
                "-r",
                str(info["fps"]),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-threads",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out),
            ],
            check=True,
        )
        return str(out.relative_to(building))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name in pool.map(encode, encodings):
            print("encoded", name, flush=True)
    jobs = [
        (
            camera,
            index,
            building
            / f"videos/chunk-000/observation.images.{camera}/episode_{index:06d}.mp4",
        )
        for camera in CAMERAS
        for index in range(info["total_episodes"])
    ]

    def audit(item):
        camera, index, video = item
        result = audit_video(video, episodes[index]["length"], info["fps"], 0.05)
        return camera, index, result

    audits = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for camera, index, result in pool.map(audit, jobs):
            audits.append(dict(camera=camera, episode_index=index, **result))
            print("verified", camera, index, result["frames"], flush=True)
    info.update(
        total_frames=total,
        total_videos=2 * len(episodes),
        total_episodes=len(episodes),
        splits={"train": f"0:{len(episodes)}"},
    )
    for name, data in [
        ("info.json", info),
        ("cut_info.json", cuts),
        ("segment_tags.json", tags),
    ]:
        (building / "meta" / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        )
    (building / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in episodes)
    )
    revision = dict(
        manual_filtered_episodes=len(episodes),
        input_frames=sum(row["input_frames"] for row in rows),
        output_frames=total,
        trimmed_frames=sum(row["trimmed_prefix"] for row in rows),
        episodes=rows,
        previous_hf_revision=previous_revision,
        method="Synchronized prefix trim through last detected startup green/tinted frame in either camera",
        green_detection=dict(
            window_frames=90, fraction_threshold=0.05, channel_excess=12
        ),
        image_stats="All output frames, each spatially sampled to 160x90 RGB",
        encoder="libx264 fast CRF18 yuv420p for trimmed videos; untouched videos copied byte-for-byte",
    )
    if timing is not None:
        timing = json.loads(Path(timing).read_text())
        if [row["trimmed_prefix"] for row in rows] != [row["trim"] for row in timing]:
            raise ValueError("Timing audit does not match selected trims")
        if any(row["invalid_after_trim"] for row in timing):
            raise ValueError(
                "Timing audit includes retained offsets greater than 50 ms"
            )
        alignment = dict(
            fps=info["fps"],
            max_source_offset_s=0.05,
            episodes=timing,
            retained_max_source_offset_ms=max(row["max_offset_ms"] for row in timing),
            retained_max_camera_skew_ms=max(row["max_skew_ms"] for row in timing),
        )
        (building / "meta/alignment.json").write_text(
            json.dumps(alignment, indent=2) + "\n"
        )
    (building / "meta/quality_revision.json").write_text(
        json.dumps(revision, ensure_ascii=False, indent=2) + "\n"
    )
    validation = dict(
        total_decoded_frames=sum(row["frames"] for row in audits),
        videos=len(audits),
        max_green_fraction=max(row["max_green_fraction"] for row in audits),
        all_frame_counts_match=True,
        statistics_rebuilt=True,
        video_audits=audits,
    )
    (building / "meta/validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    from ur5_teleop.lerobot_recording import write_batch_statistics

    write_batch_statistics(building, image_audits=audits)
    building.replace(destination)
    return revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ffmpeg", default=os.environ.get("FFMPEG", shutil.which("ffmpeg"))
    )
    parser.add_argument("--previous-revision")
    parser.add_argument("--timing-audit", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not args.ffmpeg:
        parser.error("ffmpeg is required")
    print(
        json.dumps(
            repair(
                args.source,
                args.output,
                ffmpeg=args.ffmpeg,
                previous_revision=args.previous_revision,
                timing=args.timing_audit,
                workers=args.workers,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
