"""Background LeRobot finalization with durable jobs and atomic dataset generations."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_job(path):
    job = read_json(path)
    if not isinstance(job, dict) or job.get("status") not in {
        "queued",
        "processing",
        "complete",
        "failed",
        "skipped",
        "discarded",
    }:
        raise ValueError(f"Invalid recording job object or status: {path}")
    return job


def fail_job(path, error):
    """Preserve malformed job bytes before replacing them with a failure record."""
    path = Path(path)
    try:
        job = read_job(path)
    except (ValueError, TypeError):
        backup = path.with_suffix(f".corrupt-{uuid.uuid4().hex}")
        shutil.copy2(path, backup)
        job = {"corrupt_job_backup": str(backup)}
    job.update(
        status="failed",
        error=f"{type(error).__name__}: {error}",
        failed_at=time.time(),
        log=str(path.with_suffix(".log")),
    )
    write_json(path, job)
    return job


def dataset_state(dataset):
    dataset = Path(dataset).expanduser().absolute()
    return dataset.parent / ("." + dataset.name + ".recording")


@contextmanager
def dataset_lock(dataset):
    state = dataset_state(dataset)
    state.mkdir(parents=True, exist_ok=True)
    with (state / "append.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield state


def json_lines(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def feature_stats(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite LeRobot feature")
    return dict(
        mean=values.mean(0).tolist(),
        std=values.std(0).tolist(),
        min=values.min(0).tolist(),
        max=values.max(0).tolist(),
        q01=np.quantile(values, 0.01, axis=0).tolist(),
        q99=np.quantile(values, 0.99, axis=0).tolist(),
        count=[len(values)],
    )


def image_stats(moments):
    import numpy as np

    mean = np.asarray(moments["sum"]) / moments["count"]
    variance = np.asarray(moments["square_sum"]) / moments["count"] - mean**2
    return {
        "mean": mean.reshape(3, 1, 1).tolist(),
        "std": np.sqrt(np.maximum(variance, 0)).reshape(3, 1, 1).tolist(),
        "min": np.asarray(moments["min"]).reshape(3, 1, 1).tolist(),
        "max": np.asarray(moments["max"]).reshape(3, 1, 1).tolist(),
        "count": [moments["count"]],
    }


def image_moments(batch, camera):
    import cv2
    import numpy as np

    episodes = {}
    for path in sorted((batch / "videos").glob(f"*/observation.images.{camera}/*.mp4")):
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot decode {path}")
        sums, squares = np.zeros(3), np.zeros(3)
        minimum, maximum = np.ones(3), np.zeros(3)
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        decoded = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                pixels = (
                    cv2.resize(frame, (160, 90))[:, :, ::-1].astype(np.float64) / 255
                )
                sums += pixels.mean((0, 1))
                squares += np.square(pixels).mean((0, 1))
                minimum = np.minimum(minimum, pixels.min((0, 1)))
                maximum = np.maximum(maximum, pixels.max((0, 1)))
                decoded += 1
        finally:
            capture.release()
        if decoded != expected or decoded == 0:
            raise ValueError(f"Video decode count mismatch or empty video: {path}")
        episode = int(path.stem.removeprefix("episode_"))
        episodes[episode] = {
            "count": decoded,
            "sum": sums.tolist(),
            "square_sum": squares.tolist(),
            "min": minimum.tolist(),
            "max": maximum.tolist(),
        }
    if not episodes:
        raise ValueError("No decoded images")
    values = list(episodes.values())
    return {
        "count": sum(row["count"] for row in values),
        "sum": np.sum([row["sum"] for row in values], axis=0).tolist(),
        "square_sum": np.sum([row["square_sum"] for row in values], axis=0).tolist(),
        "min": np.min([row["min"] for row in values], axis=0).tolist(),
        "max": np.max([row["max"] for row in values], axis=0).tolist(),
        "episodes": episodes,
    }


def write_batch_statistics(root, image_audits=None):
    """Build v2.1 statistics from the exported rows and the decoded RGB frames."""
    import numpy as np
    import pandas as pd

    root = Path(root)
    info = read_json(root / "meta/info.json")
    episode_stats = {}
    arrays = {}
    for episode in range(info["total_episodes"]):
        path = root / info["data_path"].format(
            episode_chunk=episode // info["chunks_size"], episode_index=episode
        )
        table = pd.read_parquet(path)
        episode_stats[episode] = {}
        for key in table.columns:
            values = (
                np.stack(table[key])
                if key in {"observation.state", "action"}
                else table[key].to_numpy().reshape(-1, 1)
            )
            episode_stats[episode][key] = feature_stats(values)
            arrays.setdefault(key, []).append(values)
    stats = {
        key: feature_stats(np.concatenate(values)) for key, values in arrays.items()
    }
    moments = {}
    for camera in ["global", "wrist"]:
        if image_audits is None:
            current = image_moments(root, camera)
        else:
            episodes = {
                row["episode_index"]: {**row, "count": row["frames"]}
                for row in image_audits
                if row["camera"] == camera
            }
            values = list(episodes.values())
            current = {
                "count": sum(row["count"] for row in values),
                "sum": np.sum([row["sum"] for row in values], axis=0).tolist(),
                "square_sum": np.sum(
                    [row["square_sum"] for row in values], axis=0
                ).tolist(),
                "min": np.min([row["min"] for row in values], axis=0).tolist(),
                "max": np.max([row["max"] for row in values], axis=0).tolist(),
                "episodes": episodes,
            }
        for episode, values in current.pop("episodes").items():
            if values["count"] != episode_stats[episode]["frame_index"]["count"][0]:
                raise ValueError(
                    f"Video and table frame counts differ in episode {episode}"
                )
            episode_stats[episode][f"observation.images.{camera}"] = image_stats(values)
        if current["count"] != info["total_frames"]:
            raise ValueError("Decoded camera frame count differs from dataset metadata")
        moments[camera] = current
        stats[f"observation.images.{camera}"] = image_stats(current)
    info["features"]["index"] = {"dtype": "int64", "shape": [1], "names": None}
    write_json(root / "meta/info.json", info)
    write_json(root / "meta/stats.json", stats)
    write_json(root / "meta/image_moments.json", moments)
    (root / "meta/episodes_stats.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": episode, "stats": values}, ensure_ascii=False)
            + "\n"
            for episode, values in episode_stats.items()
        )
    )


def append_batch(batch, dataset, source_name, metadata_sha256):
    """Called while holding the dataset lock. The visible generation changes once."""
    import numpy as np
    import pandas as pd

    batch, dataset = Path(batch), Path(dataset).absolute()
    state = dataset_state(dataset)
    generations = state / "generations"
    generations.mkdir(exist_ok=True)
    current = None
    if dataset.is_symlink():
        current = dataset.resolve(strict=True)
        if not current.is_relative_to(generations.resolve()):
            raise ValueError("Dataset link points outside its managed generations")
    elif dataset.exists():
        raise FileExistsError(
            f"Choose a new recording dataset path; this directory already exists: {dataset}"
        )
    old_sources = read_json(current / "meta/recording_sources.json") if current else {}
    if source_name in old_sources:
        receipt = old_sources[source_name]
        if receipt["metadata_sha256"] != metadata_sha256:
            raise ValueError(
                f"Previously exported source metadata changed: {source_name}"
            )
        return receipt
    new_info = read_json(batch / "meta/info.json")
    old_info = read_json(current / "meta/info.json") if current else None
    if old_info and (
        old_info["features"] != new_info["features"]
        or old_info["fps"] != new_info["fps"]
    ):
        raise ValueError(
            "Recording feature schema or FPS differs from the existing dataset"
        )
    old_count = old_info["total_episodes"] if old_info else 0
    old_frames = old_info["total_frames"] if old_info else 0
    tasks = json_lines(current / "meta/tasks.jsonl") if current else []
    task_lookup = {row["task"]: row["task_index"] for row in tasks}
    batch_tasks = json_lines(batch / "meta/tasks.jsonl")
    task_mapping = {}
    for row in batch_tasks:
        if row["task"] not in task_lookup:
            task_lookup[row["task"]] = len(tasks)
            tasks.append(dict(task_index=len(tasks), task=row["task"]))
        task_mapping[row["task_index"]] = task_lookup[row["task"]]
    # A failed or killed build never changes the currently visible dataset.
    with tempfile.TemporaryDirectory(prefix=".building-", dir=generations) as temporary:
        build = Path(temporary) / "dataset"
        build.mkdir()
        (build / "meta").mkdir()
        for name in ["data", "videos"]:
            if current:
                shutil.copytree(current / name, build / name, copy_function=os.link)
            else:
                (build / name).mkdir()
        offset = old_frames
        episode_stats = (
            json_lines(current / "meta/episodes_stats.jsonl") if current else []
        )
        added_stats = {}
        batch_stats = {
            row["episode_index"]: row["stats"]
            for row in json_lines(batch / "meta/episodes_stats.jsonl")
        }
        for episode in range(new_info["total_episodes"]):
            index = old_count + episode
            chunk = index // new_info["chunks_size"]
            source = batch / new_info["data_path"].format(
                episode_chunk=episode // new_info["chunks_size"], episode_index=episode
            )
            frame = pd.read_parquet(source)
            frame["episode_index"] = np.full(len(frame), index, dtype=np.int64)
            frame["index"] = np.arange(offset, offset + len(frame), dtype=np.int64)
            frame["task_index"] = frame["task_index"].map(task_mapping).astype(np.int64)
            target = build / new_info["data_path"].format(
                episode_chunk=chunk, episode_index=index
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(target, index=False)
            added_stats[index] = {
                **batch_stats[episode],
                **{
                    key: feature_stats(
                        np.stack(frame[key])
                        if key in {"observation.state", "action"}
                        else frame[key].to_numpy().reshape(-1, 1)
                    )
                    for key in frame.columns
                },
            }
            for camera in ["global", "wrist"]:
                key = f"observation.images.{camera}"
                source = batch / new_info["video_path"].format(
                    episode_chunk=episode // new_info["chunks_size"],
                    episode_index=episode,
                    video_key=key,
                )
                target = build / new_info["video_path"].format(
                    episode_chunk=chunk, episode_index=index, video_key=key
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, target)
            offset += len(frame)
        episodes = json_lines(current / "meta/episodes.jsonl") if current else []
        episodes += [
            {**row, "episode_index": row["episode_index"] + old_count}
            for row in json_lines(batch / "meta/episodes.jsonl")
        ]
        (build / "meta/episodes.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in episodes)
        )
        (build / "meta/tasks.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in tasks)
        )
        for name in ["cut_info.json", "segment_tags.json"]:
            previous = read_json(current / "meta" / name) if current else []
            added = [
                {**row, "episode_index": row["episode_index"] + old_count}
                for row in read_json(batch / "meta" / name)
            ]
            write_json(build / "meta" / name, previous + added)
        alignment = read_json(batch / "meta/alignment.json")
        if current:
            alignment["sources"] = (
                read_json(current / "meta/alignment.json")["sources"]
                + alignment["sources"]
            )
        write_json(build / "meta/alignment.json", alignment)
        arrays = {
            key: []
            for key in [
                "observation.state",
                "action",
                "index",
                "episode_index",
                "frame_index",
                "timestamp",
                "task_index",
            ]
        }
        for path in sorted((build / "data").glob("*/episode_*.parquet")):
            frame = pd.read_parquet(path)
            for key in arrays:
                arrays[key].append(
                    np.stack(frame[key])
                    if key in {"observation.state", "action"}
                    else frame[key].to_numpy().reshape(-1, 1)
                )
        stats = {
            key: feature_stats(np.concatenate(values)) for key, values in arrays.items()
        }
        moments = {}
        old_moments = read_json(current / "meta/image_moments.json") if current else {}
        for camera in ["global", "wrist"]:
            fresh = read_json(batch / "meta/image_moments.json")[camera]
            if fresh["count"] != new_info["total_frames"]:
                raise ValueError("Image count differs from exported rows")
            if current:
                prior = old_moments[camera]
                fresh = dict(
                    count=prior["count"] + fresh["count"],
                    sum=(np.asarray(prior["sum"]) + fresh["sum"]).tolist(),
                    square_sum=(
                        np.asarray(prior["square_sum"]) + fresh["square_sum"]
                    ).tolist(),
                    min=np.minimum(prior["min"], fresh["min"]).tolist(),
                    max=np.maximum(prior["max"], fresh["max"]).tolist(),
                )
            moments[camera] = fresh
            stats[f"observation.images.{camera}"] = image_stats(fresh)
        episode_stats += [
            {"episode_index": episode, "stats": values}
            for episode, values in added_stats.items()
        ]
        (build / "meta/episodes_stats.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in episode_stats)
        )
        write_json(build / "meta/stats.json", stats)
        write_json(build / "meta/image_moments.json", moments)
        total_episodes = len(episodes)
        new_info.update(
            total_episodes=total_episodes,
            total_frames=offset,
            total_tasks=len(tasks),
            total_videos=2 * total_episodes,
            total_chunks=max(1, math.ceil(total_episodes / new_info["chunks_size"])),
            splits={"train": f"0:{total_episodes}"},
        )
        write_json(build / "meta/info.json", new_info)
        receipt = dict(
            metadata_sha256=metadata_sha256,
            episodes=list(range(old_count, total_episodes)),
            frames=offset - old_frames,
            completed_at=time.time(),
        )
        write_json(
            build / "meta/recording_sources.json", {**old_sources, source_name: receipt}
        )
        generation = generations / f"{total_episodes:08d}-{uuid.uuid4().hex}"
        build.replace(generation)
    link = dataset.with_name("." + dataset.name + "." + uuid.uuid4().hex + ".link")
    link.symlink_to(
        os.path.relpath(generation, dataset.parent), target_is_directory=True
    )
    link.replace(dataset)
    return receipt


def validate_camera_recording(metadata):
    if metadata.get("camera_valid") is False:
        raise ValueError("Camera recording is invalid; inspect raw camera metadata")
    errors = metadata.get("camera_stats", {}).get("errors")
    if errors:
        raise ValueError(f"Camera recording failed: {errors}")


def process_job(path):
    path = Path(path)
    job = read_job(path)
    with dataset_lock(job["dataset"]) as state:
        from .recording_retraction import recover_withdrawals

        recover_withdrawals(job["dataset"])
        job = read_job(path)
        if job["status"] in {"complete", "discarded"}:
            return job
        source = Path(job["source"])
        metadata_path = source / "meta.json"
        if (
            hashlib.sha256(metadata_path.read_bytes()).hexdigest()
            != job["metadata_sha256"]
        ):
            raise ValueError("Source metadata changed after enqueue")
        metadata = read_json(metadata_path)
        if metadata.get("success") is not True:
            raise ValueError(
                "Incomplete or failed raw recording cannot be published as a demonstration"
            )
        validate_camera_recording(metadata)
        job.update(status="processing", started_at=time.time(), error=None)
        write_json(path, job)
        current = Path(job["dataset"])
        if current.is_symlink():
            sources = read_json(current.resolve() / "meta/recording_sources.json")
            if source.name in sources:
                receipt = sources[source.name]
                if receipt["metadata_sha256"] != job["metadata_sha256"]:
                    raise ValueError(
                        "Exported recording metadata differs from this job"
                    )
                job.update(status="complete", receipt=receipt, completed_at=time.time())
                write_json(path, job)
                return job
        generations = state / "generations"
        generations.mkdir(exist_ok=True)
        for stale in generations.glob(".building-*"):
            shutil.rmtree(stale)
        with tempfile.TemporaryDirectory(prefix="export-", dir=state) as temporary:
            output = Path(temporary) / "batch"
            command = [
                sys.executable,
                str(ROOT / "tools/export_lerobot.py"),
                "--episodes-dir",
                str(source.parent),
                "--include-source",
                source.name,
                "--out",
                str(output),
                "--fps",
                str(job["fps"]),
                "--cut-margin",
                str(job["cut_margin_s"]),
                "--min-segment-s",
                str(job["min_segment_s"]),
            ]
            subprocess.run(command, check=True)
            receipt = append_batch(
                output, job["dataset"], source.name, job["metadata_sha256"]
            )
        job.update(status="complete", receipt=receipt, completed_at=time.time())
        write_json(path, job)
        return job


class LeRobotRecordingWriter:
    """Queue saved raw recordings; all video/Parquet work runs in a low-priority process."""

    def __init__(
        self, dataset, *, fps=30.0, cut_margin_s=0.25, min_segment_s=1.5, printer=print
    ):
        self.dataset = Path(dataset).expanduser().absolute()
        self.state = dataset_state(self.dataset)
        (self.state / "jobs").mkdir(parents=True, exist_ok=True)
        self.fps, self.cut_margin_s, self.min_segment_s = (
            fps,
            cut_margin_s,
            min_segment_s,
        )
        if (
            not all(
                math.isfinite(value) for value in (fps, cut_margin_s, min_segment_s)
            )
            or fps <= 0
            or cut_margin_s < 0
            or min_segment_s <= 0
        ):
            raise ValueError("Invalid recording FPS or segmentation parameters")
        self.printer = printer
        self._enqueue_lock = threading.Lock()
        self._closed = False
        self._worker_errors = []
        self.queue = queue.Queue()
        from .recording_retraction import recover_withdrawals

        with dataset_lock(self.dataset):
            recover_withdrawals(self.dataset)
        for path in sorted((self.state / "jobs").glob("*.json")):
            try:
                job = read_job(path)
            except (ValueError, TypeError) as error:
                fail_job(path, error)
                self.printer(f"[lerobot] 损坏的任务已记录为失败：{path}；{error}")
                continue
            if job["status"] in {"queued", "processing"}:
                self.queue.put(path)
        self.thread = threading.Thread(
            target=self._worker, name="lerobot-finalization", daemon=True
        )
        self.thread.start()

    def recover(self, raw_root):
        for metadata_path in sorted(Path(raw_root).glob("*/meta.json")):
            request = read_json(metadata_path).get("lerobot_export", {})
            if request.get("dataset") == str(self.dataset):
                self.enqueue(metadata_path.parent)

    def enqueue(self, source):
        with self._enqueue_lock:
            if self._closed:
                raise RuntimeError("Recording writer is closed")
            # Keep enqueue fast while another process is doing a long export.
            with (self.state / "jobs.lock").open("a") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                return self._enqueue(source)

    def _enqueue(self, source):
        source = Path(source).resolve()
        identifier = hashlib.sha256(str(source).encode()).hexdigest()[:24]
        path = self.state / "jobs" / (identifier + ".json")
        if path.exists() and read_job(path)["status"] == "discarded":
            self.printer(f"[lerobot] 已撤回的录制跳过整理：{source.name}")
            return path
        metadata_path = source / "meta.json"
        metadata = read_json(metadata_path)
        checksum = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        if path.exists():
            previous = read_json(path)
            if previous["metadata_sha256"] != checksum:
                raise ValueError("A queued source metadata file changed")
            return path
        options = metadata.get("lerobot_export", {})
        job = dict(
            source=str(source),
            dataset=str(self.dataset),
            metadata_sha256=checksum,
            fps=options.get("fps", self.fps),
            cut_margin_s=options.get("cut_margin_s", self.cut_margin_s),
            min_segment_s=options.get("min_segment_s", self.min_segment_s),
            status="queued" if metadata["success"] else "skipped",
            created_at=time.time(),
            error=None,
        )
        write_json(path, job)
        if job["status"] == "queued":
            self.queue.put(path)
            self.printer(f"[lerobot] 已排队自动整理：{source.name}")
        else:
            self.printer(f"[lerobot] 未完成录制已保留在原始目录：{source.name}")
        return path

    def _worker(self):
        while True:
            path = self.queue.get()
            try:
                if path is None:
                    return
                log = path.with_suffix(".log")
                environment = dict(
                    os.environ,
                    PYTHONPATH=str(ROOT)
                    + os.pathsep
                    + os.environ.get("PYTHONPATH", ""),
                    OMP_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                )
                with log.open("a") as stream:
                    result = subprocess.run(
                        [
                            "nice",
                            "-n",
                            "10",
                            sys.executable,
                            "-m",
                            "ur5_teleop.lerobot_recording",
                            "--job",
                            str(path),
                        ],
                        check=False,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        env=environment,
                    )
                job = read_job(path)
                if job["status"] == "discarded":
                    self.printer(f"[lerobot] 已撤回：{job['source']}")
                    continue
                if result.returncode != 0 or job["status"] != "complete":
                    if job["status"] != "failed":
                        job.update(
                            status="failed",
                            error=f"Finalizer exited {result.returncode}",
                            log=str(log),
                        )
                        write_json(path, job)
                    self.printer(f"[lerobot] 整理失败：{job['error']}；日志 {log}")
                else:
                    self.printer(
                        f"[lerobot] 保存完成：episodes={job['receipt']['episodes']} → {self.dataset}"
                    )
            except Exception as error:  # noqa: BLE001 - isolate and persist each queued job failure
                try:
                    fail_job(path, error)
                    with path.with_suffix(".log").open("a") as stream:
                        traceback.print_exc(file=stream)
                except Exception as persistence_error:  # noqa: BLE001 - surface storage failures at the worker boundary
                    self._worker_errors.append(
                        f"{path}: {error}; failure persistence: {persistence_error}"
                    )
                self.printer(f"[lerobot] 后台任务失败：{path}；{error}")
            finally:
                self.queue.task_done()

    def close(self):
        with self._enqueue_lock:
            if not self._closed:
                self._closed = True
                self.queue.put(None)
        # FIFO drains accepted jobs before the sentinel. Joining the worker also
        # returns if it died unexpectedly, unlike queue.join() with pending tasks.
        self.thread.join()
        if self.queue.unfinished_tasks:
            raise RuntimeError(
                f"LeRobot worker exited with {self.queue.unfinished_tasks} unfinished queue entries; inspect {self.state / 'jobs'}"
            )
        failures = [
            path
            for path in (self.state / "jobs").glob("*.json")
            if read_job(path)["status"] == "failed"
        ]
        if self._worker_errors:
            raise RuntimeError(
                "LeRobot worker failed: " + "; ".join(self._worker_errors)
            )
        if failures:
            raise RuntimeError(
                f"LeRobot finalization failed for {len(failures)} recording(s); inspect {self.state / 'jobs'}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--job", type=Path)
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    if args.job:
        try:
            process_job(args.job)
        except Exception as error:  # noqa: BLE001 - persist failures at the job process boundary
            fail_job(args.job, error)
            traceback.print_exc()
            return 1
        return 0
    if not args.dataset:
        parser.error("--dataset is required with --status or --retry-failed")
    paths = sorted((dataset_state(args.dataset) / "jobs").glob("*.json"))
    if args.retry_failed:
        for path in paths:
            job = read_job(path)
            if job["status"] == "failed":
                job.update(status="queued", error=None)
                write_json(path, job)
        writer = LeRobotRecordingWriter(args.dataset)
        writer.close()
    print(json.dumps([read_json(path) for path in paths], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
