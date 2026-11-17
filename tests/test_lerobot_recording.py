"""Saved demonstrations must remain recoverable and publish complete LeRobot datasets."""

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from tests.test_collector import make_step
from ur5_teleop.cameras import CameraSet
from ur5_teleop.collector import COLUMNS, Collector
from ur5_teleop.config import CollectConfig
from ur5_teleop.lerobot_recording import (
    LeRobotRecordingWriter,
    append_batch,
    dataset_lock,
    dataset_state,
    read_json,
    write_json,
)
from ur5_teleop.video_quality import FrameReadiness


def make_recording(root, name, task="pick", *, success=True, fps=10):
    source = root / name
    (source / "videos").mkdir(parents=True)
    # Eight seconds with one intervention. Both retained sides exceed 1.5 s.
    for camera, shift in [("global", 0), ("wrist", 0.02)]:
        writer = cv2.VideoWriter(
            str(source / "videos" / f"{camera}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (64, 48),
        )
        if not writer.isOpened():
            raise RuntimeError("Synthetic video encoder unavailable")
        with (source / "videos" / f"{camera}_timestamps.csv").open("w") as stream:
            stamps = csv.writer(stream)
            stamps.writerow(["frame", "t_mono", "t_wall"])
            for frame in range(81):
                image = np.full((48, 64, 3), (30 + frame, 70, 110), dtype=np.uint8)
                image[10:20, 10:20] = 200
                writer.write(image)
                stamps.writerow([frame, 100 + frame / fps + shift, 1000 + frame / fps])
        writer.release()
    with (source / "robot.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for frame in range(901):
            row = dict.fromkeys(COLUMNS, 0)
            row.update(
                t_mono=100 + frame / 100,
                robot_x=frame / 1000,
                gripper_position=700,
                target_ry=-3.14,
                robot_ry=-3.14,
            )
            writer.writerow(row)
    (source / "meta.json").write_text(
        json.dumps(
            dict(
                started_at=100,
                success=success,
                task=task,
                disturbance_windows=[[103, 104]],
                camera_stats={"frames": {"global": 81, "wrist": 81}},
            )
        )
    )
    return source


class CollectFormatTest(unittest.TestCase):
    def test_lerobot_recording_paths_and_format(self):
        settings = CollectConfig(
            format="lerobot", dir="data/lerobot_live", raw_dir="data/raw_recordings"
        )
        self.assertEqual(settings.format, "lerobot")
        for fields in ({"format": "unknown"}, {"format": "lerobot"}, {"raw_dir": ""}):
            with self.assertRaises(ValueError):
                CollectConfig(**fields)


class ReadinessTest(unittest.TestCase):
    def test_rejects_green_and_white_balance_ramp_before_stable_frames(self):
        gate = FrameReadiness(0)
        green = np.full((48, 64, 3), (0, 255, 0), dtype=np.uint8)
        for index in range(35):
            self.assertFalse(gate.accept(green, index / 30))
        image = np.full((48, 64, 3), 70, dtype=np.uint8)
        image[:20, :20] = 170
        for index in range(45):
            tinted = image.copy()
            tinted[:, :, 1] += 45 - index
            self.assertFalse(gate.accept(tinted, (35 + index) / 30))
        for index in range(16):
            ready = gate.accept(image, (80 + index) / 30)
        self.assertTrue(ready)
        self.assertGreater(gate.discarded, 75)

    def test_buffered_burst_cannot_count_as_stable_wall_time(self):
        gate = FrameReadiness(0)
        image = np.full((48, 64, 3), 70, dtype=np.uint8)
        image[:20, :20] = 170
        for index in range(30):
            self.assertFalse(gate.accept(image, 2 + index / 1000))


class CameraLifecycleTest(unittest.TestCase):
    def test_worker_owns_resources_and_new_recording_resets_counts(self):
        image = np.full((48, 64, 3), 80, dtype=np.uint8)
        camera = unittest.mock.Mock()
        camera.isOpened.return_value = True

        def read_frame():
            time.sleep(0.002)
            return True, image

        camera.read.side_effect = read_frame
        video = unittest.mock.Mock()
        video.isOpened.return_value = True
        readiness = unittest.mock.Mock(ready=True, discarded=0)
        readiness.accept.return_value = True
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("ur5_teleop.cameras.cv2.VideoCapture", return_value=camera),
            patch("ur5_teleop.cameras.cv2.VideoWriter", return_value=video),
            patch("ur5_teleop.video_quality.FrameReadiness", return_value=readiness),
        ):
            cameras = CameraSet(indices=(0,), names=("global",), width=64, height=48)
            cameras.start(Path(temporary) / "first")
            deadline = time.monotonic() + 1
            while (
                cameras.latest()["global"]["frame"] < 2 and time.monotonic() < deadline
            ):
                time.sleep(0.005)
            stats = cameras.stop()
            self.assertGreater(stats["frames"]["global"], 2)
            self.assertEqual(
                cameras.latest()["global"]["frame"], stats["frames"]["global"] - 1
            )
            self.assertFalse(stats["errors"])
            video.release.assert_called_once()
            camera.release.assert_called_once()
            with (
                Path(temporary) / "first/videos/global_timestamps.csv"
            ).open() as stream:
                timestamps = list(csv.DictReader(stream))
            self.assertEqual(len(timestamps), stats["frames"]["global"])
            camera.isOpened.return_value = False
            cameras.start(Path(temporary) / "second")
            stats = cameras.stop()
            self.assertEqual(stats["frames"]["global"], 0)
            self.assertIn("global", stats["errors"])


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
class RecordingWriterTest(unittest.TestCase):
    def test_background_export_append_task_mapping_and_idempotency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "lerobot"
            first = make_recording(root / "raw", "20261114_ep001")
            second = make_recording(root / "raw", "20261114_ep002", "place")
            writer = LeRobotRecordingWriter(dataset, fps=10, printer=lambda _: None)
            job1 = writer.enqueue(first)
            self.assertEqual(job1, writer.enqueue(first))
            writer.enqueue(second)
            writer.close()
            info = read_json(dataset / "meta/info.json")
            self.assertTrue(dataset.is_symlink())
            self.assertEqual(info["total_episodes"], 4)
            self.assertEqual(info["total_tasks"], 2)
            self.assertEqual(info["total_videos"], 8)
            frames = []
            cuts = read_json(dataset / "meta/cut_info.json")
            for episode in range(4):
                table = pd.read_parquet(
                    dataset / f"data/chunk-000/episode_{episode:06d}.parquet"
                )
                frames.append(table)
                self.assertEqual(set(table["task_index"]), {episode // 2})
                self.assertEqual(set(table["episode_index"]), {episode})
                np.testing.assert_array_equal(
                    table["frame_index"], np.arange(len(table))
                )
                np.testing.assert_allclose(
                    table["timestamp"], np.arange(len(table)) / 10, atol=1e-6
                )
                cut = cuts[episode]
                actual_times = (
                    100.02 + np.arange(cut["frames"][0], cut["frames"][1] + 1) / 10
                )
                self.assertTrue(
                    np.all((actual_times < 102.75) | (actual_times > 104.25))
                )
                for camera in ["global", "wrist"]:
                    capture = cv2.VideoCapture(
                        str(
                            dataset
                            / f"videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4"
                        )
                    )
                    self.assertEqual(
                        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), len(table)
                    )
                    capture.release()
            combined = pd.concat(frames)
            np.testing.assert_array_equal(
                combined["index"], np.arange(info["total_frames"])
            )
            stats = read_json(dataset / "meta/stats.json")
            self.assertEqual(
                stats["observation.images.global"]["count"], [info["total_frames"]]
            )
            self.assertGreater(
                np.asarray(stats["observation.images.global"]["mean"]).mean(), 0
            )
            per_episode = [
                json.loads(line)
                for line in (dataset / "meta/episodes_stats.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(per_episode), 4)
            for episode, entry in enumerate(per_episode):
                self.assertEqual(entry["stats"]["episode_index"]["mean"], [episode])
                self.assertEqual(
                    entry["stats"]["observation.images.global"]["count"],
                    [len(frames[episode])],
                )
            original = dataset.resolve()
            writer = LeRobotRecordingWriter(dataset, fps=10, printer=lambda _: None)
            writer.enqueue(first)
            writer.close()
            self.assertEqual(dataset.resolve(), original)
            self.assertTrue((first / "videos/global.mp4").exists())
            # Recover a crash after publication but before job completion was recorded.
            interrupted = read_json(job1)
            interrupted["status"] = "processing"
            job1.write_text(json.dumps(interrupted))
            writer = LeRobotRecordingWriter(dataset, printer=lambda _: None)
            writer.close()
            self.assertEqual(read_json(job1)["status"], "complete")
            self.assertEqual(dataset.resolve(), original)
            # A failed new generation leaves the published version unchanged.
            checksum = hashlib.sha256((first / "meta.json").read_bytes()).hexdigest()
            with (
                dataset_lock(dataset),
                patch(
                    "ur5_teleop.lerobot_recording.feature_stats",
                    side_effect=ValueError("injected"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "injected"):
                    append_batch(dataset.resolve(), dataset, "failed-source", checksum)
            self.assertEqual(dataset.resolve(), original)
            self.assertNotIn(
                "failed-source", read_json(dataset / "meta/recording_sources.json")
            )

    def test_failed_export_is_reported_and_raw_is_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = make_recording(root / "raw", "20261114_ep001")
            metadata = read_json(source / "meta.json")
            metadata["camera_valid"] = False
            (source / "meta.json").write_text(json.dumps(metadata))
            writer = LeRobotRecordingWriter(
                root / "lerobot", fps=10, printer=lambda _: None
            )
            path = writer.enqueue(source)
            with self.assertRaisesRegex(RuntimeError, "finalization failed"):
                writer.close()
            self.assertEqual(read_json(path)["status"], "failed")
            self.assertFalse((root / "lerobot").exists())
            self.assertTrue((source / "robot.csv").exists())

    def test_failed_encoder_can_be_retried_without_changing_raw_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "lerobot"
            source = make_recording(root / "raw", "20261114_ep001")
            binary = root / "bin"
            binary.mkdir()
            encoder = binary / "ffmpeg"
            encoder.write_text("#!/bin/sh\nexit 7\n")
            encoder.chmod(0o755)
            with patch.dict(
                os.environ, {"PATH": str(binary) + os.pathsep + os.environ["PATH"]}
            ):
                writer = LeRobotRecordingWriter(dataset, fps=10, printer=lambda _: None)
                job = writer.enqueue(source)
                with self.assertRaises(RuntimeError):
                    writer.close()
            self.assertEqual(read_json(job)["status"], "failed")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ur5_teleop.lerobot_recording",
                    "--dataset",
                    str(dataset),
                    "--retry-failed",
                ],
                check=True,
                capture_output=True,
            )
            self.assertEqual(read_json(job)["status"], "complete")
            self.assertEqual(read_json(dataset / "meta/info.json")["total_episodes"], 2)

    def test_recovery_only_imports_explicit_export_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "lerobot"
            selected = make_recording(root / "raw", "20261114_ep001")
            make_recording(root / "raw", "20261114_ep002")
            metadata = read_json(selected / "meta.json")
            metadata["lerobot_export"] = {"dataset": str(dataset)}
            (selected / "meta.json").write_text(json.dumps(metadata))
            writer = LeRobotRecordingWriter(dataset, fps=10, printer=lambda _: None)
            writer.recover(root / "raw")
            writer.close()
            self.assertEqual(
                set(read_json(dataset / "meta/recording_sources.json")), {selected.name}
            )


class WorkerFailureTest(unittest.TestCase):
    def test_unexpected_exception_is_persisted_and_next_job_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = []
            for name in ["first", "second"]:
                source = root / name
                source.mkdir()
                write_json(source / "meta.json", {"success": True})
                sources.append(source)

            def run_job(command, **kwargs):
                path = Path(command[-1])
                job = read_json(path)
                if Path(job["source"]).name == "first":
                    raise RuntimeError("injected worker exception")
                job.update(status="complete", receipt={"episodes": []})
                write_json(path, job)
                return unittest.mock.Mock(returncode=0)

            with patch(
                "ur5_teleop.lerobot_recording.subprocess.run", side_effect=run_job
            ):
                writer = LeRobotRecordingWriter(
                    root / "dataset", printer=lambda _: None
                )
                first, second = [writer.enqueue(source) for source in sources]
                with self.assertRaisesRegex(RuntimeError, "finalization failed"):
                    writer.close()
            self.assertEqual(read_json(first)["status"], "failed")
            self.assertIn("injected worker exception", read_json(first)["error"])
            self.assertEqual(read_json(second)["status"], "complete")
            self.assertEqual(writer.queue.unfinished_tasks, 0)
            self.assertFalse(writer.thread.is_alive())
            with self.assertRaisesRegex(RuntimeError, "finalization failed"):
                writer.close()

    def test_corrupt_jobs_are_preserved_and_reported_at_startup(self):
        for content in ["null", "{broken", '{"status": []}']:
            with (
                self.subTest(content=content),
                tempfile.TemporaryDirectory() as temporary,
            ):
                dataset = Path(temporary) / "dataset"
                jobs = dataset_state(dataset) / "jobs"
                jobs.mkdir(parents=True)
                path = jobs / "broken.json"
                path.write_text(content)
                writer = LeRobotRecordingWriter(dataset, printer=lambda _: None)
                with self.assertRaisesRegex(RuntimeError, "finalization failed"):
                    writer.close()
                failure = read_json(path)
                self.assertEqual(failure["status"], "failed")
                self.assertEqual(
                    Path(failure["corrupt_job_backup"]).read_text(), content
                )

    def test_worker_death_does_not_block_close_with_pending_jobs(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(LeRobotRecordingWriter, "_worker", return_value=None),
        ):
            writer = LeRobotRecordingWriter(
                Path(temporary) / "dataset", printer=lambda _: None
            )
            writer.thread.join(1)
            writer.queue.put(Path(temporary) / "pending.json")
            errors = []

            def close_writer():
                try:
                    writer.close()
                except RuntimeError as error:
                    errors.append(str(error))

            closer = threading.Thread(target=close_writer, daemon=True)
            closer.start()
            closer.join(1)
            self.assertFalse(closer.is_alive(), "close blocked after worker death")
            self.assertTrue(errors)
            self.assertIn("worker exited", errors[0])

    def test_manual_export_rejects_invalid_camera_quality(self):
        from tools.export_lerobot import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = make_recording(root / "raw", "20261115_test")
            metadata = read_json(source / "meta.json")
            metadata["camera_valid"] = False
            write_json(source / "meta.json", metadata)
            with self.assertRaisesRegex(ValueError, "Camera recording is invalid"):
                main(
                    ["--episodes-dir", str(root / "raw"), "--out", str(root / "output")]
                )
            self.assertFalse((root / "output").exists())

    def test_invalid_job_cli_preserves_original_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            path.write_text("null")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ur5_teleop.lerobot_recording",
                    "--job",
                    str(path),
                ],
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            failure = read_json(path)
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(Path(failure["corrupt_job_backup"]).read_text(), "null")


class CollectorExportTest(unittest.TestCase):
    def test_camera_quality_is_explicit_for_raw_and_lerobot_formats(self):
        for use_lerobot in [False, True]:
            with (
                self.subTest(lerobot=use_lerobot),
                tempfile.TemporaryDirectory() as temporary,
                patch(
                    "ur5_teleop.lerobot_recording.LeRobotRecordingWriter"
                ) as writer_class,
            ):
                writer = writer_class.return_value
                writer.dataset = Path(temporary) / "dataset"
                writer.fps, writer.cut_margin_s, writer.min_segment_s = 30.0, 0.25, 1.5
                cameras = unittest.mock.Mock()
                cameras.start.return_value = {"ok": True}
                cameras.stop.return_value = {
                    "frames": {"global": 10, "wrist": 10},
                    "errors": {"global": "disconnected"},
                }
                collector = Collector(
                    Path(temporary) / "raw",
                    cameras=cameras,
                    lerobot_dir=writer.dataset if use_lerobot else None,
                    printer=lambda _: None,
                )
                collector.arm(make_step())
                saved = collector.save()
                metadata = read_json(saved / "meta.json")
                self.assertTrue(metadata["success"])
                self.assertFalse(metadata["camera_valid"])
                self.assertEqual(
                    metadata["camera_stats"]["errors"], {"global": "disconnected"}
                )
                collector.close()

    def test_save_enqueues_and_close_retains_failed_recording(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "ur5_teleop.lerobot_recording.LeRobotRecordingWriter"
            ) as writer_class,
        ):
            writer = writer_class.return_value
            writer.dataset = Path(temporary) / "lerobot"
            writer.fps, writer.cut_margin_s, writer.min_segment_s = 30.0, 0.25, 1.5
            cameras = unittest.mock.Mock()
            cameras.start.return_value = {"ok": True}
            cameras.stop.return_value = {"frames": {"global": 30, "wrist": 30}}
            collector = Collector(
                Path(temporary) / "raw",
                cameras=cameras,
                lerobot_dir=writer.dataset,
                printer=lambda _: None,
            )
            collector.arm(make_step())
            source = collector.save()
            writer.enqueue.assert_called_once_with(source)
            self.assertTrue(read_json(source / "meta.json")["camera_valid"])
            self.assertEqual(
                read_json(source / "meta.json")["lerobot_export"]["dataset"],
                str(writer.dataset),
            )
            collector.arm(make_step())
            incomplete = collector.directory
            collector.close()
            self.assertFalse(read_json(incomplete / "meta.json")["success"])
            writer.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
