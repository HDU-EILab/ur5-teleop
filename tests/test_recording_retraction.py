"""Withdrawing a saved source removes every derived episode and retains its raw files."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from tests.test_lerobot_recording import make_recording
from ur5_teleop.lerobot_recording import (
    LeRobotRecordingWriter,
    dataset_state,
    process_job,
    read_json,
)
from ur5_teleop.recording_retraction import discard_last_recording

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
class RetractionTest(unittest.TestCase):
    def prepare(self, root):
        first = make_recording(root / "raw", "20261115_ep001", task="pick")
        second = make_recording(root / "raw", "20261115_ep002", task="place")
        writer = LeRobotRecordingWriter(
            root / "dataset", fps=10, printer=lambda _: None
        )
        first_job = writer.enqueue(first)
        writer.queue.join()
        first_generation = (root / "dataset").resolve()
        second_job = writer.enqueue(second)
        writer.close()
        return first, second, first_job, second_job, first_generation

    def test_latest_save_retracts_all_segments_and_queue_cannot_restore_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second, _, job, first_generation = self.prepare(root)
            result = discard_last_recording(root / "raw", root / "dataset")
            self.assertEqual(result["removed_episodes"], [2, 3])
            self.assertEqual(result["remaining_episodes"], 2)
            self.assertEqual((root / "dataset").resolve(), first_generation)
            self.assertTrue((first / "robot.csv").exists())
            self.assertFalse(second.exists())
            self.assertTrue((Path(result["archived_source"]) / "robot.csv").exists())
            self.assertEqual(process_job(job)["status"], "discarded")
            writer = LeRobotRecordingWriter(root / "dataset", printer=lambda _: None)
            self.assertEqual(writer.enqueue(second), job)
            writer.recover(root / "raw")
            writer.close()
            self.assertEqual(read_json(job)["status"], "discarded")
            # A new save can append normally after restoring the preceding generation.
            third = make_recording(root / "raw", "20261115_ep003", task="new")
            writer = LeRobotRecordingWriter(
                root / "dataset", fps=10, printer=lambda _: None
            )
            writer.enqueue(third)
            writer.close()
            self.assertEqual(
                read_json(root / "dataset/meta/info.json")["total_episodes"], 4
            )

    def test_last_remaining_save_removes_live_link_and_cli_uses_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, _, _, _, _ = self.prepare(root)
            discard_last_recording(root / "raw", root / "dataset")
            config = root / "task.toml"
            config.write_text(
                '[collect]\nformat="lerobot"\ndir='
                + json.dumps(str(root / "dataset"))
                + "\nraw_dir="
                + json.dumps(str(root / "raw"))
                + "\n"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/discard_last_recording.py"),
                    "--config",
                    str(config),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("已撤回", result.stdout)
            self.assertFalse((root / "dataset").is_symlink())
            self.assertFalse(first.exists())
            self.assertEqual(
                len(list((root / "raw/_discarded").glob("*/meta.json"))), 2
            )

    def test_out_of_order_export_rebuilds_indices_tasks_and_keeps_other_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second, _, _, _ = self.prepare(root)
            old = (root / "dataset").resolve()
            original_video = (
                old / "videos/chunk-000/observation.images.global/episode_000002.mp4"
            )
            future = (second / "meta.json").stat().st_mtime_ns + 1_000_000
            os.utime(first / "meta.json", ns=(future, future))
            result = discard_last_recording(root / "raw", root / "dataset")
            self.assertEqual(result["removed_episodes"], [0, 1])
            info = read_json(root / "dataset/meta/info.json")
            self.assertEqual(info["total_episodes"], 2)
            self.assertEqual(info["total_tasks"], 1)
            tables = [
                pd.read_parquet(
                    root / f"dataset/data/chunk-000/episode_{episode:06d}.parquet"
                )
                for episode in range(2)
            ]
            combined = pd.concat(tables)
            np.testing.assert_array_equal(
                combined["index"], np.arange(info["total_frames"])
            )
            self.assertEqual(set(combined["task_index"]), {0})
            self.assertEqual(
                set(read_json(root / "dataset/meta/recording_sources.json")),
                {second.name},
            )
            self.assertEqual(
                read_json(root / "dataset/meta/recording_sources.json")[second.name][
                    "episodes"
                ],
                [0, 1],
            )
            video = (
                root
                / "dataset/videos/chunk-000/observation.images.global/episode_000000.mp4"
            )
            self.assertEqual(video.stat().st_ino, original_video.stat().st_ino)
            stats = read_json(root / "dataset/meta/stats.json")
            self.assertEqual(
                stats["observation.images.global"]["count"], [info["total_frames"]]
            )

    def test_interrupted_withdrawal_recovers_after_dataset_switch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, second, _, job, previous = self.prepare(root)
            with patch.object(
                Path, "rename", side_effect=OSError("injected archive failure")
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    discard_last_recording(root / "raw", root / "dataset")
            self.assertTrue(second.exists())
            self.assertEqual((root / "dataset").resolve(), previous)
            self.assertEqual(read_json(job)["status"], "discarded")
            writer = LeRobotRecordingWriter(root / "dataset", printer=lambda _: None)
            writer.close()
            self.assertFalse(second.exists())
            journal = next(
                (dataset_state(root / "dataset") / "withdrawals").glob("*.json")
            )
            self.assertEqual(read_json(journal)["status"], "complete")


class PendingRetractionTest(unittest.TestCase):
    def test_saved_but_not_yet_queued_source_is_cancelled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw" / "20261115_ep001"
            source.mkdir(parents=True)
            (source / "meta.json").write_text('{"success": true}')
            result = discard_last_recording(root / "raw", root / "dataset")
            self.assertEqual(result["removed_episodes"], [])
            self.assertEqual(process_job(result["job_path"])["status"], "discarded")
            writer = LeRobotRecordingWriter(root / "dataset", printer=lambda _: None)
            writer.enqueue(source)
            writer.close()
            self.assertFalse((root / "dataset").exists())

    def test_active_unsaved_recording_is_not_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active = root / "raw" / "20261115_ep001"
            active.mkdir(parents=True)
            (active / "robot.csv").write_text("active recording")
            with self.assertRaisesRegex(ValueError, "没有可撤回"):
                discard_last_recording(root / "raw", root / "dataset")
            self.assertEqual((active / "robot.csv").read_text(), "active recording")
            config = root / "task.toml"
            config.write_text(
                '[collect]\nformat="lerobot"\ndir='
                + json.dumps(str(root / "dataset"))
                + "\nraw_dir="
                + json.dumps(str(root / "raw"))
                + "\n"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/discard_last_recording.py"),
                    "--config",
                    str(config),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("没有可撤回", result.stderr)


if __name__ == "__main__":
    unittest.main()
