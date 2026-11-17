"""Service-lifetime streams must supply each episode immediately and release devices on exit."""

import csv
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from ur5_teleop.cameras import CameraSet


class ReadyGate:
    def __init__(self, started_at):
        self.started_at = started_at
        self.ready = False
        self.discarded = 0
        self.calls = 0

    def accept(self, frame, now):
        self.calls += 1
        self.ready = True
        return True


class PersistentCameraTest(unittest.TestCase):
    def make_capture(self):
        capture = Mock()
        capture.isOpened.return_value = True
        image = np.full((48, 64, 3), 90, dtype=np.uint8)

        def read():
            time.sleep(0.003)
            return True, image

        capture.read.side_effect = read
        return capture

    def wait_frames(self, cameras, count=3):
        deadline = time.monotonic() + 1
        while (
            cameras.latest()["global"]["frame"] < count - 1
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        self.assertGreaterEqual(cameras.latest()["global"]["frame"], count - 1)

    def test_two_recordings_share_one_capture_and_idle_does_not_write(self):
        capture = self.make_capture()
        writers = []

        def new_writer(*args):
            writer = Mock()
            writer.isOpened.return_value = True
            writers.append(writer)
            return writer

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "ur5_teleop.cameras.cv2.VideoCapture", return_value=capture
            ) as open_capture,
            patch("ur5_teleop.cameras.cv2.VideoWriter", side_effect=new_writer),
            patch(
                "ur5_teleop.video_quality.FrameReadiness", side_effect=ReadyGate
            ) as gates,
        ):
            cameras = CameraSet(
                indices=(0,), names=("global",), width=64, height=48, persistent=True
            )
            try:
                cameras.open()
                self.assertTrue(cameras.wait_ready(1)["ready"])
                self.assertEqual(writers, [])
                self.assertEqual(cameras.latest()["global"]["frame"], -1)
                cameras.probe()
                for episode in [1, 2]:
                    directory = Path(temporary) / str(episode)
                    status = cameras.start(directory)
                    self.assertTrue(status["ready"])
                    self.wait_frames(cameras)
                    stats = cameras.stop()
                    self.assertFalse(stats["errors"])
                    self.assertLess(stats["first_frame_delay_s"]["global"], 0.1)
                    capture.release.assert_not_called()
                    with (directory / "videos/global_timestamps.csv").open() as stream:
                        stamps = list(csv.DictReader(stream))
                    self.assertEqual(len(stamps), stats["frames"]["global"])
                    self.assertEqual(
                        [int(row["frame"]) for row in stamps], list(range(len(stamps)))
                    )
                    self.assertGreaterEqual(
                        float(stamps[0]["t_mono"]),
                        status["recording_started_at"] - 0.000001,
                    )
                    count = writers[-1].write.call_count
                    reads = capture.read.call_count
                    time.sleep(0.03)
                    self.assertGreater(capture.read.call_count, reads)
                    self.assertEqual(writers[-1].write.call_count, count)
                open_capture.assert_called_once()
                self.assertEqual(gates.call_count, 1)
                self.assertEqual(cameras._warmup["global"].calls, 1)
                for writer in writers:
                    writer.release.assert_called_once()
            finally:
                cameras.close()
            capture.release.assert_called_once()
            cameras.close()
            capture.release.assert_called_once()

    def test_latest_does_not_wait_for_an_in_progress_video_write(self):
        capture = self.make_capture()
        writer = Mock()
        writer.isOpened.return_value = True
        entered, release, read_done = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def encode(frame):
            entered.set()
            release.wait(2)

        writer.write.side_effect = encode
        observed = []
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("ur5_teleop.cameras.cv2.VideoCapture", return_value=capture),
            patch("ur5_teleop.cameras.cv2.VideoWriter", return_value=writer),
            patch("ur5_teleop.video_quality.FrameReadiness", side_effect=ReadyGate),
        ):
            cameras = CameraSet(
                indices=(0,), names=("global",), width=64, height=48, persistent=True
            )
            try:
                cameras.open()
                cameras.wait_ready(1)
                cameras.start(Path(temporary))
                self.assertTrue(entered.wait(1))

                def read_latest():
                    observed.append(cameras.latest())
                    read_done.set()

                reader = threading.Thread(target=read_latest, daemon=True)
                reader.start()
                self.assertTrue(
                    read_done.wait(0.1),
                    "video encoding blocked the control-loop reader",
                )
                self.assertEqual(observed[0]["global"], {"frame": -1, "t_mono": 0.0})
                release.set()
                reader.join(1)
                self.wait_frames(cameras)
                cameras.stop()
                snapshot = cameras.latest()
                actual = snapshot["global"]["frame"]
                snapshot["global"]["frame"] = 99999
                self.assertEqual(cameras.latest()["global"]["frame"], actual)
            finally:
                release.set()
                cameras.close()

    def test_failed_stream_reports_errors_in_readiness_and_episode(self):
        capture = self.make_capture()
        capture.read.side_effect = ValueError("disconnected")
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("ur5_teleop.cameras.cv2.VideoCapture", return_value=capture),
            patch("ur5_teleop.video_quality.FrameReadiness", side_effect=ReadyGate),
        ):
            cameras = CameraSet(
                indices=(0,), names=("global",), width=64, height=48, persistent=True
            )
            try:
                cameras.open()
                status = cameras.wait_ready(1)
                self.assertFalse(status["ready"])
                self.assertIn("disconnected", status["errors"]["global"])
                cameras.start(Path(temporary))
                self.assertIn("global", cameras.stop()["errors"])
            finally:
                cameras.close()

    def test_legacy_per_recording_fps_override_is_preserved(self):
        capture = self.make_capture()
        writer = Mock()
        writer.isOpened.return_value = True
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("ur5_teleop.cameras.cv2.VideoCapture", return_value=capture),
            patch("ur5_teleop.cameras.cv2.VideoWriter", return_value=writer) as factory,
            patch("ur5_teleop.video_quality.FrameReadiness", side_effect=ReadyGate),
        ):
            cameras = CameraSet(indices=(0,), names=("global",), width=64, height=48)
            try:
                cameras.start(Path(temporary), fps=10)
                self.wait_frames(cameras)
                stats = cameras.stop()
                self.assertFalse(stats["errors"])
                self.assertEqual(factory.call_args.args[2], 10)
                capture.release.assert_called_once()
            finally:
                cameras.close()


if __name__ == "__main__":
    unittest.main()
