"""Episode 采集状态机单测（无相机、无机器人）。"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from ur5_teleop.collector import Collector
from ur5_teleop.safety import TeleopState
from ur5_teleop.teleop import TeleopStep

POSE = (0.1, 0.2, 0.3, 0.0, 0.0, 0.0)


def make_step(buttons=None, servo=False, state=TeleopState.IDLE) -> TeleopStep:
    return TeleopStep(
        state, POSE, servo, None, "",
        robot_pose=(0.4, 0.0, 0.4, 0.0, 0.0, 0.0),
        robot_joints=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        grip_left=0.0, grip_right=0.9, trigger_left=0.0, trigger_right=0.3,
        buttons=dict(buttons or {}),
    )


class CollectorTest(unittest.TestCase):
    def test_b_starts_and_saves_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = Collector(tmp, printer=lambda msg: None)
            self.assertFalse(collector.armed)
            self.assertIsNone(collector.step(make_step({"b": True})))      # 边沿：开始录制
            self.assertTrue(collector.armed)
            self.assertIsNone(collector.step(make_step({"b": True})))      # 按住不重复
            self.assertIsNone(collector.step(make_step({})))                # 松开
            action = collector.step(make_step({"b": True}, servo=False))   # 再按：保存
            self.assertEqual(action, "save_home")
            directory = collector.save(success=True)
            self.assertIsNotNone(directory)
            self.assertFalse(collector.armed)
            self.assertTrue((directory / "robot.csv").is_file())
            self.assertTrue((directory / "meta.json").is_file())
            self.assertTrue((directory / "annotations.json").is_file())
            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            self.assertTrue(meta["success"])
            self.assertIn("red cylinder", meta["task"])
            self.assertIn("blue cube", meta["task"])
            self.assertGreaterEqual(meta["rows"], 1)
            self.assertGreaterEqual(len(list(csv.DictReader((directory / "robot.csv").open()))), 1)

    def test_b_ignored_while_moving(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = Collector(tmp, printer=lambda msg: None)
            collector.step(make_step({"b": True}))          # 开始
            collector.step(make_step({}))
            action = collector.step(make_step({"b": True}, servo=True))
            self.assertIsNone(action, "运动中的 B 不应触发保存")
            collector.discard()

    def test_stick_marks_disturbance_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = Collector(tmp, printer=lambda msg: None)
            collector.step(make_step({"b": True}))
            collector.step(make_step({}))
            collector.step(make_step({"axis_right_click": True}))
            self.assertEqual(len(collector.disturbance), 1)
            self.assertIsNone(collector.disturbance[0][1])
            collector.step(make_step({"axis_right_click": False}))
            collector.step(make_step({"axis_right_click": True}))
            self.assertIsNotNone(collector.disturbance[0][1])
            directory = collector.save(success=True)
            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(len(meta["disturbance_windows"]), 1)
            start, end = meta["disturbance_windows"][0]
            self.assertIsNotNone(start)
            self.assertIsNotNone(end)

    def test_stick_ignored_while_moving(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = Collector(tmp, printer=lambda msg: None)
            collector.step(make_step({"b": True}))
            collector.step(make_step({}))
            collector.step(make_step({"axis_right_click": True}, servo=True))
            self.assertEqual(collector.disturbance, [])
            collector.discard()

    def test_discard_moves_to_discarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = Collector(tmp, printer=lambda msg: None)
            collector.step(make_step({"b": True}))
            directory = collector.directory
            collector.discard()
            self.assertFalse(collector.armed)
            self.assertTrue((Path(tmp) / "_discarded" / directory.name).is_dir())

    def test_close_saves_incomplete_episode_as_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = Collector(tmp, printer=lambda msg: None)
            collector.step(make_step({"b": True}))
            collector.close()
            metas = list(Path(tmp).glob("*/meta.json"))
            self.assertEqual(len(metas), 1)
            meta = json.loads(metas[0].read_text(encoding="utf-8"))
            self.assertFalse(meta["success"])


if __name__ == "__main__":
    unittest.main()


class EpisodeNumberingTest(unittest.TestCase):
    def test_continues_from_existing_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path as _P
            (_P(tmp) / "20260101_000000_ep007").mkdir()
            (_P(tmp) / "20260101_010000_ep012").mkdir()
            collector = Collector(tmp, printer=lambda msg: None)
            self.assertEqual(collector.episode_count, 12)

