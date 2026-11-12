"""数据录制模块单测。"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from ur5_teleop.recording import COLUMNS, SessionRecorder
from ur5_teleop.safety import TeleopState
from ur5_teleop.teleop import TeleopStep

POSE = (0.1, 0.2, 0.3, 0.0, 0.0, 0.0)


def make_step(state: TeleopState) -> TeleopStep:
    return TeleopStep(
        state, POSE, False, None, "test",
        robot_pose=(0.4, 0.0, 0.4, 0.0, 0.0, 0.0),
        robot_joints=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        grip_left=0.1, grip_right=0.9, trigger_left=0.0, trigger_right=0.5,
    )


class SessionRecorderTest(unittest.TestCase):
    def test_writes_header_rows_and_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = SessionRecorder(tmp, meta={"robot_ip": "1.2.3.4"})
            self.assertTrue(recorder.csv_path.is_file())
            states = [TeleopState.IDLE, TeleopState.TRACKING, TeleopState.TRACKING,
                      TeleopState.IDLE, TeleopState.TRACKING]
            for state in states:
                recorder.log(make_step(state), gripper_position=250, gripper_state=1)
            recorder.close()

            with recorder.csv_path.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 5)
            self.assertEqual([row["episode"] for row in rows], ["0", "1", "1", "1", "2"])
            self.assertEqual(rows[0]["state"], "IDLE")
            self.assertEqual(rows[1]["state"], "TRACKING")
            self.assertEqual(rows[1]["gripper_position"], "250")
            self.assertEqual(rows[1]["gripper_state"], "1")
            self.assertEqual(rows[1]["robot_x"], "0.400000")
            self.assertEqual(rows[1]["q1"], "0.100000")
            self.assertEqual(rows[1]["trigger_right"], "0.5000")

            meta = json.loads(recorder.meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["robot_ip"], "1.2.3.4")
            self.assertEqual(meta["episodes"], 2)
            self.assertEqual(meta["columns"], COLUMNS)

    def test_missing_optional_fields_are_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = SessionRecorder(tmp)
            recorder.log(TeleopStep(TeleopState.IDLE, POSE, False, None))
            recorder.close()
            with recorder.csv_path.open(newline="", encoding="utf-8") as fh:
                row = next(csv.DictReader(fh))
            self.assertEqual(row["robot_x"], "")
            self.assertEqual(row["q1"], "")
            self.assertEqual(row["gripper_position"], "")

    def test_context_manager_closes(self):
        with tempfile.TemporaryDirectory() as tmp:
            with SessionRecorder(tmp) as recorder:
                recorder.log(make_step(TeleopState.IDLE))
            self.assertTrue(recorder.meta_path.is_file())
            self.assertEqual(recorder.episode, 0)


if __name__ == "__main__":
    unittest.main()
