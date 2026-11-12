"""配置解析（含新增 [gripper]/[recording] 节）单测。"""

import tempfile
import unittest
from pathlib import Path

from ur5_teleop import config as config_module


class ConfigTest(unittest.TestCase):
    def test_defaults_when_sections_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.toml"
            path.write_text("[robot]\nip = '1.2.3.4'\n", encoding="utf-8")
            cfg = config_module.load(path)
        self.assertFalse(cfg.gripper.enabled)
        self.assertEqual(cfg.gripper.baud, 115200)
        self.assertTrue(cfg.recording.enabled)
        self.assertEqual(cfg.recording.dir, "data/sessions")

    def test_gripper_and_recording_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.toml"
            path.write_text(
                "[robot]\nip = '1.2.3.4'\n\n"
                "[gripper]\nenabled = true\nport = '/dev/tty.test'\n"
                "slave_id = 2\nspeed = 80\nforce = 40\ntrigger_side = 'left'\n\n"
                "[recording]\nenabled = false\ndir = 'data/x'\n",
                encoding="utf-8")
            cfg = config_module.load(path)
        self.assertTrue(cfg.gripper.enabled)
        self.assertEqual(cfg.gripper.port, "/dev/tty.test")
        self.assertEqual(cfg.gripper.slave_id, 2)
        self.assertEqual(cfg.gripper.speed, 80)
        self.assertEqual(cfg.gripper.trigger_side, "left")
        self.assertFalse(cfg.recording.enabled)
        self.assertEqual(cfg.recording.dir, "data/x")

    def test_invalid_gripper_values_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.toml"
            path.write_text("[robot]\nip = '1.2.3.4'\n\n[gripper]\nslave_id = 0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                config_module.load(path)


if __name__ == "__main__":
    unittest.main()

    def test_rot_scale_and_trigger_mode_validation(self):
        base = Path("config/ur5.toml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            bad_rot = Path(tmp) / "bad_rot.toml"
            bad_rot.write_text(base.replace("rot_scale = 1.0", "rot_scale = 0.0"), encoding="utf-8")
            with self.assertRaises(ValueError):
                config.load(bad_rot)
            bad_trig = Path(tmp) / "bad_trig.toml"
            bad_trig.write_text(base.replace('trigger_mode = "analog"', 'trigger_mode = "magic"'),
                                encoding="utf-8")
            with self.assertRaises(ValueError):
                config.load(bad_trig)

    def test_home_mode_validation(self):
        self.assertEqual(config.load("config/ur5.toml").home.mode, "movej")
        base = Path("config/ur5.toml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.toml"
            bad.write_text(base.replace('mode = "movej"', 'mode = "magic"'), encoding="utf-8")
            with self.assertRaises(ValueError):
                config.load(bad)

    def test_home_and_collect_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.toml"
            path.write_text(
                "[robot]\nip = '1.2.3.4'\n\n"
                "[home]\njoints_deg = [-180, -90, -90, -90, 90, -90]\nlift_m = 0.2\n"
                "speed_mps = 0.08\ncountdown_s = 2.0\n\n"
                "[collect]\ndir = 'data/eps'\ncamera_indices = [1, 2]\n"
                "camera_fps = 15.0\n",
                encoding="utf-8")
            cfg = config_module.load(path)
        self.assertEqual(cfg.home.lift_m, 0.2)
        self.assertEqual(cfg.home.speed_mps, 0.08)
        self.assertEqual(cfg.home.countdown_s, 2.0)
        self.assertEqual(len(cfg.home.joints_deg), 6)
        self.assertEqual(cfg.collect.dir, "data/eps")
        self.assertEqual(cfg.collect.camera_indices, (1, 2))
        self.assertEqual(cfg.collect.camera_fps, 15.0)
