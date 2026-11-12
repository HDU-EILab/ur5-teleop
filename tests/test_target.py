"""``ur5_teleop.target`` 单测：离合锁存、目标合成、工作空间裁剪、步长限幅。"""

import math
import unittest

from ur5_teleop import frames
from ur5_teleop.config import SafetyConfig, TeleopConfig
from ur5_teleop.target import ClutchOrigin, TargetShaper, clip_pose, limit_pose_step

START_POSE = (0.4, 0.0, 0.4, 0.0, 0.0, 0.0)
START_POSE_NO_ROT = START_POSE
CONTROLLER_POSITION = (0.2, 1.2, -0.3)
IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def make_teleop(scale=1.0, mount_yaw_deg=0.0, max_pos_step_m=0.02,
                max_rot_step_rad=0.05, rot_scale=1.0):
    return TeleopConfig(scale=scale, mount_yaw_deg=mount_yaw_deg,
                        max_pos_step_m=max_pos_step_m,
                        max_rot_step_rad=max_rot_step_rad, rot_scale=rot_scale)


def make_safety(workspace_min=(-0.5, -0.5, 0.05), workspace_max=(0.6, 0.5, 0.9)):
    return SafetyConfig(workspace_min=workspace_min, workspace_max=workspace_max)


def quat_about_y(angle):
    half = 0.5 * angle
    return (0.0, math.sin(half), 0.0, math.cos(half))


class TargetShaperTest(unittest.TestCase):
    def make_shaper(self, **kwargs):
        return TargetShaper(make_teleop(**kwargs), make_safety())

    def engage(self, shaper, start_pose=START_POSE):
        return shaper.engage(start_pose, CONTROLLER_POSITION,
                             IDENTITY_QUAT, IDENTITY_QUAT)

    def converge(self, shaper, controller_position, controller_quat, steps=100):
        """反复用同一手柄位姿推进，直到收敛（用于步长限幅后的终值断言）。"""
        target = None
        for _ in range(steps):
            target = shaper.next_target(controller_position, controller_quat)
        return target

    def test_engage_latches_start(self):
        shaper = self.make_shaper()
        self.assertFalse(shaper.engaged)
        origin = self.engage(shaper)
        self.assertIsInstance(origin, ClutchOrigin)
        self.assertTrue(shaper.engaged)
        self.assertEqual(origin.start_pose, START_POSE)
        self.assertEqual(origin.controller_position, CONTROLLER_POSITION)
        self.assertEqual(shaper.last_target, START_POSE)

    def test_next_target_without_engage_returns_none(self):
        shaper = self.make_shaper()
        self.assertIsNone(shaper.next_target(CONTROLLER_POSITION, IDENTITY_QUAT))
        with self.assertRaises(RuntimeError):
            shaper.desired_target(CONTROLLER_POSITION, IDENTITY_QUAT)

    def test_no_motion_keeps_start_pose(self):
        shaper = self.make_shaper()
        self.engage(shaper)
        target = shaper.next_target(CONTROLLER_POSITION, IDENTITY_QUAT)
        for actual, expected in zip(target, START_POSE):
            self.assertAlmostEqual(actual, expected, places=9)

    def test_controller_forward_moves_robot_plus_x(self):
        shaper = self.make_shaper()
        self.engage(shaper)
        # PICO 头显前方 = -Z：控制器 z 减小 → 机器人 +X
        desired = shaper.desired_target((0.2, 1.2, -0.4), IDENTITY_QUAT)
        self.assertAlmostEqual(desired[0], 0.5, places=9)
        self.assertAlmostEqual(desired[1], 0.0, places=9)
        self.assertAlmostEqual(desired[2], 0.4, places=9)

    def test_controller_right_moves_robot_minus_y(self):
        shaper = self.make_shaper()
        self.engage(shaper)
        desired = shaper.desired_target((0.3, 1.2, -0.3), IDENTITY_QUAT)
        self.assertAlmostEqual(desired[0], 0.4, places=9)
        self.assertAlmostEqual(desired[1], -0.1, places=9)

    def test_scale_applies_to_translation_only(self):
        shaper = self.make_shaper(scale=0.5)
        self.engage(shaper)
        desired = shaper.desired_target((0.2, 1.2, -0.4), IDENTITY_QUAT)
        self.assertAlmostEqual(desired[0], 0.45, places=9)
        # 姿态仍是 1:1：手柄转 0.4 rad → 目标绕基座 +Z 0.4 rad
        desired = shaper.desired_target((0.2, 1.2, -0.3), quat_about_y(0.4))
        self.assertAlmostEqual(desired[3], 0.0, places=9)
        self.assertAlmostEqual(desired[4], 0.0, places=9)
        self.assertAlmostEqual(desired[5], 0.4, places=9)

    def test_mount_yaw_rotates_mapping(self):
        shaper = self.make_shaper(mount_yaw_deg=90.0)
        self.engage(shaper)
        desired = shaper.desired_target((0.2, 1.2, -0.4), IDENTITY_QUAT)
        self.assertAlmostEqual(desired[0], 0.4, places=9)
        self.assertAlmostEqual(desired[1], 0.1, places=9)

    def test_workspace_clips_position(self):
        shaper = self.make_shaper()
        self.engage(shaper)
        # 手柄前方移动 1 m：期望目标超工作空间盒，裁剪到 x=0.6
        desired = shaper.desired_target((0.2, 1.2, -1.3), IDENTITY_QUAT)
        self.assertAlmostEqual(desired[0], 0.6, places=9)
        # 姿态不受裁剪影响
        self.assertEqual(desired[3:], START_POSE[3:])

    def test_workspace_clips_z_axis(self):
        shaper = self.make_shaper()
        self.engage(shaper)
        desired = shaper.desired_target((0.2, 6.0, -0.3), IDENTITY_QUAT)  # PICO +Y 向上
        self.assertAlmostEqual(desired[2], 0.9, places=9)
        desired = shaper.desired_target((0.2, -6.0, -0.3), IDENTITY_QUAT)
        self.assertAlmostEqual(desired[2], 0.05, places=9)

    def test_step_limits_position(self):
        shaper = self.make_shaper(max_pos_step_m=0.02)
        self.engage(shaper)
        target = shaper.next_target((0.2, 1.2, -1.3), IDENTITY_QUAT)  # 期望走 1 m
        self.assertAlmostEqual(target[0], 0.42, places=9)  # 只走 20 mm
        self.assertAlmostEqual(target[1], 0.0, places=9)
        self.assertAlmostEqual(target[2], 0.4, places=9)

    def test_step_limit_converges_to_desired(self):
        shaper = self.make_shaper(max_pos_step_m=0.02)
        self.engage(shaper)
        target = self.converge(shaper, (0.2, 1.2, -0.4), IDENTITY_QUAT)
        # 期望 +0.1 m，20 mm 步长需要 5 步
        self.assertAlmostEqual(target[0], 0.5, places=9)

    def _rotation_angle(self, target_a, target_b):
        r1 = frames.rotvec_to_matrix(target_a[3:])
        r2 = frames.rotvec_to_matrix(target_b[3:])
        rel = frames.mat_mul(r1, frames.mat_transpose(r2))
        rv = frames.matrix_to_rotvec(rel)
        return math.sqrt(sum(v * v for v in rv))

    def test_rot_scale_reduces_rotation(self):
        q = quat_about_y(math.radians(40))
        s1 = self.make_shaper()
        self.engage(s1)
        full = self.converge(s1, CONTROLLER_POSITION, q)
        s2 = self.make_shaper(rot_scale=0.5)
        self.engage(s2)
        half = self.converge(s2, CONTROLLER_POSITION, q)
        angle_full = self._rotation_angle(full, START_POSE)
        angle_half = self._rotation_angle(half, START_POSE)
        self.assertGreater(angle_full, 0.3)          # 40° ≈ 0.70 rad
        self.assertAlmostEqual(angle_half / angle_full, 0.5, places=2)

    def test_step_limits_rotation(self):
        shaper = self.make_shaper(max_rot_step_rad=0.05)
        self.engage(shaper)
        target = shaper.next_target(CONTROLLER_POSITION, quat_about_y(1.0))
        delta_rotation = frames.mat_mul(
            frames.rotvec_to_matrix(target[3:]),
            frames.mat_transpose(frames.rotvec_to_matrix(START_POSE[3:])))
        self.assertAlmostEqual(frames.rotation_angle(delta_rotation), 0.05, places=9)

    def test_rotation_composes_onto_start_pose(self):
        shaper = self.make_shaper(max_rot_step_rad=0.05)
        self.engage(shaper)
        target = self.converge(shaper, CONTROLLER_POSITION, quat_about_y(0.3))
        self.assertAlmostEqual(target[3], 0.0, places=8)
        self.assertAlmostEqual(target[4], 0.0, places=8)
        self.assertAlmostEqual(target[5], 0.3, places=8)

    def test_disengage_keeps_last_target_and_stops(self):
        shaper = self.make_shaper()
        self.engage(shaper)
        target = shaper.next_target((0.2, 1.2, -0.4), IDENTITY_QUAT)
        shaper.disengage()
        self.assertFalse(shaper.engaged)
        self.assertEqual(shaper.last_target, target)
        self.assertIsNone(shaper.next_target((0.2, 1.2, -0.9), IDENTITY_QUAT))
        self.assertEqual(shaper.last_target, target)

    def test_reengage_latches_new_start(self):
        shaper = self.make_shaper()
        self.engage(shaper)
        shaper.next_target((0.2, 1.2, -0.4), IDENTITY_QUAT)
        shaper.disengage()
        new_start = (0.5, 0.1, 0.45, 0.0, 0.0, 0.0)
        shaper.engage(new_start, (0.1, 1.1, -0.5), IDENTITY_QUAT, IDENTITY_QUAT)
        self.assertEqual(shaper.last_target, new_start)
        # 新起点之上再前进
        delta = shaper.desired_target((0.1, 1.1, -0.6), IDENTITY_QUAT)
        self.assertAlmostEqual(delta[0], 0.6, places=9)

    def test_tilted_head_raises_from_target(self):
        shaper = self.make_shaper()
        tilted_head = (math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))  # 低头 90°
        shaper.engage(START_POSE, CONTROLLER_POSITION, IDENTITY_QUAT, tilted_head)
        with self.assertRaises(ValueError):
            shaper.next_target((0.2, 1.2, -0.4), IDENTITY_QUAT)


class HelperFunctionTest(unittest.TestCase):
    def test_clip_pose_only_clips_position(self):
        pose = (1.0, -2.0, 3.0, 0.1, 0.2, 0.3)
        clipped = clip_pose(pose, (-0.5, -0.5, 0.0), (0.5, 0.5, 1.0))
        self.assertEqual(clipped, (0.5, -0.5, 1.0, 0.1, 0.2, 0.3))

    def test_clip_pose_rejects_wrong_dimension(self):
        with self.assertRaises(ValueError):
            clip_pose((0.0, 0.0, 0.0), (-1, -1, -1), (1, 1, 1))

    def test_limit_pose_step_within_limits_returns_desired(self):
        previous = (0.4, 0.0, 0.4, 0.0, 0.0, 0.0)
        desired = (0.41, 0.0, 0.4, 0.0, 0.0, 0.03)
        target = limit_pose_step(previous, desired, 0.02, 0.05)
        for actual, expected in zip(target, desired):
            self.assertAlmostEqual(actual, expected, places=9)

    def test_limit_pose_step_limits_both(self):
        previous = (0.4, 0.0, 0.4, 0.0, 0.0, 0.0)
        desired = (0.5, 0.0, 0.4, 0.0, 0.0, 1.0)
        target = limit_pose_step(previous, desired, 0.02, 0.05)
        self.assertAlmostEqual(math.sqrt(sum(
            (target[i] - previous[i]) ** 2 for i in range(3))), 0.02, places=9)
        self.assertAlmostEqual(target[5], 0.05, places=9)


if __name__ == "__main__":
    unittest.main()
