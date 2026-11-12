"""自动回 home 路径生成单测（纯数学，不连机器人）。"""

import math
import unittest

from ur5_teleop import frames
from ur5_teleop.home import build_home_path, build_movej_program, home_tcp_pose


class MovejProgramTest(unittest.TestCase):
    def test_movej_program_contains_home_joints_in_radians(self):
        program = build_movej_program((-180, -90, -90, -90, 90, -90))
        self.assertIn("movej([", program)
        self.assertIn("go_home", program)
        # -180° = -π, +90° = +π/2，保持 6 个关节
        self.assertIn("-3.14159265", program)
        self.assertIn("1.57079633", program)
        self.assertEqual(program.count(","), 7)  # 6 个关节 + a/v 参数各 1 个逗号
        # CB3 限制：不能出现 for/abs/max 等（编译不过）
        for word in (" for ", "abs(", "max("):
            self.assertNotIn(word, program)


class HomePathTest(unittest.TestCase):
    def test_fk_matches_known_home_tcp(self):
        pose = home_tcp_pose()
        expected = (-0.4869, 0.1091, 0.4319)
        for actual, want in zip(pose[:3], expected):
            self.assertAlmostEqual(actual, want, places=3)
        # 姿态：绕 Y 轴 π（等价 ±π）
        rv = pose[3:6]
        self.assertAlmostEqual(abs(rv[1]), math.pi, places=2)

    def test_three_phases_lift_sweep_lower(self):
        current = (0.40, -0.20, 0.40, 0.0, 0.3, 0.0)
        home = home_tcp_pose()
        lift = 0.15
        path = build_home_path(current, home, lift_m=lift, speed_mps=0.10, dt=0.008)
        self.assertGreater(len(path), 50)
        poses = [p for p, _ in path]
        times = [t for _, t in path]
        # 时间单调
        self.assertTrue(all(b > a for a, b in zip(times, times[1:])))
        z_safe = max(current[2], home[2]) + lift
        # 起点 = 当前位姿；抬升段 z 单调升到 z_safe
        self.assertAlmostEqual(poses[0][2], current[2], delta=0.005)
        self.assertAlmostEqual(max(p[2] for p in poses), z_safe, places=6)
        first_safe = next(i for i, p in enumerate(poses) if abs(p[2] - z_safe) < 1e-6)
        lift_zs = [p[2] for p in poses[:first_safe + 1]]
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(lift_zs, lift_zs[1:])))
        # 中间段处于安全高度
        self.assertAlmostEqual(poses[len(poses) // 2][2], z_safe, places=6)
        # 终点 = home
        for actual, want in zip(poses[-1][:3], home[:3]):
            self.assertAlmostEqual(actual, want, places=6)
        # 每步位移不超过速度上限的 1.5 倍
        max_step = 0.10 * 0.008 * 1.5
        for a, b in zip(poses, poses[1:]):
            self.assertLessEqual(math.dist(a[:3], b[:3]), max_step + 1e-9)

    def test_end_orientation_matches_home(self):
        current = (0.40, -0.20, 0.40, 0.2, -0.4, 0.1)
        home = home_tcp_pose()
        path = build_home_path(current, home, lift_m=0.15, speed_mps=0.1, dt=0.008)
        end_rot = frames.rotvec_to_matrix(path[-1][0][3:6])
        home_rot = frames.rotvec_to_matrix(home[3:6])
        rel = frames.mat_mul(frames.mat_transpose(end_rot), home_rot)
        angle = math.sqrt(sum(v * v for v in frames.matrix_to_rotvec(rel)))
        self.assertLess(angle, 1e-3)


if __name__ == "__main__":
    unittest.main()
