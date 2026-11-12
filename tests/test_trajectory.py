"""离线关节轨迹部署测试（不需要机器人）。

覆盖：
1. cvpr-部署UR5 数据包结构与校验规则（行数、周期、起始位姿、速度上限、SHA256）
2. 播放器流程：movej 起始位姿 → 125 Hz 关节流式发送；安全/跟踪误差/未跟随 三种中止
3. 数据损坏场景（时间网格、关节限位、速度、哈希）必须被拒绝

播放器用虚拟时钟 + FakeBackend/EchoBackend，整段 55 s 轨迹毫秒级跑完。
"""

from __future__ import annotations

import hashlib
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import play_trajectory as pt  # noqa: E402

from ur5_teleop.backend import FakeBackend  # noqa: E402


def write_npy(path: Path, rows: Sequence[float] | Sequence[Sequence[float]]) -> None:
    """写 float64、C 序的 .npy（1D/2D，测试用，避免依赖 numpy）。"""
    if rows and not isinstance(rows[0], (tuple, list)):
        shape = (len(rows),)  # type: ignore[assignment]
        flat = [float(v) for v in rows]  # type: ignore[arg-type]
    else:
        shape = (len(rows), len(rows[0]))  # type: ignore[arg-type]
        flat = [float(v) for row in rows for v in row]  # type: ignore[union-attr]
    header = "{'descr': '<f8', 'fortran_order': False, 'shape': %r, }" % (shape,)
    pad = (64 - (10 + len(header) + 1) % 64) % 64
    header = header + " " * pad + "\n"
    body = struct.pack(f"<{len(flat)}d", *flat)
    path.write_bytes(
        b"\x93NUMPY" + bytes([1, 0]) + struct.pack("<H", len(header))
        + header.encode("latin1") + body)


HOME = (0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0)


class TrajectoryDataTest(unittest.TestCase):
    """真实数据包：cvpr-部署UR5/。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tr = pt.load_trajectory(pt.DEFAULT_FOLDER)

    def test_rows_and_duration(self) -> None:
        self.assertEqual(self.tr.rows, 6859)
        self.assertAlmostEqual(self.tr.dt, 0.008, places=12)
        self.assertAlmostEqual(self.tr.duration_s, 54.864, places=6)

    def test_start_pose_is_ur5_home(self) -> None:
        for got, want in zip(self.tr.q[0], HOME):
            self.assertAlmostEqual(got, want, places=9)

    def test_period_is_125hz(self) -> None:
        self.assertAlmostEqual(1.0 / self.tr.dt, 125.0, places=6)

    def test_max_joint_speed_within_deployment_limit(self) -> None:
        speed = self.tr.max_joint_speed()
        self.assertLess(speed, pt.MAX_JOINT_SPEED_RAD_S)
        self.assertAlmostEqual(speed, 0.0216, places=3)

    def test_manifest_matches_data(self) -> None:
        manifest = self.tr.manifest
        self.assertEqual(manifest.get("joint_order"),
                         ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"])
        self.assertAlmostEqual(manifest["command_duration_s"], self.tr.duration_s, places=6)

    def test_j1_offset_matches_installed_home(self) -> None:
        adjusted = self.tr.with_j1_offset(-180.0)
        for got, want in zip(adjusted.q[0], (-math.pi, -math.pi / 2, math.pi / 2,
                                           -math.pi / 2, -math.pi / 2, 0.0)):
            self.assertAlmostEqual(got, want, places=9)
        # j2..j6 不变，速度不变
        for index in range(1, 6):
            self.assertEqual(adjusted.q[0][index], self.tr.q[0][index])
        self.assertAlmostEqual(adjusted.max_joint_speed(), self.tr.max_joint_speed(), places=12)
        # 越界偏移必须拒绝
        with self.assertRaises(pt.TrajectoryError):
            self.tr.with_j1_offset(360.0)

    def test_npy_loader_round_trip(self) -> None:
        rows = [(1.0, 2.0), (3.0, 4.0)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.npy"
            write_npy(path, rows)
            self.assertEqual(pt._load_npy(path), [(1.0, 2.0), (3.0, 4.0)])


class HomeRemapTest(unittest.TestCase):
    """把数据包 home 映射到现场约定 home 的两种方式。"""

    NEW_HOME = (-180.0, -90.0, -90.0, -90.0, 90.0, -90.0)

    @classmethod
    def setUpClass(cls) -> None:
        cls.tr = pt.load_trajectory(pt.DEFAULT_FOLDER)

    def test_parse_home_joints(self) -> None:
        self.assertEqual(pt._parse_home_joints("-180,-90,-90,-90,90,-90"),
                         self.NEW_HOME)
        self.assertEqual(pt._parse_home_joints("-180 -90 -90 -90 90 -90"),
                         self.NEW_HOME)
        with self.assertRaises(pt.TrajectoryError):
            pt._parse_home_joints("1,2,3")
        with self.assertRaises(pt.TrajectoryError):
            pt._parse_home_joints("a,b,c,d,e,f")

    def test_parse_args_accepts_negative_home_joints(self) -> None:
        args = pt._parse_args(["--dry-run", "--home-joints", "-180,-90,-90,-90,90,-90",
                               "--remap", "joint"])
        self.assertEqual(pt._parse_home_joints(args.home_joints), self.NEW_HOME)
        self.assertEqual(args.remap, "joint")

    def test_joint_offset_maps_home(self) -> None:
        offsets = [self.NEW_HOME[i] - math.degrees(self.tr.q[0][i]) for i in range(6)]
        remapped = self.tr.with_joint_offsets(offsets)
        for got, want in zip(remapped.q[0], self.NEW_HOME):
            self.assertAlmostEqual(math.degrees(got), want, places=9)
        # 关节增量保持不变
        self.assertAlmostEqual(remapped.max_joint_speed(), self.tr.max_joint_speed(), places=12)
        self.assertEqual(remapped.rows, self.tr.rows)

    def test_joint_offset_out_of_limit_rejected(self) -> None:
        with self.assertRaises(pt.TrajectoryError):
            self.tr.with_joint_offsets([360.0, 0, 0, 0, 0, 0])


class BadDataTest(unittest.TestCase):
    """损坏/异常数据包必须被拒绝。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, q, t) -> None:
        write_npy(self.folder / "q_command_rad.npy", q)
        write_npy(self.folder / "t_command_s.npy", t)

    def test_missing_files(self) -> None:
        with self.assertRaises(pt.TrajectoryError):
            pt.load_trajectory(self.folder)

    def test_uneven_time_grid(self) -> None:
        self._write([HOME, HOME, HOME], [0.0, 0.008, 0.02])
        with self.assertRaisesRegex(pt.TrajectoryError, "不均匀|周期"):
            pt.load_trajectory(self.folder)

    def test_joint_limit_violation(self) -> None:
        bad = (7.0, -1.5, 1.5, -1.5, -1.5, 0.0)  # 7 rad > 360°
        self._write([HOME, bad], [0.0, 0.008])
        with self.assertRaisesRegex(pt.TrajectoryError, "360"):
            pt.load_trajectory(self.folder)

    def test_joint_speed_too_high(self) -> None:
        fast = (0.01, -1.5, 1.5, -1.5, -1.5, 0.0)  # 0.0125 rad / 8 ms = 1.56 rad/s
        self._write([HOME, fast], [0.0, 0.008])
        with self.assertRaisesRegex(pt.TrajectoryError, "速度"):
            pt.load_trajectory(self.folder)

    def test_time_not_starting_at_zero(self) -> None:
        self._write([HOME, HOME], [1.0, 1.008])
        with self.assertRaisesRegex(pt.TrajectoryError, "从 0 开始"):
            pt.load_trajectory(self.folder)

    def test_sha256_mismatch(self) -> None:
        self._write([HOME, HOME], [0.0, 0.008])
        digest = hashlib.sha256()
        with (self.folder / "q_command_rad.npy").open("rb") as handle:
            digest.update(handle.read())
        (self.folder / "SHA256SUMS").write_text(
            f"{'0' * 64}  q_command_rad.npy\n", encoding="utf-8")
        with self.assertRaisesRegex(pt.TrajectoryError, "SHA256"):
            pt.load_trajectory(self.folder)


class TcpAlignTest(unittest.TestCase):
    """TCP 刚体变换（不改安装配置）的数学与对齐逻辑。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tr = pt.load_trajectory(pt.DEFAULT_FOLDER)

    def test_fk_matches_nominal_home(self) -> None:
        matrix = pt.ur5_fk(self.tr.q[0])
        self.assertAlmostEqual(matrix[0][3], -0.4869, places=3)
        self.assertAlmostEqual(matrix[1][3], -0.1091, places=3)
        self.assertAlmostEqual(matrix[2][3], 0.4319, places=3)

    def test_rotvec_round_trip(self) -> None:
        for rv in ((0.1, -0.2, 0.3), (1.2, 0.4, -0.7), (0.0, 0.0, 0.0)):
            rot = pt._rotvec_to_rot(rv)
            back = pt._rot_to_rotvec(rot)
            for got, want in zip(back, rv):
                self.assertAlmostEqual(got, want, places=9)

    def test_align_start_pose_and_rigidity(self) -> None:
        start = pt._pose_from_matrix(pt.ur5_fk(self.tr.q[0]))
        target = pt.shifted_start_pose(start, 0.10, 0.18)
        aligned = pt.align_trajectory_to_start(self.tr, target)
        for got, want in zip(aligned.poses[0], target):
            self.assertAlmostEqual(got, want, places=6)   # 轴角反解有 1e-8 量级数值误差
        # 刚体变换保持点间距离：变换后首末距离 == 原 TCP 首末距离
        original_first = [pt.ur5_fk(self.tr.q[0])[i][3] for i in range(3)]
        original_last = [pt.ur5_fk(self.tr.q[-1])[i][3] for i in range(3)]
        aligned_last = aligned.poses[-1][:3]
        self.assertAlmostEqual(math.dist(aligned.poses[0][:3], aligned_last),
                               math.dist(original_first, original_last), places=6)
        self.assertEqual(aligned.rows, self.tr.rows)

    def test_shifted_start_moves_inward_and_down(self) -> None:
        current = (0.4, 0.2, 0.8, 0.0, 0.0, 0.0)
        shifted = pt.shifted_start_pose(current, 0.10, 0.18)
        self.assertAlmostEqual(shifted[2], 0.62, places=9)
        self.assertLess(math.hypot(shifted[0], shifted[1]),
                        math.hypot(current[0], current[1]))
        self.assertEqual(shifted[3:], current[3:])

    def test_tcp_player_dry_run(self) -> None:
        start = pt._pose_from_matrix(pt.ur5_fk(self.tr.q[0]))
        aligned = pt.align_trajectory_to_start(self.tr, pt.shifted_start_pose(start, 0.1, 0.18))
        backend = pt._EchoBackend(self.tr.q[0])
        backend.connect()
        backend.feed_state(pose=start)
        clock = pt._VirtualClock()
        player = pt.TcpPlayer(backend, aligned, clock=clock, sleep=clock.sleep,
                              progress_period_s=1e9)
        player.lead_in(seconds=2.0)
        stats = player.play(max_seconds=0.4)
        self.assertEqual(stats.rows, 51)
        self.assertLess(stats.max_error_rad, 1e-9)  # 完美跟随
        servo = [pose for pose, flag, _ in backend.sent if flag]
        self.assertGreaterEqual(len(servo), 51)


class PlayerTest(unittest.TestCase):
    """播放器流程与中止逻辑（虚拟时钟，毫秒级）。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tr = pt.load_trajectory(pt.DEFAULT_FOLDER)

    def _make_player(self, backend, **kwargs):
        clock = pt._VirtualClock()
        return pt.TrajectoryPlayer(
            backend, self.tr, clock=clock, sleep=clock.sleep,
            progress_period_s=1e9, **kwargs)

    def test_move_and_stream_frames_in_order(self) -> None:
        backend = pt._EchoBackend(self.tr.q[0])
        backend.connect()
        player = self._make_player(backend)
        player.move_to_start()
        self.assertTrue(backend.move_start_sent, "应至少发一次 cmd=3")
        stats = player.play(max_seconds=0.4)
        self.assertEqual(stats.rows, 51)  # t <= 0.4 s 共 51 帧
        self.assertLess(math.degrees(stats.max_error_rad), 1e-6)

        servo_frames = [joints for joints, servo, _ in backend.joint_sent if servo]
        self.assertGreaterEqual(len(servo_frames), 51)
        for index in range(51):
            self.assertEqual(servo_frames[index], self.tr.q[index])

    def test_lead_in_streams_from_current_pose(self) -> None:
        start = list(self.tr.q[0])
        start[2] += 0.3  # 假设机器人现在偏离起始位姿 0.3 rad（j3）
        backend = pt._EchoBackend(start)
        backend.connect()
        player = self._make_player(backend, max_track_error_rad=0.2)
        player.lead_in(min_duration_s=2.0)
        self.assertEqual(backend.move_start_sent, [], "lead-in 不应发 cmd=3（不依赖 movej）")
        servo_frames = [joints for joints, servo, _ in backend.joint_sent if servo]
        self.assertGreater(len(servo_frames), 100)
        for got, want in zip(servo_frames[-1], self.tr.q[0]):
            self.assertAlmostEqual(got, want, places=9)

    def test_aborts_when_safety_not_normal(self) -> None:
        backend = FakeBackend(initial_joints=self.tr.q[0], safety_status=4)
        backend.connect()
        player = self._make_player(backend)
        with self.assertRaisesRegex(pt.PlaybackError, "安全状态"):
            player.play(max_seconds=0.1)

    def test_aborts_on_tracking_error(self) -> None:
        backend = FakeBackend(initial_joints=self.tr.q[0])
        backend.connect()
        player = self._make_player(backend, max_track_error_rad=math.radians(1.5))
        with self.assertRaisesRegex(pt.PlaybackError, "跟踪误差"):
            player.play(max_seconds=6.0)

    def test_aborts_when_robot_not_following(self) -> None:
        backend = FakeBackend(initial_joints=self.tr.q[0])
        backend.connect()
        player = self._make_player(backend, max_track_error_rad=10.0)
        with self.assertRaisesRegex(pt.PlaybackError, "没有跟随"):
            player.play(max_seconds=15.0)

    def test_scale_speeds_up_playback(self) -> None:
        backend = pt._EchoBackend(self.tr.q[0])
        backend.connect()
        player = self._make_player(backend, scale=20.0)
        stats = player.play(max_seconds=2.0)
        self.assertEqual(stats.rows, 251)  # t <= 2 s 共 251 帧
        # 20 倍速下实际耗时约为标称的 1/20
        self.assertLess(stats.wall_s, 2.0 / 20.0 + 1.0)


if __name__ == "__main__":
    unittest.main()
