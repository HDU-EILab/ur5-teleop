"""``ur5_teleop.frames`` 纯数学单测（stdlib unittest）。"""

import math
import unittest

from ur5_teleop import frames

IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def quat_about(axis, angle):
    """绕给定轴的旋转四元数。"""
    norm = math.sqrt(sum(value * value for value in axis))
    kx, ky, kz = (value / norm for value in axis)
    half = 0.5 * angle
    sin_half = math.sin(half)
    return (kx * sin_half, ky * sin_half, kz * sin_half, math.cos(half))


def rot_x(angle):
    cos, sin = math.cos(angle), math.sin(angle)
    return ((1.0, 0.0, 0.0), (0.0, cos, -sin), (0.0, sin, cos))


def rot_y(angle):
    cos, sin = math.cos(angle), math.sin(angle)
    return ((cos, 0.0, sin), (0.0, 1.0, 0.0), (-sin, 0.0, cos))


class FrameMathTestCase(unittest.TestCase):
    """矩阵/向量断言的公共基类。"""

    def assert_matrix_almost(self, actual, expected, places=9):
        for row in range(3):
            for col in range(3):
                self.assertAlmostEqual(
                    actual[row][col], expected[row][col], places=places,
                    msg=f"矩阵元素 [{row}][{col}] 不一致")

    def assert_vector_almost(self, actual, expected, places=9):
        self.assertEqual(len(actual), len(expected))
        for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
            self.assertAlmostEqual(actual_value, expected_value, places=places,
                                   msg=f"分量 {index} 不一致")


class QuaternionMatrixTest(FrameMathTestCase):
    def test_identity_quaternion_is_identity_matrix(self):
        self.assert_matrix_almost(frames.quat_to_matrix(IDENTITY_QUAT), frames.IDENTITY_MATRIX)

    def test_quat_to_matrix_normalizes_input(self):
        # 非单位四元数（模长 2）也应得到单位旋转
        self.assert_matrix_almost(frames.quat_to_matrix((0.0, 0.0, 0.0, 2.0)),
                                  frames.IDENTITY_MATRIX)

    def test_quat_to_matrix_rotations(self):
        self.assert_matrix_almost(frames.quat_to_matrix(quat_about((1, 0, 0), 0.4)),
                                  rot_x(0.4))
        self.assert_matrix_almost(frames.quat_to_matrix(quat_about((0, 1, 0), 0.7)),
                                  rot_y(0.7))
        self.assert_matrix_almost(frames.quat_to_matrix(quat_about((0, 0, 1), -0.9)),
                                  frames.rot_z(-0.9))

    def test_matrix_quat_round_trip(self):
        cases = [
            frames.IDENTITY_MATRIX,
            frames.rot_z(0.9),
            rot_x(-1.2),
            rot_y(2.5),
            frames.mat_mul(frames.rot_z(0.3), rot_x(0.4)),
            frames.rotvec_to_matrix((math.pi, 0.0, 0.0)),
            frames.rotvec_to_matrix((0.0, math.pi, 0.0)),
            frames.rotvec_to_matrix((0.0, 0.0, math.pi)),
        ]
        for matrix in cases:
            with self.subTest(matrix=matrix):
                recovered = frames.matrix_to_quat(matrix)
                self.assert_matrix_almost(frames.quat_to_matrix(recovered), matrix)

    def test_matrix_to_quat_is_canonical(self):
        for angle in (0.5, 3.0, 4.0, 6.0):
            quat = frames.matrix_to_quat(frames.rot_z(angle))
            self.assertGreaterEqual(quat[3], 0.0, msg=f"angle={angle} 的 w 应为非负")

    def test_invalid_quaternion_raises(self):
        with self.assertRaises(ValueError):
            frames.quat_to_matrix((0.0, 0.0, 0.0, 0.0))


class RotvecTest(FrameMathTestCase):
    def test_identity_matrix_has_zero_rotvec(self):
        self.assert_vector_almost(frames.matrix_to_rotvec(frames.IDENTITY_MATRIX), (0.0, 0.0, 0.0))

    def test_rotvec_matrix_round_trip(self):
        cases = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.3),
            (0.3, -0.2, 0.5),
            (-1.0, 0.2, 0.1),
            (math.pi, 0.0, 0.0),
            (0.0, math.pi, 0.0),
            (0.0, 0.0, math.pi),
            (1e-9, 0.0, 0.0),
        ]
        for rotvec in cases:
            with self.subTest(rotvec=rotvec):
                matrix = frames.rotvec_to_matrix(rotvec)
                recovered = frames.matrix_to_rotvec(matrix)
                self.assert_matrix_almost(frames.rotvec_to_matrix(recovered), matrix)

    def test_rotvec_to_matrix_axis_angle(self):
        self.assert_matrix_almost(frames.rotvec_to_matrix((0.0, 0.0, 0.7)), frames.rot_z(0.7))
        self.assert_matrix_almost(frames.rotvec_to_matrix((0.7, 0.0, 0.0)), rot_x(0.7))

    def test_quat_rotvec_round_trip(self):
        for angle in (0.0, 0.01, 0.5, 2.0, math.pi - 1e-6):
            with self.subTest(angle=angle):
                quat = quat_about((0.0, 0.0, 1.0), angle)
                rotvec = frames.quat_to_rotvec(quat)
                self.assert_vector_almost(rotvec, (0.0, 0.0, angle), places=8)
                self.assert_matrix_almost(
                    frames.quat_to_matrix(frames.rotvec_to_quat(rotvec)),
                    frames.quat_to_matrix(quat))

    def test_rotation_angle(self):
        self.assertAlmostEqual(frames.rotation_angle(frames.rot_z(0.7)), 0.7, places=9)
        self.assertAlmostEqual(frames.rotation_angle(frames.rot_z(-0.7)), 0.7, places=9)
        self.assertAlmostEqual(
            frames.rotation_angle(frames.rotvec_to_matrix((math.pi, 0.0, 0.0))),
            math.pi, places=6)


class HeadingBasisTest(FrameMathTestCase):
    def test_identity_head_basis(self):
        self.assert_matrix_almost(
            frames.heading_basis(IDENTITY_QUAT),
            ((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))

    def test_controller_along_head_forward_maps_to_robot_plus_x(self):
        # VR 头显前方是本地 -Z：手柄往前移动（VR 世界 -Z）→ 机器人 +X
        delta = frames.robot_delta((0.0, 0.0, -1.0), (0.0, 0.0, 0.0), IDENTITY_QUAT)
        self.assert_vector_almost(delta, (1.0, 0.0, 0.0))

    def test_controller_right_maps_to_robot_minus_y(self):
        # VR +X 是右方 → UR 基座 +Y 是左方
        delta = frames.robot_delta((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), IDENTITY_QUAT)
        self.assert_vector_almost(delta, (0.0, -1.0, 0.0))

    def test_controller_up_maps_to_robot_plus_z(self):
        delta = frames.robot_delta((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), IDENTITY_QUAT)
        self.assert_vector_almost(delta, (0.0, 0.0, 1.0))

    def test_head_yaw_90_keeps_forward_mapping(self):
        head = quat_about((0.0, 1.0, 0.0), math.pi / 2)
        # 头显转向后其前方在 VR 世界是 -X；沿该方向移动仍映射到机器人 +X
        delta = frames.robot_delta((-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), head)
        self.assert_vector_almost(delta, (1.0, 0.0, 0.0))

    def test_mild_head_pitch_is_allowed(self):
        head = quat_about((1.0, 0.0, 0.0), math.radians(30.0))
        self.assert_vector_almost(
            frames.robot_delta((0.0, 0.0, -1.0), (0.0, 0.0, 0.0), head),
            (1.0, 0.0, 0.0))

    def test_head_tilted_too_far_raises(self):
        head = quat_about((1.0, 0.0, 0.0), math.pi / 2)  # 低头 90°
        with self.assertRaises(ValueError):
            frames.heading_basis(head)
        with self.assertRaises(ValueError):
            frames.robot_delta((0.0, 0.0, -1.0), (0.0, 0.0, 0.0), head)

    def test_mount_yaw_compensation(self):
        # 配置 90°：操作者前方在基座系 +Y 方向
        delta = frames.robot_delta((0.0, 0.0, -1.0), (0.0, 0.0, 0.0),
                                   IDENTITY_QUAT, mount_yaw_deg=90.0)
        self.assert_vector_almost(delta, (0.0, 1.0, 0.0))
        # 0° 与不传参数等价
        self.assert_matrix_almost(
            frames.heading_basis(IDENTITY_QUAT, mount_yaw_deg=0.0),
            frames.heading_basis(IDENTITY_QUAT))


class RobotRotationTest(FrameMathTestCase):
    def test_identity_controller_motion_gives_identity_rotation(self):
        self.assert_matrix_almost(
            frames.robot_rotation(IDENTITY_QUAT, IDENTITY_QUAT, IDENTITY_QUAT),
            frames.IDENTITY_MATRIX)

    def test_pico_yaw_maps_to_robot_base_z_yaw(self):
        current = quat_about((0.0, 1.0, 0.0), 0.3)
        rotation = frames.robot_rotation(current, IDENTITY_QUAT, IDENTITY_QUAT)
        self.assert_matrix_almost(rotation, frames.rot_z(0.3))

    def test_pico_pitch_maps_to_robot_negative_axis(self):
        # VR +X（右）在基座系是 -Y，绕 VR +X 转正角 → 绕基座 -Y 转
        current = quat_about((1.0, 0.0, 0.0), 0.3)
        rotation = frames.robot_rotation(current, IDENTITY_QUAT, IDENTITY_QUAT)
        self.assert_matrix_almost(rotation, rot_y(-0.3))

    def test_rotation_uses_head_basis_locked_at_clutch(self):
        head = quat_about((0.0, 1.0, 0.0), math.pi / 2)
        current = quat_about((0.0, 1.0, 0.0), 0.2)
        rotation = frames.robot_rotation(current, IDENTITY_QUAT, head)
        # 头显转 90° 后，手柄绕自己固有的 +Y（此时是 VR -X）转，
        # 但 heading basis 已把 VR 世界轴统一映射：仍是基座 +Z 转 0.2
        self.assert_matrix_almost(rotation, frames.rot_z(0.2))


class ComposePoseTest(FrameMathTestCase):
    def test_translation_only(self):
        pose = frames.compose_pose((0.4, 0.0, 0.4, 0.0, 0.0, 0.0), (0.1, -0.05, 0.02))
        self.assert_vector_almost(pose[:3], (0.5, -0.05, 0.42), places=9)
        self.assert_vector_almost(pose[3:], (0.0, 0.0, 0.0), places=9)

    def test_rotation_is_left_multiplied(self):
        pose = frames.compose_pose((0.4, 0.0, 0.4, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                                   frames.rot_z(math.pi / 2))
        self.assert_vector_almost(pose[3:], (0.0, 0.0, math.pi / 2), places=9)
        self.assert_vector_almost(pose[:3], (0.4, 0.0, 0.4), places=9)

    def test_rotation_applies_in_base_frame(self):
        start_rotation = frames.rotvec_to_matrix((0.0, 0.0, 1.0))
        pose = frames.compose_pose((0, 0, 0, 0.0, 0.0, 1.0), (0, 0, 0),
                                   rot_x(0.2))
        self.assert_matrix_almost(frames.rotvec_to_matrix(pose[3:]),
                                  frames.mat_mul(rot_x(0.2), start_rotation))

    def test_start_rotation_preserved_without_rotation_delta(self):
        start = (0.4, 0.0, 0.4, 0.0, math.pi / 2, 0.0)
        pose = frames.compose_pose(start, (0.01, 0.02, 0.03))
        self.assert_matrix_almost(frames.rotvec_to_matrix(pose[3:]),
                                  frames.rotvec_to_matrix(start[3:]))

    def test_wrong_dimensions_raise(self):
        with self.assertRaises(ValueError):
            frames.compose_pose((0.4, 0.0, 0.4), (0.0, 0.0, 0.0))
        self.assertRaises(ValueError, frames.compose_pose,
                          (0.4, 0.0, 0.4, 0.0, 0.0, 0.0), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
