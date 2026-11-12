"""纯函数坐标/位姿数学：四元数、旋转矩阵、轴角、PICO→UR 基座系映射。

约定（与 docs/design.md §2 一致）：

- 四元数一律为 ``(x, y, z, w)``（OpenXR / PICO 顺序）。
- 旋转矩阵是 3x3 行优先 tuple-of-tuple（沿用 upstream 项目的纯 stdlib 风格），
  作用方式 ``v_out = R @ v_in``。
- 旋转向量（轴角）为 ``(rx, ry, rz)``：方向是转轴，模长是转角（rad），
  与 UR ``actual_TCP_pose`` 的旋转表示一致。
- PICO 世界系：+Y 向上，头显本地前方是 -Z；
  UR 基座系：+X 前、+Y 左、+Z 上（右手系）。
- ``heading_basis()`` 输出的矩阵 M 满足 ``v_ur = M @ v_pico``，
  即把“操作者前方/左方/上方”旋到 UR 基座系。

映射实现对照 upstream ``pico-controller-bridge/dry_run.py`` 的
``_heading_basis`` / ``robot_delta`` / ``robot_rotation``，数值语义保持一致，
额外补了完整四元数/矩阵/轴角互转与 ``mount_yaw_deg`` 装夹偏航补偿。
所有函数只依赖标准库，出错抛 ``ValueError``。
"""

from __future__ import annotations

import math

Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
Quaternion = tuple[float, float, float, float]
Vector3 = tuple[float, float, float]
Pose6 = tuple[float, float, float, float, float, float]

IDENTITY_MATRIX: Matrix3 = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

#: 判定“小角/零模长”的阈值
_EPS = 1e-12


def mat_mul(left: Matrix3, right: Matrix3) -> Matrix3:
    """3x3 矩阵乘法。"""
    return tuple(
        tuple(sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )


def mat_transpose(matrix: Matrix3) -> Matrix3:
    """3x3 矩阵转置。"""
    return tuple(tuple(matrix[row][col] for row in range(3)) for col in range(3))


def mat_vec(matrix: Matrix3, vector) -> Vector3:
    """3x3 矩阵乘 3 维列向量。"""
    x, y, z = (float(value) for value in vector)
    return tuple(
        matrix[row][0] * x + matrix[row][1] * y + matrix[row][2] * z
        for row in range(3)
    )


def rot_z(angle_rad: float) -> Matrix3:
    """绕 +Z 旋转的矩阵。"""
    cos = math.cos(angle_rad)
    sin = math.sin(angle_rad)
    return (
        (cos, -sin, 0.0),
        (sin, cos, 0.0),
        (0.0, 0.0, 1.0),
    )


def _normalize_quaternion(quaternion) -> Quaternion:
    """归一化四元数；模长接近 0 时抛 ValueError。"""
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-8:
        raise ValueError("无效的四元数（模长接近 0）")
    return (x / norm, y / norm, z / norm, w / norm)


def quat_to_matrix(quaternion) -> Matrix3:
    """四元数 → 旋转矩阵（内部先归一化，输入不必是单位四元数）。"""
    x, y, z, w = _normalize_quaternion(quaternion)
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def matrix_to_quat(matrix: Matrix3) -> Quaternion:
    """旋转矩阵 → 四元数，输出归一化且 w >= 0（转角取主值 0..pi）。"""
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        quat = ((m21 - m12) / scale, (m02 - m20) / scale,
                (m10 - m01) / scale, 0.25 * scale)
    elif m00 > m11 and m00 > m22:
        scale = 2.0 * math.sqrt(max(1.0 + m00 - m11 - m22, _EPS))
        quat = (0.25 * scale, (m01 + m10) / scale,
                (m02 + m20) / scale, (m21 - m12) / scale)
    elif m11 > m22:
        scale = 2.0 * math.sqrt(max(1.0 + m11 - m00 - m22, _EPS))
        quat = ((m01 + m10) / scale, 0.25 * scale,
                (m12 + m21) / scale, (m02 - m20) / scale)
    else:
        scale = 2.0 * math.sqrt(max(1.0 + m22 - m00 - m11, _EPS))
        quat = ((m02 + m20) / scale, (m12 + m21) / scale,
                0.25 * scale, (m10 - m01) / scale)
    x, y, z, w = _normalize_quaternion(quat)
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    return (x, y, z, w)


def quat_to_rotvec(quaternion) -> Vector3:
    """四元数 → 旋转向量（主值转角 0..pi）。"""
    x, y, z, w = _normalize_quaternion(quaternion)
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    sin_half = math.sqrt(x * x + y * y + z * z)
    if sin_half < _EPS:
        return (0.0, 0.0, 0.0)
    scale = 2.0 * math.atan2(sin_half, w) / sin_half
    return (x * scale, y * scale, z * scale)


def rotvec_to_quat(rotvec) -> Quaternion:
    """旋转向量 → 四元数（小角退化为单位四元数）。"""
    rx, ry, rz = (float(value) for value in rotvec)
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < _EPS:
        return (0.0, 0.0, 0.0, 1.0)
    half = 0.5 * angle
    scale = math.sin(half) / angle
    return (rx * scale, ry * scale, rz * scale, math.cos(half))


def rotvec_to_matrix(rotvec) -> Matrix3:
    """旋转向量 → 旋转矩阵（Rodrigues 公式；小角退化为单位矩阵）。"""
    rx, ry, rz = (float(value) for value in rotvec)
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < _EPS:
        return IDENTITY_MATRIX
    kx, ky, kz = rx / angle, ry / angle, rz / angle
    cos = math.cos(angle)
    sin = math.sin(angle)
    one_minus_cos = 1.0 - cos
    return (
        (kx * kx * one_minus_cos + cos, kx * ky * one_minus_cos - kz * sin,
         kx * kz * one_minus_cos + ky * sin),
        (ky * kx * one_minus_cos + kz * sin, ky * ky * one_minus_cos + cos,
         ky * kz * one_minus_cos - kx * sin),
        (kz * kx * one_minus_cos - ky * sin, kz * ky * one_minus_cos + kx * sin,
         kz * kz * one_minus_cos + cos),
    )


def matrix_to_rotvec(matrix: Matrix3) -> Vector3:
    """旋转矩阵 → 旋转向量。"""
    return quat_to_rotvec(matrix_to_quat(matrix))


def heading_basis(head_quaternion, mount_yaw_deg: float = 0.0) -> Matrix3:
    """头显水平朝向基：把 PICO 世界向量旋到 UR 基座系。

    取头显前方 (0, 0, -1) 的水平投影为机器人 +X（前方），其左方为 +Y，
    PICO +Y（上）为 +Z；输出矩阵 M 满足 ``v_ur = M @ v_pico``。
    头显俯仰超过 60 度（水平投影 < 0.5）时无法定义前方，抛 ValueError。

    mount_yaw_deg：操作者前方在 UR 基座系中的偏航角（度，绕基座 +Z 逆时针
    为正）。基座相对操作者装了偏航角时，整段映射再左乘 Rz(mount_yaw_deg)，
    即配置值就是“操作者朝向在基座系里的角度”。
    """
    matrix = quat_to_matrix(head_quaternion)
    # 头显本地前方 (0, 0, -1) 旋到 PICO 世界 = -R 的第三列
    forward_x = -matrix[0][2]
    forward_z = -matrix[2][2]
    horizontal = math.hypot(forward_x, forward_z)
    if horizontal < 0.5:
        raise ValueError("头显朝向过度倾斜，无法定义前方")
    forward_x /= horizontal
    forward_z /= horizontal
    # 右手系：right = forward x up；UR 基座 +Y 是左方，所以取 -right
    basis: Matrix3 = (
        (forward_x, 0.0, forward_z),
        (forward_z, 0.0, -forward_x),
        (0.0, 1.0, 0.0),
    )
    if mount_yaw_deg:
        basis = mat_mul(rot_z(math.radians(float(mount_yaw_deg))), basis)
    return basis


def robot_delta(current, origin, head_quaternion, mount_yaw_deg: float = 0.0) -> Vector3:
    """手柄位移 → UR 基座系位移（用离合按下瞬间锁存的头显朝向 basis）。"""
    dx = float(current[0]) - float(origin[0])
    dy = float(current[1]) - float(origin[1])
    dz = float(current[2]) - float(origin[2])
    basis = heading_basis(head_quaternion, mount_yaw_deg)
    return (
        basis[0][0] * dx + basis[0][1] * dy + basis[0][2] * dz,
        basis[1][0] * dx + basis[1][1] * dy + basis[1][2] * dz,
        basis[2][0] * dx + basis[2][1] * dy + basis[2][2] * dz,
    )


def robot_rotation(current_quaternion, origin_quaternion, head_quaternion,
                   mount_yaw_deg: float = 0.0) -> Matrix3:
    """手柄姿态变化 → UR 基座系旋转增量。

    先算 PICO 世界系的相对旋转 ``R_now @ R_origin^T``，再用同一 heading basis
    共轭到基座系：``B @ delta @ B^T``。返回值左乘到起始位姿的旋转上。
    """
    basis = heading_basis(head_quaternion, mount_yaw_deg)
    delta = mat_mul(quat_to_matrix(current_quaternion),
                    mat_transpose(quat_to_matrix(origin_quaternion)))
    return mat_mul(basis, mat_mul(delta, mat_transpose(basis)))


def compose_pose(start_pose, delta_xyz, rotation: Matrix3 | None = None) -> Pose6:
    """目标位姿 = 起始位姿 ⊕ (位移增量, 旋转增量)。

    ``delta_xyz`` 是 UR 基座系位移（m），``rotation`` 是基座系旋转增量矩阵，
    左乘到起始旋转上（与 upstream ``solve_position`` 的语义一致）。
    """
    start = tuple(float(value) for value in start_pose)
    if len(start) != 6:
        raise ValueError(f"起始位姿必须是 6 维 [x,y,z,rx,ry,rz]，收到 {len(start)} 维")
    delta = tuple(float(value) for value in delta_xyz)
    if len(delta) != 3:
        raise ValueError(f"位移增量必须是 3 维 [x,y,z]，收到 {len(delta)} 维")
    position = (
        start[0] + delta[0],
        start[1] + delta[1],
        start[2] + delta[2],
    )
    start_rotation = rotvec_to_matrix(start[3:])
    target_rotation = start_rotation if rotation is None else mat_mul(rotation, start_rotation)
    return position + matrix_to_rotvec(target_rotation)


def rotation_angle(rotation: Matrix3) -> float:
    """旋转矩阵对应的最小转角（rad，0..pi）。"""
    rx, ry, rz = matrix_to_rotvec(rotation)
    return math.sqrt(rx * rx + ry * ry + rz * rz)
