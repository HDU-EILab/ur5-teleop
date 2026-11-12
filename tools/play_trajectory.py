#!/usr/bin/env python3
"""在真机上复现离线关节轨迹（cvpr-部署UR5 数据包）。

数据：q_command_rad.npy（6859×6，rad，125 Hz）+ t_command_s.npy，总时长 54.864 s。
动作：从 UR5 home 姿出发、慢速小范围关节运动（TCP 路径约 31 cm，最大 TCP 速度 1.07 cm/s）。

流程：
1. 校验数据包（SHA256、时间网格、关节限位、速度上限、起始位姿）
2. Dashboard / RTDE 预检（RUNNING + NORMAL）
3. 下发 urscript/ur5_traj.script（下发后立即关闭脚本通道，程序继续运行）
4. cmd=3：movej 到轨迹首帧关节位姿（低速 0.15 rad/s），等待到位
5. cmd=2：按标称 125 Hz 流式发送 6859 帧关节目标（可用 --scale 改变速度）
6. cmd=0：idle，dashboard stop，打印跟踪误差统计

用法：
    # 空跑：不连机器人，验证数据包与播放流程
    python3 tools/play_trajectory.py --dry-run

    # 真机：默认先 movej 到起始位姿，再完整播放 55 s 轨迹
    python3 tools/play_trajectory.py --config config/ur5.toml --yes

    # 首次现场：只走到起始位姿，人工检查后再播放
    python3 tools/play_trajectory.py --config config/ur5.toml --yes --move-only

    # 只播放前 10 s（轨迹时间）
    python3 tools/play_trajectory.py --config config/ur5.toml --yes --skip-move --max-seconds 10

安全约定：
- 不自动上电/松闸/解锁保护停；机器人必须在 RUNNING + NORMAL 下运行。
- movej 与播放期间人手不离急停；轨迹来自未标定的标称 UR5 模型（最小离障间隙 1.1 cm），
  现场工装/工具/线缆可能与模型不同，先低速、先 `--move-only` 检查。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ur5_teleop import config as config_module  # noqa: E402
from ur5_teleop.backend import (  # noqa: E402
    BackendError,
    DisconnectedError,
    FakeBackend,
)
from ur5_teleop.ur_client import (  # noqa: E402
    CMD_IDLE,
    CMD_JOINT_SERVO,
    CMD_MOVE_START,
    ROBOT_MODE_NAMES,
    SAFETY_MODE_NAMES,
    DashboardClient,
    RTDEBackend,
    open_script_channel,
    send_program,
)

#: cvpr-部署UR5 数据包（q_command_rad.npy + t_command_s.npy）
DEFAULT_FOLDER = ROOT / "cvpr-部署UR5"
#: 机器人端关节轨迹播放程序
DEFAULT_SCRIPT = ROOT / "urscript" / "ur5_traj.script"

MODE_RUNNING = 7
SAFETY_NORMAL = 1
#: 数据包标称控制周期（manifest.assumed_period_s）
NOMINAL_PERIOD_S = 0.008
#: 关节限位（UR5 各关节 ±360°）
JOINT_LIMIT_RAD = 2.0 * math.pi
#: 部署安全上限：本数据包实测最大关节速度 0.0216 rad/s，超过此值视为数据异常
MAX_JOINT_SPEED_RAD_S = 0.5


class TrajectoryError(ValueError):
    """数据包校验失败。"""


class PlaybackError(RuntimeError):
    """播放中止（通信/安全/跟踪误差）。"""


# --------------------------------------------------------------------------- 数据加载
def _load_npy(path: Path) -> list[float] | list[tuple[float, ...]]:
    """读取 .npy（float64、C 序、1D/2D），不依赖 numpy。"""
    data = path.read_bytes()
    if len(data) < 12 or data[:6] != b"\x93NUMPY":
        raise TrajectoryError(f"{path.name} 不是 .npy 文件")
    major = data[6]
    if major == 1:
        header_len = struct.unpack("<H", data[8:10])[0]
        offset = 10
    elif major in (2, 3):
        fmt = "<I" if major == 2 else "<I"
        header_len = struct.unpack(fmt, data[8:12])[0]
        offset = 12
    else:
        raise TrajectoryError(f"{path.name} 的 .npy 版本不支持：{major}")
    header = ast.literal_eval(data[offset:offset + header_len].decode("latin1"))
    descr = header.get("descr")
    if descr not in ("<f8", "=f8"):
        raise TrajectoryError(f"{path.name} 必须是 float64，实际 {descr!r}")
    if header.get("fortran_order"):
        raise TrajectoryError(f"{path.name} 不能是 Fortran 序")
    shape = tuple(int(v) for v in header.get("shape", ()))
    if len(shape) not in (1, 2):
        raise TrajectoryError(f"{path.name} 维度不支持：{shape}")
    count = 1
    for value in shape:
        count *= value
    body = data[offset + header_len:]
    if len(body) < count * 8:
        raise TrajectoryError(f"{path.name} 数据长度不足")
    flat = struct.unpack(f"<{count}d", body[:count * 8])
    if len(shape) == 1:
        return list(flat)
    rows, cols = shape
    return [tuple(flat[i * cols:(i + 1) * cols]) for i in range(rows)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksums(folder: Path, names: Sequence[str]) -> None:
    sums = folder / "SHA256SUMS"
    if not sums.is_file():
        return
    expected: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            expected[parts[1]] = parts[0]
    for name in names:
        want = expected.get(name)
        if want is None:
            continue
        got = _sha256(folder / name)
        if got != want:
            raise TrajectoryError(f"{name} SHA256 不匹配：期望 {want}，实际 {got}")


@dataclass(frozen=True)
class Trajectory:
    """离线关节轨迹（rad / s）。"""

    q: list[tuple[float, ...]]
    t: list[float]
    folder: Path
    manifest: dict = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return len(self.q)

    @property
    def duration_s(self) -> float:
        return self.t[-1] - self.t[0]

    @property
    def dt(self) -> float:
        return self.t[1] - self.t[0]

    def max_joint_speed(self) -> float:
        return max(
            abs(self.q[k + 1][j] - self.q[k][j]) / self.dt
            for k in range(self.rows - 1)
            for j in range(6)
        )

    def with_joint_offsets(self, offsets_deg: Sequence[float]) -> "Trajectory":
        """每个关节加常量偏移（deg），用于把数据包的 home 映射到现场约定的 home。"""
        offsets = [math.radians(float(v)) for v in offsets_deg]
        if len(offsets) != 6:
            raise TrajectoryError(f"需要 6 个关节偏移，收到 {len(offsets)}")
        q = [tuple(row[i] + offsets[i] for i in range(6)) for row in self.q]
        for row in q:
            for value in row:
                if abs(value) > JOINT_LIMIT_RAD + 1e-9:
                    raise TrajectoryError(
                        f"关节偏移后超出 ±360°：{math.degrees(value):.2f}°")
        return Trajectory(q=q, t=list(self.t), folder=self.folder, manifest=self.manifest)

    def with_j1_offset(self, offset_deg: float) -> "Trajectory":
        """基座安装方位补偿：只给第 1 关节加常量偏差（j2..j6 不变）。

        机器人整体绕基座 Z 轴安装（如本机 -180°）时，用这个把轨迹的 home 对齐到实机 home；
        偏移量不改变关节速度，但会改变 TCP 路径在基座系中的方位（镜像到另一侧）。
        """
        offset = math.radians(float(offset_deg))
        q = [(row[0] + offset,) + tuple(row[1:]) for row in self.q]
        for row in q:
            for value in row:
                if abs(value) > JOINT_LIMIT_RAD + 1e-9:
                    raise TrajectoryError(
                        f"j1 偏移 {offset_deg:+.1f}° 后超出 ±360°：{math.degrees(value):.2f}°")
        return Trajectory(q=q, t=list(self.t), folder=self.folder, manifest=self.manifest)


def load_trajectory(folder: str | Path = DEFAULT_FOLDER) -> Trajectory:
    """读取并校验 cvpr-部署UR5 数据包；任何结构/限位/速度异常都抛 TrajectoryError。"""
    folder = Path(folder)
    q_path = folder / "q_command_rad.npy"
    t_path = folder / "t_command_s.npy"
    for path in (q_path, t_path):
        if not path.is_file():
            raise TrajectoryError(f"缺少数据文件：{path}")

    _verify_checksums(folder, ("q_command_rad.npy", "t_command_s.npy"))

    manifest: dict = {}
    manifest_path = folder / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    raw_q = _load_npy(q_path)
    raw_t = _load_npy(t_path)
    if not raw_q or not isinstance(raw_q[0], tuple):
        raise TrajectoryError("q_command_rad.npy 必须是二维 (N, 6)")
    q = [tuple(float(v) for v in row) for row in raw_q]  # type: ignore[arg-type]
    t = [float(v) for v in raw_t]  # type: ignore[arg-type]

    if len(q) != len(t):
        raise TrajectoryError(f"q/t 行数不一致：{len(q)} vs {len(t)}")
    if len(q) < 2:
        raise TrajectoryError("轨迹至少需要 2 行")
    for row in q:
        if len(row) != 6:
            raise TrajectoryError(f"关节数必须是 6，实际 {len(row)}")
        if not all(math.isfinite(v) for v in row):
            raise TrajectoryError("关节角含非有限值")
    if not all(math.isfinite(v) for v in t):
        raise TrajectoryError("时间含非有限值")
    if abs(t[0]) > 1e-9:
        raise TrajectoryError(f"时间应从 0 开始，实际 {t[0]}")
    dt = t[1] - t[0]
    if dt <= 0:
        raise TrajectoryError(f"时间必须递增，dt={dt}")
    for k in range(len(t) - 1):
        if abs((t[k + 1] - t[k]) - dt) > 1e-9:
            raise TrajectoryError(f"时间网格不均匀：第 {k} 行 dt={t[k + 1] - t[k]:.9f}")
    if not 0.001 <= dt <= 0.02:
        raise TrajectoryError(f"控制周期异常：{dt:.4f} s")

    for row in q:
        for value in row:
            if abs(value) > JOINT_LIMIT_RAD + 1e-9:
                raise TrajectoryError(f"关节角超出 ±360°：{math.degrees(value):.2f}°")
    trajectory = Trajectory(q=q, t=t, folder=folder, manifest=manifest)
    speed = trajectory.max_joint_speed()
    if speed > MAX_JOINT_SPEED_RAD_S:
        raise TrajectoryError(
            f"最大关节速度 {speed:.3f} rad/s 超过部署上限 {MAX_JOINT_SPEED_RAD_S} rad/s")

    if manifest:
        period = manifest.get("assumed_period_s")
        if period is not None and abs(period - dt) > 1e-9:
            raise TrajectoryError(f"manifest 周期 {period} 与数据 dt={dt} 不一致")
        duration = manifest.get("command_duration_s")
        if duration is not None and abs(duration - trajectory.duration_s) > 1e-6:
            raise TrajectoryError(
                f"manifest 时长 {duration} 与数据 {trajectory.duration_s} 不一致")
    return trajectory


# --------------------------------------------------------------------------- 播放器
@dataclass
class PlaybackStats:
    rows: int = 0
    max_error_rad: float = 0.0
    sum_error_rad: float = 0.0
    wall_s: float = 0.0
    samples: int = 0

    @property
    def mean_error_rad(self) -> float:
        return self.sum_error_rad / self.samples if self.samples else 0.0


# --------------------------------------------------------------------------- TCP 对齐（不改安装配置）
# UR5 正运动学（modified DH，与真机实测 TCP 误差 <2 mm，用于离线轨迹的 TCP 刚体变换）
_UR5_DH = (
    (0.089159, 0.0, math.pi / 2),        # d1, a1, alpha1
    (0.0, -0.425, 0.0),                  # d2, a2, alpha2
    (0.0, -0.39225, 0.0),                # d3, a3, alpha3
    (0.10915, 0.0, math.pi / 2),         # d4, a4, alpha4
    (0.09465, 0.0, -math.pi / 2),        # d5, a5, alpha5
    (0.0823, 0.0, 0.0),                  # d6, a6, alpha6
)


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def ur5_fk(q_rad: Sequence[float]) -> list[list[float]]:
    """UR5 正运动学：6 个关节角（rad）→ tool0 齐次矩阵（基座系）。"""
    transform = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    for q, (d, a, alpha) in zip(q_rad, _UR5_DH):
        theta = float(q)
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        link = [[ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0.0, sa, ca, d],
                [0.0, 0.0, 0.0, 1.0]]
        transform = _mat_mul(transform, link)
    return transform


def _rot_to_rotvec(r: list[list[float]]) -> tuple[float, float, float]:
    """3×3 旋转矩阵 → 轴角旋转向量（UR TCP 姿态表示）。"""
    trace = r[0][0] + r[1][1] + r[2][2]
    angle = math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))
    if angle < 1e-9:
        return (0.0, 0.0, 0.0)
    if abs(math.pi - angle) < 1e-6:  # 180°：用对角元素恢复轴
        axis = [math.sqrt(max(0.0, (r[i][i] + 1.0) / 2.0)) for i in range(3)]
        if r[0][1] < 0:
            axis[1] = -axis[1]
        if r[0][2] < 0:
            axis[2] = -axis[2]
        norm = math.sqrt(sum(v * v for v in axis)) or 1.0
        return tuple(angle * v / norm for v in axis)  # type: ignore[return-value]
    scale = angle / (2.0 * math.sin(angle))
    return (scale * (r[2][1] - r[1][2]),
            scale * (r[0][2] - r[2][0]),
            scale * (r[1][0] - r[0][1]))


def _rotvec_to_rot(rv: Sequence[float]) -> list[list[float]]:
    """轴角旋转向量 → 3×3 旋转矩阵。"""
    x, y, z = (float(v) for v in rv)
    angle = math.sqrt(x * x + y * y + z * z)
    if angle < 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    kx, ky, kz = x / angle, y / angle, z / angle
    c, s = math.cos(angle), math.sin(angle)
    v = 1.0 - c
    return [[c + kx * kx * v, kx * ky * v - kz * s, kx * kz * v + ky * s],
            [ky * kx * v + kz * s, c + ky * ky * v, ky * kz * v - kx * s],
            [kz * kx * v - ky * s, kz * ky * v + kx * s, c + kz * kz * v]]


def _pose_from_matrix(transform: list[list[float]]) -> tuple[float, ...]:
    rot = [[transform[i][j] for j in range(3)] for i in range(3)]
    rv = _rot_to_rotvec(rot)
    return (transform[0][3], transform[1][3], transform[2][3], rv[0], rv[1], rv[2])


def _matrix_from_pose(pose: Sequence[float]) -> list[list[float]]:
    rot = _rotvec_to_rot(pose[3:])
    return [[rot[0][0], rot[0][1], rot[0][2], float(pose[0])],
            [rot[1][0], rot[1][1], rot[1][2], float(pose[1])],
            [rot[2][0], rot[2][1], rot[2][2], float(pose[2])],
            [0.0, 0.0, 0.0, 1.0]]


def _mat_transpose3(m: list[list[float]]) -> list[list[float]]:
    return [[m[j][i] for j in range(3)] for i in range(3)]


def _rot_angle_between(a: Sequence[float], b: Sequence[float]) -> float:
    """两个轴角姿态之间的旋转角（rad）。"""
    ra, rb = _rotvec_to_rot(a), _rotvec_to_rot(b)
    rel = [[sum(_mat_transpose3(ra)[i][k] * rb[k][j] for k in range(3)) for j in range(3)]
           for i in range(3)]
    trace = rel[0][0] + rel[1][1] + rel[2][2]
    return math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))


@dataclass(frozen=True)
class TcpTrajectory:
    """刚体变换后的 TCP 轨迹（位置 + 轴角旋转向量，基座系）。"""

    poses: list[tuple[float, ...]]
    t: list[float]
    transform: list[list[float]]
    target_start: tuple[float, ...]
    source: Trajectory

    @property
    def rows(self) -> int:
        return len(self.poses)

    @property
    def dt(self) -> float:
        return self.t[1] - self.t[0]

    @property
    def duration_s(self) -> float:
        return self.t[-1] - self.t[0]


def align_trajectory_to_start(trajectory: Trajectory,
                              target_start: Sequence[float]) -> TcpTrajectory:
    """把整条 TCP 路径刚体变换，使其起点位姿等于 target_start（姿态取目标姿态）。"""
    start_matrix = ur5_fk(trajectory.q[0])
    start_pose = _pose_from_matrix(start_matrix)
    target_matrix = _matrix_from_pose(target_start)
    r0 = [[start_matrix[i][j] for j in range(3)] for i in range(3)]
    rt = [[target_matrix[i][j] for j in range(3)] for i in range(3)]
    rot_delta = [[sum(rt[i][k] * _mat_transpose3(r0)[k][j] for k in range(3)) for j in range(3)]
                 for i in range(3)]
    pos_delta = [float(target_start[i]) - sum(rot_delta[i][k] * start_pose[k] for k in range(3))
                 for i in range(3)]
    poses = []
    for row in trajectory.q:
        frame = ur5_fk(row)
        rot = [[frame[i][j] for j in range(3)] for i in range(3)]
        new_rot = [[sum(rot_delta[i][k] * rot[k][j] for k in range(3)) for j in range(3)]
                   for i in range(3)]
        new_pos = [sum(rot_delta[i][k] * frame[k][3] for k in range(3)) + pos_delta[i]
                   for i in range(3)]
        rv = _rot_to_rotvec(new_rot)
        poses.append((new_pos[0], new_pos[1], new_pos[2], rv[0], rv[1], rv[2]))
    transform = [[rot_delta[0][0], rot_delta[0][1], rot_delta[0][2], pos_delta[0]],
                 [rot_delta[1][0], rot_delta[1][1], rot_delta[1][2], pos_delta[1]],
                 [rot_delta[2][0], rot_delta[2][1], rot_delta[2][2], pos_delta[2]],
                 [0.0, 0.0, 0.0, 1.0]]
    return TcpTrajectory(poses=poses, t=list(trajectory.t), transform=transform,
                         target_start=tuple(float(v) for v in target_start),
                         source=trajectory)


def shifted_start_pose(current_pose: Sequence[float], inward_m: float,
                       drop_m: float) -> tuple[float, ...]:
    """当前位置沿 XY 往基座收 inward_m、Z 下降 drop_m，姿态保持当前工具姿态。"""
    x, y, z = (float(v) for v in current_pose[:3])
    norm = math.hypot(x, y)
    ux, uy = (x / norm, y / norm) if norm > 1e-6 else (1.0, 0.0)
    return (x - ux * inward_m, y - uy * inward_m, z - drop_m,
            float(current_pose[3]), float(current_pose[4]), float(current_pose[5]))


class TcpPlayer:
    """把刚体变换后的 TCP 轨迹流式发给 ur5_teleop.script（机器人端 IK + servoj）。"""

    def __init__(self, backend, trajectory: TcpTrajectory, *, scale: float = 1.0,
                 max_pos_error_m: float = 0.10, max_rot_error_rad: float = 0.5,
                 printer: Callable[[str], None] = print,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 progress_period_s: float = 2.0) -> None:
        self.backend = backend
        self.trajectory = trajectory
        self.scale = float(scale)
        self.max_pos_error_m = float(max_pos_error_m)
        self.max_rot_error_rad = float(max_rot_error_rad)
        self.printer = printer
        self.clock = clock
        self.sleep = sleep
        self.progress_period_s = float(progress_period_s)

    def _state(self):
        try:
            state = self.backend.read_state()
        except DisconnectedError as exc:
            raise PlaybackError(f"RTDE 状态读取失败：{exc}") from exc
        if state.safety_status != SAFETY_NORMAL:
            name = SAFETY_MODE_NAMES.get(state.safety_status, str(state.safety_status))
            raise PlaybackError(f"安全状态变为 {name}，已中止（检查示教器/现场）")
        if state.robot_mode != MODE_RUNNING:
            name = ROBOT_MODE_NAMES.get(state.robot_mode, str(state.robot_mode))
            raise PlaybackError(f"机器人不在 RUNNING（当前 {name}），已中止")
        return state

    def _error(self, actual: Sequence[float], target: Sequence[float]) -> tuple[float, float]:
        pos = math.dist([actual[i] for i in range(3)], [target[i] for i in range(3)])
        rot = _rot_angle_between(actual[3:6], target[3:6])
        return pos, rot

    def lead_in(self, seconds: float = 10.0) -> float:
        """从当前 TCP 位姿平滑过渡到变换后的轨迹起点（只做平移，姿态不变）。"""
        state = self._state()
        start = tuple(state.tcp_pose)
        target = self.trajectory.target_start
        distance = math.dist(start[:3], target[:3])
        if distance < 0.005:
            self.printer(f"[lead-in] 已在目标起点附近（{distance * 1000:.1f} mm）")
            return 0.0
        duration = max(float(seconds), distance / 0.02)   # 峰值速度 ≤ ~1.2×0.02 m/s
        steps = max(int(round(duration / 0.008)), 1)
        self.printer(f"[lead-in] 当前→起点：{distance * 1000:.0f} mm，用 {duration:.1f} s 平滑过渡")
        started = self.clock()
        last_print = started
        for k in range(1, steps + 1):
            ratio = k / steps
            smooth = ratio * ratio * ratio * (10.0 - 15.0 * ratio + 6.0 * ratio * ratio)
            pose = tuple(float(start[i]) + (float(target[i]) - float(start[i])) * smooth
                         for i in range(6))
            self.backend.send_sets(pose, servo=True)
            state = self._state()
            pos_err, rot_err = self._error(state.tcp_pose, pose)
            if pos_err > self.max_pos_error_m or rot_err > self.max_rot_error_rad:
                raise PlaybackError(
                    f"lead-in 第 {k} 步偏差 {pos_err * 1000:.0f} mm/{math.degrees(rot_err):.1f}° 超阈值，已中止")
            now = self.clock()
            if now - last_print >= self.progress_period_s:
                self.printer(f"[lead-in] {100.0 * ratio:5.1f}%  偏差 {pos_err * 1000:.0f} mm")
                last_print = now
            self.sleep(0.008)
        return self.clock() - started

    def play(self, max_seconds: float | None = None, hold_s: float = 0.5) -> PlaybackStats:
        trajectory = self.trajectory
        poses, t = trajectory.poses, trajectory.t
        end_index = len(poses)
        if max_seconds is not None:
            end_index = 1
            while end_index < len(poses) and t[end_index] <= max_seconds:
                end_index += 1
        self.printer(f"[play-tcp] 开始播放 {end_index}/{len(poses)} 帧，"
                     f"标称 {t[end_index - 1] / self.scale:.1f} s")
        started = self.clock()
        last_print = started
        stats = PlaybackStats(rows=end_index)
        start_actual = None
        for k in range(end_index):
            target_time = started + t[k] / self.scale
            delay = target_time - self.clock()
            if delay > 0:
                self.sleep(delay)
            self.backend.send_sets(poses[k], servo=True)
            state = self._state()
            pos_err, rot_err = self._error(state.tcp_pose, poses[k])
            if pos_err > self.max_pos_error_m or rot_err > self.max_rot_error_rad:
                raise PlaybackError(
                    f"第 {k} 帧偏差 {pos_err * 1000:.0f} mm/{math.degrees(rot_err):.1f}° "
                    f"超过阈值，已中止")
            stats.max_error_rad = max(stats.max_error_rad, pos_err)  # 复用字段存位置误差（m）
            stats.sum_error_rad += pos_err
            stats.samples += 1
            if start_actual is None and k * trajectory.dt >= 0.2:
                start_actual = state.tcp_pose
            if start_actual is not None and k * trajectory.dt >= 3.0 and k % 25 == 0:
                moved_cmd = math.dist(poses[k][:3], poses[0][:3])
                moved_act = math.dist(state.tcp_pose[:3], start_actual[:3])
                if moved_cmd > 0.03 and moved_act < 0.003:
                    raise PlaybackError(
                        "机器人没有跟随指令（3 s 内几乎未动）：检查 ur5_teleop.script 是否在运行")
            now = self.clock()
            if now - last_print >= self.progress_period_s:
                self.printer(f"[play-tcp] {100.0 * (k + 1) / end_index:5.1f}%  "
                             f"偏差 {pos_err * 1000:.1f} mm/{math.degrees(rot_err):.2f}°  "
                             f"安全 {SAFETY_MODE_NAMES.get(state.safety_status, state.safety_status)}")
                last_print = now
        final = poses[end_index - 1]
        hold_until = self.clock() + hold_s
        while self.clock() < hold_until:
            self.backend.send_sets(final, servo=True)
            self._state()
            self.sleep(0.008)
        self.backend.send_sets(final, servo=False)
        stats.wall_s = self.clock() - started
        final_state = self._state()
        pos_err, rot_err = self._error(final_state.tcp_pose, final)
        self.printer(f"[play-tcp] 完成：{end_index} 帧，用时 {stats.wall_s:.1f} s，"
                     f"平均偏差 {stats.mean_error_rad * 1000:.1f} mm，"
                     f"最大偏差 {stats.max_error_rad * 1000:.1f} mm，"
                     f"终点偏差 {pos_err * 1000:.1f} mm / {math.degrees(rot_err):.2f}°")
        return stats


class TrajectoryPlayer:
    """把预览过的关节轨迹按标称周期流式发给机器人端 ur5_traj.script。"""

    def __init__(
        self,
        backend,
        trajectory: Trajectory,
        *,
        scale: float = 1.0,
        max_track_error_rad: float = 0.15,
        printer: Callable[[str], None] = print,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        progress_period_s: float = 2.0,
    ) -> None:
        if scale <= 0:
            raise ValueError("scale 必须 > 0")
        self.backend = backend
        self.trajectory = trajectory
        self.scale = float(scale)
        self.max_track_error_rad = float(max_track_error_rad)
        self.printer = printer
        self.clock = clock
        self.sleep = sleep
        self.progress_period_s = float(progress_period_s)

    # ------------------------------------------------------------------ 工具
    def _state(self):
        try:
            state = self.backend.read_state()
        except DisconnectedError as exc:
            raise PlaybackError(f"RTDE 状态读取失败：{exc}") from exc
        if state.safety_status != SAFETY_NORMAL:
            name = SAFETY_MODE_NAMES.get(state.safety_status, str(state.safety_status))
            raise PlaybackError(f"安全状态变为 {name}，已中止（检查示教器/现场）")
        if state.robot_mode != MODE_RUNNING:
            name = ROBOT_MODE_NAMES.get(state.robot_mode, str(state.robot_mode))
            raise PlaybackError(f"机器人不在 RUNNING（当前 {name}），已中止")
        return state

    @staticmethod
    def _joint_error(q_a: Sequence[float], q_b: Sequence[float]) -> float:
        return max(abs(q_a[i] - q_b[i]) for i in range(6))

    # ------------------------------------------------------------------ 步骤
    def lead_in(self, min_duration_s: float = 20.0, max_speed_rad_s: float = 0.06,
                tolerance_rad: float = 0.02, max_error_rad: float = 0.40) -> float:
        """用 servoj 从当前关节位姿平滑走到轨迹首帧（不依赖 movej / 速度滑块）。

        CB3 上 movej 受速度滑块缩放（滑块 0% 时几乎不动）；本方法直接流式 servoj，
        时长按最大关节差自适应，峰值关节速度不超过 max_speed_rad_s。
        """
        state = self._state()
        q_from = tuple(state.joints)
        q_to = tuple(self.trajectory.q[0])
        delta = max(abs(q_to[i] - q_from[i]) for i in range(6))
        if delta <= tolerance_rad:
            self.printer(f"[lead-in] 已在起始位姿（最大误差 {math.degrees(delta):.2f}°）")
            return 0.0
        duration = max(float(min_duration_s), delta / max(max_speed_rad_s, 1e-6))
        steps = max(int(round(duration / self.trajectory.dt)), 1)
        self.printer(f"[lead-in] 当前→起始位姿：最大关节差 {math.degrees(delta):.1f}°，"
                     f"用 {duration:.1f} s 平滑过渡（峰值 ≈ {1.875 * delta / duration:.3f} rad/s）")
        started = self.clock()
        last_print = started
        for k in range(1, steps + 1):
            ratio = k / steps
            smooth = ratio * ratio * ratio * (10.0 - 15.0 * ratio + 6.0 * ratio * ratio)
            target = tuple(q_from[i] + (q_to[i] - q_from[i]) * smooth for i in range(6))
            self.backend.send_joint_sets(target, servo=True)
            state = self._state()
            error = self._joint_error(state.joints, target)
            if error > max_error_rad:
                raise PlaybackError(
                    f"lead-in 第 {k} 步跟踪误差 {math.degrees(error):.1f}° 超过阈值，已中止")
            now = self.clock()
            if now - last_print >= self.progress_period_s:
                self.printer(f"[lead-in] {100.0 * ratio:5.1f}%  误差 "
                             f"{math.degrees(error):.2f}°  安全 "
                             f"{SAFETY_MODE_NAMES.get(state.safety_status, state.safety_status)}")
                last_print = now
            self.sleep(self.trajectory.dt)
        # 收尾：保持终点 0.3 s 让伺服收敛
        hold_until = self.clock() + 0.3
        while self.clock() < hold_until:
            self.backend.send_joint_sets(q_to, servo=True)
            self._state()
            self.sleep(self.trajectory.dt)
        final_error = self._joint_error(self._state().joints, q_to)
        self.printer(f"[lead-in] 完成：用时 {self.clock() - started:.1f} s，"
                     f"终点误差 {math.degrees(final_error):.2f}°")
        return self.clock() - started

    def move_to_start(self, timeout_s: float = 180.0, tolerance_rad: float = 0.02) -> float:
        """cmd=3：低速 movej 到轨迹首帧，等待到位（最大误差 tolerance_rad）。"""
        q0 = self.trajectory.q[0]
        started = self.clock()
        last_print = started
        self.printer("[movej] 正在移动到轨迹起始位姿（v=0.15 rad/s，阻塞）...")
        while True:
            state = self._state()
            error = self._joint_error(state.joints, q0)
            if error <= tolerance_rad and self.clock() - started > 0.3:
                self.printer(f"[movej] 到位：最大关节误差 {math.degrees(error):.2f}°"
                             f"（{self.clock() - started:.1f} s）")
                break
            self.backend.send_move_to_start(q0)
            now = self.clock()
            if now - started > timeout_s:
                raise PlaybackError(
                    f"movej 超时（{timeout_s:.0f} s），当前最大误差 {math.degrees(error):.2f}°")
            if now - last_print >= self.progress_period_s:
                self.printer(f"[movej] 误差 {math.degrees(error):.2f}° （{now - started:.1f} s）")
                last_print = now
            self.sleep(0.008)
        self.backend.send_joint_sets(q0, servo=False)
        self.sleep(0.3)
        return self.clock() - started

    def play(self, max_seconds: float | None = None, hold_s: float = 0.5) -> PlaybackStats:
        """cmd=2：按标称 125 Hz 流式播放；超限立即抛 PlaybackError。"""
        trajectory = self.trajectory
        q, t = trajectory.q, trajectory.t
        end_index = len(q)
        if max_seconds is not None:
            end_index = 1
            while end_index < len(q) and t[end_index] <= max_seconds:
                end_index += 1
        planned_wall = t[end_index - 1] / self.scale
        self.printer(f"[play] 开始播放 {end_index}/{len(q)} 帧，标称 {planned_wall:.1f} s"
                     f"（scale={self.scale:g}，上限误差 {math.degrees(self.max_track_error_rad):.1f}°）")

        started = self.clock()
        last_print = started
        stats = PlaybackStats(rows=end_index)
        start_actual = None
        for k in range(end_index):
            target_time = started + t[k] / self.scale
            delay = target_time - self.clock()
            if delay > 0:
                self.sleep(delay)
            self.backend.send_joint_sets(q[k], servo=True)
            state = self._state()
            error = self._joint_error(state.joints, q[k])
            stats.max_error_rad = max(stats.max_error_rad, error)
            stats.sum_error_rad += error
            stats.samples += 1
            if error > self.max_track_error_rad:
                raise PlaybackError(
                    f"第 {k} 帧跟踪误差 {math.degrees(error):.1f}° 超过阈值 "
                    f"{math.degrees(self.max_track_error_rad):.1f}°，已中止")
            # 3 s 内指令已走 >0.05 rad 而实测几乎没动 → 脚本没在跑/未接管
            if start_actual is None and k * trajectory.dt >= 0.2:
                start_actual = state.joints
            if start_actual is not None and k * trajectory.dt >= 3.0 and k % 25 == 0:
                moved_cmd = self._joint_error(q[k], q[0])
                moved_act = self._joint_error(state.joints, start_actual)
                if moved_cmd > 0.05 and moved_act < 0.005:
                    raise PlaybackError(
                        "机器人没有跟随指令（3 s 内几乎未动）：检查 ur5_traj.script 是否在运行")
            now = self.clock()
            if now - last_print >= self.progress_period_s:
                self.printer(
                    f"[play] {100.0 * (k + 1) / end_index:5.1f}%  帧 {k + 1}/{end_index}  "
                    f"误差 {math.degrees(error):.2f}°  "
                    f"安全 {SAFETY_MODE_NAMES.get(state.safety_status, state.safety_status)}")
                last_print = now

        # 保持最终目标，让伺服收敛后再 idle
        final = q[end_index - 1]
        hold_until = self.clock() + hold_s
        while self.clock() < hold_until:
            self.backend.send_joint_sets(final, servo=True)
            self._state()
            self.sleep(0.008)
        self.backend.send_joint_sets(final, servo=False)
        stats.wall_s = self.clock() - started
        final_state = self._state()
        final_error = self._joint_error(final_state.joints, final)
        self.printer(f"[play] 完成：{end_index} 帧，用时 {stats.wall_s:.1f} s，"
                     f"平均误差 {math.degrees(stats.mean_error_rad):.2f}°，"
                     f"最大误差 {math.degrees(stats.max_error_rad):.2f}°，"
                     f"终点误差 {math.degrees(final_error):.2f}°")
        return stats

    def stop_program(self) -> None:
        """尽力发一帧 idle（避免机器人继续推进最后目标）。"""
        try:
            self.backend.send_joint_sets(self.trajectory.q[-1], servo=False)
        except Exception:  # noqa: BLE001 - 收尾阶段吞掉异常
            pass


class _EchoBackend(FakeBackend):
    """--dry-run 用：把关节指令当作实际状态回显（完美跟随），验证播放逻辑。"""

    def __init__(self, q0: Sequence[float]) -> None:
        super().__init__(initial_joints=q0)

    def send_joint_sets(self, joints6: Sequence[float], servo: bool = True) -> None:
        super().send_joint_sets(joints6, servo)
        if servo:
            self.feed_state(joints=joints6)

    def send_move_to_start(self, joints6: Sequence[float]) -> None:
        super().send_move_to_start(joints6)
        self.feed_state(joints=joints6)

    def send_sets(self, pose6: Sequence[float], servo: bool) -> None:
        super().send_sets(pose6, servo)
        if servo:
            self.feed_state(pose=pose6)


class _VirtualClock:
    """--dry-run 用的虚拟时钟：每次 sleep 前进一个控制周期，跑完整流程只需毫秒级。"""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, _seconds: float) -> None:
        self.value += NOMINAL_PERIOD_S


# --------------------------------------------------------------------------- 入口
def _summary(trajectory: Trajectory) -> str:
    q0 = [math.degrees(v) for v in trajectory.q[0]]
    q1 = [math.degrees(v) for v in trajectory.q[-1]]
    return (
        f"数据：{trajectory.folder.name}，{trajectory.rows} 帧 / {trajectory.duration_s:.3f} s"
        f" @ {1.0 / trajectory.dt:.0f} Hz\n"
        f"起始关节(deg)：[{', '.join(f'{v:+.2f}' for v in q0)}]\n"
        f"终止关节(deg)：[{', '.join(f'{v:+.2f}' for v in q1)}]\n"
        f"最大关节速度：{trajectory.max_joint_speed():.4f} rad/s")


def _parse_home_joints(text: str) -> tuple[float, ...]:
    """解析 "--home-joints -180,-90,-90,-90,90,-90"（deg）。"""
    parts = [part.strip() for part in text.replace(" ", ",").split(",") if part.strip()]
    if len(parts) != 6:
        raise TrajectoryError(f"--home-joints 需要 6 个角度（deg），收到 {len(parts)} 个")
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise TrajectoryError(f"--home-joints 含非数字：{text}") from exc
    for value in values:
        if abs(value) > 360.0 + 1e-9:
            raise TrajectoryError(f"--home-joints 超出 ±360°：{value}")
    return values


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    # argparse 会把 "-180,..." 当成选项：把 "--home-joints -180,..." 合并成 "=" 形式
    merged: list[str] = []
    index = 0
    while index < len(raw):
        if raw[index] == "--home-joints" and index + 1 < len(raw):
            merged.append("--home-joints=" + raw[index + 1])
            index += 2
            continue
        merged.append(raw[index])
        index += 1
    argv = merged
    parser = argparse.ArgumentParser(
        description="在真机上复现 cvpr-部署UR5 离线关节轨迹",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--config", default=None, help="配置文件，默认 config/ur5.toml")
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER, help="数据包目录")
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT, help="机器人端 URScript")
    parser.add_argument("--port", type=int, default=30001, help="脚本端口，默认 30001")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="时间缩放：1=标称 55 s，2=快一倍，0.5=慢一倍")
    parser.add_argument("--j1-offset-deg", type=float, default=0.0,
                        help="基座安装方位补偿（只加到 j1）：本机基座转 180° 用 -180")
    parser.add_argument("--home-joints", default=None,
                        help="现场约定的 home 关节角（deg，逗号分隔 6 个）：把轨迹的 home 映射到它")
    parser.add_argument("--remap", choices=("joint", "tcp"), default="tcp",
                        help="home 映射方式：joint=关节常量偏移（关节运动与数据包一致）；"
                             "tcp=TCP 刚体变换（TCP 路径形状与数据包一致）")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="只播放轨迹前 N 秒（轨迹时间），便于分段试跑")
    parser.add_argument("--max-track-error", type=float, default=0.15,
                        help="跟踪误差上限（rad），超过即中止，默认 0.15")
    parser.add_argument("--move-tolerance", type=float, default=0.02,
                        help="movej 到位判定（rad），默认 0.02")
    parser.add_argument("--tcp-align", action="store_true",
                        help="把轨迹 TCP 路径刚体变换到当前位姿附近（不改安装配置；机器人端用 IK）")
    parser.add_argument("--align-inward", type=float, default=0.10,
                        help="TCP 对齐模式的起点：沿 XY 往基座收多少米，默认 0.10")
    parser.add_argument("--align-drop", type=float, default=0.18,
                        help="TCP 对齐模式的起点：Z 下降多少米，默认 0.18")
    parser.add_argument("--tcp-script", type=Path,
                        default=ROOT / "urscript" / "ur5_teleop.script",
                        help="TCP 对齐模式使用的机器人端脚本（IK+servoj）")
    parser.add_argument("--max-pos-error", type=float, default=0.10,
                        help="TCP 对齐模式的跟踪位置偏差上限（m），默认 0.10")
    parser.add_argument("--lead-in", type=float, default=None, metavar="SECONDS",
                        help="不用 movej，改用 servoj 从当前位姿平滑走到起始位姿（最小秒数；"
                             "不受速度滑块影响）")
    parser.add_argument("--skip-move", action="store_true",
                        help="跳过移动到起始位姿（假定已就位；仍会校验误差）")
    parser.add_argument("--move-only", action="store_true",
                        help="只 movej 到起始位姿，不播放轨迹")
    parser.add_argument("--dry-run", action="store_true", help="不连机器人，验证数据包与流程")
    parser.add_argument("--yes", action="store_true", help="确认现场安全，允许驱动真机")
    return parser.parse_args(argv)


def _run_dry_run_tcp(trajectory: Trajectory, args: argparse.Namespace) -> int:
    start_matrix = ur5_fk(trajectory.q[0])
    current = _pose_from_matrix(start_matrix)
    if getattr(args, "home_pose", None) is not None:
        target_start = args.home_pose
        current = args.home_pose   # 假设机器人已停在目标 home
    else:
        target_start = shifted_start_pose(current, args.align_inward, args.align_drop)
    tcp_traj = align_trajectory_to_start(trajectory, target_start)
    print(f"[dry-run] TCP 对齐：当前(TCP)={tuple(round(v, 4) for v in current)}")
    print(f"[dry-run] TCP 对齐：目标起点={tuple(round(v, 4) for v in tcp_traj.target_start)}")
    backend = _EchoBackend(trajectory.q[0])
    backend.connect()
    backend.feed_state(pose=current)
    clock = _VirtualClock()
    player = TcpPlayer(backend, tcp_traj, scale=args.scale,
                       max_pos_error_m=args.max_pos_error,
                       clock=clock, sleep=clock.sleep, progress_period_s=5.0)
    try:
        if not args.skip_move:
            player.lead_in(seconds=2.0)
        if args.move_only:
            print("[dry-run] TCP 对齐 move-only：lead-in 模拟完成")
        else:
            stats = player.play(max_seconds=args.max_seconds)
            print(f"[dry-run] TCP 对齐通过：{stats.rows} 帧、模拟最大偏差 "
                  f"{stats.max_error_rad * 1000:.1f} mm")
    finally:
        backend.close()
    print("[dry-run] 不连接机器人，未发送任何真实指令。")
    return 0


def _run_dry_run(trajectory: Trajectory, args: argparse.Namespace) -> int:
    backend = _EchoBackend(trajectory.q[0])
    backend.connect()
    clock = _VirtualClock()
    player = TrajectoryPlayer(
        backend, trajectory, scale=args.scale,
        max_track_error_rad=args.max_track_error,
        clock=clock, sleep=clock.sleep, progress_period_s=5.0)
    try:
        if not args.skip_move:
            if args.lead_in is not None:
                player.lead_in(min_duration_s=args.lead_in)
            else:
                player.move_to_start(timeout_s=180.0, tolerance_rad=args.move_tolerance)
        else:
            print("[dry-run] --skip-move：跳过移动到起始位姿")
        if args.move_only:
            stats = PlaybackStats()
        else:
            stats = player.play(max_seconds=args.max_seconds)
    finally:
        player.stop_program()
        backend.close()
    expected = stats.rows
    if expected and len(backend.joint_sent) < expected:
        print(f"[dry-run] 失败：只发出 {len(backend.joint_sent)}/{expected} 帧关节目标")
        return 1
    if args.move_only:
        print(f"[dry-run] movej 模拟完成：发起 cmd=3 {len(backend.move_start_sent)} 次")
    else:
        print(f"[dry-run] 通过：{stats.rows} 帧关节目标、模拟最大误差 "
              f"{math.degrees(stats.max_error_rad):.2f}°、movej 发起 "
              f"{len(backend.move_start_sent)} 次")
    print("[dry-run] 不连接机器人，未发送任何真实指令。")
    return 0


def _check_ready(dashboard: DashboardClient) -> None:
    safety = dashboard.command("safetymode")
    mode = dashboard.command("robotmode")
    print(f"[预检] {safety} | {mode}")
    if "NORMAL" not in safety:
        raise PlaybackError(
            f"安全状态不是 NORMAL（{safety}）。先查现场，必要时执行："
            "python3 tools/ur_ctl.py --ip <IP> --yes unlock-protective-stop")
    if "RUNNING" not in mode:
        raise PlaybackError(
            f"机器人不在 RUNNING（{mode}）。先上电/松闸："
            "python3 tools/ur_ctl.py --ip <IP> --yes power-on，再 --yes brake-release，再 play")


def _run_real(trajectory: Trajectory, args: argparse.Namespace) -> int:
    if not args.yes:
        print("该工具会驱动机械臂。确认现场安全后加 --yes 重跑。")
        return 2
    cfg = config_module.load(args.config)
    script_path = args.tcp_script if args.tcp_align else args.script
    script_text = script_path.read_text(encoding="utf-8")

    dashboard: DashboardClient | None = None
    backend: RTDEBackend | None = None
    player: TrajectoryPlayer | None = None
    tcp_player: TcpPlayer | None = None
    try:
        dashboard = DashboardClient(cfg.robot.ip)
        print(f"[dashboard] {dashboard.connect()}")
        _check_ready(dashboard)
        print(f"[dashboard] stop: {dashboard.command('stop')}")
        time.sleep(1.0)

        print(f"[program] 下发 {script_path.name} 到 {cfg.robot.ip}:{args.port}")
        channel = open_script_channel(cfg.robot.ip, args.port)
        send_program(channel, script_text)
        time.sleep(0.8)
        channel.close()   # 程序继续运行；长期占着不读的脚本通道会拖死控制器 TCP 发送
        print("[program] 已下发并关闭脚本通道，程序 idle 运行")

        backend = RTDEBackend(
            cfg.robot.ip,
            protocol_version=cfg.robot.protocol_version,
            frequency_hz=cfg.robot.rtde_frequency_hz,
            state_timeout_s=max(0.3, cfg.safety.comm_timeout_s * 1.5),
        )
        print(backend.connect())
        deadline = time.monotonic() + 5.0
        state = None
        while time.monotonic() < deadline:
            try:
                state = backend.read_state()
                break
            except DisconnectedError:
                time.sleep(0.05)
        if state is None:
            raise PlaybackError("5 s 内未收到 RTDE 状态帧")
        if state.safety_status != SAFETY_NORMAL:
            raise PlaybackError(
                f"RTDE 读到安全状态 {SAFETY_MODE_NAMES.get(state.safety_status)}")
        if state.robot_mode != MODE_RUNNING:
            raise PlaybackError(
                f"RTDE 读到 robot_mode={ROBOT_MODE_NAMES.get(state.robot_mode)}")

        if args.tcp_align:
            if getattr(args, "home_pose", None) is not None:
                target_start = args.home_pose
            else:
                target_start = shifted_start_pose(
                    state.tcp_pose, args.align_inward, args.align_drop)
            tcp_traj = align_trajectory_to_start(trajectory, target_start)
            rot = tcp_traj.transform
            angle = math.degrees(math.acos(max(-1.0, min(1.0,
                (rot[0][0] + rot[1][1] + rot[2][2] - 1.0) / 2.0))))
            distance = math.dist([state.tcp_pose[i] for i in range(3)],
                                 [tcp_traj.target_start[i] for i in range(3)])
            print(f"[tcp-align] 当前 TCP={tuple(round(v, 4) for v in state.tcp_pose[:3])}，"
                  f"轨迹起点变换到 {tuple(round(v, 4) for v in tcp_traj.target_start[:3])}"
                  f"（需移动 {distance * 1000:.0f} mm）")
            print(f"[tcp-align] 刚体变换：旋转 {angle:.1f}°，路径长度与形状保持不变；"
                  "机器人端用 get_inverse_kin+servoj 跟随，安装配置不变")
            tcp_player = TcpPlayer(backend, tcp_traj, scale=args.scale,
                                   max_pos_error_m=args.max_pos_error, printer=print)
            if not args.skip_move:
                seconds = args.lead_in if args.lead_in is not None else 10.0
                print(f"[安全检查] 即将从当前位姿平滑移动到轨迹起点"
                      f"（约 {distance * 1000:.0f} mm、{seconds:.0f} s）：确认周围无障碍、手在急停上。3 s 后开始…")
                for countdown in (3, 2, 1):
                    print(f"  {countdown}…")
                    time.sleep(1.0)
                tcp_player.lead_in(seconds=seconds)
            else:
                print("[skip-move] 跳过 lead-in（会从当前位置直接开始跟随轨迹，前几秒有步长限幅过渡）")
            if args.move_only:
                print("[move-only] 已到轨迹起点，未播放轨迹；下一步可加 --skip-move 播放。")
                return 0
            tcp_player.play(max_seconds=args.max_seconds)
            return 0

        player = TrajectoryPlayer(
            backend, trajectory, scale=args.scale,
            max_track_error_rad=args.max_track_error)
        print(f"[当前] 关节误差到起始位姿 "
              f"{math.degrees(player._joint_error(state.joints, trajectory.q[0])):.2f}°")

        if not args.skip_move and args.lead_in is not None:
            print("[安全检查] 即将用 servoj 平滑走到起始 home 姿：确认机器人周围无障碍、手在急停上。3 s 后开始…")
            for countdown in (3, 2, 1):
                print(f"  {countdown}…")
                time.sleep(1.0)
            player.lead_in(min_duration_s=args.lead_in)
        elif not args.skip_move:
            print("[安全检查] 即将 movej 到起始 home 姿：确认机器人周围无障碍、手在急停上。3 s 后开始…")
            for countdown in (3, 2, 1):
                print(f"  {countdown}…")
                time.sleep(1.0)
            player.move_to_start(tolerance_rad=args.move_tolerance)
        else:
            error = player._joint_error(state.joints, trajectory.q[0])
            if error > math.radians(2.0):
                print(f"[warn] --skip-move 但当前与起始位姿差 {math.degrees(error):.1f}°，"
                      "脚本会直接伺服到轨迹首帧附近，请确认无障碍")
            else:
                print("[skip-move] 已在起始位姿附近")

        if args.move_only:
            print("[move-only] 已到起始位姿，未播放轨迹；下一步可加 --skip-move 播放。")
            return 0

        player.play(max_seconds=args.max_seconds)
        return 0
    except KeyboardInterrupt:
        print("\n[中止] Ctrl-C：发 idle 并停程序")
        return 130
    except PlaybackError as exc:
        print(f"[中止] {exc}")
        return 1
    finally:
        if backend is not None:
            backend.close()
        if dashboard is not None:
            try:
                print(f"[dashboard] stop: {dashboard.command('stop')}")
            except OSError:
                pass
            dashboard.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        trajectory = load_trajectory(args.folder)
    except TrajectoryError as exc:
        print(f"[数据校验失败] {exc}")
        return 1
    if args.j1_offset_deg:
        trajectory = trajectory.with_j1_offset(args.j1_offset_deg)
        print(f"[安装补偿] j1 偏移 {args.j1_offset_deg:+.1f}°（基座安装方位）")
    args.home_pose = None
    if args.home_joints:
        home = _parse_home_joints(args.home_joints)
        print(f"[home 映射] 新 home 关节(deg): {list(home)}")
        if args.remap == "joint":
            offsets = [home[i] - math.degrees(trajectory.q[0][i]) for i in range(6)]
            print(f"[home 映射] 方式 joint：关节偏移(deg) = {[round(v, 1) for v in offsets]}")
            print("[home 映射] 关节增量与数据包完全一致；TCP 路径形状会随支路变化")
            trajectory = trajectory.with_joint_offsets(offsets)
        else:
            args.home_pose = _pose_from_matrix(ur5_fk([math.radians(v) for v in home]))
            args.tcp_align = True   # tcp 映射走 TCP 对齐流程（IK+servoj）
            print("[home 映射] 方式 tcp：刚体变换，TCP 路径形状与数据包一致；机器人端 IK 跟随")
            print(f"[home 映射] 新 home TCP(m/rad) = {tuple(round(v, 4) for v in args.home_pose)}")
    print(_summary(trajectory))
    if args.dry_run:
        if args.tcp_align:
            return _run_dry_run_tcp(trajectory, args)
        return _run_dry_run(trajectory, args)
    return _run_real(trajectory, args)


if __name__ == "__main__":
    raise SystemExit(main())
