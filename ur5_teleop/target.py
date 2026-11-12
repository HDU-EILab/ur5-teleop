"""离合起点锁存、目标位姿合成、工作空间裁剪与每帧步长限幅。

流程（docs/design.md §2）：

1. grip 按下瞬间 ``engage()``：锁存机器人当前 ``actual_TCP_pose``（起点位姿）、
   手柄位置/姿态、头显四元数；
2. 每帧用锁存的头显水平 basis 把位移/姿态增量映射到 UR 基座系，与起点合成；
3. 目标位置裁进基座系 XYZ 工作空间盒；
4. 相对上一帧目标做位置/转角步长限幅，防止单帧跳变。

``scale`` 只作用于平移增量（与 upstream 一致），姿态 1:1。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import frames
from .config import SafetyConfig, TeleopConfig

Pose6 = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class ClutchOrigin:
    """一次离合锁存的全部起点信息。"""

    start_pose: Pose6            # grip 按下瞬间的机器人 TCP 位姿（基座系）
    controller_position: tuple[float, float, float]   # 手柄位置（PICO 世界系）
    controller_quaternion: tuple[float, float, float, float]  # 手柄姿态（PICO 世界系）
    head_quaternion: tuple[float, float, float, float]        # 头显姿态（定义 basis）


def clip_pose(pose, workspace_min, workspace_max) -> Pose6:
    """把 pose6 的位置裁进 XYZ 盒；姿态原样保留。"""
    values = tuple(float(value) for value in pose)
    if len(values) != 6:
        raise ValueError(f"位姿必须是 6 维 [x,y,z,rx,ry,rz]，收到 {len(values)} 维")
    return tuple(
        min(max(values[axis], float(workspace_min[axis])), float(workspace_max[axis]))
        if axis < 3 else values[axis]
        for axis in range(6)
    )


def limit_pose_step(previous, desired, max_pos_step_m: float, max_rot_step_rad: float) -> Pose6:
    """相对 ``previous`` 做每帧位置/转角限幅，返回不超过限幅的目标位姿。"""
    previous_values = tuple(float(value) for value in previous)
    desired_values = tuple(float(value) for value in desired)

    step = [desired_values[i] - previous_values[i] for i in range(3)]
    distance = math.sqrt(sum(value * value for value in step))
    if distance > max_pos_step_m:
        scale = max_pos_step_m / distance
        step = [value * scale for value in step]
    position = tuple(previous_values[i] + step[i] for i in range(3))

    previous_rotation = frames.rotvec_to_matrix(previous_values[3:])
    desired_rotation = frames.rotvec_to_matrix(desired_values[3:])
    relative = frames.mat_mul(desired_rotation, frames.mat_transpose(previous_rotation))
    rotvec = frames.matrix_to_rotvec(relative)
    angle = math.sqrt(sum(value * value for value in rotvec))
    if angle > max_rot_step_rad:
        scale = max_rot_step_rad / angle
        relative = frames.rotvec_to_matrix(tuple(value * scale for value in rotvec))
    rotation = frames.mat_mul(relative, previous_rotation)
    return position + frames.matrix_to_rotvec(rotation)


class TargetShaper:
    """离合 + 目标合成 + 工作空间裁剪 + 步长限幅。

    未离合时 ``next_target()`` 返回 None；``last_target`` 保留最后一次下发的
    目标，供 idle 心跳携带（机器人收到 idle 时不采纳位姿，仅用于日志/调试）。
    """

    def __init__(self, teleop: TeleopConfig, safety: SafetyConfig) -> None:
        self._scale = float(teleop.scale)
        self._rot_scale = float(teleop.rot_scale)
        self._mount_yaw_deg = float(teleop.mount_yaw_deg)
        self._max_pos_step_m = float(teleop.max_pos_step_m)
        self._max_rot_step_rad = float(teleop.max_rot_step_rad)
        self._workspace_min = tuple(float(v) for v in safety.workspace_min)
        self._workspace_max = tuple(float(v) for v in safety.workspace_max)
        self._origin: ClutchOrigin | None = None
        self._last_target: Pose6 | None = None

    @property
    def engaged(self) -> bool:
        return self._origin is not None

    @property
    def origin(self) -> ClutchOrigin | None:
        return self._origin

    @property
    def last_target(self) -> Pose6 | None:
        return self._last_target

    def engage(self, start_pose, controller_position, controller_quaternion,
               head_quaternion) -> ClutchOrigin:
        """锁存离合起点；起始位姿应是机器人当前实际 TCP 位姿。"""
        start = tuple(float(value) for value in start_pose)
        if len(start) != 6:
            raise ValueError(f"起始位姿必须是 6 维 [x,y,z,rx,ry,rz]，收到 {len(start)} 维")
        self._origin = ClutchOrigin(
            start_pose=start,  # type: ignore[arg-type]
            controller_position=tuple(float(v) for v in controller_position),
            controller_quaternion=tuple(float(v) for v in controller_quaternion),
            head_quaternion=tuple(float(v) for v in head_quaternion),
        )
        self._last_target = self._origin.start_pose
        return self._origin

    def disengage(self) -> None:
        """松开离合；保留 ``last_target`` 供 idle 保持。"""
        self._origin = None

    def desired_target(self, controller_position, controller_quaternion) -> Pose6:
        """由锁存起点与当前手柄位姿合成期望目标（已做工作空间裁剪，未做步长限幅）。"""
        origin = self._origin
        if origin is None:
            raise RuntimeError("离合未按下，没有可用的起始位姿")
        delta = frames.robot_delta(controller_position, origin.controller_position,
                                   origin.head_quaternion, self._mount_yaw_deg)
        delta = tuple(value * self._scale for value in delta)
        rotation = frames.robot_rotation(controller_quaternion, origin.controller_quaternion,
                                         origin.head_quaternion, self._mount_yaw_deg)
        if self._rot_scale != 1.0:
            rv = frames.matrix_to_rotvec(rotation)
            rotation = frames.rotvec_to_matrix(tuple(v * self._rot_scale for v in rv))
        pose = frames.compose_pose(origin.start_pose, delta, rotation)
        return clip_pose(pose, self._workspace_min, self._workspace_max)

    def next_target(self, controller_position, controller_quaternion) -> Pose6 | None:
        """合成并做每帧步长限幅后的目标；未离合时返回 None。"""
        if self._origin is None:
            return None
        desired = self.desired_target(controller_position, controller_quaternion)
        assert self._last_target is not None
        target = limit_pose_step(self._last_target, desired,
                                 self._max_pos_step_m, self._max_rot_step_rad)
        self._last_target = target
        return target
