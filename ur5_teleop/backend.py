"""机器人后端契约。

遥操作主循环只依赖这里的 ``URBackend`` 协议，不直接依赖 RTDE。
真实实现见 ``ur_client.RTDEBackend``；``FakeBackend`` 供 dry-run 与单元测试使用。

契约修改需 Lead 确认（见 docs/design.md §4）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


class BackendError(RuntimeError):
    """后端连接/配置/协议错误。"""


class DisconnectedError(BackendError):
    """与机器人（或模拟器）的连接已断开，或状态数据过期。"""


@dataclass(frozen=True)
class RobotState:
    """一帧机器人状态。

    tcp_pose: 基座系 TCP 位姿 [x, y, z, rx, ry, rz]，单位 m / rad（轴角旋转向量）。
    joints:   关节角（rad），UR5 为 6 个。
    timestamp: 主机收到该帧时的 time.monotonic()。
    """

    tcp_pose: tuple[float, float, float, float, float, float]
    joints: tuple[float, ...]
    robot_mode: int
    safety_status: int
    timestamp: float
    speed_scaling: float = -1.0     # RTDE speed_scaling（轨迹限速器缩放）；-1 表示未订阅/不可用
    runtime_state: int = -1         # RTDE runtime_state（1=程序已停止 2=运行中）；-1 表示未订阅


@runtime_checkable
class URBackend(Protocol):
    """机器人后端接口。"""

    def connect(self) -> str:
        """建立连接并启动数据通道；返回人类可读的描述串。失败抛 BackendError。"""

    def read_state(self) -> RobotState:
        """返回最近一帧状态；无数据或已断开时抛 DisconnectedError。"""

    def send_sets(self, pose6: Sequence[float], servo: bool) -> None:
        """写一帧目标。

        pose6: 目标 TCP 位姿（基座系，m / rad 轴角）。
        servo: True = 伺服推进；False = idle（机器人保持，不采纳该位姿）。
        心跳计数在此处自增，调用方无需关心。
        """

    def send_joint_sets(self, joints6: Sequence[float], servo: bool = True) -> None:
        """写一帧关节目标（6 个关节角，rad）。

        servo=True 进入关节伺服（ur5_traj.script cmd=2）；False 回 idle。
        与 send_sets 共用同一组寄存器，靠命令字区分模式。
        """

    def send_move_to_start(self, joints6: Sequence[float]) -> None:
        """请求机器人端用 movej 走到给定关节角（rad，低速阻塞移动）。"""

    def close(self) -> None:
        """释放连接；重复调用必须安全。"""


class FakeBackend:
    """不做任何网络 I/O 的假后端：记录收到的指令，状态可由测试直接喂入。"""

    def __init__(
        self,
        initial_pose: Sequence[float] = (0.4, 0.0, 0.4, 0.0, 3.14, 0.0),
        initial_joints: Sequence[float] = (-1.57, -1.2, 1.2, -1.57, -1.57, 0.0),
        robot_mode: int = 7,
        safety_status: int = 1,
    ) -> None:
        self._pose = tuple(float(v) for v in initial_pose)
        self._joints = tuple(float(v) for v in initial_joints)
        self._robot_mode = int(robot_mode)
        self._safety_status = int(safety_status)
        self._timestamp = time.monotonic()
        self.connected = False
        #: (pose6, servo, timestamp) 的发送记录，测试用
        self.sent: list[tuple[tuple[float, ...], bool, float]] = []
        #: (joints6, servo, timestamp) 的关节模式发送记录，测试用
        self.joint_sent: list[tuple[tuple[float, ...], bool, float]] = []
        #: movej 起始位姿请求记录，测试用
        self.move_start_sent: list[tuple[float, ...]] = []
        self.closed = False

    # -- URBackend --
    def connect(self) -> str:
        self.connected = True
        return "FakeBackend"

    def read_state(self) -> RobotState:
        if not self.connected:
            raise DisconnectedError("FakeBackend 未连接")
        return RobotState(
            tcp_pose=self._pose,  # type: ignore[arg-type]
            joints=self._joints,
            robot_mode=self._robot_mode,
            safety_status=self._safety_status,
            timestamp=self._timestamp,
            runtime_state=2,
        )

    def send_sets(self, pose6: Sequence[float], servo: bool) -> None:
        if not self.connected:
            raise DisconnectedError("FakeBackend 未连接")
        self.sent.append((tuple(float(v) for v in pose6), bool(servo), time.monotonic()))

    def send_joint_sets(self, joints6: Sequence[float], servo: bool = True) -> None:
        if not self.connected:
            raise DisconnectedError("FakeBackend 未连接")
        self.joint_sent.append(
            (tuple(float(v) for v in joints6), bool(servo), time.monotonic()))

    def send_move_to_start(self, joints6: Sequence[float]) -> None:
        if not self.connected:
            raise DisconnectedError("FakeBackend 未连接")
        self.move_start_sent.append(tuple(float(v) for v in joints6))

    def close(self) -> None:
        self.connected = False
        self.closed = True

    # -- 测试辅助 --
    def feed_state(
        self,
        pose: Sequence[float] | None = None,
        joints: Sequence[float] | None = None,
        robot_mode: int | None = None,
        safety_status: int | None = None,
    ) -> None:
        """模拟机器人反馈更新。"""
        if pose is not None:
            self._pose = tuple(float(v) for v in pose)
        if joints is not None:
            self._joints = tuple(float(v) for v in joints)
        if robot_mode is not None:
            self._robot_mode = int(robot_mode)
        if safety_status is not None:
            self._safety_status = int(safety_status)
        self._timestamp = time.monotonic()

    @property
    def last_command(self) -> tuple[tuple[float, ...], bool] | None:
        if not self.sent:
            return None
        pose, servo, _ = self.sent[-1]
        return pose, servo
