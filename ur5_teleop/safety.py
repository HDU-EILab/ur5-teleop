"""通信/偏差看门狗与遥操作状态机判定（纯逻辑，不碰网络/机器人，可单测）。

状态语义（docs/design.md §2、§5）：

- ``IDLE``：没有跟踪，持续发送 idle 心跳，机器人保持最后目标。
- ``TRACKING``：grip 按住、输入流与状态流都新鲜，发送 servo 目标。
- ``HOLD``：PICO 输入帧断流，立即 idle 停止推进；恢复后需要松开 grip 再重新离合。
- ``FAULT``：机器人状态流断流或目标/实测偏差超限，锁存故障，必须调
  ``TeleopLoop.reset_fault()`` 复位（复位后同样需要松开 grip 再重新离合）。
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

from . import frames


class TeleopState(enum.Enum):
    """遥操作主循环状态。"""

    IDLE = "idle"
    HOLD = "hold"
    TRACKING = "tracking"
    FAULT = "fault"


@dataclass(frozen=True)
class CommStatus:
    """一次通信看门狗判定的结果。

    ``*_age_s`` 为 None 表示“本轮从未收到”（区分于收到过但已超时）。
    """

    input_fresh: bool
    state_fresh: bool
    input_age_s: float | None
    state_age_s: float | None


class CommWatchdog:
    """通信看门狗：判定 PICO 输入帧与机器人状态帧是否超时。

    输入帧超时阈值取 ``input.frame_timeout_s``（PICO 断流），状态帧超时阈值
    取 ``safety.comm_timeout_s``（RTDE 状态断流）。
    """

    def __init__(self, input_timeout_s: float, state_timeout_s: float) -> None:
        self.input_timeout_s = float(input_timeout_s)
        self.state_timeout_s = float(state_timeout_s)

    def update(self, *, now: float, last_input_receive: float | None,
               last_state_receive: float | None) -> CommStatus:
        input_age = None if last_input_receive is None else max(0.0, now - last_input_receive)
        state_age = None if last_state_receive is None else max(0.0, now - last_state_receive)
        return CommStatus(
            input_fresh=input_age is not None and input_age <= self.input_timeout_s,
            state_fresh=state_age is not None and state_age <= self.state_timeout_s,
            input_age_s=input_age,
            state_age_s=state_age,
        )


def decide_state(*, comm: CommStatus, tracking: bool, fault: bool) -> TeleopState:
    """状态机判定（纯函数）。

    优先级：锁存故障 > 状态帧超时（FAULT）> 输入帧超时（HOLD）> TRACKING/IDLE。
    输入帧“从未收到”（age 为 None）不算 HOLD，返回 IDLE 继续等待。
    """
    if fault:
        return TeleopState.FAULT
    if not comm.state_fresh:
        return TeleopState.FAULT
    if comm.input_age_s is not None and not comm.input_fresh:
        return TeleopState.HOLD
    return TeleopState.TRACKING if tracking else TeleopState.IDLE


def pose_error(target_pose, actual_pose) -> tuple[float, float]:
    """两个 pose6 的误差：``(位置误差 m, 姿态误差 rad)``。"""
    position_error = math.sqrt(sum(
        (float(target_pose[i]) - float(actual_pose[i])) ** 2 for i in range(3)))
    target_rotation = frames.rotvec_to_matrix(target_pose[3:])
    actual_rotation = frames.rotvec_to_matrix(actual_pose[3:])
    relative = frames.mat_mul(target_rotation, frames.mat_transpose(actual_rotation))
    rotation_error = math.sqrt(sum(value * value for value in frames.matrix_to_rotvec(relative)))
    return position_error, rotation_error


class DeviationWatchdog:
    """偏差看门狗：目标 vs 实测超阈值并持续 ``hold_s`` 后锁存故障。

    “持续”指超限状态不间断；中途回到阈值内会清零计时。故障一旦锁存，
    只能 ``reset()`` 复位。
    """

    def __init__(self, pos_threshold_m: float, rot_threshold_rad: float, hold_s: float) -> None:
        self.pos_threshold_m = float(pos_threshold_m)
        self.rot_threshold_rad = float(rot_threshold_rad)
        self.hold_s = float(hold_s)
        self._since: float | None = None
        self._faulted = False

    @property
    def faulted(self) -> bool:
        return self._faulted

    @property
    def since(self) -> float | None:
        """本次超限开始的时刻；未超限为 None。"""
        return self._since

    def reset(self) -> None:
        self._since = None
        self._faulted = False

    def update(self, target_pose, actual_pose, now: float) -> bool:
        """喂一帧目标/实测；返回是否已判定故障（锁存）。"""
        if self._faulted:
            return True
        position_error, rotation_error = pose_error(target_pose, actual_pose)
        exceeded = (position_error > self.pos_threshold_m
                    or rotation_error > self.rot_threshold_rad)
        if not exceeded:
            self._since = None
            return False
        if self._since is None:
            self._since = float(now)
        if now - self._since >= self.hold_s:
            self._faulted = True
        return self._faulted
