"""遥操作主循环：收帧 → 状态机 → 目标合成/限幅 → ``backend.send_sets``。

只依赖 ``backend.URBackend`` 契约与 ``tracking.FrameSource``，不做任何真实
网络/机器人 I/O（RTDE 在 ``ur_client.RTDEBackend``，PICO 在 ``UdpFrameSource``）。

每步（``step()``）做四件事：发一帧心跳（idle 或 servo）、读一帧机器人状态、
收 PICO 帧、按状态机决定目标；``run()`` 是带 Ctrl-C 安全退出的循环版本：

- grip 按下（≥ ``grip_threshold``）：锁存机器人实际 TCP 位姿 + 手柄位置/姿态 +
  头显四元数为起点，进入 TRACKING；
- grip 松开（≤ ``grip_release_threshold``，回差防抖）：发 idle 保持最后目标；
- PICO 输入断流 → HOLD；机器人状态断流/偏差超限 → FAULT（锁存，需
  ``reset_fault()``）；HOLD/FAULT 恢复后都必须先松开 grip 才能重新离合；
- Ctrl-C：发一帧 idle 后退出（``run()``）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .backend import BackendError, URBackend
from .config import Config
from .safety import CommWatchdog, DeviationWatchdog, TeleopState, decide_state
from .target import TargetShaper
from .tracking import FrameSource, PicoFrame

#: 从未拿到机器人位姿时的占位 idle 位姿（servo=False 时机器人端不采纳该位姿）
_ZERO_POSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class TeleopStep:
    """一步的执行结果（dry-run 打印、测试断言用）。"""

    state: TeleopState
    pose: tuple[float, ...]      # 本步下发的位姿（idle 时为保持位姿）
    servo: bool                  # True = 推进；False = idle
    frame: PicoFrame | None      # 本步新收到的 PICO 帧（没有新帧为 None）
    note: str = ""               # 人读的说明/原因
    robot_pose: tuple[float, ...] = ()      # 本步实测 TCP 位姿（录制用）
    robot_joints: tuple[float, ...] = ()    # 本步实测关节角（录制用）
    grip_left: float = 0.0                  # 最近一帧手柄量（录制用）
    grip_right: float = 0.0
    trigger_left: float = 0.0
    trigger_right: float = 0.0
    buttons: dict[str, bool] = field(default_factory=dict)   # 手柄按键快照（采集状态机用）
    axis: dict[str, tuple[float, float]] = field(default_factory=dict)


class TeleopLoop:
    """遥操作主循环。"""

    def __init__(self, config: Config, backend: URBackend, source: FrameSource,
                 *, clock: Callable[[], float] = time.monotonic) -> None:
        self.config = config
        self._backend = backend
        self._source = source
        self._clock = clock
        self._shaper = TargetShaper(config.teleop, config.safety)
        self._comm = CommWatchdog(config.input.frame_timeout_s, config.safety.comm_timeout_s)
        self._deviation = DeviationWatchdog(
            config.safety.deviation_pos_m,
            config.safety.deviation_rot_rad,
            config.safety.deviation_hold_s,
        )
        self._state = TeleopState.IDLE
        self._faulted = False
        self._await_release = False
        self._resume_pending = False
        self._grip_low_since: float | None = None
        self._grip_high_since: float | None = None
        self._program_stopped_since: float | None = None
        self._last_frame: PicoFrame | None = None
        self._last_robot_pose: tuple[float, ...] | None = None
        self._last_robot_joints: tuple[float, ...] = ()
        self._period_s = 1.0 / config.robot.rtde_frequency_hz

    # -- 只读属性 --
    @property
    def state(self) -> TeleopState:
        return self._state

    @property
    def faulted(self) -> bool:
        return self._faulted

    @property
    def engaged(self) -> bool:
        return self._shaper.engaged

    @property
    def target(self) -> tuple[float, ...] | None:
        """最近一次下发/保持的目标位姿。"""
        return self._shaper.last_target

    # -- 对外操作 --
    def reset_fault(self) -> bool:
        """复位锁存故障（偏差/状态断流后由操作者调用）；返回是否真的复位了。

        复位后要求先松开 grip 再重新按下方可恢复跟踪，避免故障瞬间手还按着
        就直接续动。
        """
        if not self._faulted:
            return False
        self._faulted = False
        self._deviation.reset()
        self._shaper.disengage()
        self._await_release = True
        self._resume_pending = False
        self._state = TeleopState.IDLE
        return True

    def step(self) -> TeleopStep | None:
        """执行一步；回放源耗尽时返回 None（实时源不会）。"""
        if self._source.exhausted:
            return None
        now = self._clock()
        frame = self._source.poll()
        if frame is not None:
            self._last_frame = frame

        robot_state = self._read_state()
        if robot_state is None:
            self._faulted = True
            self._shaper.disengage()
            return self._send_idle(frame, TeleopState.FAULT, "机器人状态读取失败")
        self._last_robot_pose = tuple(float(v) for v in robot_state.tcp_pose)
        self._last_robot_joints = tuple(float(v) for v in robot_state.joints)

        # 机器人端程序监护：程序停了（崩溃/被外部停止）时 grip 必然无效，
        # 连续 1 s 仍未恢复就 FAULT，由上层重新下发程序（tools/teleop_supervisor.py）。
        if int(getattr(robot_state, "runtime_state", -1)) == 1:
            if self._program_stopped_since is None:
                self._program_stopped_since = now
            elif now - self._program_stopped_since > 1.0:
                self._faulted = True
                self._shaper.disengage()
                return self._send_idle(
                    frame, TeleopState.FAULT,
                    f"机器人端程序已停止（runtime_state=1，{now - self._program_stopped_since:.1f}s）")
        else:
            self._program_stopped_since = None

        comm = self._comm.update(
            now=now,
            last_input_receive=self._source.last_receive,
            last_state_receive=robot_state.timestamp,
        )

        # 通信/故障层面的状态判定（纯函数在 safety.decide_state）
        state_hint = decide_state(comm=comm, tracking=self._shaper.engaged,
                                  fault=self._faulted)
        if state_hint is TeleopState.FAULT:
            if not self._faulted:
                self._faulted = True
                self._shaper.disengage()
                return self._send_idle(
                    frame, TeleopState.FAULT,
                    f"机器人状态帧超时（{comm.state_age_s:.3f}s）")
            return self._send_idle(frame, TeleopState.FAULT, "故障未复位，保持 idle")
        if state_hint is TeleopState.HOLD:
            # PICO 断流：立即 idle 停止推进；恢复时若 grip 仍按着，自动以当前位姿重新锁存继续
            if self._shaper.engaged:
                self._shaper.disengage()
                self._resume_pending = True
            return self._send_idle(
                frame, TeleopState.HOLD,
                f"PICO 输入帧超时（{comm.input_age_s:.3f}s），HOLD")
        if self._last_frame is None:
            return self._send_idle(frame, TeleopState.IDLE, "等待 PICO 帧")

        current = self._last_frame
        side = self.config.input.side
        grip = current.grip(side)

        if self._resume_pending:
            self._resume_pending = False
            if grip >= self.config.input.grip_threshold:
                # 输入恢复且手还按着：以当前机器人位姿重新锁存起点，自动续跟（不会跳变）
                self._deviation.reset()
                self._shaper.engage(robot_state.tcp_pose, current.pose(side)[:3],
                                    current.pose(side)[3:], current.head[3:])
                return self._send(frame, TeleopState.TRACKING, self._shaper.last_target,
                                  True, "PICO 恢复，自动重新锁存起点")

        if self._await_release:
            if grip <= self.config.input.grip_release_threshold:
                self._await_release = False
                return self._send_idle(frame, TeleopState.IDLE, "grip 已松开，可重新离合")
            return self._send_idle(frame, TeleopState.IDLE, "等待松开 grip 后重新离合")

        hard = self.config.input.grip_threshold
        soft = self.config.input.grip_release_threshold

        if self._shaper.engaged:
            if grip <= soft:
                debounce = float(self.config.input.grip_release_debounce_s)
                if debounce <= 0.0:
                    self._shaper.disengage()
                    self._deviation.reset()
                    return self._send_idle(frame, TeleopState.IDLE, "grip 松开，保持最后目标")
                if self._grip_low_since is None:
                    self._grip_low_since = now
                if now - self._grip_low_since >= debounce:
                    self._grip_low_since = None
                    self._shaper.disengage()
                    self._deviation.reset()
                    return self._send_idle(frame, TeleopState.IDLE, "grip 松开，保持最后目标")
                # 去抖窗口内的短暂低电平：视为 PICO grip 抖动，继续跟随
            else:
                self._grip_low_since = None
        else:
            self._grip_low_since = None
            if grip < hard:
                # 软阈值续跟：PICO 跟踪短暂丢失后恢复时 grip 可能只回到 0.6~0.75，
                # 不经过"松开→重新按下"也自动重新锁存（持续 grip_reengage_debounce_s）
                debounce = float(self.config.input.grip_reengage_debounce_s)
                if debounce > 0.0 and grip >= soft:
                    if self._grip_high_since is None:
                        self._grip_high_since = now
                    if now - self._grip_high_since < debounce:
                        return self._send_idle(frame, TeleopState.IDLE, "等待 grip 稳定")
                else:
                    self._grip_high_since = None
                    return self._send_idle(frame, TeleopState.IDLE, "等待按下 grip")
            self._grip_high_since = None
            # 锁存机器人实际 TCP 位姿 + 手柄位置/姿态 + 头显姿态
            self._deviation.reset()
            self._shaper.engage(robot_state.tcp_pose, current.pose(side)[:3],
                                current.pose(side)[3:], current.head[3:])
            return self._send(frame, TeleopState.TRACKING, self._shaper.last_target,
                              True, "grip 按下，已锁存起点")

        controller_pose = current.pose(side)
        try:
            target = self._shaper.next_target(controller_pose[:3], controller_pose[3:])
        except (ValueError, RuntimeError) as exc:
            # 例如头显过度倾斜无法定义 heading basis：停止推进，要求重新离合
            self._shaper.disengage()
            self._await_release = True
            return self._send_idle(frame, TeleopState.HOLD, f"无法计算目标（{exc}），HOLD")
        assert target is not None
        self._deviation.update(target, robot_state.tcp_pose, now)
        if self._deviation.since is not None:
            # 大动作/快动作时目标必然领先实测较多：这里不做故障停机，
            # 直接以当前实测位姿重新锁存起点（相当于离合打滑），机器人继续跟随。
            self._deviation.reset()
            self._shaper.engage(robot_state.tcp_pose, controller_pose[:3],
                                controller_pose[3:], current.head[3:])
            return self._send(frame, TeleopState.TRACKING, self._shaper.last_target,
                              True, "目标领先实测过多，自动重新锁存继续")
        return self._send(frame, TeleopState.TRACKING, target, True, "")

    def run(self, on_step: Callable[[TeleopStep], None] | None = None) -> TeleopState:
        """阻塞运行主循环；Ctrl-C 时发 idle 退出。返回退出时的状态。"""
        try:
            while True:
                step = self.step()
                if step is None:
                    break
                if on_step is not None:
                    on_step(step)
                if step.state is TeleopState.FAULT:
                    break
                time.sleep(self._period_s)
        except KeyboardInterrupt:
            pass
        finally:
            self._send_idle_heartbeat()
        return self._state

    # -- 内部辅助 --
    def _read_state(self):
        try:
            return self._backend.read_state()
        except BackendError:
            return None

    def _hold_pose(self) -> tuple[float, ...]:
        """idle 心跳携带的位姿：优先最后目标，其次最近实测，最后占位零位姿。"""
        if self._shaper.last_target is not None:
            return tuple(self._shaper.last_target)
        if self._last_robot_pose is not None:
            return self._last_robot_pose
        return _ZERO_POSE

    def _extras(self) -> dict:
        """录制用的附加状态：实测位姿/关节 + 最近一帧手柄量。"""
        frame = self._last_frame
        return {
            "robot_pose": self._last_robot_pose or (),
            "robot_joints": self._last_robot_joints,
            "grip_left": frame.left_grip if frame is not None else 0.0,
            "grip_right": frame.right_grip if frame is not None else 0.0,
            "trigger_left": frame.left_trigger if frame is not None else 0.0,
            "trigger_right": frame.right_trigger if frame is not None else 0.0,
            "buttons": dict(frame.buttons) if frame is not None else {},
            "axis": dict(frame.axis) if frame is not None else {},
        }

    def _send(self, frame: PicoFrame | None, state: TeleopState, pose,
              servo: bool, note: str) -> TeleopStep:
        pose_tuple = tuple(float(value) for value in pose)
        extras = self._extras()
        try:
            self._backend.send_sets(pose_tuple, servo)
        except BackendError as exc:
            # 发送路径出错同样按故障处理，避免“以为在发、其实没发”
            self._faulted = True
            self._shaper.disengage()
            self._state = TeleopState.FAULT
            return TeleopStep(TeleopState.FAULT, pose_tuple, servo, frame,
                              f"发送失败（{exc}）", **extras)
        self._state = state
        return TeleopStep(state, pose_tuple, servo, frame, note, **extras)

    def _send_idle(self, frame: PicoFrame | None, state: TeleopState, note: str) -> TeleopStep:
        return self._send(frame, state, self._hold_pose(), False, note)

    def _send_idle_heartbeat(self) -> None:
        """退出时的最后一帧 idle（尽力而为，不再改状态）。"""
        try:
            self._backend.send_sets(self._hold_pose(), False)
        except BackendError:
            pass
