"""``ur5_teleop.teleop`` 主循环单测：grip 语义、HOLD/FAULT、Ctrl-C、dry-run 冒烟。

用可控假时钟 + 脚本化后端/帧来源，不产生任何真实网络或机器人 I/O。
"""

import math
import pathlib
import subprocess
import sys
import unittest

from ur5_teleop.backend import DisconnectedError, RobotState
from ur5_teleop.config import Config, InputConfig, RobotConfig, SafetyConfig, TeleopConfig
from ur5_teleop.safety import TeleopState
from ur5_teleop.teleop import TeleopLoop, TeleopStep
from ur5_teleop.tracking import PicoFrame, ReplayFrameSource, parse_frame

ROOT = pathlib.Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "tests" / "data" / "sample_frames.jsonl"

START_POSE = (0.4, 0.0, 0.4, 0.0, 0.0, 0.0)
CONTROLLER_POSITION = (0.25, 1.2, -0.35)
HEAD = (0.0, 1.6, 0.0, 0.0, 0.0, 0.0, 1.0)
IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


class FakeClock:
    """可手动推进的单调时钟。"""

    def __init__(self, start=1000.0):
        self.now = float(start)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class ScriptedBackend:
    """按契约实现的最小后端：状态可注入、记录下发指令。"""

    def __init__(self, clock, pose=START_POSE):
        self.clock = clock
        self.pose = tuple(float(value) for value in pose)
        self.state_age = 0.0       # read_state 返回的状态“有多旧”
        self.connected = True
        self.sent: list[tuple[tuple[float, ...], bool]] = []

    def read_state(self):
        if not self.connected:
            raise DisconnectedError("脚本后端未连接")
        return RobotState(
            tcp_pose=self.pose,  # type: ignore[arg-type]
            joints=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            robot_mode=7,
            safety_status=1,
            timestamp=self.clock() - self.state_age,
        )

    def send_sets(self, pose6, servo):
        if not self.connected:
            raise DisconnectedError("脚本后端未连接")
        self.sent.append((tuple(float(value) for value in pose6), bool(servo)))

    def close(self):
        self.connected = False

    @property
    def last_command(self):
        return self.sent[-1] if self.sent else None


class ScriptedSource:
    """按顺序放出预先排好的帧；没有帧时 poll 返回 None。"""

    def __init__(self, clock):
        self.clock = clock
        self.frames: list[PicoFrame] = []
        self.last_receive: float | None = None

    def push(self, frame):
        self.frames.append(frame)

    def poll(self):
        if not self.frames:
            return None
        frame = self.frames.pop(0)
        self.last_receive = self.clock()
        return frame

    @property
    def exhausted(self):
        return False


class InterruptSource:
    """poll 直接抛 KeyboardInterrupt，用来测 Ctrl-C 安全退出。"""

    last_receive = None

    @property
    def exhausted(self):
        return False

    def poll(self):
        raise KeyboardInterrupt


def make_frame(stamp, controller_z, grip, controller_yaw=0.0):
    """构造一帧右侧手柄数据；控制器沿 PICO -Z 前进 → 机器人 +X。"""
    half = 0.5 * controller_yaw
    right = (CONTROLLER_POSITION[0], CONTROLLER_POSITION[1], float(controller_z),
             0.0, math.sin(half), 0.0, math.cos(half))
    left = (-0.25, 1.2, -0.3, 0.0, 0.0, 0.0, 1.0)
    return PicoFrame(
        pico_time_ns=int(stamp),
        left=left,
        right=right,
        head=HEAD,
        left_grip=0.1,
        right_grip=float(grip),
    )


def make_config(max_pos_step_m=0.02, input_cfg=None):
    return Config(
        robot=RobotConfig(),
        input=input_cfg or InputConfig(),
        teleop=TeleopConfig(scale=1.0, mount_yaw_deg=0.0,
                            max_pos_step_m=max_pos_step_m, max_rot_step_rad=0.05),
        safety=SafetyConfig(workspace_min=(-0.8, -0.8, 0.0),
                            workspace_max=(0.8, 0.8, 1.2),
                            deviation_pos_m=0.08, deviation_rot_rad=0.35,
                            deviation_hold_s=0.3, comm_timeout_s=0.1),
    )


class TeleopLoopTestBase(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.backend = ScriptedBackend(self.clock)
        self.source = ScriptedSource(self.clock)
        self.loop = TeleopLoop(make_config(), self.backend, self.source,
                               clock=self.clock)
        self.stamp = 1_000_000_000

    def step_with_frame(self, controller_z=CONTROLLER_POSITION[2], grip=0.9,
                        controller_yaw=0.0):
        self.stamp += 11_111_111
        self.source.push(make_frame(self.stamp, controller_z, grip, controller_yaw))
        return self.loop.step()

    def press(self, controller_z=CONTROLLER_POSITION[2]):
        return self.step_with_frame(controller_z=controller_z, grip=0.9)

    def release(self):
        return self.step_with_frame(grip=0.5)


class IdleAndClutchTest(TeleopLoopTestBase):
    def test_no_frames_sends_idle(self):
        step = self.loop.step()
        self.assertIsNotNone(step)
        self.assertIs(step.state, TeleopState.IDLE)
        self.assertFalse(step.servo)
        self.assertEqual(step.pose, START_POSE)  # 从未拿到目标时携带实测位姿
        self.assertEqual(len(self.backend.sent), 1)

    def test_grip_press_latches_start_pose(self):
        step = self.press()
        self.assertTrue(step.servo)
        self.assertTrue(self.loop.engaged)
        self.assertEqual(step.pose, START_POSE)
        self.assertIn("锁存", step.note)

    def test_controller_forward_moves_robot_plus_x(self):
        self.press()
        # 控制器沿 PICO -Z 前进 2 cm → 机器人 +X 2 cm（在 20 mm 限幅内）
        step = self.step_with_frame(controller_z=CONTROLLER_POSITION[2] - 0.02)
        self.assertAlmostEqual(step.pose[0], 0.42, places=6)
        self.assertAlmostEqual(step.pose[1], 0.0, places=6)
        self.assertAlmostEqual(step.pose[2], 0.4, places=6)

    def test_step_limit_caps_single_frame_jump(self):
        self.press()
        step = self.step_with_frame(controller_z=CONTROLLER_POSITION[2] - 0.5)
        self.assertAlmostEqual(step.pose[0], 0.42, places=6)  # 只允许 20 mm

    def test_every_step_sends_exactly_one_heartbeat(self):
        for _ in range(3):
            self.loop.step()
        self.assertEqual(len(self.backend.sent), 3)

    def test_grip_release_holds_last_target(self):
        self.press()
        tracking = self.step_with_frame(controller_z=CONTROLLER_POSITION[2] - 0.02)
        released = self.release()
        self.assertFalse(released.servo)
        self.assertEqual(released.pose, tracking.pose)
        # 松开后手继续动，目标不再推进
        for _ in range(3):
            step = self.step_with_frame(controller_z=CONTROLLER_POSITION[2] - 0.3, grip=0.5)
            self.assertFalse(step.servo)
            self.assertEqual(step.pose, tracking.pose)
        self.assertFalse(self.loop.engaged)

    def test_reengage_latches_new_origin(self):
        self.press()
        self.step_with_frame(controller_z=CONTROLLER_POSITION[2] - 0.02)
        self.release()
        # 松开期间机器人被别人移动/手动示教
        self.backend.pose = (0.5, 0.1, 0.45, 0.0, 0.0, 0.0)
        step = self.press(controller_z=CONTROLLER_POSITION[2] - 0.5)
        self.assertTrue(step.servo)
        self.assertAlmostEqual(step.pose[0], 0.5, places=6)
        self.assertAlmostEqual(step.pose[1], 0.1, places=6)

    def test_hysteresis_keeps_tracking_between_thresholds(self):
        self.press()
        # 0.7 在 stop(0.8) 与 release(0.6) 之间：保持跟踪
        step = self.step_with_frame(controller_z=CONTROLLER_POSITION[2] - 0.02, grip=0.7)
        self.assertTrue(step.servo)
        self.assertTrue(self.loop.engaged)


class WatchdogTest(TeleopLoopTestBase):
    def test_input_timeout_enters_hold(self):
        self.press()
        # 人为把“最近收帧时间”推到超时之前
        self.source.last_receive = self.clock() - 1.0
        step = self.loop.step()
        self.assertIs(step.state.value, "hold")
        self.assertFalse(step.servo)
        self.assertFalse(self.loop.engaged)
        self.assertIn("HOLD", step.note)

    def test_hold_auto_resumes_when_grip_held(self):
        self.press()
        self.source.last_receive = self.clock() - 1.0
        self.loop.step()
        # 帧恢复且 grip 仍按着：自动以当前位姿重新锁存起点继续跟踪（不要求松手重按）
        step = self.press()
        self.assertIs(step.state.value, "tracking")
        self.assertTrue(step.servo)
        self.assertTrue(self.loop.engaged)
        self.assertIn("自动重新锁存", step.note)

    def test_hold_then_release_stays_idle(self):
        self.press()
        self.source.last_receive = self.clock() - 1.0
        self.loop.step()
        # 帧恢复但已松开 grip：保持 idle，等待重新按下
        step = self.release()
        self.assertIs(step.state.value, "idle")
        self.assertFalse(step.servo)
        step = self.press()
        self.assertTrue(step.servo)
        self.assertTrue(self.loop.engaged)

    def test_state_timeout_enters_fault(self):
        self.press()
        self.backend.state_age = 1.0  # 状态帧过期 (> comm_timeout_s=0.1)
        step = self.loop.step()
        self.assertIs(step.state.value, "fault")
        self.assertFalse(step.servo)
        self.assertTrue(self.loop.faulted)

    def test_fault_is_latched_until_reset(self):
        self.press()
        self.backend.state_age = 1.0
        self.loop.step()
        self.backend.state_age = 0.0
        step = self.loop.step()
        self.assertIs(step.state.value, "fault")
        self.assertTrue(self.loop.faulted)
        self.assertTrue(self.loop.reset_fault())
        # 复位后要求先松开 grip
        step = self.press()
        self.assertFalse(step.servo)
        self.assertIn("松开 grip", step.note)
        self.release()
        self.assertTrue(self.press().servo)

    def test_reset_fault_without_fault_returns_false(self):
        self.assertFalse(self.loop.reset_fault())

    def test_deviation_watchdog_relatches_instead_of_faulting(self):
        self.press()
        # 机器人实际位置离目标很远（> deviation_pos_m）：不再锁存 FAULT，
        # 而是以当前实测位姿重新锁存起点继续跟随（大动作不中断）
        self.backend.pose = (0.6, 0.2, 0.4, 0.0, 0.0, 0.0)
        self.clock.advance(0.1)
        step = self.press()
        self.assertIs(step.state.value, "tracking")
        self.assertTrue(step.servo)
        self.assertFalse(self.loop.faulted)
        self.assertIn("自动重新锁存", step.note)
        # 重新锁存后目标应贴着实测位姿
        self.assertAlmostEqual(step.pose[0], 0.6, places=6)
        self.assertAlmostEqual(step.pose[1], 0.2, places=6)

    def test_disconnected_backend_faults(self):
        self.backend.connected = False
        step = self.loop.step()
        self.assertIs(step.state.value, "fault")
        self.assertTrue(self.loop.faulted)


class RunLoopTest(TeleopLoopTestBase):
    def test_ctrl_c_sends_idle_and_exits(self):
        loop = TeleopLoop(make_config(), self.backend, InterruptSource(),
                          clock=self.clock)
        state = loop.run()
        self.assertEqual(state.value, "idle")
        self.assertTrue(self.backend.sent)
        pose, servo = self.backend.sent[-1]
        self.assertFalse(servo)

    def test_run_stops_when_replay_exhausted(self):
        source = ReplayFrameSource([
            parse_frame({
                "pico_time_ns": 1_000,
                "left": [-0.25, 1.2, -0.3, 0.0, 0.0, 0.0, 1.0],
                "right": [0.25, 1.2, -0.35, 0.0, 0.0, 0.0, 1.0],
                "head": [0.0, 1.6, 0.0, 0.0, 0.0, 0.0, 1.0],
                "left_grip": 0.1,
                "right_grip": 0.9,
            }),
        ])
        loop = TeleopLoop(make_config(), self.backend, source, clock=self.clock)
        state = loop.run()
        self.assertEqual(state.value, "tracking")  # 退出前最后一帧在跟踪
        self.assertFalse(self.backend.sent[-1][1])  # 但额外发了一帧 idle 收尾

    def test_replay_source_exhaustion_ends_step(self):
        source = ReplayFrameSource([make_frame(1_000, -0.35, 0.9)])
        loop = TeleopLoop(make_config(), self.backend, source, clock=self.clock)
        self.assertIsNotNone(loop.step())
        self.assertIsNone(loop.step())


class DryRunSmokeTest(unittest.TestCase):
    def test_replay_cli_runs_and_prints_targets(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "dry_run.py"),
             "--replay", str(SAMPLE_PATH)],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("回放结束：共 20 帧", result.stdout)
        self.assertIn("tracking", result.stdout)
        # 打印了目标位姿与增量
        self.assertIn("pose=[", result.stdout)
        self.assertIn("dpos=", result.stdout)


if __name__ == "__main__":
    unittest.main()


class GripperTriggerMappingTest(unittest.TestCase):
    """扳机映射：analog 连续行程 / binary 过半闭合。"""

    def _cfg(self, **kwargs):
        from ur5_teleop.config import GripperConfig
        base = dict(enabled=False, port="/dev/null", trigger_side="right")
        base.update(kwargs)
        return GripperConfig(**base)

    def _pos(self, gripper_cfg, trigger):
        from types import SimpleNamespace
        from tools.teleop_supervisor import gripper_position_from_step
        step = TeleopStep(state=TeleopState.TRACKING, pose=(0, 0, 0, 0, 0, 0),
                          servo=True, frame=None, note="", trigger_left=trigger,
                          trigger_right=trigger)
        return gripper_position_from_step(SimpleNamespace(gripper=gripper_cfg), step)

    def test_analog_maps_trigger_continuously(self):
        cfg = self._cfg(trigger_mode="analog")
        self.assertEqual(self._pos(cfg, 0.0), 1000)
        self.assertEqual(self._pos(cfg, 0.5), 500)
        self.assertEqual(self._pos(cfg, 1.0), 0)

    def test_binary_uses_half_threshold(self):
        cfg = self._cfg(trigger_mode="binary")
        self.assertEqual(self._pos(cfg, 0.0), 1000)   # 松开=全开
        self.assertEqual(self._pos(cfg, 0.49), 1000)
        self.assertEqual(self._pos(cfg, 0.5), 0)      # 过半=闭合
        self.assertEqual(self._pos(cfg, 1.0), 0)


class GripperTriggerFilterTest(unittest.TestCase):
    """释放去抖：PICO 丢包时 trigger 掉 0 不应立刻张开。"""

    def _filter(self, mode="binary", debounce=1.0):
        from types import SimpleNamespace
        from tools.teleop_supervisor import GripperTriggerFilter
        cfg = SimpleNamespace(trigger_mode=mode, invert_trigger=False,
                              release_debounce_s=debounce)
        now = [0.0]
        f = GripperTriggerFilter(cfg, clock=lambda: now[0])
        return f, now

    def test_close_is_immediate_and_open_waits_debounce(self):
        f, now = self._filter(debounce=1.0)
        self.assertEqual(f.update(0.9), 0)      # 闭合立即
        self.assertEqual(f.update(0.0), 0)      # 低电平刚开始：保持闭合
        now[0] = 0.5
        self.assertEqual(f.update(0.0), 0)      # 0.5s：仍在去抖窗口
        now[0] = 1.0
        self.assertEqual(f.update(0.0), 1000)   # 持续 1s 才张开

    def test_short_zero_glitch_keeps_closed(self):
        """模拟丢包导致的短时 trigger=0：不应张开。"""
        f, now = self._filter(debounce=1.0)
        f.update(1.0)
        now[0] = 0.1; f.update(0.0)             # 丢包 0.3s
        now[0] = 0.3; f.update(0.0)
        now[0] = 0.5; self.assertEqual(f.update(1.0), 0)   # 恢复，仍闭合
        now[0] = 0.6; self.assertEqual(f.update(0.0), 0)   # 低电平重新计时

    def test_analog_mode_maps_directly(self):
        f, _ = self._filter(mode="analog")
        self.assertEqual(f.update(0.0), 1000)
        self.assertEqual(f.update(0.5), 500)
        self.assertEqual(f.update(1.0), 0)


class GripDebounceTest(TeleopLoopTestBase):
    """PICO grip 抖动/跟踪丢失时的离合去抖。"""

    def setUp(self):
        super().setUp()
        from ur5_teleop.config import InputConfig as _InputConfig
        cfg = make_config(input_cfg=_InputConfig(grip_release_debounce_s=0.5,
                                                 grip_reengage_debounce_s=0.25))
        self.loop = TeleopLoop(cfg, self.backend, self.source, clock=self.clock)

    def test_short_grip_dip_keeps_tracking(self):
        self.press()
        # 一次短暂掉到 0.5（PICO 抖动），未超过 0.5s 去抖窗口 → 继续跟随
        self.clock.advance(0.01)
        step = self.step_with_frame(grip=0.5)
        self.assertTrue(step.servo)
        self.assertTrue(self.loop.engaged)
        self.clock.advance(0.2)
        step = self.step_with_frame(grip=0.5)
        self.assertTrue(step.servo)
        self.assertTrue(self.loop.engaged)
        # 恢复按住 → 去抖计时清零
        self.clock.advance(0.01)
        step = self.step_with_frame(grip=0.9)
        self.assertTrue(step.servo)

    def test_sustained_grip_release_disengages(self):
        self.press()
        self.clock.advance(0.01)
        self.step_with_frame(grip=0.5)
        self.clock.advance(0.3)
        self.step_with_frame(grip=0.5)
        self.clock.advance(0.25)   # 累计 0.56s > 0.5s
        step = self.step_with_frame(grip=0.5)
        self.assertFalse(step.servo)
        self.assertFalse(self.loop.engaged)

    def test_soft_grip_reengages_without_repress(self):
        """跟踪丢失恢复后 grip 只回到 0.65：等 0.25s 自动续跟，不需松开再按。"""
        self.press()
        self.clock.advance(0.01)
        self.step_with_frame(grip=0.0)
        self.clock.advance(0.6)
        step = self.step_with_frame(grip=0.0)     # 超过去抖 → 松开
        self.assertFalse(self.loop.engaged)
        self.clock.advance(0.01)
        step = self.step_with_frame(grip=0.65)    # 软阈值：等待稳定
        self.assertFalse(step.servo)
        self.clock.advance(0.3)
        step = self.step_with_frame(grip=0.65)    # 0.3s ≥ 0.25s → 自动续跟
        self.assertTrue(step.servo)
        self.assertTrue(self.loop.engaged)
