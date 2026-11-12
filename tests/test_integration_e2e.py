"""端到端链路测试（无真机）：mock RTDE 服务端 + 回放 PICO 帧 + TeleopLoop + RTDEBackend。

验证的是"主机侧全链路"：PICO 帧 → 离合/位姿数学/限幅 → RTDE 输入寄存器 → mock 收到的值。
不验证 servoj / IK / 真机安全行为（那些只能在 URSim 或真机上验证）。
"""

from __future__ import annotations

import sys
import time
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mock_rtde_server import MockRTDEServer  # noqa: E402
from ur5_teleop import config as config_module  # noqa: E402
from ur5_teleop.teleop import TeleopLoop, TeleopStep  # noqa: E402
from ur5_teleop.tracking import ReplayFrameSource  # noqa: E402
from ur5_teleop.ur_client import RTDEBackend  # noqa: E402

SAMPLE_FRAMES = ROOT / "tests" / "data" / "sample_frames.jsonl"
CONFIG_PATH = ROOT / "config" / "ur5.toml"


def load_tolerant_config():
    """测试用配置：放宽通信超时——回放远快于墙钟，整套测试跑起来机器有负载时

    0.1 s 的状态帧超时会让主循环偶发进 FAULT（不是产品 bug，是测试时序假设）。
    """
    cfg = config_module.load(CONFIG_PATH)
    return config_module.Config(
        robot=cfg.robot,
        input=replace(cfg.input, frame_timeout_s=1.0),
        teleop=cfg.teleop,
        safety=replace(cfg.safety, comm_timeout_s=1.0),
        source=cfg.source,
    )


def wait_for(predicate, timeout_s: float = 2.0, interval_s: float = 0.01) -> bool:
    """mock 的输入包在会话线程里异步记录，断言前等它落盘。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


class EndToEndReplayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records: list[dict[str, object]] = []
        self.server = MockRTDEServer(port=0, on_input=self.records.append)
        self.server.start()
        self.addCleanup(self.server.close)
        self.backend = RTDEBackend(
            "127.0.0.1",
            protocol_version=2,
            frequency_hz=125.0,
            port=self.server.port,
        )
        self.addCleanup(self.backend.close)

    def test_replay_drives_rtde_registers(self) -> None:
        description = self.backend.connect()
        self.assertIn(f"127.0.0.1:{self.server.port}", description)
        cfg = load_tolerant_config()
        source = ReplayFrameSource.from_jsonl(SAMPLE_FRAMES)
        loop = TeleopLoop(cfg, self.backend, source)
        steps: list[TeleopStep] = []

        def on_step(step: TeleopStep) -> None:
            steps.append(step)
            # 让 mock 的实测位姿跟上刚下发的位置，模拟“机器人刚好跟上”
            if step.servo:
                self.server.set_state(tcp_pose=step.pose)

        state = loop.run(on_step)
        self.assertEqual(state.name, "TRACKING")  # 回放末帧 grip 仍按住
        self.assertFalse(loop.faulted)

        # 退出时 run() 会在 finally 里补一帧 idle；等 mock 把它记下来
        self.assertTrue(wait_for(lambda: bool(self.records)
                                 and int(self.records[-1]["input_int_register_1"]) == 0),
                        "退出前应补发 idle 心跳")

        # 1) RTDE 输入包确实到达，且含 6 个位姿 double + 命令字 + 心跳
        self.assertGreater(len(self.records), 0)
        for index in range(6):
            self.assertIn(f"input_double_register_{index}", self.records[-1])
        self.assertIn("input_int_register_1", self.records[-1])

        # 2) 命令字：至少出现过一次 servo(1)；退出时最后一帧是 idle(0)
        commands = [int(record["input_int_register_1"]) for record in self.records]
        self.assertIn(1, commands)
        self.assertEqual(commands[-1], 0, "退出时应发 idle")

        # 3) 心跳严格自增，且等于输入包序号
        heartbeats = [int(record["input_int_register_0"]) for record in self.records]
        self.assertEqual(heartbeats, list(range(1, len(heartbeats) + 1)))

        # 4) servo 帧的目标位姿：x 单调递增（回放里手柄沿头显前方前移）
        servo_poses = [
            tuple(float(record[f"input_double_register_{i}"]) for i in range(6))
            for record in self.records
            if int(record["input_int_register_1"]) == 1
        ]
        self.assertGreaterEqual(len(servo_poses), 5)
        xs = [pose[0] for pose in servo_poses]
        # 回放有两段离合：先朝一个方向移动、松开保持、再继续移动；
        # 方向取决于 config 的 mount_yaw（现场 180° 时为 -X），因此断言"单向单调"
        deltas = [later - earlier for earlier, later in zip(xs, xs[1:])]
        moved_positive = sum(1 for d in deltas if d > 0) >= sum(1 for d in deltas if d < 0)
        if moved_positive:
            self.assertTrue(all(d >= -1e-9 for d in deltas), f"x 不应回退：{xs}")
        else:
            self.assertTrue(all(d <= 1e-9 for d in deltas), f"x 不应回退：{xs}")
        self.assertGreaterEqual(sum(1 for d in deltas if abs(d) > 1e-9), 5)
        self.assertGreater(abs(xs[-1] - xs[0]), 0.03)

        # 5) 主机下发的目标与 mock 收到的一致（同一帧内无串包）
        tracking_steps = [step for step in steps if step.servo and step.state.name == "TRACKING"]
        self.assertGreaterEqual(len(tracking_steps), 5)
        last_pose = tracking_steps[-1].pose
        last_servo_record = next(
            record for record in reversed(self.records)
            if int(record["input_int_register_1"]) == 1
        )
        for index in range(6):
            self.assertAlmostEqual(
                float(last_servo_record[f"input_double_register_{index}"]),
                last_pose[index],
                places=6,
            )

        # 6) 状态反馈与 speed_scaling 可用
        robot_state = self.backend.read_state()
        self.assertEqual(robot_state.robot_mode, 7)
        self.assertEqual(robot_state.safety_status, 1)
        self.assertAlmostEqual(robot_state.speed_scaling, 1.0)


class IdleOnlyTest(unittest.TestCase):
    """没有 PICO 帧时，主循环只发 idle 心跳，不推进（安全默认）。"""

    def test_no_frames_means_idle_only(self) -> None:
        records: list[dict[str, object]] = []
        server = MockRTDEServer(port=0, on_input=records.append)
        server.start()
        self.addCleanup(server.close)
        backend = RTDEBackend("127.0.0.1", frequency_hz=125.0, port=server.port)
        self.addCleanup(backend.close)
        backend.connect()
        cfg = load_tolerant_config()
        source = ReplayFrameSource([])  # 空回放：一帧都没有
        loop = TeleopLoop(cfg, backend, source)
        state = loop.run()
        self.assertEqual(state.name, "IDLE")
        self.assertTrue(wait_for(lambda: bool(records)), "至少应发过一帧 idle 心跳")
        self.assertTrue(all(int(record["input_int_register_1"]) == 0 for record in records),
                        "没有任何 PICO 输入时不应出现 servo 命令")


if __name__ == "__main__":
    unittest.main()
