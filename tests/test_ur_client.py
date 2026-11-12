"""RTDEBackend 集成测试：跑在 tests/mock_rtde_server.py 的模拟控制柜上。

只使用 ur5_teleop.ur_client 的公开 API（RTDEBackend(ip, protocol_version, frequency_hz,
port, ...) / connect() / read_state() / send_sets() / close()），不修改实现。

一旦 ur_client 尚未落地或不可导入，整组测试自动 skip，不影响 mock 自身的回环测试。
"""

from __future__ import annotations

import socket
import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mock_rtde_server import MockRTDEServer  # noqa: E402

from ur5_teleop.backend import (  # noqa: E402
    BackendError,
    DisconnectedError,
    RobotState,
    URBackend,
)

try:  # Lead 并行实现 ur_client.py；不存在/导入失败时整组测试 skip
    from ur5_teleop.ur_client import RTDEBackend

    _UR_CLIENT_IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001 - 缺文件或依赖问题都在这里降级为 skip
    RTDEBackend = None  # type: ignore[assignment,misc]
    _UR_CLIENT_IMPORT_ERROR = exc

HOST = "127.0.0.1"
POSE = (0.4, 0.0, 0.4, 0.0, 3.14, 0.0)
JOINTS = (-1.57, -1.2, 1.2, -1.57, -1.57, 0.0)

OUTPUT_NAMES = [
    "actual_TCP_pose",
    "actual_q",
    "robot_mode",
    "safety_status",
    "speed_scaling",
    "runtime_state",
]
INPUT_NAMES = [f"input_double_register_{i}" for i in range(6)] + [
    "input_int_register_0",
    "input_int_register_1",
]

HEARTBEAT_MASK = 0x7FFFFFFF


def wait_for(predicate, timeout: float = 2.0, interval: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def free_port() -> int:
    """拿一个空闲端口（不占用 30004）。"""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((HOST, 0))
        return int(probe.getsockname()[1])
    finally:
        probe.close()


def _pose_of(frame: dict) -> tuple[float, ...]:
    return tuple(frame[f"input_double_register_{i}"] for i in range(6))


class _BackendTestBase(unittest.TestCase):
    """起一个 mock 控制柜 + 一个 RTDEBackend，提供收集输入帧的辅助方法。"""

    def setUp(self) -> None:
        self.frames: list[dict] = []
        self._frames_lock = threading.Lock()
        self.server = MockRTDEServer(
            port=0,
            tcp_pose=POSE,
            joints=JOINTS,
            robot_mode=7,
            safety_status=1,
            on_input=self._on_input,
        )
        self.server.start()
        self.addCleanup(self.server.stop)
        self.backend = RTDEBackend(
            HOST, protocol_version=2, frequency_hz=125.0, port=self.server.port
        )
        self.addCleanup(self._safe_close)

    def _on_input(self, values: dict) -> None:
        with self._frames_lock:
            self.frames.append(dict(values))

    def _safe_close(self) -> None:
        try:
            self.backend.close()
        except Exception:  # noqa: BLE001 - cleanup 阶段不掩盖测试结果
            pass

    # -- 辅助 ----------------------------------------------------------------
    def _connect(self) -> str:
        return self.backend.connect()

    def _wait_state(self, predicate, timeout: float = 2.0) -> RobotState:
        """轮询 read_state 直到满足条件；首次收帧前 DisconnectedError 属于预期。"""
        deadline = time.monotonic() + timeout
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                state = self.backend.read_state()
            except DisconnectedError as exc:
                last_error = exc
            else:
                if predicate(state):
                    return state
            time.sleep(0.005)
        self.fail(f"{timeout}s 内未拿到满足条件的 RobotState（最后错误：{last_error!r}）")

    def _snapshot_frames(self) -> list[dict]:
        with self._frames_lock:
            return list(self.frames)

    def _wait_frame(self, predicate, timeout: float = 2.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for frame in reversed(self._snapshot_frames()):
                if predicate(frame):
                    return frame
            time.sleep(0.005)
        self.fail(f"{timeout}s 内未收到满足条件的输入帧")


@unittest.skipUnless(
    RTDEBackend is not None,
    f"ur5_teleop.ur_client 不可用（{_UR_CLIENT_IMPORT_ERROR!r}），先完成 mock 回环测试",
)
class RTDEBackendIntegrationTest(_BackendTestBase):
    """RTDEBackend ↔ mock 控制柜的端到端集成测试。"""

    def test_implements_contract_and_connect_description(self) -> None:
        description = self._connect()
        self.assertIsInstance(description, str)
        self.assertTrue(description)
        self.assertIn(HOST, description)
        self.assertIn(str(self.server.port), description)
        self.assertTrue(wait_for(lambda: self.server.started))
        self.assertEqual(self.server.output_names, OUTPUT_NAMES)
        self.assertTrue(self.server.input_recipes, "RTDE 输入 recipe 未配置")
        self.assertIsInstance(self.backend, URBackend)

    def test_read_state_matches_injected_robot_state(self) -> None:
        self._connect()
        state = self._wait_state(lambda item: item.tcp_pose == POSE)
        self.assertIsInstance(state, RobotState)
        self.assertEqual(state.tcp_pose, POSE)
        self.assertEqual(state.joints, JOINTS)
        self.assertEqual(state.robot_mode, 7)
        self.assertEqual(state.safety_status, 1)
        self.assertLessEqual(abs(state.timestamp - time.monotonic()), 2.0)

        new_pose = (0.45, -0.05, 0.3, 0.02, 3.0, -0.03)
        new_joints = (-1.5, -1.1, 1.1, -1.5, -1.5, 0.1)
        self.server.set_state(
            tcp_pose=new_pose,
            joints=new_joints,
            robot_mode=5,
            safety_status=2,
            speed_scaling=0.5,
        )
        state = self._wait_state(lambda item: item.tcp_pose == new_pose)
        self.assertEqual(state.joints, new_joints)
        self.assertEqual(state.robot_mode, 5)
        self.assertEqual(state.safety_status, 2)

    def test_send_sets_writes_pose_heartbeat_and_servo_flag(self) -> None:
        self._connect()
        self._wait_state(lambda item: item.tcp_pose == POSE)
        # 20 Hz 保活线程会先产生 idle 输入帧，确认通道已 STARTED
        self.assertTrue(wait_for(lambda: self.server.input_count >= 1), "保活输入帧未到达")

        target = (0.5, 0.1, 0.2, 0.0, 3.14, 0.0)
        self.backend.send_sets(target, True)
        servo_frame = self._wait_frame(
            lambda frame: frame["input_int_register_1"] == 1 and _pose_of(frame) == target
        )

        target2 = (0.51, 0.1, 0.2, 0.0, 3.14, 0.0)
        self.backend.send_sets(target2, False)
        idle_frame = self._wait_frame(
            lambda frame: frame["input_int_register_1"] == 0 and _pose_of(frame) == target2
        )

        # 心跳每写一帧 +1（含保活 idle 帧），且单调递增
        heartbeat_1 = servo_frame["input_int_register_0"]
        heartbeat_2 = idle_frame["input_int_register_0"]
        self.assertIsInstance(heartbeat_1, int)
        self.assertGreater(heartbeat_1, 0)
        self.assertGreater((heartbeat_2 - heartbeat_1) % (HEARTBEAT_MASK + 1), 0)
        frames = self._snapshot_frames()
        for previous, following in zip(frames, frames[1:]):
            self.assertEqual(
                following["input_int_register_0"],
                (previous["input_int_register_0"] + 1) & HEARTBEAT_MASK,
                "输入心跳必须每帧严格自增 1",
            )

    def test_read_state_times_out_when_output_stalls(self) -> None:
        self._connect()
        self._wait_state(lambda item: item.tcp_pose == POSE)
        self.server.pause_output()
        self.assertTrue(
            wait_for(self._read_state_raises_disconnected, timeout=2.0),
            "输出停推后 read_state 未在 2s 内抛 DisconnectedError",
        )
        self.server.resume_output()
        self._wait_state(lambda item: item.tcp_pose == POSE, timeout=2.0)

    def test_close_is_idempotent(self) -> None:
        self._connect()
        self._wait_state(lambda item: item.tcp_pose == POSE)
        self.backend.close()
        self.backend.close()  # 第二次必须安全
        with self.assertRaises(DisconnectedError):
            self.backend.read_state()
        self.assertTrue(wait_for(lambda: not self.server.started, timeout=2.0))

    def test_close_sends_final_idle_frame(self) -> None:
        self._connect()
        self._wait_state(lambda item: item.tcp_pose == POSE)
        target = (0.52, 0.02, 0.21, 0.0, 3.14, 0.0)
        self.backend.send_sets(target, True)
        self._wait_frame(
            lambda frame: frame["input_int_register_1"] == 1 and _pose_of(frame) == target
        )
        self.backend.close()
        # close() 契约：先发一帧 idle（保持最后目标位姿、命令字回 0）再断开
        self.assertTrue(wait_for(lambda: not self.server.started, timeout=2.0))
        last = self.server.last_input
        self.assertIsNotNone(last)
        self.assertEqual(last["input_int_register_1"], 0, "close() 应先发一帧 idle 再断开")
        self.assertEqual(_pose_of(last), target)

    def test_send_sets_rejects_bad_pose_length(self) -> None:
        self._connect()
        self._wait_state(lambda item: item.tcp_pose == POSE)
        with self.assertRaises(ValueError):
            self.backend.send_sets((0.1, 0.2, 0.3), True)

    def _read_state_raises_disconnected(self) -> bool:
        try:
            self.backend.read_state()
        except DisconnectedError:
            return True
        except Exception:  # noqa: BLE001 - 其它异常不算超时路径
            return False
        return False


@unittest.skipUnless(
    RTDEBackend is not None,
    f"ur5_teleop.ur_client 不可用（{_UR_CLIENT_IMPORT_ERROR!r}）",
)
class RTDEBackendFailureTest(unittest.TestCase):
    """连接失败与未连接路径（不需要 mock 服务端）。"""

    def test_connect_to_closed_port_raises_backend_error(self) -> None:
        backend = RTDEBackend(HOST, port=free_port())
        self.addCleanup(backend.close)
        with self.assertRaises(BackendError):
            backend.connect()

    def test_send_sets_before_connect_raises_disconnected(self) -> None:
        backend = RTDEBackend(HOST, port=free_port())
        self.addCleanup(backend.close)
        with self.assertRaises(DisconnectedError):
            backend.send_sets((0.0,) * 6, True)

    def test_read_state_before_connect_raises_disconnected(self) -> None:
        backend = RTDEBackend(HOST, port=free_port())
        self.addCleanup(backend.close)
        with self.assertRaises(DisconnectedError):
            backend.read_state()


if __name__ == "__main__":
    unittest.main()
