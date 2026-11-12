"""mock RTDE 服务端测试：原始协议帧 + vendor 官方 Python 客户端回环。

覆盖：
1. 握手命令字/包长/字节序（裸 socket，逐字节断言）
2. 输出 recipe 类型串、输入 recipe id 分配
3. 官方客户端 ``rtde.rtde.RTDE`` 完整回环：connect → setup → start → receive / send
4. 机器人状态注入、输入寄存器落库、START/PAUSE 语义、端口占用与释放
"""

from __future__ import annotations

import socket
import struct
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "RTDE_Python_Client_Library"
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mock_rtde_server import MockRTDEServer  # noqa: E402

import rtde.rtde as rtde  # noqa: E402

# 与 docs/design.md §3.1 一致
OUTPUT_NAMES = [
    "actual_TCP_pose",
    "actual_q",
    "robot_mode",
    "safety_status",
    "speed_scaling",
]
OUTPUT_TYPES = ["VECTOR6D", "VECTOR6D", "UINT32", "UINT32", "DOUBLE"]
INPUT_NAMES = [f"input_double_register_{i}" for i in range(6)] + [
    "input_int_register_0",
    "input_int_register_1",
]
INPUT_TYPES = ["DOUBLE"] * 6 + ["INT32", "INT32"]

POSE = (0.4, 0.0, 0.4, 0.0, 3.14, 0.0)
JOINTS = (-1.57, -1.2, 1.2, -1.57, -1.57, 0.0)

CMD_V = ord("V")
CMD_v = ord("v")
CMD_O = ord("O")
CMD_I = ord("I")
CMD_S = ord("S")
CMD_P = ord("P")
CMD_U = ord("U")


def wait_for(predicate, timeout: float = 2.0, interval: float = 0.005) -> bool:
    """轮询等待条件成立。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError(f"连接关闭，还差 {size - len(data)} 字节")
        data += chunk
    return data


class RawRTDE:
    """裸 socket 客户端，逐包发 RTDE 命令（用来钉死线上格式）。"""

    def __init__(self, port: int) -> None:
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=2.0)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def send(self, command: int, payload: bytes = b"") -> None:
        self.sock.sendall(struct.pack(">HB", 3 + len(payload), command) + payload)

    def recv_packet(self) -> tuple[int, bytes]:
        header = recv_exact(self.sock, 3)
        size, command = struct.unpack(">HB", header)
        assert size >= 3, f"非法包长 {size}"
        return command, recv_exact(self.sock, size - 3)

    def close(self) -> None:
        self.sock.close()


class MockRTDEServerProtocolTest(unittest.TestCase):
    """裸 socket 验证握手/recipe/数据包格式。"""

    def setUp(self) -> None:
        self.server = MockRTDEServer(port=0, tcp_pose=POSE, joints=JOINTS)
        self.server.start()
        self.addCleanup(self.server.stop)
        self.client = RawRTDE(self.server.port)
        self.addCleanup(self.client.close)

    def _negotiate(self, version: int = 2) -> None:
        self.client.send(CMD_V, struct.pack(">H", version))
        command, payload = self.client.recv_packet()
        self.assertEqual(command, CMD_V)
        self.assertEqual(payload, b"\x01")

    def _setup_outputs(self, names: list[str], frequency: float = 125.0) -> tuple[int, list[str]]:
        payload = struct.pack(">d", frequency) + ",".join(names).encode("utf-8")
        self.client.send(CMD_O, payload)
        command, reply = self.client.recv_packet()
        self.assertEqual(command, CMD_O)
        recipe_id = reply[0]
        types = reply[1:].decode("utf-8").split(",")
        return recipe_id, types

    def _setup_inputs(self, names: list[str]) -> tuple[int, list[str]]:
        self.client.send(CMD_I, ",".join(names).encode("utf-8"))
        command, reply = self.client.recv_packet()
        self.assertEqual(command, CMD_I)
        recipe_id = reply[0]
        types = reply[1:].decode("utf-8").split(",")
        return recipe_id, types

    def test_handshake_packet_framing(self) -> None:
        # REQUEST_PROTOCOL_VERSION: u16 size(=4) + u8 'V' + u8 accepted
        self._negotiate(2)
        self.assertEqual(self.server.protocol_version, 2)

        # GET_URCONTROL_VERSION: payload = 4×u32 大端
        self.client.send(CMD_v)
        command, payload = self.client.recv_packet()
        self.assertEqual(command, CMD_v)
        self.assertEqual(len(payload), 16)
        self.assertEqual(struct.unpack(">IIII", payload), (5, 26, 0, 0))

    def test_protocol_version_v1_v2_v3_accepted_and_garbage_rejected(self) -> None:
        for version in (1, 2, 3):
            client = RawRTDE(self.server.port)
            try:
                client.send(CMD_V, struct.pack(">H", version))
                command, payload = client.recv_packet()
                self.assertEqual((command, payload), (CMD_V, b"\x01"))
            finally:
                client.close()
        client = RawRTDE(self.server.port)
        try:
            client.send(CMD_V, struct.pack(">H", 9))
            self.assertEqual(client.recv_packet(), (CMD_V, b"\x00"))
        finally:
            client.close()

    def test_setup_outputs_returns_recipe_id_zero_and_types(self) -> None:
        self._negotiate()
        recipe_id, types = self._setup_outputs(OUTPUT_NAMES, frequency=125.0)
        self.assertEqual(recipe_id, 0)
        self.assertEqual(types, OUTPUT_TYPES)
        self.assertEqual(self.server.output_names, OUTPUT_NAMES)
        self.assertEqual(self.server.output_types, OUTPUT_TYPES)
        self.assertAlmostEqual(self.server.requested_frequency_hz, 125.0)

    def test_setup_inputs_allocates_incrementing_recipe_ids(self) -> None:
        self._negotiate()
        first_id, types = self._setup_inputs(INPUT_NAMES)
        self.assertEqual(first_id, 1)
        self.assertEqual(types, INPUT_TYPES)
        second_id, _ = self._setup_inputs(["input_double_register_0"])
        self.assertEqual(second_id, 2)
        self.assertIn(1, self.server.input_recipes)

    def test_unknown_variable_replies_not_found(self) -> None:
        self._negotiate()
        self.client.send(CMD_O, struct.pack(">d", 125.0) + b"no_such_output")
        self.assertEqual(
            self.client.recv_packet(), (CMD_O, b"\x00NOT_FOUND")
        )
        self.client.send(CMD_I, b"no_such_input")
        self.assertEqual(
            self.client.recv_packet(), (CMD_I, b"\x00NOT_FOUND")
        )

    def test_start_requires_output_recipe(self) -> None:
        self._negotiate()
        self.client.send(CMD_S)
        self.assertEqual(self.client.recv_packet(), (CMD_S, b"\x00"))
        self.assertFalse(self.server.started)

        self._setup_outputs(OUTPUT_NAMES)
        self.client.send(CMD_S)
        self.assertEqual(self.client.recv_packet(), (CMD_S, b"\x01"))
        self.assertTrue(self.server.started)

    def test_data_package_format_and_state_injection(self) -> None:
        self._negotiate()
        self._setup_outputs(OUTPUT_NAMES)
        self.client.send(CMD_S)
        self.assertEqual(self.client.recv_packet(), (CMD_S, b"\x01"))

        command, payload = self.client.recv_packet()
        self.assertEqual(command, CMD_U)
        # payload = u8 recipe_id + 6d(VECTOR6D) + 6d(VECTOR6D) + I + I + d
        self.assertEqual(payload[0], 0)
        values = struct.unpack(">6d6dIId", payload[1:])
        self.assertEqual(values[:6], POSE)
        self.assertEqual(values[6:12], JOINTS)
        self.assertEqual(values[12:], (7, 1, 1.0))

        # 注入新状态后，下一包必须反映新值
        new_pose = (0.5, -0.1, 0.2, 0.0, 2.9, 0.1)
        self.server.set_state(tcp_pose=new_pose, robot_mode=5, safety_status=2)
        seen = None
        for _ in range(50):
            command, payload = self.client.recv_packet()
            if command != CMD_U:
                continue
            values = struct.unpack(">6d6dIId", payload[1:])
            if values[:6] == new_pose:
                seen = values
                break
        self.assertIsNotNone(seen, "注入新位姿后 50 个包内未看到更新")
        self.assertEqual(seen[12:], (5, 2, 1.0))

    def test_input_data_package_is_parsed_into_last_input(self) -> None:
        self._negotiate()
        self._setup_outputs(OUTPUT_NAMES)
        input_id, _types = self._setup_inputs(INPUT_NAMES)
        self.client.send(CMD_S)
        self.client.recv_packet()

        target = (0.31, 0.02, 0.42, 0.01, 3.0, -0.02)
        payload = struct.pack(">B6dii", input_id, *target, 123, 1)
        self.client.send(CMD_U, payload)

        self.assertTrue(wait_for(lambda: self.server.input_count >= 1))
        self.assertEqual(self.server.last_input["input_double_register_0"], target[0])
        self.assertEqual(self.server.last_input["input_double_register_5"], target[5])
        self.assertEqual(self.server.last_input["input_int_register_0"], 123)
        self.assertEqual(self.server.last_input["input_int_register_1"], 1)

    def test_pause_stops_data_stream(self) -> None:
        self._negotiate()
        self._setup_outputs(OUTPUT_NAMES)
        self.client.send(CMD_S)
        self.assertEqual(self.client.recv_packet(), (CMD_S, b"\x01"))
        self.client.recv_packet()  # 至少收到一包数据
        self.client.send(CMD_P)
        self.assertEqual(self.client.recv_packet(), (CMD_P, b"\x01"))
        self.assertFalse(self.server.started)


class MockRTDEServerLifecycleTest(unittest.TestCase):
    """start/stop/close、随机端口、端口占用。"""

    def test_ephemeral_port_is_random_and_not_30004(self) -> None:
        server = MockRTDEServer(port=0)
        server.start()
        self.addCleanup(server.stop)
        self.assertGreater(server.port, 0)
        self.assertNotEqual(server.port, 30004)

    def test_port_busy_raises_os_error(self) -> None:
        first = MockRTDEServer(port=0)
        first.start()
        self.addCleanup(first.stop)
        second = MockRTDEServer(port=first.port)
        with self.assertRaises(OSError):
            second.start()
        second.stop()  # 没启动过也必须能安全 stop

    def test_stop_and_close_are_idempotent_and_release_port(self) -> None:
        server = MockRTDEServer(port=0)
        server.start()
        port = server.port
        server.stop()
        server.stop()
        server.close()
        self.assertFalse(server.started)

        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        finally:
            probe.close()

    def test_defaults_and_state_injection_validation(self) -> None:
        server = MockRTDEServer()
        self.assertIsNone(server.last_input)
        self.assertEqual(server.input_count, 0)
        self.assertEqual(server.tcp_pose, POSE)
        with self.assertRaises(ValueError):
            server.set_state(tcp_pose=(1.0, 2.0))
        with self.assertRaises(ValueError):
            server.set_state(joints=(1.0, 2.0, 3.0))


class MockRTDEServerOfficialClientTest(unittest.TestCase):
    """用 vendor 官方 Python 客户端做完整回环（验收硬要求）。"""

    def setUp(self) -> None:
        self.server = MockRTDEServer(port=0, tcp_pose=POSE, joints=JOINTS)
        self.server.start()
        self.addCleanup(self.server.stop)
        self.con: rtde.RTDE | None = None

    def tearDown(self) -> None:
        if self.con is not None:
            self.con.disconnect()

    def _handshake(self, frequency: float = 125.0):
        self.con = rtde.RTDE("127.0.0.1", self.server.port)
        self.con.connect()
        self.assertEqual(self.con.get_controller_version(), (5, 26, 0, 0))
        self.assertTrue(
            self.con.send_output_setup(OUTPUT_NAMES, OUTPUT_TYPES, frequency=frequency)
        )
        input_data = self.con.send_input_setup(INPUT_NAMES, INPUT_TYPES)
        self.assertIsNotNone(input_data)
        self.assertTrue(self.con.send_start())
        return input_data

    def test_official_client_receives_injected_state(self) -> None:
        self._handshake()
        state = self.con.receive()
        self.assertIsNotNone(state, "官方客户端 1 秒内没有收到 DATA_PACKAGE")
        self.assertEqual(list(state.actual_TCP_pose), list(POSE))
        self.assertEqual(list(state.actual_q), list(JOINTS))
        self.assertEqual(state.robot_mode, 7)
        self.assertEqual(state.safety_status, 1)
        self.assertAlmostEqual(state.speed_scaling, 1.0)

        new_pose = (0.45, -0.05, 0.3, 0.02, 3.0, -0.03)
        self.server.set_state(tcp_pose=new_pose)
        for _ in range(50):
            state = self.con.receive()
            if state is not None and list(state.actual_TCP_pose) == list(new_pose):
                break
        else:
            self.fail("注入新位姿后 50 帧内未收到更新")
        self.assertEqual(list(state.actual_q), list(JOINTS))

    def test_official_client_writes_input_registers(self) -> None:
        input_data = self._handshake()
        pose = (0.41, -0.12, 0.33, 0.01, 3.13, -0.02)
        for i, value in enumerate(pose):
            setattr(input_data, f"input_double_register_{i}", value)
        input_data.input_int_register_0 = 42
        input_data.input_int_register_1 = 1
        self.assertTrue(self.con.send(input_data))

        self.assertTrue(wait_for(lambda: self.server.input_count >= 1))
        self.assertEqual(
            [self.server.last_input[f"input_double_register_{i}"] for i in range(6)],
            list(pose),
        )
        self.assertEqual(self.server.last_input["input_int_register_0"], 42)
        self.assertEqual(self.server.last_input["input_int_register_1"], 1)

    def test_official_client_pause(self) -> None:
        self._handshake()
        self.assertTrue(self.con.send_pause())
        self.assertTrue(wait_for(lambda: not self.server.started))


if __name__ == "__main__":
    unittest.main()
