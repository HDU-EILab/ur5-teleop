#!/usr/bin/env python3
"""纯标准库的 RTDE 服务端模拟器（无真机联调 / 单元测试用）。

它模拟 UR 控制柜的 RTDE 接口（默认 TCP 30004），让主机侧 RTDE 客户端
（``ur5_teleop.ur_client.RTDEBackend`` 或 vendor 里的官方 Python 客户端）
在没有真机、没有 URSim 的情况下也能跑通握手、recipe 配置、状态流与寄存器写入。

协议依据（只读参考，不修改 vendor）：
- ``docs/research/ur5-external-control.md`` §1.4（握手/包格式/协议版本）、§2.2（寄存器分配）
- ``vendor/RTDE_Python_Client_Library/rtde/rtde.py`` 与 ``serialize.py``（字节序/打包格式以官方客户端为准）

线上格式（全部网络字节序，大端）::

    包头 = u16 size + u8 command          # size 含包头本身
    握手：V=86 请求协议版本，v=118 控制柜版本，O=79 输出 recipe，
          I=73 输入 recipe，S=83 开始，P=80 暂停
    数据包：U=85，payload = u8 recipe_id + 按 recipe 类型打包的值

输出变量（固定可用）：
    actual_TCP_pose VECTOR6D / actual_q VECTOR6D / robot_mode UINT32 /
    safety_status UINT32 / speed_scaling DOUBLE
    （另提供 runtime_state UINT32、timestamp DOUBLE 两个可选常见字段）

用法：
    # 单元测试
    server = MockRTDEServer(port=0)       # 0 = 随机空闲端口
    server.start()
    ...                                   # 客户端连 127.0.0.1:server.port
    server.tcp_pose = (0.4, 0.0, 0.4, 0.0, 3.14, 0.0)   # 注入机器人位姿
    server.stop()

    # 手动联调（CLI，打印收到的输入包）
    python3 tests/mock_rtde_server.py --port 30004

注意：mock 只验证“主机 → 机器人”的 RTDE 数据面，不能验证 servoj / IK / 真机安全行为。
"""

from __future__ import annotations

import argparse
import logging
import re
import socket
import struct
import sys
import threading
import time
from typing import Callable

_log = logging.getLogger("mock_rtde")

# ---------------------------------------------------------------------------
# RTDE 命令字（与官方客户端 rtde.Command 一致）
# ---------------------------------------------------------------------------
CMD_REQUEST_PROTOCOL_VERSION = ord("V")  # 86
CMD_GET_URCONTROL_VERSION = ord("v")  # 118
CMD_TEXT_MESSAGE = ord("M")  # 77
CMD_DATA_PACKAGE = ord("U")  # 85
CMD_CONTROL_PACKAGE_SETUP_OUTPUTS = ord("O")  # 79
CMD_CONTROL_PACKAGE_SETUP_INPUTS = ord("I")  # 73
CMD_CONTROL_PACKAGE_START = ord("S")  # 83
CMD_CONTROL_PACKAGE_PAUSE = ord("P")  # 80

#: 输出变量 → 线上类型。前 5 个是 mock 固定提供的状态字段。
OUTPUT_VARIABLE_TYPES: dict[str, str] = {
    "actual_TCP_pose": "VECTOR6D",
    "actual_q": "VECTOR6D",
    "robot_mode": "UINT32",
    "safety_status": "UINT32",
    "speed_scaling": "DOUBLE",
    # 可选字段：真实控制器也有；若客户端订阅就一并回，不订阅则无关紧要。
    "runtime_state": "UINT32",
    "timestamp": "DOUBLE",
}

#: 线上类型 → (struct 单值格式字符, 单值个数)
_WIRE_FORMAT: dict[str, tuple[str, int]] = {
    "BOOL": ("?", 1),
    "UINT8": ("B", 1),
    "INT32": ("i", 1),
    "UINT32": ("I", 1),
    "UINT64": ("Q", 1),
    "DOUBLE": ("d", 1),
    "VECTOR3D": ("d", 3),
    "VECTOR6D": ("d", 6),
    "VECTOR6INT32": ("i", 6),
    "VECTOR6UINT32": ("I", 6),
}

#: 输入寄存器名 → 线上类型（docs/design.md §3.1 用的就是 input_*_register_N）
_INPUT_REGISTER_RE = re.compile(r"^input_(double|int|bit)_register_(\d+)$")
_INPUT_REGISTER_TYPES = {"double": "DOUBLE", "int": "INT32", "bit": "BOOL"}


def _input_register_type(name: str) -> str | None:
    match = _INPUT_REGISTER_RE.match(name)
    if match is None:
        return None
    return _INPUT_REGISTER_TYPES[match.group(1)]


def _struct_format(types: list[str]) -> str:
    """把 recipe 类型列表拼成 struct 格式串（不含 recipe_id）。"""
    fmt = ">"
    for wire_type in types:
        char, count = _WIRE_FORMAT[wire_type]
        fmt += f"{count}{char}" if count > 1 else char
    return fmt


def _value_items(wire_type: str, value: object) -> tuple:
    """把单个 recipe 字段的标准值拆成 struct 参数序列。"""
    _, count = _WIRE_FORMAT[wire_type]
    if count == 1:
        return (value,)
    items = tuple(value)  # type: ignore[arg-type]
    if len(items) != count:
        raise ValueError(f"{wire_type} 需要 {count} 个值，收到 {len(items)} 个")
    return items


def _clamp_frequency(value: float, fallback: float) -> float:
    if not (value == value) or value in (float("inf"), float("-inf")):  # NaN / inf
        return fallback
    return min(500.0, max(1.0, float(value)))


class _Session:
    """一条客户端 TCP 连接上的 RTDE 会话状态。"""

    def __init__(self, server: "MockRTDEServer", conn: socket.socket) -> None:
        self.server = server
        self.conn: socket.socket | None = conn
        self.buf = b""
        # 协议版本：None = 客户端还没发 V，按 v2 解析
        self.protocol_version: int | None = None
        # 输出 recipe（多个客户端时各自独立）
        self.output_names: list[str] = []
        self.output_types: list[str] = []
        self.output_recipe_id = 0
        self.requested_frequency_hz = server.frequency_hz
        self.actual_frequency_hz = server.frequency_hz
        # 输入 recipe：recipe_id → (names, types)
        self.input_recipes: dict[int, tuple[list[str], list[str]]] = {}
        self.started = False
        self._send_lock = threading.Lock()
        self._push_stop = threading.Event()
        self._push_thread: threading.Thread | None = None

    # -- 网络 ---------------------------------------------------------------
    def send_packet(self, command: int, payload: bytes = b"") -> None:
        conn = self.conn
        if conn is None:
            raise OSError("连接已关闭")
        packet = struct.pack(">HB", 3 + len(payload), command) + payload
        with self._send_lock:
            conn.sendall(packet)

    def close(self) -> None:
        self._push_stop.set()
        conn, self.conn = self.conn, None
        if conn is not None:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
        self._join_push()
        self.started = False

    # -- 会话主循环 ----------------------------------------------------------
    def run(self) -> None:
        conn = self.conn
        assert conn is not None
        try:
            conn.settimeout(0.1)
            while not self.server._stop_event.is_set() and self.conn is not None:
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                self.buf += chunk
                try:
                    self._process_buffer()
                except ConnectionError as exc:
                    _log.error("协议错误，断开客户端：%s", exc)
                    break
        finally:
            self.close()
            self.server._forget_session(self)

    def _process_buffer(self) -> None:
        while len(self.buf) >= 3:
            size, command = struct.unpack_from(">HB", self.buf)
            if size < 3:
                raise ConnectionError(f"非法包长 {size}")
            if len(self.buf) < size:
                return
            payload = self.buf[3:size]
            self.buf = self.buf[size:]
            self._handle(command, payload)

    # -- 命令分派 ------------------------------------------------------------
    def _handle(self, command: int, payload: bytes) -> None:
        if command == CMD_REQUEST_PROTOCOL_VERSION:
            self._handle_protocol_version(payload)
        elif command == CMD_GET_URCONTROL_VERSION:
            self.send_packet(
                CMD_GET_URCONTROL_VERSION,
                struct.pack(">IIII", *self.server.controller_version),
            )
        elif command == CMD_CONTROL_PACKAGE_SETUP_OUTPUTS:
            self._handle_setup_outputs(payload)
        elif command == CMD_CONTROL_PACKAGE_SETUP_INPUTS:
            self._handle_setup_inputs(payload)
        elif command == CMD_CONTROL_PACKAGE_START:
            self._handle_start()
        elif command == CMD_CONTROL_PACKAGE_PAUSE:
            self._handle_pause()
        elif command == CMD_DATA_PACKAGE:
            self._handle_input_package(payload)
        elif command == CMD_TEXT_MESSAGE:
            self._handle_text_message(payload)
        else:
            _log.warning("未知 RTDE 命令字 %d，已忽略", command)

    def _handle_protocol_version(self, payload: bytes) -> None:
        if len(payload) < 2:
            self.send_packet(CMD_REQUEST_PROTOCOL_VERSION, b"\x00")
            return
        (version,) = struct.unpack_from(">H", payload)
        accepted = version in (1, 2, 3)
        if accepted:
            self.protocol_version = version
            self.server.protocol_version = version
        self.send_packet(CMD_REQUEST_PROTOCOL_VERSION, struct.pack(">B", int(accepted)))

    def _handle_setup_outputs(self, payload: bytes) -> None:
        # v2/v3: payload = f64 frequency + "var1,var2,..."；v1: 只有变量名
        version = self.protocol_version if self.protocol_version is not None else 2
        if version >= 2 and len(payload) >= 8:
            (frequency,) = struct.unpack_from(">d", payload)
            names_bytes = payload[8:]
            self.requested_frequency_hz = _clamp_frequency(
                frequency, self.server.frequency_hz
            )
        else:
            names_bytes = payload
            self.requested_frequency_hz = self.server.frequency_hz
        names = [name for name in names_bytes.decode("utf-8").split(",") if name]
        types: list[str] = []
        for name in names:
            wire_type = OUTPUT_VARIABLE_TYPES.get(name)
            if wire_type is None:
                _log.warning("输出 recipe 含未知变量 %r，回复 NOT_FOUND", name)
                self.send_packet(CMD_CONTROL_PACKAGE_SETUP_OUTPUTS, b"\x00NOT_FOUND")
                return
            types.append(wire_type)
        if not names:
            self.send_packet(CMD_CONTROL_PACKAGE_SETUP_OUTPUTS, b"\x00NOT_FOUND")
            return
        self.output_names = names
        self.output_types = types
        self.output_recipe_id = 0
        self.actual_frequency_hz = self.requested_frequency_hz
        self.server.output_names = list(names)
        self.server.output_types = list(types)
        self.server.requested_frequency_hz = self.requested_frequency_hz
        self.server.actual_frequency_hz = self.actual_frequency_hz
        recipe = bytes([self.output_recipe_id]) + ",".join(types).encode("utf-8")
        self.send_packet(CMD_CONTROL_PACKAGE_SETUP_OUTPUTS, recipe)
        _log.info(
            "输出 recipe 已配置：%.1f Hz，变量 %s", self.actual_frequency_hz, names
        )

    def _handle_setup_inputs(self, payload: bytes) -> None:
        names = [name for name in payload.decode("utf-8").split(",") if name]
        types: list[str] = []
        for name in names:
            wire_type = _input_register_type(name)
            if wire_type is None:
                _log.warning("输入 recipe 含未知变量 %r，回复 NOT_FOUND", name)
                self.send_packet(CMD_CONTROL_PACKAGE_SETUP_INPUTS, b"\x00NOT_FOUND")
                return
            types.append(wire_type)
        if not names:
            self.send_packet(CMD_CONTROL_PACKAGE_SETUP_INPUTS, b"\x00NOT_FOUND")
            return
        recipe_id = max(self.input_recipes, default=0) + 1
        self.input_recipes[recipe_id] = (names, types)
        self.server.input_recipes[recipe_id] = (list(names), list(types))
        recipe = bytes([recipe_id]) + ",".join(types).encode("utf-8")
        self.send_packet(CMD_CONTROL_PACKAGE_SETUP_INPUTS, recipe)
        _log.info("输入 recipe id=%d 已配置：变量 %s", recipe_id, names)

    def _handle_start(self) -> None:
        if not self.output_names:
            self.send_packet(CMD_CONTROL_PACKAGE_START, b"\x00")
            return
        self.started = True
        self.server.started = True
        self.send_packet(CMD_CONTROL_PACKAGE_START, b"\x01")
        _log.info("RTDE 同步已启动，按 %.1f Hz 推送数据", self.actual_frequency_hz)
        self._start_push()

    def _handle_pause(self) -> None:
        if not self.started:
            self.send_packet(CMD_CONTROL_PACKAGE_PAUSE, b"\x00")
            return
        self.started = False
        self.server.started = False
        self._push_stop.set()
        self._join_push()  # PAUSE 后立刻 START 能可靠重启推送线程
        self.send_packet(CMD_CONTROL_PACKAGE_PAUSE, b"\x01")
        _log.info("RTDE 同步已暂停")

    def _handle_input_package(self, payload: bytes) -> None:
        if len(payload) < 1:
            return
        recipe_id = payload[0]
        recipe = self.input_recipes.get(recipe_id)
        if recipe is None:
            _log.warning("收到未知输入 recipe id=%d 的数据包，已忽略", recipe_id)
            return
        names, types = recipe
        fmt = _struct_format(types)
        expected = struct.calcsize(fmt)
        data = payload[1:]
        if len(data) != expected:
            _log.warning(
                "输入包长度不符：recipe %d 需要 %d 字节，收到 %d 字节",
                recipe_id,
                expected,
                len(data),
            )
            return
        items = struct.unpack(fmt, data)
        values: dict[str, object] = {}
        offset = 0
        for name, wire_type in zip(names, types):
            _, count = _WIRE_FORMAT[wire_type]
            if count == 1:
                values[name] = items[offset]
                offset += 1
            else:
                values[name] = list(items[offset : offset + count])
                offset += count
        self.server._record_input(values)

    def _handle_text_message(self, payload: bytes) -> None:
        try:
            message_length = payload[0]
            message = payload[1 : 1 + message_length].decode("utf-8", "replace")
            rest = payload[1 + message_length :]
            source_length = rest[0]
            source = rest[1 : 1 + source_length].decode("utf-8", "replace")
            level = rest[1 + source_length]
        except IndexError:
            return
        self.server.last_text_message = {
            "level": level,
            "message": message,
            "source": source,
        }
        _log.info("收到 TEXT_MESSAGE：%s", message)

    # -- 数据推送 ------------------------------------------------------------
    def _start_push(self) -> None:
        if self._push_thread is not None and self._push_thread.is_alive():
            return
        self._push_stop.clear()
        thread = threading.Thread(
            target=self._push_loop, name="mock-rtde-push", daemon=True
        )
        self._push_thread = thread
        thread.start()

    def _join_push(self) -> None:
        thread = self._push_thread
        self._push_thread = None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def _push_loop(self) -> None:
        period = 1.0 / self.actual_frequency_hz if self.actual_frequency_hz > 0 else 0.008
        next_at = time.monotonic()
        while not self._push_stop.is_set() and not self.server._stop_event.is_set():
            try:
                self.send_packet(
                    CMD_DATA_PACKAGE,
                    self.server._build_output_payload(
                        self.output_names, self.output_types, self.output_recipe_id
                    ),
                )
            except OSError:
                break
            next_at += period
            delay = next_at - time.monotonic()
            if delay > 0:
                self._push_stop.wait(delay)
            else:
                # 落后了就不要追帧，避免恢复后突发一串包
                next_at = time.monotonic()


class MockRTDEServer:
    """RTDE 服务端模拟器。默认只绑 127.0.0.1（测试请用随机端口）。"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        frequency_hz: float = 125.0,
        controller_version: tuple[int, int, int, int] = (5, 26, 0, 0),
        tcp_pose: tuple[float, ...] = (0.4, 0.0, 0.4, 0.0, 3.14, 0.0),
        joints: tuple[float, ...] = (-1.57, -1.2, 1.2, -1.57, -1.57, 0.0),
        robot_mode: int = 7,
        safety_status: int = 1,
        speed_scaling: float = 1.0,
        runtime_state: int = 2,
        on_input: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        if len(tcp_pose) != 6:
            raise ValueError(f"tcp_pose 必须是 6 维，收到 {len(tcp_pose)} 维")
        if len(joints) != 6:
            raise ValueError(f"joints 必须是 6 维，收到 {len(joints)} 维")
        self.host = host
        self._requested_port = port
        self.port = port
        self.frequency_hz = float(frequency_hz)
        self.controller_version = tuple(int(v) for v in controller_version)

        # -- 机器人状态（可直接赋值注入）--
        self.tcp_pose: tuple[float, ...] = tuple(float(v) for v in tcp_pose)
        self.joints: tuple[float, ...] = tuple(float(v) for v in joints)
        self.robot_mode = int(robot_mode)
        self.safety_status = int(safety_status)
        self.speed_scaling = float(speed_scaling)
        self.runtime_state = int(runtime_state)

        # -- 测试可读的会话镜像 --
        self.protocol_version: int | None = None
        self.output_names: list[str] = []
        self.output_types: list[str] = []
        self.requested_frequency_hz = self.frequency_hz
        self.actual_frequency_hz = self.frequency_hz
        self.input_recipes: dict[int, tuple[list[str], list[str]]] = {}
        self.started = False
        self.last_input: dict[str, object] | None = None
        self.input_count = 0
        self.last_input_time: float | None = None
        self.last_text_message: dict[str, object] | None = None

        self.on_input = on_input
        self._input_lock = threading.Lock()
        self._listener: socket.socket | None = None
        self._stop_event = threading.Event()
        self._sessions: set[_Session] = set()
        self._threads: list[threading.Thread] = []

    # -- 生命周期 ------------------------------------------------------------
    def start(self) -> "MockRTDEServer":
        if self._listener is not None:
            return self
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((self.host, self._requested_port))
            listener.listen(2)
            listener.settimeout(0.1)
        except OSError:
            listener.close()
            raise
        self._listener = listener
        self.port = listener.getsockname()[1]
        self._stop_event.clear()
        thread = threading.Thread(
            target=self._accept_loop, name="mock-rtde-accept", daemon=True
        )
        self._threads.append(thread)
        thread.start()
        _log.info("mock RTDE server 监听 %s:%d", self.host, self.port)
        return self

    def stop(self) -> None:
        """停止监听并断开所有客户端；重复调用安全。"""
        self._stop_event.set()
        listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        for session in list(self._sessions):
            session.close()
        self._sessions.clear()
        for thread in list(self._threads):
            if thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=2.0)
        self._threads.clear()
        self.started = False
        _log.info("mock RTDE server 已停止")

    def close(self) -> None:
        """``stop()`` 的别名。"""
        self.stop()

    def pause_output(self) -> None:
        """停止推送输出帧但保持 TCP 连接（用于验证客户端的状态帧超时逻辑）。"""
        for session in list(self._sessions):
            session._push_stop.set()
            session._join_push()

    def resume_output(self) -> None:
        """恢复推送输出帧（客户端仍处于 STARTED 状态时有效）。"""
        for session in list(self._sessions):
            if session.started:
                session._start_push()

    # -- 状态注入 ------------------------------------------------------------
    def set_state(
        self,
        *,
        tcp_pose: tuple[float, ...] | None = None,
        joints: tuple[float, ...] | None = None,
        robot_mode: int | None = None,
        safety_status: int | None = None,
        speed_scaling: float | None = None,
        runtime_state: int | None = None,
    ) -> None:
        """按脚本注入机器人状态；未给出的字段保持不变。"""
        if tcp_pose is not None:
            pose = tuple(float(v) for v in tcp_pose)
            if len(pose) != 6:
                raise ValueError(f"tcp_pose 必须是 6 维，收到 {len(pose)} 维")
            self.tcp_pose = pose
        if joints is not None:
            values = tuple(float(v) for v in joints)
            if len(values) != 6:
                raise ValueError(f"joints 必须是 6 维，收到 {len(values)} 维")
            self.joints = values
        if robot_mode is not None:
            self.robot_mode = int(robot_mode)
        if safety_status is not None:
            self.safety_status = int(safety_status)
        if speed_scaling is not None:
            self.speed_scaling = float(speed_scaling)
        if runtime_state is not None:
            self.runtime_state = int(runtime_state)

    # -- 内部 ----------------------------------------------------------------
    def _accept_loop(self) -> None:
        while not self._stop_event.is_set():
            listener = self._listener
            if listener is None:
                break
            try:
                conn, _addr = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
            session = _Session(self, conn)
            self._sessions.add(session)
            thread = threading.Thread(
                target=session.run, name="mock-rtde-session", daemon=True
            )
            self._threads.append(thread)
            thread.start()

    def _forget_session(self, session: _Session) -> None:
        self._sessions.discard(session)

    def _record_input(self, values: dict[str, object]) -> None:
        with self._input_lock:
            self.input_count += 1
            self.last_input = values
            self.last_input_time = time.monotonic()
        callback = self.on_input
        if callback is not None:
            try:
                callback(values)
            except Exception:  # 回调异常不应影响服务端
                _log.exception("on_input 回调异常")

    def _current_output_values(self) -> dict[str, object]:
        return {
            "actual_TCP_pose": tuple(self.tcp_pose),
            "actual_q": tuple(self.joints),
            "robot_mode": self.robot_mode,
            "safety_status": self.safety_status,
            "speed_scaling": self.speed_scaling,
            "runtime_state": self.runtime_state,
            "timestamp": time.monotonic(),
        }

    def _build_output_payload(
        self, names: list[str], types: list[str], recipe_id: int
    ) -> bytes:
        values = self._current_output_values()
        fmt = ">B"
        args: list[object] = [recipe_id]
        for name, wire_type in zip(names, types):
            fmt += _struct_format([wire_type])[1:]
            args.extend(_value_items(wire_type, values[name]))
        return struct.pack(fmt, *args)


# ---------------------------------------------------------------------------
# CLI：手动联调，打印收到的输入包
# ---------------------------------------------------------------------------
def _format_value(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(f"{float(item):.6f}" for item in value) + "]"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="纯标准库 RTDE 服务端模拟器（手动联调时打印收到的输入包）"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="监听地址，默认 127.0.0.1（不要改成 0.0.0.0 暴露到 LAN）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30004,
        help="监听端口，默认 30004（与真机相同）；测试请用 0 或随机端口",
    )
    parser.add_argument(
        "--frequency-hz",
        type=float,
        default=125.0,
        help="客户端未指定时的默认输出频率（Hz），默认 125",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="不逐包打印输入，只在退出时打印统计",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [mock-rtde] %(message)s",
        datefmt="%H:%M:%S",
    )

    counter = {"count": 0}

    def print_input(values: dict[str, object]) -> None:
        counter["count"] += 1
        fields = " ".join(f"{name}={_format_value(value)}" for name, value in values.items())
        print(f"[{counter['count']:6d}] {fields}", flush=True)

    server = MockRTDEServer(
        host=args.host,
        port=args.port,
        frequency_hz=args.frequency_hz,
        on_input=None if args.quiet else print_input,
    )
    try:
        server.start()
    except OSError as exc:
        print(f"无法监听 {args.host}:{args.port}：{exc}", file=sys.stderr)
        return 1

    print(f"mock RTDE server 已监听 {server.host}:{server.port}", flush=True)
    print("输出变量：" + ", ".join(f"{name}({kind})" for name, kind in OUTPUT_VARIABLE_TYPES.items()), flush=True)
    print("提示：RTDEBackend / 官方客户端连 127.0.0.1:%d；Ctrl+C 退出" % server.port, flush=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print(flush=True)
        print(f"退出：共收到 {counter['count']} 个输入包", flush=True)
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
