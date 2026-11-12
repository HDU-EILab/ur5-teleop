"""RTDE 后端与 Dashboard 客户端（纯 Python，无原生依赖）。

- RTDEBackend：实现 ur5_teleop.backend.URBackend 契约
  * 后台线程持续读 RTDE 输出帧（actual_TCP_pose / actual_q / robot_mode / safety_status / speed_scaling）
  * 保活线程以 keepalive_hz 发 idle 心跳，避免主机卡顿时机器人端 watchdog 误停
  * send_sets() 写一帧输入包（6 个目标位姿 double + 心跳 + 命令字），心跳自增
- DashboardClient：Dashboard Server 29999 的文本命令封装（只在上机运维工具里用）
- 辅助函数：send_program()/open_script_channel() 通过 30001/30002/30003 写 URScript

依据：docs/research/ur5-external-control.md §2.1/§2.2/§3.2；RTDE 线协议细节以
vendor/RTDE_Python_Client_Library（官方 BSD-3-Clause）为准。
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

_VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "RTDE_Python_Client_Library"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from rtde import rtde as _rtde  # noqa: E402

from .backend import BackendError, DisconnectedError, RobotState

RTDE_PORT = 30004
DASHBOARD_PORT = 29999

# 命令字（input_int_register_1）
#   0 = idle，1 = TCP 位姿伺服（ur5_teleop.script 遥操作）
#   2 = 关节伺服，3 = movej 到寄存器给定关节角（ur5_traj.script 离线轨迹播放）
CMD_IDLE = 0
CMD_SERVO = 1
CMD_JOINT_SERVO = 2
CMD_MOVE_START = 3

# 输入 recipe（与 urscript/ur5_teleop.script 的寄存器契约一致，见 docs/design.md §3.1）
INPUT_NAMES = tuple(f"input_double_register_{i}" for i in range(6)) + (
    "input_int_register_0",
    "input_int_register_1",
)
INPUT_TYPES = ("DOUBLE",) * 6 + ("INT32", "INT32")

# 输出 recipe 首选与降级（不同固件的 robot_mode/safety_status/runtime_state 类型不同：
# 实测 CB3 3.15.8 为 INT32/INT32/UINT32，e-Series 文档与 mock 为全 UINT32；
# 还要保留没有 runtime_state / speed_scaling 的老固件降级路径）
OUTPUT_RECIPES = (
    (
        ("actual_TCP_pose", "actual_q", "robot_mode", "safety_status", "speed_scaling",
         "runtime_state"),
        ("VECTOR6D", "VECTOR6D", "INT32", "INT32", "DOUBLE", "UINT32"),
    ),
    (
        ("actual_TCP_pose", "actual_q", "robot_mode", "safety_status", "speed_scaling",
         "runtime_state"),
        ("VECTOR6D", "VECTOR6D", "UINT32", "UINT32", "DOUBLE", "UINT32"),
    ),
    (
        ("actual_TCP_pose", "actual_q", "robot_mode", "safety_status", "speed_scaling"),
        ("VECTOR6D", "VECTOR6D", "INT32", "INT32", "DOUBLE"),
    ),
    (
        ("actual_TCP_pose", "actual_q", "robot_mode", "safety_status"),
        ("VECTOR6D", "VECTOR6D", "INT32", "INT32"),
    ),
    (
        ("actual_TCP_pose", "actual_q"),
        ("VECTOR6D", "VECTOR6D"),
    ),
)

ROBOT_MODE_NAMES = {
    0: "DISCONNECTED",
    1: "CONFIRM_SAFETY",
    2: "BOOTING",
    3: "POWER_OFF",
    4: "POWER_ON",
    5: "IDLE",
    6: "BACKDRIVE",
    7: "RUNNING",
}

SAFETY_MODE_NAMES = {
    1: "NORMAL",
    2: "REDUCED",
    3: "PROTECTIVE_STOP",
    4: "RECOVERY",
    5: "SAFEGUARD_STOP",
    6: "SYSTEM_EMERGENCY_STOP",
    7: "ROBOT_EMERGENCY_STOP",
    8: "VIOLATION",
    9: "FAULT",
}


def describe_generation(controller_version_major: int) -> str:
    """按控制器主版本号判断世代。"""
    if controller_version_major == 3:
        return "CB3 (PolyScope 3.x)"
    if controller_version_major == 5:
        return "e-Series (PolyScope 5.x)"
    if controller_version_major >= 10:
        return "PolyScope X (gSeries)"
    return "未知"


class _RTDE(_rtde.RTDE):
    """让 connect() 内的协议版本协商使用配置值（官方库默认 v2）。"""

    def __init__(self, hostname: str, port: int = RTDE_PORT, protocol_version: int = 2) -> None:
        super().__init__(hostname, port)
        self._protocol_version = int(protocol_version)

    def negotiate_protocol_version(self, version=None):  # type: ignore[override]
        return super().negotiate_protocol_version(
            self._protocol_version if version is None else version)


class RTDEBackend:
    """UR5 RTDE 后端，实现 backend.URBackend。"""

    def __init__(
        self,
        ip: str,
        protocol_version: int = 2,
        frequency_hz: float = 125.0,
        port: int = RTDE_PORT,
        state_timeout_s: float = 0.2,
        keepalive_hz: float = 20.0,
    ) -> None:
        if not ip:
            raise ValueError("ip 不能为空")
        if not 1.0 <= frequency_hz <= 500.0:
            raise ValueError(f"frequency_hz 必须在 [1, 500]，收到 {frequency_hz}")
        if keepalive_hz <= 0 or 1.0 / keepalive_hz >= state_timeout_s:
            raise ValueError("keepalive_hz 必须足够高（周期需明显小于 state_timeout_s）")
        self.ip = ip
        self.port = int(port)
        self.protocol_version = int(protocol_version)
        self.frequency_hz = float(frequency_hz)
        self.state_timeout_s = float(state_timeout_s)
        self.keepalive_hz = float(keepalive_hz)

        self._con: _rtde.RTDE | None = None
        self._input = None
        self._output_names: tuple[str, ...] = ()
        self._state: RobotState | None = None
        self._state_lock = threading.Lock()
        self._input_lock = threading.Lock()
        self._heartbeat = 0
        self._last_input: tuple[tuple[float, ...], int] = ((0.0,) * 6, CMD_IDLE)
        self._last_send = 0.0
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._connected = False
        self._controller = ""

    # ------------------------------------------------------------------ 连接
    def connect(self) -> str:
        if self._connected:
            return self._controller
        con = _RTDE(self.ip, self.port, self.protocol_version)
        try:
            con.connect()
            version = con.get_controller_version()
            if version[0] is None:
                raise BackendError(f"{self.ip}: 读取控制器版本失败")
            major, minor, bugfix, build = version
            recipe_ok = False
            for names, types in OUTPUT_RECIPES:
                if con.send_output_setup(list(names), list(types), self.frequency_hz):
                    self._output_names = tuple(names)
                    recipe_ok = True
                    break
            if not recipe_ok:
                raise BackendError(f"{self.ip}: RTDE 输出 recipe 配置失败（固件版本过旧？）")
            self._input = con.send_input_setup(list(INPUT_NAMES), list(INPUT_TYPES))
            if self._input is None:
                raise BackendError(f"{self.ip}: RTDE 输入 recipe 配置失败（寄存器被占用？）")
            if not con.send_start():
                raise BackendError(f"{self.ip}: RTDE 同步启动失败")
        except BackendError:
            _close_quietly(con)
            raise
        except _rtde.RTDEException as exc:
            _close_quietly(con)
            raise BackendError(f"{self.ip}: RTDE 协议错误：{exc}") from exc
        except (socket.timeout, OSError) as exc:
            _close_quietly(con)
            raise BackendError(f"{self.ip}: 连接 RTDE {self.port} 失败：{exc}") from exc

        self._con = con
        self._heartbeat = 0
        self._last_input = ((0.0,) * 6, CMD_IDLE)
        self._last_send = 0.0
        self._error = None
        self._stop.clear()
        self._connected = True
        self._controller = (
            f"RTDE 已连接 {self.ip}:{self.port}，控制器 "
            f"{major}.{minor}.{bugfix}-{build}"
            f"（{describe_generation(int(major))}），输出 {self.frequency_hz:g} Hz，"
            f"recipe={','.join(self._output_names)}")
        self._threads = [
            threading.Thread(target=self._receive_loop, name="rtde-receive", daemon=True),
            threading.Thread(target=self._keepalive_loop, name="rtde-keepalive", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        return self._controller

    # ------------------------------------------------------------------ 契约
    def read_state(self) -> RobotState:
        if not self._connected:
            raise DisconnectedError("RTDE 未连接")
        if self._error is not None:
            raise DisconnectedError(f"RTDE 连接异常：{self._error}") from self._error
        with self._state_lock:
            state = self._state
        if state is None:
            raise DisconnectedError("尚未收到 RTDE 状态帧")
        age = time.monotonic() - state.timestamp
        if age > self.state_timeout_s:
            raise DisconnectedError(f"RTDE 状态帧已 {age * 1000:.0f} ms 未更新")
        return state

    def send_sets(self, pose6: Sequence[float], servo: bool) -> None:
        self._write_input(_as_six(pose6), CMD_SERVO if servo else CMD_IDLE)

    def send_joint_sets(self, joints6: Sequence[float], servo: bool = True) -> None:
        """关节模式写一帧：6 个目标关节角（rad）。

        与 send_sets 共用 input_double_register_0..5，靠命令字区分：
        cmd=2 关节伺服 / cmd=0 idle（urscript/ur5_traj.script）。
        """
        self._write_input(_as_six(joints6), CMD_JOINT_SERVO if servo else CMD_IDLE)

    def send_move_to_start(self, joints6: Sequence[float]) -> None:
        """cmd=3：请求 ur5_traj.script 用 movej 走到这 6 个关节角（低速、阻塞）。"""
        self._write_input(_as_six(joints6), CMD_MOVE_START)

    def close(self) -> None:
        if not self._connected and self._con is None:
            return
        self._stop.set()
        # 先发一帧 idle（此时 connected 仍为 True，_write_input 才会真正下发），
        # 尽量让机器人保持而不是继续推进最后的目标
        try:
            if self._con is not None and self._input is not None:
                values, _ = self._last_input
                self._write_input(values, CMD_IDLE)
        except Exception:  # noqa: BLE001 - 关闭阶段吞掉所有异常
            pass
        self._connected = False
        for thread in self._threads:
            thread.join(timeout=0.5)
        self._threads = []
        con = self._con
        self._con = None
        self._input = None
        if con is not None:
            try:
                con.send_pause()
            except Exception:  # noqa: BLE001
                pass
            _close_quietly(con)

    # ------------------------------------------------------------------ 内部
    def _write_input(self, values: tuple[float, ...], command: int) -> None:
        con = self._con
        data = self._input
        if con is None or data is None or not self._connected:
            raise DisconnectedError("RTDE 未连接，无法发送目标")
        with self._input_lock:
            self._heartbeat = (self._heartbeat + 1) & 0x7FFFFFFF
            for index in range(6):
                setattr(data, INPUT_NAMES[index], float(values[index]))
            setattr(data, "input_int_register_0", self._heartbeat)
            setattr(data, "input_int_register_1", int(command))
            ok = con.send(data)
            if ok is not True:
                self._error = BackendError("RTDE 输入包发送失败")
                raise DisconnectedError("RTDE 输入包发送失败（连接可能已断开）")
            self._last_send = time.monotonic()
            self._last_input = (values, int(command))

    def _receive_loop(self) -> None:
        con = self._con
        if con is None:
            return
        while not self._stop.is_set():
            try:
                data = con.receive()
            except Exception as exc:  # noqa: BLE001 - 断连时官方库可能抛多种异常
                if not self._stop.is_set():
                    self._error = exc
                return
            if data is None:
                continue
            state = _state_from_data(data, self._output_names)
            with self._state_lock:
                self._state = state

    def _keepalive_loop(self) -> None:
        period = 1.0 / self.keepalive_hz
        while not self._stop.wait(period):
            if not self._connected:
                return
            if time.monotonic() - self._last_send < 2.0 * period:
                continue
            try:
                values, _ = self._last_input
                self._write_input(values, CMD_IDLE)
            except Exception:  # noqa: BLE001
                return


def _state_from_data(data, names: tuple[str, ...]) -> RobotState:
    pose = tuple(float(v) for v in data.actual_TCP_pose[:6])
    joints = tuple(float(v) for v in data.actual_q)
    robot_mode = int(getattr(data, "robot_mode", -1)) if "robot_mode" in names else -1
    safety = int(getattr(data, "safety_status", -1)) if "safety_status" in names else -1
    speed_scaling = float(getattr(data, "speed_scaling", -1.0)) if "speed_scaling" in names else -1.0
    runtime_state = int(getattr(data, "runtime_state", -1)) if "runtime_state" in names else -1
    return RobotState(
        tcp_pose=pose,  # type: ignore[arg-type]
        joints=joints,
        robot_mode=robot_mode,
        safety_status=safety,
        timestamp=time.monotonic(),
        speed_scaling=speed_scaling,
        runtime_state=runtime_state,
    )


def _as_six(values6: Sequence[float]) -> tuple[float, ...]:
    """校验并转成 6 元组（TCP 位姿和关节角共用）。"""
    values = tuple(float(v) for v in values6)
    if len(values) != 6:
        raise ValueError(f"必须是 6 维，收到 {len(values)}")
    return values


def _close_quietly(con) -> None:
    try:
        con.disconnect()
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------- Dashboard
class DashboardClient:
    """Dashboard Server（29999）文本协议封装。仅运维工具使用。"""

    def __init__(self, ip: str, port: int = DASHBOARD_PORT, timeout: float = 2.0) -> None:
        self.ip = ip
        self.port = int(port)
        self.timeout = float(timeout)
        self._sock: socket.socket | None = None
        self._banner = ""

    def connect(self) -> str:
        sock = socket.create_connection((self.ip, self.port), timeout=self.timeout)
        sock.settimeout(self.timeout)
        self._sock = sock
        self._banner = self._readline()
        return self._banner

    def command(self, cmd: str) -> str:
        if self._sock is None:
            self.connect()
        assert self._sock is not None
        self._sock.sendall((cmd.strip() + "\n").encode())
        return self._readline()

    def _readline(self) -> str:
        assert self._sock is not None
        chunks = bytearray()
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                chunk = self._sock.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                break
            chunks.extend(chunk)
            if b"\n" in chunks:
                break
        text = chunks.decode("utf-8", "replace").strip()
        return text

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self) -> "DashboardClient":
        self.connect()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # 只读
    def version(self) -> str:
        return self.command("version")

    def polyscope_version(self) -> str:
        return self.command("PolyscopeVersion")

    def robot_mode(self) -> str:
        return self.command("robotmode")

    def safety_mode(self) -> str:
        return self.command("safetymode")

    def program_state(self) -> str:
        return self.command("programState")

    def serial_number(self) -> str:
        return self.command("get serial number")

    def robot_model(self) -> str:
        return self.command("get robot model")

    # 状态操作（上机运维用，调用前确认安全）
    def power_on(self) -> str:
        return self.command("power on")

    def power_off(self) -> str:
        return self.command("power off")

    def brake_release(self) -> str:
        return self.command("brake release")

    def play(self) -> str:
        return self.command("play")

    def stop(self) -> str:
        return self.command("stop")

    def pause(self) -> str:
        return self.command("pause")

    def load(self, program: str) -> str:
        return self.command(f"load {program}")

    def unlock_protective_stop(self) -> str:
        return self.command("unlock protective stop")

    def close_safety_popup(self) -> str:
        return self.command("close safety popup")


# ------------------------------------------------------------------ URScript 下发
SCRIPT_PORTS = (30003, 30002, 30001)


def open_script_channel(ip: str, port: int, timeout: float = 3.0) -> socket.socket:
    sock = socket.create_connection((ip, port), timeout=timeout)
    sock.settimeout(timeout)
    return sock


def send_program(sock: socket.socket, program: str) -> None:
    """把 URScript 程序文本写入脚本端口。程序需自包含（定义函数并在末尾调用）。"""
    text = program if program.endswith("\n") else program + "\n"
    sock.sendall(text.encode("utf-8"))
