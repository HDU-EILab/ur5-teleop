"""大寰 PGC 系列电爪：RS485 / Modbus RTU 控制 + 状态回读。

现场实测（2026-11-10，PGC-300-60）：串口 ``/dev/cu.usbserial-BG022E33``，
``115200 8N1``，从站 ``1``；寄存器与 PGE/PGSE 系列一致：

================  ======  =====================================
寄存器             地址     说明
================  ======  =====================================
力                 0x0101  20-100（%）
目标位置           0x0103  0-1000（0=闭合，1000=全开）
速度               0x0104  0-100（%）
使能               0x0106  写 1 使能
夹持状态           0x0201  0=运动中 1=到位 2=夹到物体 3=掉落
实时位置           0x0202  0-1000
================  ======  =====================================

设计要点：

- 所有串口 I/O 都在后台线程里做，遥操作主循环只调用 :meth:`Gripper.set_position`；
- 串口异常只写进 ``last_error`` / ``healthy``，不把异常抛进主循环；
- 只有遥操作处于 TRACKING 时才下发目标（见 ``tools/teleop_supervisor.py``），
  其余状态调用 :meth:`Gripper.hold`，夹爪保持当前位置不动。
"""

from __future__ import annotations

import threading
import logging
import time
from types import MappingProxyType
from typing import Sequence

#: 寄存器地址（16 位，见模块说明）
REG_FORCE = 0x0101
REG_TARGET_POSITION = 0x0103
REG_SPEED = 0x0104
REG_ENABLE = 0x0106
REG_GRIP_STATE = 0x0201
REG_POSITION = 0x0202

LOG = logging.getLogger(__name__)

POSITION_MIN = 0
POSITION_MAX = 1000

#: 夹持状态（0x0201）含义
GRIP_STATE_NAMES = {0: "moving", 1: "arrived", 2: "grasped", 3: "dropped"}


class GripperError(RuntimeError):
    """夹爪通信/协议错误。"""


class ModbusResponseError(GripperError):
    """No complete, CRC-valid response arrived for the transaction."""


def crc16(data: bytes) -> int:
    """Modbus RTU CRC16（多项式 0xA001，初值 0xFFFF）。"""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


def _with_crc(frame: bytes) -> bytes:
    return frame + crc16(frame).to_bytes(2, "little")


def build_read(slave_id: int, address: int, count: int) -> bytes:
    """构造 FC03（读保持寄存器）请求帧。"""
    if not 1 <= count <= 125:
        raise ValueError(f"读取寄存器数量必须在 1..125，收到 {count}")
    return _with_crc(bytes([slave_id, 0x03, (address >> 8) & 0xFF, address & 0xFF,
                            0x00, count]))


def build_write(slave_id: int, address: int, values: Sequence[int]) -> bytes:
    """构造 FC06/FC10（写单个/多个寄存器）请求帧。"""
    values = list(values)
    if not values:
        raise ValueError("写入值不能为空")
    for value in values:
        if not 0 <= value <= 0xFFFF:
            raise ValueError(f"寄存器值必须在 0..0xFFFF，收到 {value}")
    if len(values) == 1:
        return _with_crc(bytes([slave_id, 0x06, (address >> 8) & 0xFF, address & 0xFF,
                                (values[0] >> 8) & 0xFF, values[0] & 0xFF]))
    frame = bytes([slave_id, 0x10, (address >> 8) & 0xFF, address & 0xFF,
                   (len(values) >> 8) & 0xFF, len(values) & 0xFF, len(values) * 2])
    for value in values:
        frame += bytes([(value >> 8) & 0xFF, value & 0xFF])
    return _with_crc(frame)


class SerialTransport:
    """pyserial 的薄封装；只在真机使用（测试用假 transport）。"""

    def __init__(self, port: str, baud: int, timeout_s: float = 0.05) -> None:
        import serial  # 延迟导入：单测/无夹爪环境不需要 pyserial

        self._serial = serial.Serial(port, baudrate=baud, bytesize=8, parity="N",
                                     stopbits=1, timeout=timeout_s, exclusive=True)

    def write(self, data: bytes) -> None:
        self._serial.write(data)
        self._serial.flush()

    def read(self, size: int) -> bytes:
        # Block for the first byte only; drain already buffered bytes together.
        # Waiting for an entire requested frame would delay shorter exception replies.
        available = self._serial.in_waiting
        return self._serial.read(min(size, available) if available else 1)

    def reset_input_buffer(self) -> None:
        self._serial.reset_input_buffer()

    def close(self) -> None:
        self._serial.close()


class Gripper:
    """大寰 PGC 电爪控制器（后台线程轮询/下发）。

    :param transport: 注入自定义传输层（测试用）；为 None 时按 ``port``/``baud`` 打开串口。
    """

    def __init__(self, port: str = "/dev/cu.usbserial-BG022E33", *, baud: int = 115200,
                 slave_id: int = 1, speed: int = 100, force: int = 20,
                 rate_hz: float = 40.0, read_hz: float = 10.0,
                 transport=None, read_only: bool = False) -> None:
        if not 1 <= slave_id <= 247:
            raise ValueError(f"slave_id 必须在 1..247，收到 {slave_id}")
        self.port = port
        self.baud = int(baud)
        self.slave_id = int(slave_id)
        self.speed = max(0, min(100, int(speed)))
        self.force = max(20, min(100, int(force)))
        self._period_s = 1.0 / max(1.0, float(rate_hz))
        self._read_period_s = 1.0 / max(1.0, float(read_hz))
        self._transport = transport
        self._own_transport = transport is None
        self.read_only = read_only
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._hold_requested = threading.Event()
        self._held = threading.Event()
        self._thread: threading.Thread | None = None

        self._target: int | None = None      # 待下发目标（None = 不下发，保持）
        self._last_delivered: int | None = None
        # Delivery metadata belongs to the serial I/O owner. It is exposed only
        # when a subsequent full status read is published, never by relabeling
        # cached feedback at write time.
        self._delivery_generation = 0
        self._delivered_at = 0.0
        self._feedback_generation = 0
        self._feedback = MappingProxyType({
            "target": None, "delivery_generation": 0, "delivered_at": 0.0,
            "generation": 0, "read_at": 0.0, "state": None, "position": None,
        })
        self._motion_status = None
        self.position_raw: int | None = None
        self.grip_state: int | None = None
        self.healthy = False
        self.last_error: str | None = None
        self.last_read_at = 0.0
        self._next_read = 0.0
        self._response_timeout_s = 0.1
        self._last_exchange_end = 0.0
        # Modbus RTU recommends a fixed 1.75 ms inter-frame gap above 19200 baud.
        self._frame_gap_s = max(3.5 * 11 / self.baud, 0.00175)
        self.communication_resyncs = 0
        self.communication_retries = 0

    # -- 生命周期 --
    def connect(self) -> str:
        """打开串口、发使能/速度/力设置并启动后台线程；失败抛 :class:`GripperError`。"""
        if self._thread is not None:
            return self._describe()
        try:
            if self._transport is None:
                self._transport = SerialTransport(self.port, self.baud)
            if not self.read_only:
                self._write_register(REG_ENABLE, 1)
                self._write_register(REG_SPEED, self.speed)
                self._write_register(REG_FORCE, self.force)
            self._read_status()
        except Exception as exc:  # noqa: BLE001 - 统一包装成 GripperError
            self._safe_close_transport()
            raise GripperError(f"夹爪初始化失败：{exc}") from exc
        with self._lock:
            self.healthy = True
        self._stop.clear()
        self._hold_requested.clear()
        self._held.clear()
        self._thread = threading.Thread(target=self._run, name="gripper", daemon=True)
        self._thread.start()
        return self._describe()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        self._thread = None
        self._safe_close_transport()
        with self._lock:
            self.healthy = False

    @property
    def description(self) -> str:
        return self._describe()

    def _describe(self) -> str:
        state = GRIP_STATE_NAMES.get(self.grip_state or -1, "?")
        position = "?" if self.position_raw is None else str(self.position_raw)
        return (f"大寰 {self.port} {self.baud} slave={self.slave_id} "
                f"位置={position}/1000 状态={state} 速度={self.speed}% 力={self.force}%")

    # -- 主循环接口 --
    def set_position(self, position_raw: int) -> None:
        """请求运动到 0..1000 的目标位置；超出范围会被裁剪。"""
        if self.read_only:
            raise GripperError("read_only gripper cannot send a position target")
        with self._lock:
            self._target = max(POSITION_MIN, min(POSITION_MAX, int(position_raw)))

    def hold(self) -> None:
        """取消待下发目标（夹爪保持当前位置）。"""
        with self._lock:
            self._target = None

    def motion_complete(self, position_raw: int) -> bool:
        """Acknowledge command delivery and a subsequent terminal device status."""
        status = self._motion_status
        return bool(self.healthy and status is not None and status[0] == position_raw
                    and status[1] in (1, 2, 3))

    def feedback_snapshot(self) -> dict:
        """Return coherent feedback without performing I/O.

        All fields except ``healthy`` describe one successful full status read.
        ``target`` is the target delivered before that read began, not a queued
        request. ``delivery_generation`` increments after each acknowledged
        target write (including stop/hold writes); ``generation`` increments
        after each successful read. Timestamps use ``time.monotonic()``.
        Current health is included under the same lock so failures and closure
        invalidate an otherwise unchanged cached record. Caller edits cannot
        modify the internally immutable record.
        """
        with self._lock:
            return {"healthy": self.healthy, **self._feedback}

    def stop_motion(self, timeout_s: float = 0.5) -> None:
        """Ask the serial owner to read and hold the current position, then await acknowledgement."""
        if self.read_only:
            raise GripperError("read_only gripper cannot stop motion")
        if self._thread is None or not self._thread.is_alive():
            raise GripperError("Gripper worker is not running")
        self.hold()
        self._held.clear()
        self._hold_requested.set()
        if not self._held.wait(timeout_s):
            raise GripperError(f"Gripper hold was not acknowledged: {self.last_error}")

    # -- 内部 --
    def _run(self) -> None:
        next_read = 0.0
        while not self._stop.is_set():
            with self._lock:
                target = self._target
            try:
                if self._hold_requested.is_set():
                    self._read_status()
                    if self.position_raw is None:
                        raise GripperError("Gripper position is unavailable")
                    self._write_register(REG_TARGET_POSITION, self.position_raw)
                    self._hold_requested.clear()
                    self._held.set()
                elif target is not None and target != self._last_delivered:
                    self._write_register(REG_TARGET_POSITION, target)
                now = time.monotonic()
                if now >= next_read:
                    self._read_status()
                    next_read = now + self._read_period_s
            except Exception as exc:  # noqa: BLE001 - 串口异常不向上抛
                with self._lock:
                    self.healthy = False
                    if str(exc) != self.last_error:
                        self.last_error = str(exc)
            time.sleep(self._period_s)

    def _exchange(self, request: bytes, response_size: int) -> bytes:
        # Register reads and absolute register writes are idempotent. Retry only
        # a missing/corrupt response, preserving valid device exceptions.
        try:
            return self._exchange_once(request, response_size)
        except ModbusResponseError as error:
            self.communication_retries += 1
            LOG.warning("Modbus RTU retry 1/1 after invalid response: %s", error)
            return self._exchange_once(request, response_size)

    def _exchange_once(self, request: bytes, response_size: int) -> bytes:
        """Receive one matching, CRC-verified RTU response across partial USB reads."""
        time.sleep(max(0.0, self._frame_gap_s - (time.monotonic() - self._last_exchange_end)))
        self._transport.reset_input_buffer()
        self._transport.write(request)
        deadline = time.monotonic() + self._response_timeout_s
        function = request[1]
        buffer = bytearray()
        received = bytearray()
        discarded = 0
        try:
            while time.monotonic() < deadline:
                chunk = self._transport.read(response_size)
                if not chunk:
                    time.sleep(0.001)
                    continue
                buffer.extend(chunk)
                received.extend(chunk)
                if len(received) > 256:
                    del received[:-256]
                while len(buffer) >= 2:
                    if buffer[0] != self.slave_id or buffer[1] not in (function, function | 0x80):
                        del buffer[0]
                        discarded += 1
                        continue
                    length = 5 if buffer[1] == function | 0x80 else response_size
                    if len(buffer) < length:
                        break
                    response = bytes(buffer[:length])
                    if crc16(response[:-2]) != int.from_bytes(response[-2:], 'little'):
                        del buffer[0]
                        discarded += 1
                        continue
                    if discarded:
                        self.communication_resyncs += 1
                        LOG.warning('Modbus RTU resynchronized after %d bytes; rx=%s',
                                    discarded, received.hex())
                    if response[1] == function | 0x80:
                        raise GripperError(f'Modbus 异常响应：0x{response[2]:02X}; frame={response.hex()}')
                    return response
            raise ModbusResponseError(f'Modbus RTU response timeout/invalid frame; '
                               f'function=0x{function:02X}, discarded={discarded}, rx={received.hex()}')
        finally:
            self._last_exchange_end = time.monotonic()

    def _write_register(self, address: int, value: int) -> None:
        frame = build_write(self.slave_id, address, [value])
        echo = self._exchange(frame, 8)
        if echo != frame:
            raise GripperError(f'写寄存器 0x{address:04X} 回显不匹配：'
                               f'tx={frame.hex()}, rx={echo.hex()}')
        if address == REG_TARGET_POSITION:
            # Only the serial owner reaches here. A queued set_position() and
            # an unsuccessful write must not claim delivery.
            self._last_delivered = value
            self._delivery_generation += 1
            self._delivered_at = time.monotonic()

    def _read_status(self) -> None:
        read_started_at = time.monotonic()
        target = self._last_delivered
        delivery_generation = self._delivery_generation
        delivered_at = self._delivered_at
        if read_started_at < delivered_at:
            raise GripperError("Status read started before the delivered command")
        response = self._exchange(build_read(self.slave_id, REG_GRIP_STATE, 2), 9)
        if response[2] != 4:
            raise GripperError(f'读状态响应格式错误：{response.hex()}')
        state = int.from_bytes(response[3:5], 'big')
        position = int.from_bytes(response[5:7], 'big')
        if state not in GRIP_STATE_NAMES or not POSITION_MIN <= position <= POSITION_MAX:
            raise GripperError(f'读状态寄存器值错误：{response.hex()}')
        if delivery_generation != self._delivery_generation:
            raise GripperError("Delivered command changed during a status read")
        read_at = time.monotonic()
        with self._lock:
            self.grip_state = state
            self.position_raw = position
            self._motion_status = (target, self.grip_state)
            self.last_read_at = read_at
            self.healthy = True
            self.last_error = None
            self._feedback_generation += 1
            self._feedback = MappingProxyType({
                "target": target, "delivery_generation": delivery_generation,
                "delivered_at": delivered_at, "generation": self._feedback_generation,
                "read_at": read_at, "state": state, "position": position,
            })

    def _safe_close_transport(self) -> None:
        transport = self._transport
        if transport is None:
            return
        try:
            transport.close()
        except Exception:  # noqa: BLE001
            pass
        if self._own_transport:
            self._transport = None

