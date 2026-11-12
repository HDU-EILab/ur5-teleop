"""PICO UDP JSON 帧解析/校验与帧来源（实时 UDP / JSONL 回放）。

输入协议沿用 upstream ``pico-controller-bridge``：``probe.py`` 每帧发一个 JSON
（UDP 9121），字段见下；校验规则迁移自 upstream ``dry_run.validated_frame``，
并补了帧时间戳单调与来源封装：

```json
{
  "pico_time_ns": 1234567890,
  "left":  [x, y, z, qx, qy, qz, qw],
  "right": [x, y, z, qx, qy, qz, qw],
  "head":  [x, y, z, qx, qy, qz, qw],
  "left_grip": 0.0, "right_grip": 1.0,
  "left_trigger": 0.0, "right_trigger": 0.0
}
```

位置单位 m，四元数顺序 ``(x, y, z, w)``（OpenXR 顺序），grip/trigger 为 0..1。
"""

from __future__ import annotations

import json
import math
import socket
import time
from dataclasses import dataclass, field
from typing import Iterable, Protocol, runtime_checkable

#: 手柄侧的键名
SIDES = ("left", "right")
#: 帧里必须携带的位姿键名
POSES = ("left", "right", "head")

#: 四元数模长允许范围（PICO 发送端数据可能略有浮点误差）
_QUATERNION_NORM_RANGE = (0.8, 1.2)


class InvalidFrame(ValueError):
    """PICO 帧格式或数值不合法。"""


@dataclass(frozen=True)
class PicoFrame:
    """一帧通过校验的 PICO 数据。"""

    pico_time_ns: int
    left: tuple[float, ...]
    right: tuple[float, ...]
    head: tuple[float, ...]
    left_grip: float
    right_grip: float
    left_trigger: float = 0.0
    right_trigger: float = 0.0
    buttons: dict[str, bool] = field(default_factory=dict)   # A/B/X/Y/menu/摇杆按下
    axis: dict[str, tuple[float, float]] = field(default_factory=dict)  # 左右摇杆 (x, y)

    def pose(self, side: str) -> tuple[float, ...]:
        """取某侧位姿；side 为 ``left`` / ``right`` / ``head``。"""
        if side == "left":
            return self.left
        if side == "right":
            return self.right
        if side == "head":
            return self.head
        raise ValueError(f"未知的位姿名：{side!r}")

    def grip(self, side: str) -> float:
        """取某侧 grip 值（0..1）；side 为 ``left`` / ``right``。"""
        if side == "left":
            return self.left_grip
        if side == "right":
            return self.right_grip
        raise ValueError(f"未知的手柄侧：{side!r}")

    def trigger(self, side: str) -> float:
        """取某侧扳机值（0..1，0=松开 1=按到底）。"""
        if side == "left":
            return self.left_trigger
        if side == "right":
            return self.right_trigger
        raise ValueError(f"未知的手柄侧：{side!r}")

    def button(self, name: str) -> bool:
        """取按键状态（a/b/x/y/menu_left/menu_right/axis_left_click/axis_right_click）。"""
        return bool(self.buttons.get(name, False))


def _parse_pose(raw_frame: dict, name: str) -> tuple[float, ...]:
    if name not in raw_frame:
        raise InvalidFrame(f"缺少 {name} 位姿")
    raw_pose = raw_frame[name]
    if not isinstance(raw_pose, (list, tuple)) or len(raw_pose) != 7:
        raise InvalidFrame(f"无效的 {name} 位姿（必须是 7 维数组）")
    try:
        pose = tuple(float(value) for value in raw_pose)
    except (TypeError, ValueError) as exc:
        raise InvalidFrame(f"无效的 {name} 位姿（含非数值）") from exc
    if not all(math.isfinite(value) for value in pose):
        raise InvalidFrame(f"无效的 {name} 位姿（含 NaN/Inf）")
    norm = math.sqrt(sum(value * value for value in pose[3:7]))
    if not _QUATERNION_NORM_RANGE[0] < norm < _QUATERNION_NORM_RANGE[1]:
        raise InvalidFrame(f"无效的 {name} 朝向（四元数模长 {norm:.3f}）")
    # 位置、四元数虚部全 0 且 w 为 ±1：PICO 还没拿到该设备的追踪数据
    if (max(abs(value) for value in pose[:3]) < 1e-4
            and max(abs(value) for value in pose[3:6]) < 1e-4
            and abs(abs(pose[6]) - 1) < 1e-4):
        raise InvalidFrame(f"{name} 没有有效追踪位置")
    return pose


def _parse_grip(raw_frame: dict, name: str) -> float:
    key = f"{name}_grip"
    if key not in raw_frame:
        raise InvalidFrame(f"缺少 {key}")
    try:
        grip = float(raw_frame[key])
    except (TypeError, ValueError) as exc:
        raise InvalidFrame(f"无效的 {key} 值") from exc
    if not 0.0 <= grip <= 1.0:
        raise InvalidFrame(f"无效的 {key} 值：{raw_frame[key]!r}（必须在 0..1）")
    return grip


def _parse_buttons(raw_frame: dict) -> dict[str, bool]:
    """解析按键字典；老版本 bridge 无该字段时返回空字典（全部视为未按）。"""
    raw = raw_frame.get("buttons")
    if not isinstance(raw, dict):
        return {}
    result: dict[str, bool] = {}
    for key, value in raw.items():
        if isinstance(key, str):
            result[key] = bool(value)
    return result


def _parse_axis(raw_frame: dict) -> dict[str, tuple[float, float]]:
    """解析左右摇杆；缺失/非法时跳过（下游默认 0）。"""
    raw = raw_frame.get("axis")
    if not isinstance(raw, dict):
        return {}
    result: dict[str, tuple[float, float]] = {}
    for side in ("left", "right"):
        value = raw.get(side)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                result[side] = (float(value[0]), float(value[1]))
            except (TypeError, ValueError):
                continue
    return result


def _parse_trigger(raw_frame: dict, name: str) -> float:
    """解析扳机值；老版本 bridge 可能不带该字段，缺省按 0 处理。"""
    key = f"{name}_trigger"
    if key not in raw_frame:
        return 0.0
    try:
        value = float(raw_frame[key])
    except (TypeError, ValueError) as exc:
        raise InvalidFrame(f"无效的 {key} 值") from exc
    if not 0.0 <= value <= 1.0:
        raise InvalidFrame(f"无效的 {key} 值：{raw_frame[key]!r}（必须在 0..1）")
    return value


def parse_frame(packet) -> PicoFrame:
    """解析并校验一帧 PICO 数据；非法时抛 ``InvalidFrame``。

    ``packet`` 可以是 UDP 收到的 bytes、JSON 字符串或已解析的 dict。
    """
    if isinstance(packet, dict):
        raw = packet
    elif isinstance(packet, (bytes, bytearray)):
        try:
            raw = json.loads(packet)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise InvalidFrame(f"JSON 解析失败：{exc}") from exc
    elif isinstance(packet, str):
        try:
            raw = json.loads(packet)
        except json.JSONDecodeError as exc:
            raise InvalidFrame(f"JSON 解析失败：{exc}") from exc
    else:
        raise InvalidFrame(f"不支持的帧类型：{type(packet).__name__}")
    if not isinstance(raw, dict):
        raise InvalidFrame("帧必须是 JSON 对象")

    stamp = raw.get("pico_time_ns")
    if isinstance(stamp, bool) or not isinstance(stamp, int) or stamp <= 0:
        raise InvalidFrame(f"无效的 PICO 时间戳：{stamp!r}")

    poses = {name: _parse_pose(raw, name) for name in POSES}
    grips = {name: _parse_grip(raw, name) for name in SIDES}
    triggers = {name: _parse_trigger(raw, name) for name in SIDES}
    return PicoFrame(
        pico_time_ns=stamp,
        left=poses["left"],
        right=poses["right"],
        head=poses["head"],
        left_grip=grips["left"],
        right_grip=grips["right"],
        left_trigger=triggers["left"],
        right_trigger=triggers["right"],
        buttons=_parse_buttons(raw),
        axis=_parse_axis(raw),
    )


@runtime_checkable
class FrameSource(Protocol):
    """帧来源：遥操作主循环只依赖这个接口。

    ``poll()`` 返回最新一帧或 None；``last_receive`` 是最近一次收到有效帧的
    ``time.monotonic()``（从未收到为 None）；``exhausted`` 表示不会再有新帧
    （回放源用完），实时源恒为 False。
    """

    last_receive: float | None

    @property
    def exhausted(self) -> bool:
        ...

    def poll(self) -> PicoFrame | None:
        ...


class UdpFrameSource:
    """实时 UDP 帧来源：绑定本地端口，非阻塞 poll，记录收帧时间与断流状态。"""

    def __init__(self, port: int, host: str = "0.0.0.0") -> None:
        self.host = host
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self._sock.bind((host, int(port)))
        #: 实际绑定的端口（构造时传 0 可让系统分配，测试用）
        self.port: int = self._sock.getsockname()[1]
        #: 最近一次收到“有效新帧”的时间戳（time.monotonic）
        self.last_receive: float | None = None
        #: 最近一帧的 pico_time_ns，用于丢弃乱序/重复帧
        self.last_pico_time_ns = 0
        #: 被校验拒绝的包数
        self.invalid_packets = 0
        self._closed = False

    def poll(self) -> PicoFrame | None:
        """非阻塞收包：排空缓冲区，返回其中时间戳最新的有效帧。"""
        latest: PicoFrame | None = None
        while True:
            try:
                packet, _address = self._sock.recvfrom(4096)
            except (BlockingIOError, InterruptedError):
                break
            try:
                frame = parse_frame(packet)
            except InvalidFrame:
                self.invalid_packets += 1
                continue
            if frame.pico_time_ns <= self.last_pico_time_ns:
                continue
            self.last_pico_time_ns = frame.pico_time_ns
            self.last_receive = time.monotonic()
            latest = frame
        return latest

    def is_stale(self, timeout_s: float, now: float | None = None) -> bool:
        """是否超过 ``timeout_s`` 没收到有效帧（从未收到也算断流）。"""
        if self.last_receive is None:
            return True
        current = time.monotonic() if now is None else float(now)
        return current - self.last_receive > timeout_s

    @property
    def exhausted(self) -> bool:
        return False

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._sock.close()

    def __enter__(self) -> "UdpFrameSource":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()


class ReplayFrameSource:
    """JSONL 录制回放帧来源：按顺序返回帧，用完 ``exhausted`` 为 True。

    每行是一帧 ``probe.py`` 格式的 JSON；用于 ``tools/dry_run.py --replay``
    和单元测试。
    """

    def __init__(self, frames: Iterable[PicoFrame]) -> None:
        self._frames = list(frames)
        self._index = 0
        self.last_receive: float | None = None

    @classmethod
    def from_jsonl(cls, path) -> "ReplayFrameSource":
        """从 JSONL 文件构造；坏行会带上行号抛 ``InvalidFrame``。"""
        frames: list[PicoFrame] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    frames.append(parse_frame(line))
                except InvalidFrame as exc:
                    raise InvalidFrame(f"{path}:{line_number}: {exc}") from exc
        return cls(frames)

    def poll(self) -> PicoFrame | None:
        if self._index >= len(self._frames):
            return None
        frame = self._frames[self._index]
        self._index += 1
        self.last_receive = time.monotonic()
        return frame

    @property
    def exhausted(self) -> bool:
        return self._index >= len(self._frames)
