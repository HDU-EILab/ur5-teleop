#!/usr/bin/env python3
"""只读查看 UR5 状态（不发送任何运动指令）。

用法：
    python3 tools/status.py --ip 192.168.1.100
    python3 tools/status.py --config config/ur5.toml --interval 0.5
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ur5_teleop import config as config_module  # noqa: E402
from ur5_teleop.backend import DisconnectedError  # noqa: E402
from ur5_teleop.ur_client import (  # noqa: E402
    ROBOT_MODE_NAMES,
    SAFETY_MODE_NAMES,
    RTDEBackend,
)


def _deg(value: float) -> float:
    return math.degrees(value)


def format_state(state) -> str:
    joints = " ".join(f"{_deg(j):+7.2f}" for j in state.joints[:6])
    pose = " ".join(
        f"{state.tcp_pose[i] * 1000:+7.1f}" if i < 3 else f"{_deg(state.tcp_pose[i]):+7.1f}"
        for i in range(6)
    )
    mode = ROBOT_MODE_NAMES.get(state.robot_mode, f"?({state.robot_mode})")
    safety = SAFETY_MODE_NAMES.get(state.safety_status, f"?({state.safety_status})")
    scaling = "未订阅" if state.speed_scaling < 0 else f"{state.speed_scaling * 100:.0f}%"
    return (f"关节(deg) [{joints}]  TCP(mm/deg) [{pose}]  "
            f"robot_mode={mode}  safety={safety}  speed_scaling={scaling}")


def main() -> int:
    parser = argparse.ArgumentParser(description="只读查看 UR5 状态")
    parser.add_argument("--ip")
    parser.add_argument("--config", default=None, help="配置文件路径（默认 config/ur5.toml）")
    parser.add_argument("--port", type=int, default=30004, help="RTDE 端口（默认 30004，测 mock 时可改）")
    parser.add_argument("--interval", type=float, default=1.0, help="打印间隔秒")
    parser.add_argument("--count", type=int, default=0, help="打印次数，0 = 一直打印")
    args = parser.parse_args()

    ip = args.ip
    protocol_version = 2
    frequency = 125.0
    if ip is None:
        cfg = config_module.load(args.config)
        ip = cfg.robot.ip
        protocol_version = cfg.robot.protocol_version
        frequency = cfg.robot.rtde_frequency_hz

    backend = RTDEBackend(ip, protocol_version=protocol_version, frequency_hz=frequency,
                          port=args.port)
    try:
        print(backend.connect())
    except Exception as exc:  # noqa: BLE001
        print(f"连接失败：{exc}")
        return 1

    printed = 0
    try:
        while True:
            try:
                state = backend.read_state()
            except DisconnectedError as exc:
                print(f"状态不可用：{exc}")
                time.sleep(args.interval)
                continue
            print(format_state(state), flush=True)
            printed += 1
            if args.count and printed >= args.count:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
