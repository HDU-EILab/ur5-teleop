#!/usr/bin/env python3
"""真机分阶段自检：加载机器人端程序 → 0 mm servo → 微小位移。

用途：第一次上机 / 每次改完 urscript 后的快速验证，**不会**做大幅动作。
前置：机器人已上电、松闸、安全模式 NORMAL（本工具不会自动上电）。
安全：每一步都打印前后位姿；需要显式 --yes 才会发运动指令。

用法：
    python3 tools/servo_check.py --ip 192.168.1.100 --yes              # 只验证 0 mm servo
    python3 tools/servo_check.py --ip 192.168.1.100 --yes --step-mm 2  # 再走 2 mm（基座 +Z）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ur5_teleop.backend import BackendError  # noqa: E402
from ur5_teleop.ur_client import (  # noqa: E402
    DashboardClient,
    RTDEBackend,
    open_script_channel,
    send_program,
)

DEFAULT_SCRIPT = ROOT / "urscript" / "ur5_teleop.script"


def dashboard_command(ip: str, cmd: str) -> str:
    try:
        client = DashboardClient(ip)
        client.connect()
        reply = client.command(cmd)
        client.close()
        return reply
    except OSError as exc:
        return f"<{exc}>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ip", default="192.168.1.100")
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--step-mm", type=float, default=0.0,
                        help="验证位移（基座 +Z，mm，默认 0 = 只验证 0 位移 servo）")
    parser.add_argument("--hold-s", type=float, default=1.0, help="每个阶段持续秒数")
    parser.add_argument("--yes", action="store_true", help="确认允许发送运动指令")
    args = parser.parse_args()

    if not args.yes:
        print("该工具会向机器人发送 servoj 指令。确认现场安全后加 --yes 重跑。")
        return 2

    print(f"[1/4] 停止现有程序：{dashboard_command(args.ip, 'stop')}")
    time.sleep(1.0)
    print(f"[2/4] 通过 {args.port} 端口下发脚本：{args.script.name}")
    try:
        channel = open_script_channel(args.ip, args.port)
        send_program(channel, args.script.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"下发失败：{exc}")
        return 1
    time.sleep(1.0)

    backend = RTDEBackend(args.ip, frequency_hz=125.0)
    try:
        print(backend.connect())
        time.sleep(0.5)
        start = backend.read_state().tcp_pose
        print(f"初始 TCP: {[round(v, 5) for v in start]}")

        print(f"[3/4] 0 mm servo，{args.hold_s:g}s")
        deadline = time.monotonic() + args.hold_s
        while time.monotonic() < deadline:
            backend.send_sets(start, servo=True)
            time.sleep(0.02)
        time.sleep(0.3)
        held = backend.read_state()
        print(f"  位移(mm): {[round((held.tcp_pose[i]-start[i])*1000, 3) for i in range(3)]}  "
              f"robot_mode={held.robot_mode} safety={held.safety_status}")

        target = list(start)
        if args.step_mm:
            target[2] += args.step_mm / 1000.0
            print(f"[4/5] servo 到基座 +Z {args.step_mm:g} mm，{args.hold_s:g}s")
            deadline = time.monotonic() + args.hold_s
            while time.monotonic() < deadline:
                backend.send_sets(target, servo=True)
                time.sleep(0.02)
            time.sleep(0.4)
            final = backend.read_state()
            print(f"  位移(mm): {[round((final.tcp_pose[i]-start[i])*1000, 3) for i in range(3)]}  "
                  f"robot_mode={final.robot_mode} safety={final.safety_status}")
            print(f"[5/5] idle + 停止程序：{dashboard_command(args.ip, 'stop')}")
        else:
            print("[4/4] 跳过位移验证（--step-mm 0）")
            print(f"       idle + 停止程序：{dashboard_command(args.ip, 'stop')}")
        return 0
    except BackendError as exc:
        print(f"RTDE 失败：{exc}")
        return 1
    finally:
        try:
            backend.send_sets(backend.read_state().tcp_pose, servo=False)
        except Exception:  # noqa: BLE001
            pass
        backend.close()
        try:
            channel.close()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
