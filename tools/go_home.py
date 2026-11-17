#!/usr/bin/env python3
"""把 UR5 用 movej 回 HOME（调用前需先停 ur5-teleop，避免 RTDE/程序冲突）。

一般用配套脚本：tools/go_home.sh（自动停/启 teleop）。

单独使用：
    ~/ur5/.venv/bin/python tools/go_home.py --config config/ur5-orin.toml
    ~/ur5/.venv/bin/python tools/go_home.py --dry-run
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
from ur5_teleop.home import run_home_movej  # noqa: E402
from ur5_teleop.ur_client import DashboardClient, RTDEBackend  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "ur5-orin.toml"))
    ap.add_argument("--port", type=int, default=30001)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = config_module.load(args.config)
    if args.dry_run:
        print(f"[dry-run] 目标关节(deg): {list(cfg.home.joints_deg)}")
        print(f"[dry-run] 机器人: {cfg.robot.ip}")
        return 0

    # 先用 Dashboard 停掉机器人端程序，释放 input 寄存器（否则 RTDE setup_inputs 会报
    # "An input parameter is already in use."）
    try:
        dash = DashboardClient(cfg.robot.ip)
        dash.connect()
        print("[go_home] dashboard stop:", dash.command("stop"))
        dash.close()
    except Exception as exc:  # noqa: BLE001
        print("[go_home] dashboard stop 失败（继续尝试）:", exc)

    backend = RTDEBackend(cfg.robot.ip, frequency_hz=125.0)
    print(backend.connect())
    try:
        st = None
        for _ in range(30):
            try:
                st = backend.read_state()
                break
            except Exception:
                time.sleep(0.2)
        if st is None:
            raise RuntimeError("RTDE 连接成功但 6 秒内没收到状态帧")

        print("当前关节(deg):", [round(math.degrees(v), 1) for v in st.joints])
        print("当前TCP:", [round(v, 4) for v in st.tcp_pose])
        run_home_movej(cfg.robot.ip, cfg.home.joints_deg, backend=backend, port=args.port)
        st = backend.read_state()
        print("回 HOME 后关节(deg):", [round(math.degrees(v), 1) for v in st.joints])
        print("回 HOME 后TCP:", [round(v, 4) for v in st.tcp_pose])
        print("✅ 已回 HOME")
        return 0
    finally:
        backend.close()


if __name__ == "__main__":
    sys.exit(main())
