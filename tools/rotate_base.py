#!/usr/bin/env python3
"""只把 UR5 第 1 关节（基座 J1）转到指定角度，其余关节保持不动。

用途：机器人经 IK 停在"等价支路"后（基座绕了 360° 的倍数/额外角度），需要把基座摆回
-180° 之类便于安装/对齐的角度，同时保持肩/肘/腕姿态不变。用 ur5_traj.script 的
servoj 流式发送关节目标（不受示教器速度滑块影响）。

用法：
    python3 tools/rotate_base.py --target-deg -180 --dry-run   # 只读当前关节，打印路径
    python3 tools/rotate_base.py --target-deg -180 --yes --duration 30
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ur5_teleop import config as config_module  # noqa: E402
from ur5_teleop.ur_client import (  # noqa: E402
    DashboardClient,
    RTDEBackend,
    open_script_channel,
    send_program,
)

SCRIPT = ROOT / "urscript" / "ur5_traj.script"
CTRL_PERIOD = 0.008
MODE_RUNNING = 7
SAFETY_NORMAL = 1


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只旋转 UR5 基座 J1（其余关节不动）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--target-deg", type=float, required=True, help="J1 目标角度（deg）")
    parser.add_argument("--duration", type=float, default=30.0, help="旋转时长（s），默认 30")
    parser.add_argument("--max-error-deg", type=float, default=5.0,
                        help="关节跟随偏差上限（deg），默认 5")
    parser.add_argument("--dry-run", action="store_true", help="只读当前关节并打印路径，不发指令")
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args(argv)


def _read_state(cfg) -> object:
    backend = RTDEBackend(
        cfg.robot.ip,
        protocol_version=cfg.robot.protocol_version,
        frequency_hz=cfg.robot.rtde_frequency_hz,
        state_timeout_s=max(0.3, cfg.safety.comm_timeout_s * 1.5),
    )
    print(backend.connect())
    deadline = time.monotonic() + 5.0
    state = None
    while time.monotonic() < deadline:
        try:
            state = backend.read_state()
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.05)
    return backend, state


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = config_module.load(args.config)

    if args.dry_run:
        backend, state = _read_state(cfg)
        try:
            if state is None:
                print("[dry-run] 5 s 内未收到 RTDE 状态帧")
                return 1
            current = [math.degrees(v) for v in state.joints]
        finally:
            backend.close()
        delta = args.target_deg - current[0]
        peak = 1.875 * abs(math.radians(delta)) / max(args.duration, 0.1)
        print(f"[dry-run] 当前关节(deg): {[round(v, 2) for v in current]}")
        print(f"[dry-run] 目标：J1 {current[0]:.2f}° → {args.target_deg:.2f}°（Δ {delta:+.2f}°），"
              f"其余关节不变；{args.duration:.0f} s 平滑转动，峰值 ≈ {peak:.3f} rad/s")
        print("[dry-run] 未发送任何运动指令。")
        return 0

    if not args.yes:
        print("该工具会旋转机器人基座。确认现场安全后加 --yes 重跑。")
        return 2

    dashboard: DashboardClient | None = None
    backend = None
    try:
        dashboard = DashboardClient(cfg.robot.ip)
        print(f"[dashboard] {dashboard.connect()}")
        safety = dashboard.command("safemode") if False else dashboard.command("safetymode")
        mode = dashboard.command("robotmode")
        print(f"[预检] {safety} | {mode}")
        if "NORMAL" not in safety:
            raise RuntimeError(f"安全状态不是 NORMAL（{safety}）")
        if "RUNNING" not in mode:
            raise RuntimeError(f"机器人不在 RUNNING（{mode}）")
        print(f"[dashboard] stop: {dashboard.command('stop')}")
        time.sleep(1.0)

        print(f"[program] 下发 {SCRIPT.name}")
        channel = open_script_channel(cfg.robot.ip, 30001)
        send_program(channel, SCRIPT.read_text(encoding="utf-8"))
        time.sleep(0.8)
        channel.close()

        backend, state = _read_state(cfg)
        if state is None:
            raise RuntimeError("5 s 内未收到 RTDE 状态帧")
        current = [float(v) for v in state.joints]
        target = [math.radians(args.target_deg)] + current[1:]
        delta = target[0] - current[0]
        peak = 1.875 * abs(delta) / max(args.duration, 0.1)
        print(f"[当前] J1 {math.degrees(current[0]):.2f}°，目标 {args.target_deg:.2f}°"
              f"（Δ {math.degrees(delta):+.2f}°），{args.duration:.0f} s，峰值 {peak:.3f} rad/s")
        if math.degrees(delta) > 300:
            print("[警告] 单次转动超过 300°，请确认方向正确（可用 --target-deg ±180 选择等价角）")
        print("[安全检查] 即将只转动基座 J1：确认扫掠区域无人无障碍、手在急停上。3 s 后开始…")
        for countdown in (3, 2, 1):
            print(f"  {countdown}…")
            time.sleep(1.0)

        steps = max(int(round(args.duration / CTRL_PERIOD)), 1)
        max_error_deg = 0.0
        for k in range(1, steps + 1):
            ratio = k / steps
            smooth = ratio * ratio * ratio * (10.0 - 15.0 * ratio + 6.0 * ratio * ratio)
            q = tuple(current[i] + (target[i] - current[i]) * smooth for i in range(6))
            backend.send_joint_sets(q, servo=True)
            actual = backend.read_state()
            if actual.safety_status != SAFETY_NORMAL:
                raise RuntimeError("安全状态异常，已中止")
            error = max(abs(actual.joints[i] - q[i]) for i in range(6))
            max_error_deg = max(max_error_deg, math.degrees(error))
            if math.degrees(error) > args.max_error_deg:
                raise RuntimeError(f"关节跟随偏差 {math.degrees(error):.1f}° 超限，已中止")
            if k % max(int(2.0 / CTRL_PERIOD), 1) == 0:
                print(f"  {100.0 * ratio:5.1f}%  J1={math.degrees(actual.joints[0]):+8.2f}°  "
                      f"偏差 {math.degrees(error):.2f}°")
            time.sleep(CTRL_PERIOD)
        backend.send_joint_sets(target, servo=False)
        time.sleep(0.3)
        final = backend.read_state()
        print(f"[完成] J1={math.degrees(final.joints[0]):.2f}°（目标 {args.target_deg:.2f}°），"
              f"最大跟随偏差 {max_error_deg:.2f}°，其余关节保持不变")
        return 0
    except KeyboardInterrupt:
        print("\n[中止] Ctrl-C")
        return 130
    except RuntimeError as exc:
        print(f"[中止] {exc}")
        return 1
    finally:
        if backend is not None:
            backend.close()
        if dashboard is not None:
            try:
                print(f"[dashboard] stop: {dashboard.command('stop')}")
            except OSError:
                pass
            dashboard.close()


if __name__ == "__main__":
    raise SystemExit(main())
