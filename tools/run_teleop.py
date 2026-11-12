#!/usr/bin/env python3
"""实机遥操作入口：PICO UDP 帧 → 目标 TCP 位姿 → RTDE。

用法：
    # 实机（会发运动指令，必须显式 --yes）
    python3 tools/run_teleop.py --config config/ur5.toml --scale 0.3 --yes

    # 空跑：不连机器人，用 FakeBackend 只打印目标
    python3 tools/run_teleop.py --dry-run
    python3 tools/run_teleop.py --dry-run --replay tests/data/sample_frames.jsonl

安全约定：
- 不自动上电/松闸/播放程序；机器人端程序需先跑起来（见 docs/bringup.md）。
- 首次联调必须低倍速（--scale 0.3）、空载、人手不离急停。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ur5_teleop import config as config_module  # noqa: E402
from ur5_teleop.backend import FakeBackend, URBackend  # noqa: E402
from ur5_teleop.teleop import TeleopLoop, TeleopStep  # noqa: E402
from ur5_teleop.tracking import ReplayFrameSource, UdpFrameSource  # noqa: E402
from ur5_teleop.ur_client import (  # noqa: E402
    ROBOT_MODE_NAMES,
    SAFETY_MODE_NAMES,
    RTDEBackend,
)

#: 打印节流（秒）
PRINT_PERIOD_S = 0.25
#: RUNNING 的 robot_mode 值
MODE_RUNNING = 7


def format_pose(pose) -> str:
    return "[" + ", ".join(f"{value:+.4f}" for value in pose) + "]"


def make_printer():
    last = {"time": 0.0, "state": None, "note": ""}

    def on_step(step: TeleopStep) -> None:
        now = time.monotonic()
        state_changed = step.state is not last["state"]
        if not state_changed and now - last["time"] < PRINT_PERIOD_S:
            return
        last["time"] = now
        last["state"] = step.state
        if step.note and step.note != last["note"]:
            print(f"[{step.note}]")
            last["note"] = step.note
        elif not step.note:
            last["note"] = ""
        target = format_pose(step.pose)
        servo = "servo" if step.servo else "idle "
        grip = ""
        if step.frame is not None:
            side = step.frame.grip("left"), step.frame.grip("right")
            grip = f" grip(L/R)={side[0]:.2f}/{side[1]:.2f}"
        print(f"{step.state.name:<8} {servo} target={target}{grip}", flush=True)

    return on_step


def build_source(args, cfg):
    if args.replay is not None:
        return ReplayFrameSource.from_jsonl(args.replay)
    return UdpFrameSource(cfg.input.udp_port)


def run_dry(args, cfg) -> int:
    source = build_source(args, cfg)
    backend: URBackend = FakeBackend()
    backend.connect()
    loop = TeleopLoop(cfg, backend, source)
    printer = make_printer()

    def on_step(step: TeleopStep) -> None:
        # 回放远快于墙钟：每步刷新一次假机器人状态时间戳，模拟“机器人刚好跟上”
        backend.feed_state(pose=step.pose)
        printer(step)

    print(f"dry-run：FakeBackend 不连机器人；PICO 源 = {type(source).__name__}")
    print("grip 按下锁定起点，松开保持；Ctrl-C 退出。")
    try:
        loop.run(on_step)
    finally:
        close = getattr(source, "close", None)
        if callable(close):
            close()
    print(f"结束状态：{loop.state.name}" + ("（FAULT）" if loop.faulted else ""))
    return 0


def run_real(args, cfg) -> int:
    if not args.yes:
        print("实机模式会向机器人发送运动指令。确认现场安全后加 --yes 重跑。")
        return 2
    backend = RTDEBackend(
        cfg.robot.ip,
        protocol_version=cfg.robot.protocol_version,
        frequency_hz=cfg.robot.rtde_frequency_hz,
        state_timeout_s=max(0.3, cfg.safety.comm_timeout_s * 1.5),
    )
    source = None
    try:
        print(backend.connect())
        # 预检：只读状态 + 模式检查
        deadline = time.monotonic() + 3.0
        state = None
        while time.monotonic() < deadline:
            try:
                state = backend.read_state()
                break
            except Exception:  # noqa: BLE001 - 等第一帧
                time.sleep(0.05)
        if state is None:
            print("3 秒内未收到机器人状态帧，退出。")
            return 1
        mode = ROBOT_MODE_NAMES.get(state.robot_mode, f"?({state.robot_mode})")
        safety = SAFETY_MODE_NAMES.get(state.safety_status, f"?({state.safety_status})")
        print(f"预检：robot_mode={mode}  safety={safety}")
        if state.robot_mode != MODE_RUNNING:
            print("机器人当前不是 RUNNING（需先在示教器/ur_ctl 上电 + 松闸 + 播放程序）。"
                  "本工具不会自动上电。退出。")
            return 1
        if state.safety_status != 1:
            print(f"安全状态不是 NORMAL（{safety}），先现场确认并复位保护性停止。退出。")
            return 1

        source = build_source(args, cfg)
        loop = TeleopLoop(cfg, backend, source)
        side = cfg.input.side
        print(f"PICO 输入：UDP :{cfg.input.udp_port}，{side} 手柄 grip 离合；"
              f"scale={cfg.teleop.scale:g}；工作空间 "
              f"{cfg.safety.workspace_min} → {cfg.safety.workspace_max}")
        print("按 grip 开始跟随；松开保持；Ctrl-C 停止（会先发 idle）。")
        loop.run(make_printer())
        if loop.faulted:
            print("遥操作因故障停止（FAULT）。先松开 grip，排查后重新运行。")
            return 1
        print(f"正常退出，最终状态 {loop.state.name}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"运行失败：{exc}")
        return 1
    finally:
        if source is not None:
            close = getattr(source, "close", None)
            if callable(close):
                close()
        backend.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="配置文件路径（默认 config/ur5.toml）")
    parser.add_argument("--scale", type=float, default=None, help="覆盖 teleop.scale（首次联调 0.3）")
    parser.add_argument("--ip", default=None, help="覆盖 robot.ip")
    parser.add_argument("--dry-run", action="store_true", help="不连机器人，只打印目标")
    parser.add_argument("--replay", type=Path, default=None, help="回放 JSONL 帧文件（dry-run 用）")
    parser.add_argument("--yes", action="store_true", help="确认执行实机运动")
    args = parser.parse_args()

    cfg = config_module.load(args.config)
    if args.ip is not None or args.scale is not None:
        from dataclasses import replace  # noqa: PLC0415

        robot = replace(cfg.robot, ip=args.ip) if args.ip else cfg.robot
        teleop = replace(cfg.teleop, scale=args.scale) if args.scale is not None else cfg.teleop
        cfg = config_module.Config(robot=robot, input=cfg.input, teleop=teleop,
                                   safety=cfg.safety, source=cfg.source)
    if args.replay is not None and not args.dry_run:
        print("--replay 只能与 --dry-run 一起用")
        return 2
    return run_dry(args, cfg) if args.dry_run else run_real(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
