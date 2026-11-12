#!/usr/bin/env python3
"""UR5 运维命令：Dashboard 操作 + URScript 程序下发。

用法：
    python3 tools/ur_ctl.py --ip 192.168.1.100 info
    python3 tools/ur_ctl.py --ip 192.168.1.100 --yes power-on
    python3 tools/ur_ctl.py --ip 192.168.1.100 --yes brake-release
    python3 tools/ur_ctl.py --ip 192.168.1.100 send-program --file urscript/ur5_teleop.script
    python3 tools/ur_ctl.py --ip 192.168.1.100 --yes stop

说明：
- Dashboard 29999 的所有命令在 e-Series 上要求已 Enable Remote Control 且切到 Remote 模式；
  CB3 无此概念（见 docs/research/ur5-external-control.md §1.2/§3.2）。
- send-program 通过 30001（可 --port 改 30002/30003）写入 URScript 程序，**默认下发后立即关闭通道**
  （本机实测：关闭后程序仍继续运行）。注意：不要用 --hold 长时间保持一个不读数据的脚本连接——
  控制器往该连接发程序状态，缓冲约 30 秒塞满后会拖死控制器的 TCP 发送线程，连带 RTDE 掉线。
- 改变机器人状态（上电/松闸/播放/停止/解锁保护停）必须显式加 --yes。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ur5_teleop import config as config_module  # noqa: E402
from ur5_teleop.ur_client import (  # noqa: E402
    DashboardClient,
    open_script_channel,
    send_program,
)

READ_ONLY = {
    "version": "软件版本",
    "PolyscopeVersion": "PolyScope 版本",
    "robotmode": "机器人模式",
    "safetymode": "安全模式",
    "programState": "程序状态",
    "get robot model": "型号",
    "get serial number": "序列号",
    "is in remote control": "是否 Remote 模式",
}

DANGEROUS = {
    "power-on": "power on",
    "power-off": "power off",
    "brake-release": "brake release",
    "play": "play",
    "stop": "stop",
    "pause": "pause",
    "unlock-protective-stop": "unlock protective stop",
    "close-safety-popup": "close safety popup",
}


def cmd_info(ip: str, timeout: float) -> int:
    client = DashboardClient(ip, timeout=timeout)
    try:
        client.connect()
    except OSError as exc:
        print(f"连接 Dashboard {ip}:29999 失败：{exc}")
        return 1
    try:
        print(f"=== {ip} Dashboard 信息（只读）===")
        for command, label in READ_ONLY.items():
            try:
                reply = client.command(command)
            except OSError as exc:
                reply = f"<{exc}>"
            print(f"{label:<12}: {reply}")
        if "Remote 模式" in client.command("is in remote control"):
            pass
    finally:
        client.close()
    return 0


def cmd_send_program(ip: str, file_path: Path, port: int, hold: bool, timeout: float) -> int:
    if not file_path.is_file():
        print(f"找不到程序文件：{file_path}")
        return 1
    program = file_path.read_text(encoding="utf-8")
    try:
        channel = open_script_channel(ip, port, timeout=timeout)
    except OSError as exc:
        print(f"连接脚本端口 {ip}:{port} 失败：{exc}")
        print("提示：e-Series 需 Remote 模式；CB3 检查网络/防火墙；可试 --port 30002 或 30001")
        return 1
    try:
        send_program(channel, program)
        print(f"程序已写入 {ip}:{port}（{len(program)} 字节）")
        time.sleep(0.8)
        # 用 Dashboard 看一下程序是否在跑（失败不影响下发）
        try:
            client = DashboardClient(ip, timeout=timeout)
            client.connect()
            print("programState:", client.command("programState"))
            client.close()
        except OSError:
            pass
        if not hold:
            print("已关闭脚本连接（实测程序继续运行，进入 idle 等待主机命令）")
            return 0
        print("⚠️ --hold：保持脚本连接中。长时间不读该连接会让控制器 TCP 发送积压，"
              "约 30 秒后拖死其它连接（含 RTDE）。一般不需要 --hold。按 Ctrl-C 断开。")
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n断开脚本连接")
        return 0
    finally:
        try:
            channel.close()
        except OSError:
            pass


def cmd_dashboard(ip: str, command: str, timeout: float, yes: bool) -> int:
    if not yes:
        print(f"该命令会改变机器人状态：'{command}'。确认安全后加 --yes 重跑。")
        return 2
    client = DashboardClient(ip, timeout=timeout)
    try:
        client.connect()
        print(client.command(command))
    except OSError as exc:
        print(f"执行失败：{exc}")
        return 1
    finally:
        client.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ip")
    parser.add_argument("--config", default=None, help="配置文件路径（默认 config/ur5.toml）")
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--yes", action="store_true", help="确认执行会改变机器人状态的命令")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="只读信息")
    for name in DANGEROUS:
        sub.add_parser(name, help=f"Dashboard: {DANGEROUS[name]}")
    load = sub.add_parser("load", help="Dashboard: load <程序名>（程序需已在机器人上）")
    load.add_argument("program", help="机器人上的程序名，如 myprogram.urp")
    send = sub.add_parser("send-program", help="通过脚本端口下发 URScript 程序并运行")
    send.add_argument("--file", required=True, type=Path, help="本地 .script 文件")
    send.add_argument("--port", type=int, default=30001, choices=(30001, 30002, 30003),
                      help="脚本端口（默认 30001；本项目 CB3 实测可用，30002/30003 未在本机验证）")
    send.add_argument("--hold", action="store_true",
                      help="下发后保持脚本连接（不推荐，见文件头说明）")
    args = parser.parse_args()

    ip = args.ip
    if ip is None:
        cfg = config_module.load(args.config)
        ip = cfg.robot.ip

    if args.command == "info":
        return cmd_info(ip, args.timeout)
    if args.command == "send-program":
        return cmd_send_program(ip, args.file, args.port, args.hold, args.timeout)
    if args.command == "load":
        return cmd_dashboard(ip, f"load {args.program}", args.timeout, args.yes)
    command = DANGEROUS[args.command]
    return cmd_dashboard(ip, command, args.timeout, args.yes)


if __name__ == "__main__":
    raise SystemExit(main())
