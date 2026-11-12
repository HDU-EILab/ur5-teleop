#!/usr/bin/env python3
"""UR5 连接探测：找 IP、判断控制柜世代、读只读状态。

用法：
    python3 tools/probe_ur.py --interfaces                  # 列出候选网口
    python3 tools/probe_ur.py --scan 192.168.0.0/24         # 扫描段内 29999/30004
    python3 tools/probe_ur.py --ip 192.168.1.100            # 详细探测
    python3 tools/probe_ur.py --ip 192.168.1.100 --json

只发只读命令（version/robotmode/safetymode/programState 等），不改变机器人状态。
依据：docs/research/ur5-external-control.md §1.1/§1.2/§3.2。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ipaddress
import json
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ur5_teleop.backend import BackendError  # noqa: E402
from ur5_teleop.ur_client import (  # noqa: E402
    DASHBOARD_PORT,
    RTDE_PORT,
    DashboardClient,
    describe_generation,
)

CANDIDATE_PORTS = (DASHBOARD_PORT, RTDE_PORT)


def probe_tcp(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def probe_rtde(ip: str) -> dict:
    sys.path.insert(0, str(ROOT / "vendor" / "RTDE_Python_Client_Library"))
    from rtde import rtde  # noqa: PLC0415

    con = rtde.RTDE(ip, RTDE_PORT)
    try:
        con.connect()
        version = con.get_controller_version()  # tuple: (major, minor, bugfix, build)
        if version[0] is None:
            return {"reachable": False, "error": "控制器版本读取失败"}
        major, minor, bugfix, build = version
        return {
            "reachable": True,
            "version": f"{major}.{minor}.{bugfix}-{build}",
            "major": int(major),
        }
    except Exception as exc:  # noqa: BLE001 - 探测工具需要容错
        return {"reachable": False, "error": str(exc)}
    finally:
        try:
            con.disconnect()
        except Exception:  # noqa: BLE001
            pass


def probe_dashboard(ip: str, timeout: float = 2.0) -> dict:
    result: dict = {"reachable": False}
    client = DashboardClient(ip, timeout=timeout)
    try:
        banner = client.connect()
        result["reachable"] = True
        result["banner"] = banner
        # 依次尝试只读命令；老固件可能不支持个别命令
        for key, command in (
            ("software", client.version),
            ("polyscope", client.polyscope_version),
            ("robotmode", client.robot_mode),
            ("safetymode", client.safety_mode),
            ("programState", client.program_state),
            ("model", client.robot_model),
            ("serial", client.serial_number),
        ):
            try:
                result[key] = command()
            except Exception as exc:  # noqa: BLE001
                result[key] = f"<{exc}>"
    except OSError as exc:
        result["error"] = str(exc)
    finally:
        client.close()
    return result


def detect_generation(rtde_info: dict, dash_info: dict) -> str:
    major = rtde_info.get("major")
    if isinstance(major, int) and major > 0:
        return describe_generation(major)
    text = " ".join(
        str(dash_info.get(key, "")) for key in ("software", "polyscope", "model")
    )
    for prefix, label in (("URSoftware 3.", "CB3 (PolyScope 3.x)"),
                          ("URSoftware 5.", "e-Series (PolyScope 5.x)"),
                          ("URSoftware 10.", "PolyScope X (gSeries)")):
        if prefix in text:
            return label
    return "未知"


def run_interfaces() -> int:
    try:
        output = subprocess.run(
            ["ifconfig", "-a"], capture_output=True, text=True, check=False
        ).stdout
    except OSError as exc:
        print(f"执行 ifconfig 失败：{exc}")
        return 1
    # 按接口分块收集，输出时 status 可能出现在 inet 之后，需要先缓存再打印
    interfaces: list[dict] = []
    current: dict | None = None
    for line in output.splitlines():
        if line and not line[0].isspace() and ":" in line:
            current = {"name": line.split(":")[0], "inet": None, "netmask": None, "status": "?"}
            interfaces.append(current)
            continue
        if current is None:
            continue
        stripped = line.strip()
        if stripped.startswith("status:"):
            current["status"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("inet "):
            parts = stripped.split()
            if len(parts) > 3:
                current["inet"] = parts[1]
                current["netmask"] = parts[3]
    print("候选网口（非回环、有 IPv4）：")
    for item in interfaces:
        address = item["inet"]
        if not address or address.startswith("127."):
            continue
        print(f"  {item['name']:<6} {address:<16} netmask={item['netmask']:<12} "
              f"status={item['status']}")
    print("\n下一步：python3 tools/probe_ur.py --scan <网口所在网段，如 192.168.0.0/24>")
    return 0


def run_scan(cidr: str, timeout: float) -> int:
    try:
        network = ipaddress.ip_network(cidr, strict=False)
    except ValueError as exc:
        print(f"网段无效：{exc}")
        return 1
    if network.num_addresses > 1024:
        print(f"网段过大（{network.num_addresses} 个地址），请缩小到 /22 以内")
        return 1
    hosts = [str(host) for host in network.hosts()]
    print(f"扫描 {network}（{len(hosts)} 个地址，端口 {CANDIDATE_PORTS}）...")
    found: list[tuple[str, list[int]]] = []

    def worker(host: str):
        open_ports = [port for port in CANDIDATE_PORTS if probe_tcp(host, port, timeout)]
        return host, open_ports

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
        for host, open_ports in pool.map(worker, hosts):
            if open_ports:
                found.append((host, open_ports))
    if not found:
        print("未发现开放 29999/30004 的主机。检查：机器人是否上电、网线是否接在 UR 网口、"
              "Mac 静态 IP 是否与机器人同网段。")
        return 1
    print("\n发现候选主机：")
    for host, open_ports in sorted(found, key=lambda item: ipaddress.ip_address(item[0])):
        print(f"  {host:<16} 开放端口 {open_ports}")
    print("\n下一步：python3 tools/probe_ur.py --ip <主机>")
    return 0


def run_probe(ip: str, timeout: float, as_json: bool) -> int:
    rtde_info = probe_rtde(ip)
    dash_info = probe_dashboard(ip, timeout=timeout)
    generation = detect_generation(rtde_info, dash_info)
    payload = {
        "ip": ip,
        "rtde": rtde_info,
        "dashboard": dash_info,
        "generation": generation,
    }
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if rtde_info.get("reachable") or dash_info.get("reachable") else 1

    print(f"=== UR 控制柜探测 {ip} ===")
    if rtde_info.get("reachable"):
        print(f"RTDE {RTDE_PORT}        : 可达  控制器 {rtde_info['version']}")
    else:
        print(f"RTDE {RTDE_PORT}        : 不可达（{rtde_info.get('error', '未知原因')}）")
    if dash_info.get("reachable"):
        print(f"Dashboard {DASHBOARD_PORT}   : 可达  {dash_info.get('banner', '')}")
        for key, label in (("software", "软件版本"), ("polyscope", "PolyScope"),
                           ("robotmode", "机器人模式"), ("safetymode", "安全模式"),
                           ("programState", "程序状态"), ("model", "型号"),
                           ("serial", "序列号")):
            if key in dash_info:
                print(f"  {label:<8}: {dash_info[key]}")
    else:
        print(f"Dashboard {DASHBOARD_PORT}   : 不可达（{dash_info.get('error', '未知原因')}）")
    print(f"判定世代              : {generation}")
    if generation.startswith("e-Series"):
        print("提示：e-Series 需在示教器 Settings → System → Remote Control → Enable 并切到 Remote 模式，"
              "否则 30001-30003 断连、Dashboard 拒绝执行命令。")
    elif generation.startswith("CB3"):
        print("提示：CB3 无 Remote Control 模式；RTDE 上限 125 Hz，寄存器建议 0-23。")
    print("\n下一步：python3 tools/status.py --ip " + ip)
    return 0 if (rtde_info.get("reachable") or dash_info.get("reachable")) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ip", help="机器人 IP，详细探测")
    group.add_argument("--scan", metavar="CIDR", help="扫描网段（如 192.168.0.0/24）")
    group.add_argument("--interfaces", action="store_true", help="列出候选网口")
    parser.add_argument("--timeout", type=float, default=2.0, help="单次探测超时（默认 2 s）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出 --ip 结果")
    args = parser.parse_args()
    if args.interfaces:
        return run_interfaces()
    if args.scan:
        return run_scan(args.scan, min(args.timeout, 1.0))
    return run_probe(args.ip, args.timeout, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
