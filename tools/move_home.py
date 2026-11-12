#!/usr/bin/env python3
"""把 UR5 从当前位姿平滑移动到标称 home 姿（不改安装配置）。

标称 home（基座安装方位 -180°）关节角：[-180, -90, 90, -90, -90, 0]°
对应 TCP 约 (+0.487, +0.109, +0.432) m（tool0）。

为什么不用 movej/关节插值：从常见停放姿到 home 需要 J3 翻 ~185°、J5 翻 ~186°，关节空间插值会把
手臂甩到 TCP 半径 ~1.0 m、z ~1.0 m（接近甚至超出 UR5 可达范围），既危险又容易撞顶。
本工具改用"水平圆弧扫掠"：保持 TCP 高度与半径基本不变，绕基座把 TCP 从当前方位角扫到 home 方位角，
姿态用旋转插值平滑过渡；机器人端仍用 ur5_teleop.script 的 get_inverse_kin+servoj 跟随。

用法：
    python3 tools/move_home.py --dry-run          # 只打印路径包络，不连机器人
    python3 tools/move_home.py --yes              # 实机移动（约 15 s，2 倍速）
    python3 tools/move_home.py --yes --duration 30   # 恢复原速

安全：移动期间人手不离急停；扫掠圆弧经过的区域必须无人无障碍。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from play_trajectory import (  # noqa: E402
    PlaybackError,
    _parse_home_joints,
    SAFETY_MODE_NAMES,
    TcpPlayer,
    TcpTrajectory,
    _matrix_from_pose,
    _mat_transpose3,
    _pose_from_matrix,
    _rot_to_rotvec,
    _rotvec_to_rot,
    ur5_fk,
)
from ur5_teleop import config as config_module  # noqa: E402
from ur5_teleop.ur_client import (  # noqa: E402
    DashboardClient,
    RTDEBackend,
    open_script_channel,
    send_program,
)

#: 现场约定 home 关节角（deg，2026-11-09 现场调整：基座 -180°、肘/腕为现场支路）
HOME_JOINTS_DEG = (-180.0, -90.0, -90.0, -90.0, 90.0, -90.0)
#: 机器人端 IK+servoj 脚本
TELEOP_SCRIPT = ROOT / "urscript" / "ur5_teleop.script"
MODE_RUNNING = 7
SAFETY_NORMAL = 1
CTRL_PERIOD = 0.008


def home_tcp_pose(joints_deg: Sequence[float] | None = None) -> tuple[float, ...]:
    """由 home 关节角正解出 home TCP 位姿（基地系，m / rad）。"""
    joints = joints_deg if joints_deg is not None else HOME_JOINTS_DEG
    matrix = ur5_fk([math.radians(v) for v in joints])
    return _pose_from_matrix(matrix)


def _orientation_step(r_from: Sequence[float], r_to: Sequence[float], ratio: float) -> list[list[float]]:
    """朝向插值：R(t) = R_from · exp(t · log(R_fromᵀ R_to))。"""
    r0, r1 = _rotvec_to_rot(r_from), _rotvec_to_rot(r_to)
    rel = [[sum(_mat_transpose3(r0)[i][k] * r1[k][j] for k in range(3)) for j in range(3)]
           for i in range(3)]
    rv = _rot_to_rotvec(rel)
    step = _rotvec_to_rot([ratio * v for v in rv])
    return [[sum(r0[i][k] * step[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def build_path(current: Sequence[float], target: Sequence[float],
               duration_s: float) -> tuple[list[tuple[float, ...]], list[float]]:
    """当前 TCP → home TCP 的水平圆弧扫掠（极坐标插值 + 朝向旋转插值）。"""
    steps = max(int(round(duration_s / CTRL_PERIOD)), 1)
    x0, y0, z0 = current[0], current[1], current[2]
    radius0 = math.hypot(x0, y0)
    angle0 = math.atan2(y0, x0)
    x1, y1, z1 = target[0], target[1], target[2]
    radius1 = math.hypot(x1, y1)
    angle1 = math.atan2(y1, x1)
    # 取最短扫掠方向
    delta_angle = (angle1 - angle0 + math.pi) % (2.0 * math.pi) - math.pi
    poses: list[tuple[float, ...]] = []
    times: list[float] = []
    for k in range(1, steps + 1):
        ratio = k / steps
        smooth = ratio * ratio * ratio * (10.0 - 15.0 * ratio + 6.0 * ratio * ratio)
        radius = radius0 + (radius1 - radius0) * smooth
        angle = angle0 + delta_angle * smooth
        z = z0 + (z1 - z0) * smooth
        rot = _orientation_step(current[3:6], target[3:6], smooth)
        rv = _rot_to_rotvec(rot)
        poses.append((radius * math.cos(angle), radius * math.sin(angle), z,
                      rv[0], rv[1], rv[2]))
        times.append(k * CTRL_PERIOD)
    return poses, times


def _path_envelope(poses: Sequence[Sequence[float]]) -> str:
    radii = [math.sqrt(p[0] ** 2 + p[1] ** 2) for p in poses]
    zs = [p[2] for p in poses]
    r3 = [math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) for p in poses]
    return (f"路径 {len(poses)} 点 / {len(poses) * CTRL_PERIOD:.0f} s："
            f"XY 半径 {min(radii):.3f}~{max(radii):.3f} m，z {min(zs):.3f}~{max(zs):.3f} m，"
            f"基座壳半径 {min(r3):.3f}~{max(r3):.3f} m")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="沿水平圆弧把 UR5 移回标称 home 姿（不改安装配置）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--duration", type=float, default=15.0,
                        help="扫掠时长（s），默认 15（2026-11-11 起 2 倍速；用 30 恢复原速）")
    parser.add_argument("--home-joints", default=None,
                        help="自定义 home 关节角（deg，逗号分隔 6 个）；默认现场约定 home")
    parser.add_argument("--max-pos-error", type=float, default=0.05,
                        help="TCP 跟随偏差上限（m），默认 0.05")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    joints = _parse_home_joints(args.home_joints) if args.home_joints else HOME_JOINTS_DEG
    target = home_tcp_pose(joints)
    print(f"目标 home 关节(deg): {list(joints)}")
    print(f"目标 home TCP(m/rad): {tuple(round(v, 4) for v in target)}")

    if args.dry_run:
        current = tuple(target[i] + (0.9 if i == 0 else 0.0) for i in range(6))
        # 用一个"典型停放姿"演示包络：把当前位姿取为 home 绕基座转 160°、半径放大 8%
        r = math.hypot(target[0], target[1]) * 1.08
        ang = math.atan2(target[1], target[0]) + math.radians(160)
        current = (r * math.cos(ang), r * math.sin(ang), target[2] - 0.02,
                   target[3], target[4], target[5])
        poses, _ = build_path(current, target, args.duration)
        print(f"[dry-run] 示例当前 TCP: {tuple(round(v, 3) for v in current[:3])}")
        print(f"[dry-run] {_path_envelope(poses)}")
        print("[dry-run] 不连接机器人。")
        return 0

    if not args.yes:
        print("该工具会移动机械臂。确认现场安全后加 --yes 重跑。")
        return 2

    cfg = config_module.load(args.config)
    script = TELEOP_SCRIPT.read_text(encoding="utf-8")
    dashboard: DashboardClient | None = None
    backend: RTDEBackend | None = None
    try:
        dashboard = DashboardClient(cfg.robot.ip)
        print(f"[dashboard] {dashboard.connect()}")
        safety = dashboard.command("safetymode")
        mode = dashboard.command("robotmode")
        print(f"[预检] {safety} | {mode}")
        if "NORMAL" not in safety:
            raise PlaybackError(f"安全状态不是 NORMAL（{safety}）")
        if "RUNNING" not in mode:
            raise PlaybackError(f"机器人不在 RUNNING（{mode}）")
        print(f"[dashboard] stop: {dashboard.command('stop')}")
        time.sleep(1.0)

        print(f"[program] 下发 {TELEOP_SCRIPT.name}")
        channel = open_script_channel(cfg.robot.ip, 30001)
        send_program(channel, script)
        time.sleep(0.8)
        channel.close()

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
        if state is None:
            raise PlaybackError("5 s 内未收到 RTDE 状态帧")

        current = tuple(state.tcp_pose)
        distance = math.dist(current[:3], target[:3])
        print(f"[当前] TCP={tuple(round(v, 4) for v in current[:3])}，"
              f"到 home 直线距离 {distance * 1000:.0f} mm")
        poses, times = build_path(current, target, args.duration)
        print(f"[路径] {_path_envelope(poses)}")

        playback = TcpTrajectory(poses=poses, t=times,
                                 transform=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                                            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                                 target_start=tuple(target), source=None)  # type: ignore[arg-type]
        player = TcpPlayer(backend, playback, max_pos_error_m=args.max_pos_error)
        print("[安全检查] 即将沿水平圆弧回 home：确认扫掠区域无人无障碍、手在急停上。3 s 后开始…")
        for countdown in (3, 2, 1):
            print(f"  {countdown}…")
            time.sleep(1.0)
        player.play()
        final = backend.read_state()
        error = math.dist(final.tcp_pose[:3], target[:3])
        joint_error_deg = max(
            abs(math.degrees(final.joints[i] - math.radians(HOME_JOINTS_DEG[i])))
            for i in range(6))
        print(f"[完成] 终点 TCP 距 home {error * 1000:.1f} mm；关节误差 {joint_error_deg:.2f}°")
        return 0
    except KeyboardInterrupt:
        print("\n[中止] Ctrl-C")
        return 130
    except PlaybackError as exc:
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
