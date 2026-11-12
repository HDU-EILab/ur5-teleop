"""自动回 home。

两种模式（配置 ``[home] mode``）：

- ``movej``（默认，2026-11-11 实测更稳）：下发 URScript ``movej``，由控制器直接
  做关节插值到 ``home_joints_deg``。之前用 servo 弧线路径时，从高位（z≈0.76 m，
  接近伸展极限）会因为 IK 跳变卡死、偏差累积中止，``movej`` 则 ~1 s 精确到位。
- ``servo``（备用）：抬升 → 水平弧线扫掠 → 垂直落下（流式 servo，速度要慢，
  实测 40 mm/s 时跟随误差约 5 mm；80 mm/s 会累积偏差触发接触保护）。
"""

from __future__ import annotations

import math
import time
from typing import Callable, Iterable

from . import frames
from .backend import BackendError, URBackend
from .ur_client import DashboardClient, open_script_channel, send_program

#: 现场约定 home 关节角（deg），支持从配置覆盖
HOME_JOINTS_DEG = (-180.0, -90.0, -90.0, -90.0, 90.0, -90.0)

#: UR5 DH 参数（m），用于 home 关节角 → TCP 正解
_DH = {
    "d1": 0.089159, "a2": -0.425, "a3": -0.39225,
    "d4": 0.10915, "d5": 0.09465, "d6": 0.0823,
}

SAFETY_NORMAL = 1
ROBOT_MODE_RUNNING = 7


class HomeError(RuntimeError):
    """回 home 失败/中止。"""


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
            for i in range(len(a))]


def _homogeneous(x, y, z, rx, ry, rz):
    """由位置 + 轴角旋转向量构造 4x4 齐次矩阵。"""
    m = frames.rotvec_to_matrix((rx, ry, rz))
    return [[m[0][0], m[0][1], m[0][2], x],
            [m[1][0], m[1][1], m[1][2], y],
            [m[2][0], m[2][1], m[2][2], z],
            [0.0, 0.0, 0.0, 1.0]]


def ur5_fk(joints_rad: Iterable[float]) -> list[list[float]]:
    """UR5 标准 DH 正运动学 → 4x4 齐次矩阵（基座系）。"""
    q = [float(v) for v in joints_rad]
    d = _DH
    c = [math.cos(v) for v in q]
    s = [math.sin(v) for v in q]
    # 逐关节变换：T = A1 A2 ... A6（UR 标准 DH，alpha = [90,0,0,90,-90,0]）
    alphas = (math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0)
    dvals = (d["d1"], 0.0, 0.0, d["d4"], d["d5"], d["d6"])
    avals = (0.0, d["a2"], d["a3"], 0.0, 0.0, 0.0)
    T = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    for i in range(6):
        ca, sa = math.cos(alphas[i]), math.sin(alphas[i])
        A = [[c[i], -s[i] * ca, s[i] * sa, avals[i] * c[i]],
             [s[i], c[i] * ca, -c[i] * sa, avals[i] * s[i]],
             [0.0, sa, ca, dvals[i]],
             [0.0, 0.0, 0.0, 1.0]]
        T = _matmul(T, A)
    return T


def home_tcp_pose(joints_deg: Iterable[float] = HOME_JOINTS_DEG) -> tuple[float, ...]:
    """home 关节角 → TCP 位姿 [x,y,z,rx,ry,rz]（m, rad）。"""
    T = ur5_fk([math.radians(float(v)) for v in joints_deg])
    rot = [[T[i][j] for j in range(3)] for i in range(3)]
    rv = frames.matrix_to_rotvec(rot)
    return (T[0][3], T[1][3], T[2][3], rv[0], rv[1], rv[2])


def _rot_between(start_rot, end_rot):
    """R_start^T · R_end 的旋转向量（测地线）。"""
    rel = _matmul(frames.mat_transpose(start_rot), end_rot)
    return frames.matrix_to_rotvec(rel)


def _rot_apply(start_rot, rotvec, ratio: float):
    step = frames.rotvec_to_matrix(tuple(ratio * v for v in rotvec))
    return _matmul(start_rot, step)


def build_home_path(current: tuple[float, ...], target: tuple[float, ...], *,
                    lift_m: float = 0.15, speed_mps: float = 0.04,
                    dt: float = 0.008) -> list[tuple[tuple[float, ...], float]]:
    """构造三段式路径：抬升 → 弧线扫掠 → 落下；返回 [(pose, t), ...]。

    若当前已高于"目标高度 + 抬升量"，不再继续抬升（避免接近机械上限），
    直接把当前高度作为安全高度。
    """
    x0, y0, z0 = current[0], current[1], current[2]
    x1, y1, z1 = target[0], target[1], target[2]
    lift = max(0.0, float(lift_m))
    # 只有当需要"抬高去越过障碍"时才抬升：当前已高于 home 就保持当前高度
    z_safe = z0 if z0 >= z1 - 1e-3 else z1 + lift

    r0, a0 = math.hypot(x0, y0), math.atan2(y0, x0)
    r1, a1 = math.hypot(x1, y1), math.atan2(y1, x1)
    da = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi

    length_lift = abs(z_safe - z0)
    length_arc = abs(da) * max(r0, r1) + abs(r1 - r0)
    length_lower = abs(z1 - z_safe)
    total = max(length_lift + length_arc + length_lower, 1e-3)

    start_rot = frames.rotvec_to_matrix(current[3:6])
    end_rot = frames.rotvec_to_matrix(target[3:6])
    rotvec = _rot_between(start_rot, end_rot)

    steps = max(int(round((total / max(speed_mps, 1e-3)) / dt)), 1)
    poses: list[tuple[tuple[float, ...], float]] = []
    for k in range(1, steps + 1):
        s = k / steps
        travelled = s * total
        if travelled <= length_lift:
            frac = travelled / length_lift if length_lift > 1e-9 else 1.0
            x, y = x0, y0
            z = z0 + (z_safe - z0) * frac
        elif travelled <= length_lift + length_arc:
            frac = (travelled - length_lift) / length_arc if length_arc > 1e-9 else 1.0
            radius = r0 + (r1 - r0) * frac
            angle = a0 + da * frac
            x, y = radius * math.cos(angle), radius * math.sin(angle)
            z = z_safe
        else:
            frac = (travelled - length_lift - length_arc) / length_lower \
                if length_lower > 1e-9 else 1.0
            x, y = x1, y1
            z = z_safe + (z1 - z_safe) * frac
        rot = _rot_apply(start_rot, rotvec, s)
        rv = frames.matrix_to_rotvec(rot)
        poses.append(((x, y, z, rv[0], rv[1], rv[2]), s * total / max(speed_mps, 1e-3)))
    return poses


def run_home(backend: URBackend, *, printer: Callable[[str], None] = print,
             lift_m: float = 0.15, speed_mps: float = 0.04,
             home_joints_deg: Iterable[float] = HOME_JOINTS_DEG,
             max_pos_error_m: float = 0.15,
             dt: float = 0.008) -> None:
    """执行回 home；安全状态异常/偏差过大/通信失败时抛 HomeError。"""
    try:
        state = backend.read_state()
    except BackendError as exc:
        raise HomeError(f"读取机器人状态失败：{exc}") from exc
    # 速度滑块只在程序运行时才反映真实值（停机时为 0），等程序真正跑起来再判断
    deadline = time.monotonic() + 2.0
    while 0.0 <= state.speed_scaling < 0.05 and time.monotonic() < deadline:
        time.sleep(0.1)
        try:
            state = backend.read_state()
        except BackendError:
            pass
    if 0.0 <= state.speed_scaling < 0.05:
        raise HomeError(
            f"示教器速度滑块为 {state.speed_scaling * 100:.0f}%（<5%），机器人不会运动；请先拉到 100%")
    current = tuple(float(v) for v in state.tcp_pose)
    target = home_tcp_pose(home_joints_deg)
    poses = build_home_path(current, target, lift_m=lift_m, speed_mps=speed_mps, dt=dt)
    printer(f"[home] 当前→home：{len(poses)} 点 / {poses[-1][1]:.1f} s，"
            f"安全高度 +{lift_m:.2f} m，速度 {speed_mps:.2f} m/s")
    started = time.monotonic()
    for pose, t in poses:
        delay = started + t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        try:
            backend.send_sets(pose, servo=True)
            state = backend.read_state()
        except BackendError as exc:
            raise HomeError(f"通信中断：{exc}") from exc
        if state.safety_status != SAFETY_NORMAL:
            raise HomeError(f"安全状态异常（{state.safety_status}），回 home 中止")
        if state.robot_mode != ROBOT_MODE_RUNNING:
            raise HomeError(f"机器人不在 RUNNING（{state.robot_mode}），回 home 中止")
        err = math.dist([state.tcp_pose[i] for i in range(3)], pose[:3])
        if err > max_pos_error_m:
            raise HomeError(f"跟随偏差 {err * 1000:.0f} mm 超过阈值，回 home 中止")
    # 末段保持：servo 有跟随延迟，直接切 idle 会让机械臂停在 home 上方，
    # 这里再以 125 Hz 保持最终目标 1.5 s，确保真正到位
    settle_s = 1.5
    settle_end = time.monotonic() + settle_s
    while time.monotonic() < settle_end:
        try:
            backend.send_sets(poses[-1][0], servo=True)
            state = backend.read_state()
        except BackendError as exc:
            raise HomeError(f"通信中断：{exc}") from exc
        if state.safety_status != SAFETY_NORMAL:
            raise HomeError(f"安全状态异常（{state.safety_status}），回 home 中止")
        time.sleep(dt)
    try:
        backend.send_sets(poses[-1][0], servo=False)
    except BackendError:
        pass
    printer("[home] 已到达 home")


def build_movej_program(joints_deg: Iterable[float], *, a: float = 0.6, v: float = 0.4) -> str:
    """生成用 URScript movej 回 home 的程序文本（控制器做关节插值，不经过 servo/IK 路径）。"""
    q = ", ".join(f"{math.radians(float(x)):.8f}" for x in joints_deg)
    return (f"def go_home():\n"
            f"  textmsg(\"go_home: movej to home joints\")\n"
            f"  movej([{q}], a={a:.3f}, v={v:.3f})\n"
            f"  textmsg(\"go_home: done\")\n"
            f"end\n"
            f"go_home()\n")


def run_home_movej(ip: str, joints_deg: Iterable[float], *,
                   backend=None, port: int = 30001,
                   printer: Callable[[str], None] = print,
                   timeout_s: float = 60.0, a: float = 0.6, v: float = 0.4,
                   tolerance_deg: float = 1.0) -> None:
    """用 movej 程序回 home（更稳：不经过 servo 路径，避免高位 IK 跳变）。

    需要传入已连接的 ``backend`` 用于监视 runtime_state / 关节角；完成后机器人端
    遥操作程序已被替换，调用方需重新下发遥操作脚本。
    """
    target = [float(x) for x in joints_deg]
    printer(f"[home] movej 回 home：{[round(x, 1) for x in target]}")
    dash = DashboardClient(ip)
    dash.connect()
    try:
        printer(f"[home] stop：{dash.command('stop')}")
        if "NORMAL" not in dash.command("safetymode"):
            raise HomeError(f"安全状态异常（{dash.command('safetymode')}），回 home 中止")
        channel = open_script_channel(ip, port)
        send_program(channel, build_movej_program(target, a=a, v=v))
        # 必须等控制器把脚本收完再关连接，否则程序会被截断：
        # 之前出现过 "Compile error: name 'go_home' is not defined"
        time.sleep(0.8)
        channel.close()
    finally:
        dash.close()

    started = time.monotonic()
    moved = False
    started_program = False
    while time.monotonic() - started < timeout_s:
        time.sleep(0.1)
        try:
            state = backend.read_state()
        except BackendError as exc:
            raise HomeError(f"通信中断：{exc}") from exc
        if state.safety_status != SAFETY_NORMAL:
            raise HomeError(f"安全状态异常（{state.safety_status}），回 home 中止")
        if state.robot_mode != ROBOT_MODE_RUNNING:
            raise HomeError(f"机器人不在 RUNNING（{state.robot_mode}），回 home 中止")
        if state.runtime_state == 2:
            moved = True
            started_program = True
        elif not started_program and state.runtime_state == 1 and time.monotonic() - started > 3.0:
            raise HomeError("movej 程序未启动（可能编译失败），回 home 中止")
        joints = [math.degrees(float(v)) for v in state.joints]
        if len(joints) == 6 and all(abs(joints[i] - target[i]) <= tolerance_deg for i in range(6)):
            printer(f"[home] 已到达 home（movej）")
            return
        if moved and state.runtime_state == 1:
            raise HomeError(
                "movej 程序已结束但关节未到位：" +
                str([round(v, 1) for v in joints]))
    raise HomeError(f"movej 回 home 超时（{timeout_s:.0f}s）")
