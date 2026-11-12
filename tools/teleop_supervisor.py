#!/usr/bin/env python3
"""遥操作守护进程：断线自动重连，不自动做任何安全操作。

行为：
- 每轮先读 Dashboard 安全状态；不是 NORMAL（如 C207A0 保护停）就**只提醒、不自动解锁**，等待人工处理；
- 下发机器人端脚本后**立即关闭脚本通道**（实测程序继续运行；长期占着不读的连接会拖死控制器
  其它 TCP 发送，约 30 s 后 RTDE 掉线），再连 RTDE 跑 TeleopLoop；
- 主循环因通信故障（FAULT）退出 → 关闭连接、等待 2 s、重新下发程序并重连；
- PICO UDP 输入源跨轮复用，不重复绑定端口；
- Ctrl-C 退出。

用法：
    python3 tools/teleop_supervisor.py --config config/ur5.toml --yes
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ur5_teleop import config as config_module  # noqa: E402
from ur5_teleop.backend import BackendError  # noqa: E402
from ur5_teleop.cameras import CamcapCameraSet, CameraSet  # noqa: E402
from ur5_teleop.collector import Collector  # noqa: E402
from ur5_teleop.gripper import Gripper, GripperError  # noqa: E402
from ur5_teleop.home import HomeError, run_home, run_home_movej  # noqa: E402
from ur5_teleop.safety import TeleopState  # noqa: E402
from ur5_teleop.teleop import TeleopLoop, TeleopStep  # noqa: E402
from ur5_teleop.tracking import UdpFrameSource  # noqa: E402
from ur5_teleop.ur_client import (  # noqa: E402
    DashboardClient,
    RTDEBackend,
    open_script_channel,
    send_program,
)

DEFAULT_SCRIPT = ROOT / "urscript" / "ur5_teleop.script"
PRINT_PERIOD_S = 0.25
RECONNECT_DELAY_S = 2.0
SAFETY_POLL_S = 3.0


def dashboard_command(ip: str, command: str) -> None:
    try:
        client = DashboardClient(ip)
        client.connect()
        print(f"[supervisor] dashboard {command}: {client.command(command)}", flush=True)
        client.close()
    except OSError as exc:
        print(f"[supervisor] dashboard {command} 失败：{exc}", flush=True)


def dashboard_safety(ip: str) -> str:
    try:
        client = DashboardClient(ip)
        client.connect()
        reply = client.command("safetymode")
        client.close()
        return reply
    except OSError as exc:
        return f"<无法读取安全状态: {exc}>"


def wait_until_safe(ip: str) -> None:
    while True:
        status = dashboard_safety(ip)
        if "NORMAL" in status:
            return
        if status.startswith("<无法读取"):
            print(f"[supervisor] 机器人不可达（{status.strip('<>')}）—— 检查网线/控制柜上电；"
                  f"恢复后会自动继续", flush=True)
        else:
            print(f"[supervisor] 安全状态：{status} —— 等待人工解锁 "
                  f"(python3 tools/ur_ctl.py --ip {ip} --yes unlock-protective-stop)",
                  flush=True)
        time.sleep(SAFETY_POLL_S)


def make_printer():
    last = {"time": 0.0, "state": None, "note": ""}

    def on_step(step: TeleopStep) -> None:
        now = time.monotonic()
        changed = step.state is not last["state"]
        if not changed and now - last["time"] < PRINT_PERIOD_S:
            return
        last["time"] = now
        last["state"] = step.state
        if step.note and step.note != last["note"]:
            print(f"[{step.note}]", flush=True)
            last["note"] = step.note
        elif not step.note:
            last["note"] = ""
        servo = "servo" if step.servo else "idle "
        grip = ""
        if step.frame is not None:
            grip = f" grip(L/R)={step.frame.grip('left'):.2f}/{step.frame.grip('right'):.2f}"
        print(f"{step.state.name:<8} {servo} target={[round(v, 4) for v in step.pose]}{grip}",
              flush=True)

    return on_step


def gripper_position_from_step(cfg, step: TeleopStep) -> int:
    """扳机（0..1）→ 夹爪位置（0..1000）：默认扳机越按越闭合。"""
    trigger = step.trigger_right if cfg.gripper.trigger_side == "right" else step.trigger_left
    trigger = max(0.0, min(1.0, float(trigger)))
    if cfg.gripper.trigger_mode == "binary":
        # 扳机过半即视为"闭合"（0），松开即全开（1000）；比连续行程好控制
        closed = trigger >= 0.5
        if cfg.gripper.invert_trigger:
            closed = not closed
        return 0 if closed else 1000
    if cfg.gripper.invert_trigger:
        return int(round(trigger * 1000.0))
    return int(round((1.0 - trigger) * 1000.0))


class GripperTriggerFilter:
    """夹爪扳机指令滤波。

    binary 模式：闭合立即生效；**张开需要扳机低电平持续 ``release_debounce_s``**。
    原因：PICO 控制器跟踪丢失/丢包时 SDK 会把 trigger 报成 0（不是真的松开），
    binary 直接映射会让夹爪在空中张开、物体掉落（2026-11-11 实测两次）。
    """

    def __init__(self, gripper_cfg, clock=time.monotonic) -> None:
        self._cfg = gripper_cfg
        self._clock = clock
        self._cmd = 1000              # 初始全开
        self._low_since: float | None = None

    @property
    def command(self) -> int:
        return self._cmd

    def update(self, trigger: float) -> int:
        trigger = max(0.0, min(1.0, float(trigger)))
        if self._cfg.trigger_mode == "analog":
            if self._cfg.invert_trigger:
                self._cmd = int(round(trigger * 1000.0))
            else:
                self._cmd = int(round((1.0 - trigger) * 1000.0))
            return self._cmd
        closed = trigger >= 0.5
        if self._cfg.invert_trigger:
            closed = not closed
        now = self._clock()
        if closed:
            self._cmd = 0
            self._low_since = None
        else:
            if self._low_since is None:
                self._low_since = now
            if now - self._low_since >= float(self._cfg.release_debounce_s):
                self._cmd = 1000
        return self._cmd


class SaveHomeRequested(Exception):
    """B 短按保存 episode 后请求自动回 home（main 循环捕获）。"""


def _play_sound(name: str) -> None:
    try:
        subprocess.Popen(["afplay", f"/System/Library/Sounds/{name}.aiff"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


def _make_cameras(cfg):
    """按配置选择相机后端（macOS 默认 camcap 原生采集）。"""
    if cfg.collect.camera_backend == "camcap":
        return CamcapCameraSet(app_path=cfg.collect.camcap_app,
                               width=cfg.collect.camera_width,
                               height=cfg.collect.camera_height,
                               fps=cfg.collect.camera_fps,
                               global_name=cfg.collect.global_camera_name,
                               wrist_name=cfg.collect.wrist_camera_name)
    return CameraSet(indices=tuple(cfg.collect.camera_indices),
                     width=cfg.collect.camera_width,
                     height=cfg.collect.camera_height,
                     fps=cfg.collect.camera_fps)


def build_step_callback(cfg, printer, gripper: Gripper | None, collector: Collector):
    """主循环每步：打印 → 采集状态机（按键/录制）→ 夹爪跟随/保持。"""

    trigger_filter = GripperTriggerFilter(cfg.gripper)

    def on_step(step: TeleopStep) -> None:
        printer(step)
        action = collector.step(
            step,
            gripper_position=gripper.position_raw if gripper is not None else None,
            gripper_state=gripper.grip_state if gripper is not None else None,
        )
        if gripper is not None:
            if step.state is TeleopState.TRACKING and step.servo:
                # 只用"新到的 PICO 帧"更新夹爪：丢包时沿用旧帧不会误触发；
                # 张开还要求低电平持续 release_debounce_s（见 GripperTriggerFilter）
                if step.frame is not None:
                    trigger = (step.trigger_right if cfg.gripper.trigger_side == "right"
                               else step.trigger_left)
                    gripper.set_position(trigger_filter.update(trigger))
            else:
                gripper.hold()      # 非跟踪状态保持位置，避免松手后夹爪误动
        if action == "save_home":
            raise SaveHomeRequested()

    return on_step


def _send_teleop_program(cfg, program: str, port: int) -> None:
    """重新下发遥操作脚本（movej 回 home 会替换掉机器人端程序）。"""
    channel = open_script_channel(cfg.robot.ip, port)
    try:
        send_program(channel, program)
        time.sleep(0.8)
    finally:
        channel.close()


def countdown_then_home(cfg, backend, source, *, program: str = "", port: int = 30001) -> bool:
    """保存后倒计时（再按 B 取消）→ 自动回 home；返回是否执行了回 home。"""
    hold_pose = None
    try:
        hold_pose = tuple(backend.read_state().tcp_pose)
    except BackendError:
        pass
    print(f"[collect] {cfg.home.countdown_s:.0f}s 后自动回 home；再按一次 B 取消", flush=True)
    deadline = time.monotonic() + max(0.0, cfg.home.countdown_s)
    prev_b = True          # 保存时 B 刚按下，先等松开再检测"再次按下"
    next_tick = time.monotonic()
    while time.monotonic() < deadline:
        if hold_pose is not None:
            try:
                backend.send_sets(hold_pose, servo=False)
            except BackendError:
                break
        frame = source.poll()  # type: ignore[attr-defined]
        if frame is not None:
            pressed = frame.button("b")
            if pressed and not prev_b:
                _play_sound("Basso")
                print("[collect] 已取消回 home，机器人保持原地", flush=True)
                return False
            prev_b = pressed
        now = time.monotonic()
        if now >= next_tick:
            _play_sound("Tink")
            next_tick = now + 1.0
        time.sleep(0.05)
    _play_sound("Glass")
    if cfg.home.mode == "movej":
        run_home_movej(cfg.robot.ip, cfg.home.joints_deg, backend=backend,
                       port=port, printer=lambda msg: print(msg, flush=True))
        # movej 替换了机器人端程序，恢复遥操作脚本供下一 episode 使用
        _send_teleop_program(cfg, program, port)
    else:
        run_home(backend, printer=lambda msg: print(msg, flush=True),
                 lift_m=cfg.home.lift_m, speed_mps=cfg.home.speed_mps,
                 home_joints_deg=cfg.home.joints_deg)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--yes", action="store_true", help="确认允许发送运动指令")
    args = parser.parse_args()
    if not args.yes:
        print("该工具会发送运动指令。确认现场安全后加 --yes 重跑。")
        return 2

    cfg = config_module.load(args.config)
    program = args.script.read_text(encoding="utf-8")
    source = UdpFrameSource(cfg.input.udp_port)
    printer = make_printer()

    gripper: Gripper | None = None
    if cfg.gripper.enabled:
        try:
            gripper = Gripper(
                cfg.gripper.port, baud=cfg.gripper.baud, slave_id=cfg.gripper.slave_id,
                speed=cfg.gripper.speed, force=cfg.gripper.force,
                rate_hz=cfg.gripper.rate_hz)
            print(f"[supervisor] 夹爪：{gripper.connect()}", flush=True)
        except GripperError as exc:
            print(f"[supervisor] 夹爪不可用，按无夹爪继续：{exc}", flush=True)
            gripper = None

    collector = Collector(
        ROOT / cfg.collect.dir,
        task_en=cfg.collect.task_en,
        task_zh=cfg.collect.task_zh,
        cameras=_make_cameras(cfg),
        printer=lambda msg: print(msg, flush=True))
    print(f"[supervisor] 采集目录：{collector.root}", flush=True)
    print(f"[supervisor] 相机探测：{collector.cameras.probe()}", flush=True)

    on_step = build_step_callback(cfg, printer, gripper, collector)
    rounds = 0
    print(f"[supervisor] PICO UDP :{cfg.input.udp_port}，scale={cfg.teleop.scale:g}，"
          f"mount_yaw={cfg.teleop.mount_yaw_deg:g}°，机器人 {cfg.robot.ip}", flush=True)
    try:
        while True:
            rounds += 1
            wait_until_safe(cfg.robot.ip)
            print(f"[supervisor] 第 {rounds} 轮：下发机器人端程序", flush=True)
            channel = None
            backend = None
            try:
                channel = open_script_channel(cfg.robot.ip, args.port)
                send_program(channel, program)
                time.sleep(0.8)
                channel.close()   # 程序会继续运行；不能长期占着不读的脚本连接
                channel = None
                backend = RTDEBackend(
                    cfg.robot.ip,
                    protocol_version=cfg.robot.protocol_version,
                    frequency_hz=cfg.robot.rtde_frequency_hz,
                    state_timeout_s=max(0.3, cfg.safety.comm_timeout_s * 1.5),
                )
                print(backend.connect(), flush=True)
                # 等第一帧 RTDE 状态再进主循环（否则首步会误判状态读取失败）
                deadline = time.monotonic() + 3.0
                first_state = None
                while time.monotonic() < deadline:
                    try:
                        first_state = backend.read_state()
                        break
                    except BackendError:
                        time.sleep(0.05)
                if first_state is None:
                    raise RuntimeError("3 秒内未收到 RTDE 状态帧")
                if 0.0 <= first_state.speed_scaling < 0.05:
                    print(f"[supervisor] ⚠️ 示教器速度滑块为 {first_state.speed_scaling * 100:.0f}%，"
                          f"机械臂不会运动；请在示教器上拉到 100%", flush=True)
                while True:
                    loop = TeleopLoop(cfg, backend, source)
                    try:
                        loop.run(on_step)
                    except SaveHomeRequested:
                        collector.save(success=True)
                        try:
                            countdown_then_home(cfg, backend, source,
                                                program=program, port=args.port)
                        except HomeError as exc:
                            print(f"[supervisor] 回 home 中止：{exc}", flush=True)
                            break
                        print("[supervisor] 已回 home；按 B 开始下一 episode", flush=True)
                        continue
                    if loop.faulted:
                        if collector.armed:
                            collector.save(success=False)
                        print("[supervisor] 遥操作故障（通信/偏差），准备重连", flush=True)
                    break
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                print(f"[supervisor] 本轮异常：{exc}", flush=True)
            finally:
                # 机器人端看门狗在 CB3 上会以 C207A0 保护停形式锁停，已关闭；
                # 主机侧故障/退出时显式停程序，保证不再继续执行 idle 循环。
                dashboard_command(cfg.robot.ip, "stop")
                if backend is not None:
                    try:
                        backend.send_sets(backend.read_state().tcp_pose, servo=False)
                    except Exception:  # noqa: BLE001
                        pass
                    backend.close()
                if channel is not None:
                    try:
                        channel.close()
                    except OSError:
                        pass
            print(f"[supervisor] {RECONNECT_DELAY_S:g}s 后重连…", flush=True)
            time.sleep(RECONNECT_DELAY_S)
    except KeyboardInterrupt:
        print("\n[supervisor] 已退出", flush=True)
        return 0
    finally:
        if gripper is not None:
            gripper.hold()
            gripper.close()
        collector.close()
        source.close()


if __name__ == "__main__":
    raise SystemExit(main())
