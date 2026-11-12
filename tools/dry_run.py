"""离线 dry-run：不连机器人，把 PICO 帧变成 UR 目标位姿并逐帧打印。

两种输入（其它行为相同）：

- 默认：实时监听 ``config.input.udp_port`` 的 PICO UDP JSON 帧（同 upstream dry_run），
  Ctrl-C 退出，退出前发一帧 idle；
- ``--replay FILE``：回放 JSONL 录制（每行一个 ``probe.py`` 格式的帧），逐帧打印后退出。

机器人后端固定用 ``FakeBackend``：不产生任何网络/机器人 I/O。每步把假机器人的
实测位姿更新成刚下发的目标（模拟“机器人刚好跟上”），让偏差看门狗按真实逻辑工作。
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ur5_teleop import config as config_module  # noqa: E402
from ur5_teleop import frames  # noqa: E402
from ur5_teleop.backend import FakeBackend  # noqa: E402
from ur5_teleop.teleop import TeleopLoop, TeleopStep  # noqa: E402
from ur5_teleop.tracking import ReplayFrameSource, UdpFrameSource  # noqa: E402

#: dry-run 假机器人的起始位姿（工作空间盒内；不连任何真实设备）
FAKE_START_POSE = (0.4, 0.0, 0.4, 0.0, 3.14, 0.0)

#: 实时模式最快打印间隔（秒），避免刷屏
PRINT_PERIOD_S = 0.2


def format_pose(pose) -> str:
    """格式化 pose6。"""
    return "[" + ", ".join(f"{value:+.4f}" for value in pose) + "]"


def format_delta(previous, pose) -> str:
    """打印相邻两帧目标的增量（m / rad）。"""
    if previous is None:
        return ""
    change = tuple(pose[i] - previous[i] for i in range(3))
    relative = frames.mat_mul(
        frames.rotvec_to_matrix(pose[3:]),
        frames.mat_transpose(frames.rotvec_to_matrix(previous[3:])),
    )
    return (" | dpos=(" + ", ".join(f"{value:+.4f}" for value in change)
            + f") m drot={frames.rotation_angle(relative):.4f} rad")


class StepPrinter:
    """逐帧打印目标位姿/增量/状态；实时模式下按状态变化或节流打印。"""

    def __init__(self) -> None:
        self.index = 0
        self.previous_pose: tuple[float, ...] | None = None
        self._last_state = None
        self._last_note: str | None = None
        self._last_print = 0.0

    def show(self, step: TeleopStep, *, throttle: bool = False) -> None:
        now = time.monotonic()
        state_changed = step.state is not self._last_state
        if throttle and not state_changed and now - self._last_print < PRINT_PERIOD_S:
            return
        self.index += 1
        stamp = "" if step.frame is None else f" stamp={step.frame.pico_time_ns}"
        line = (f"#{self.index:03d} {step.state.value:<8s} servo={int(step.servo)}"
                f" pose={format_pose(step.pose)}{format_delta(self.previous_pose, step.pose)}{stamp}")
        if step.note and step.note != self._last_note:
            line += f"  # {step.note}"
        print(line, flush=True)
        self.previous_pose = step.pose
        self._last_state = step.state
        self._last_note = step.note if step.note else None
        self._last_print = now


def print_header(config, mode: str) -> None:
    """打印本次 dry-run 的关键配置，便于核对。"""
    teleop = config.teleop
    safety = config.safety
    print(f"dry-run 模式：{mode}（FakeBackend，不连机器人）", flush=True)
    print(f"手柄侧={config.input.side} grip 阈值={config.input.grip_threshold}/"
          f"{config.input.grip_release_threshold} scale={teleop.scale} "
          f"mount_yaw_deg={teleop.mount_yaw_deg}", flush=True)
    print(f"每帧限幅：{teleop.max_pos_step_m} m / {teleop.max_rot_step_rad} rad；"
          f"工作空间 {safety.workspace_min} ~ {safety.workspace_max}", flush=True)
    print(f"假机器人起始位姿：{format_pose(FAKE_START_POSE)}", flush=True)
    print("序号 状态      servo 目标位姿 [x,y,z,rx,ry,rz] / 相邻帧增量 / 说明", flush=True)


def run_replay(path, config) -> int:
    """回放 JSONL 录制，逐帧打印后返回。"""
    source = ReplayFrameSource.from_jsonl(path)
    backend = FakeBackend(initial_pose=FAKE_START_POSE)
    backend.connect()
    loop = TeleopLoop(config, backend, source)
    printer = StepPrinter()
    print_header(config, f"回放 {path}")
    while True:
        step = loop.step()
        if step is None:
            break
        printer.show(step)
        # 假机器人“完美跟随”上一步目标，供偏差看门狗按真实逻辑工作
        backend.feed_state(pose=step.pose)
    backend.close()
    print(f"回放结束：共 {printer.index} 帧，最终状态 {loop.state.value}", flush=True)
    return 0


def run_live(config) -> int:
    """监听实时 PICO UDP 帧；Ctrl-C 退出（退出前发 idle）。"""
    source = UdpFrameSource(config.input.udp_port)
    backend = FakeBackend(initial_pose=FAKE_START_POSE)
    backend.connect()
    loop = TeleopLoop(config, backend, source)
    printer = StepPrinter()

    def on_step(step: TeleopStep) -> None:
        printer.show(step, throttle=True)
        # 实时模式下也让假机器人跟住目标，否则偏差看门狗会误报
        backend.feed_state(pose=step.pose)

    print_header(config, f"实时 UDP :{config.input.udp_port}")
    print("等待 PICO 数据；按住手柄 grip 开始跟踪，Ctrl-C 退出。", flush=True)
    try:
        state = loop.run(on_step=on_step)
    finally:
        source.close()
        backend.close()
    print(f"结束：{printer.index} 帧，最终状态 {state.value}", flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PICO → UR5 目标位姿 dry-run（不连机器人）")
    parser.add_argument("--replay", metavar="JSONL",
                        help="回放 JSONL 录制文件；省略则监听实时 UDP")
    parser.add_argument("--config", metavar="TOML", default=None,
                        help="配置文件路径（默认 config/ur5.toml）")
    args = parser.parse_args(argv)
    config = config_module.load(args.config)
    if args.replay:
        return run_replay(args.replay, config)
    return run_live(config)


if __name__ == "__main__":
    raise SystemExit(main())
