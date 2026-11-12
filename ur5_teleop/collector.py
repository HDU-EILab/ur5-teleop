"""Episode 采集状态机（PICO 按键驱动）。

交互（与 docs/task1-control-mapping.md 一致）：

- **B 短按**：未录制 → 开始一个新 episode；录制中（且机器人未在运动）→ 保存 episode，
  返回 ``"save_home"`` 让上层执行 3 s 倒计时 + 自动回 home。
- **右摇杆按下**：标记"人工扰动窗口"开始/结束（仅在机器人静止、录制中时有效）。
- 录制期间双相机持续采集；扰动窗口内动作标签在导出阶段屏蔽（本模块只记录窗口时间）。

每个 episode 目录::

    data/episodes/<YYYYmmdd_HHMMSS>_ep<NNN>/
        meta.json          # 任务文本、配置快照、相机状态、成功/失败
        robot.csv          # 每步机器人状态/目标/夹爪/相机帧索引
        annotations.json   # 扰动窗口、按键事件、起止时间
        videos/global.mp4, wrist.mp4, {name}_timestamps.csv
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable

from .cameras import CameraSet

TASK_TEXT_EN = ("Put the red cylinder and the blue cube into the storage box. Leave all other objects outside the box.")
TASK_TEXT_ZH = "将红色圆柱和蓝色方块放入收纳盒，其他无关物体留在盒外。"

COLUMNS = [
    "t_wall", "t_mono", "state", "servo",
    "grip_left", "grip_right", "trigger_left", "trigger_right",
    "robot_x", "robot_y", "robot_z", "robot_rx", "robot_ry", "robot_rz",
    "q1", "q2", "q3", "q4", "q5", "q6",
    "target_x", "target_y", "target_z", "target_rx", "target_ry", "target_rz",
    "gripper_position", "gripper_state",
    "cam_global_frame", "cam_global_t", "cam_wrist_frame", "cam_wrist_t",
]

#: 触发采集动作的按键名（来自 PicoFrame.buttons）
BUTTON_SAVE = "b"
BUTTON_DISTURB = "axis_right_click"


def _fmt(values, n: int) -> list[str]:
    values = list(values or ())
    return [f"{float(v):.6f}" for v in values[:n]] + [""] * (n - len(values))


class Collector:
    """采集状态机；不直接控制机器人，只产出一致的数据与动作请求。"""

    def __init__(self, root: str | Path, *,
                 task_en: str = TASK_TEXT_EN, task_zh: str = TASK_TEXT_ZH,
                 cameras: CameraSet | None = None,
                 printer: Callable[[str], None] = print,
                 clock: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], float] = time.time) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.task_en = task_en
        self.task_zh = task_zh
        self.cameras = cameras if cameras is not None else CameraSet()
        self.printer = printer
        self._clock = clock
        self._wall = wall_clock
        self._prev_buttons: dict[str, bool] = {}

        self.armed = False
        self.episode_count = self._discover_last_episode()
        self.directory: Path | None = None
        self.disturbance: list[list[float]] = []
        self.events: list[dict[str, Any]] = []
        self._csv_file = None
        self._csv_writer = None
        self._rows = 0
        self._started_at: float | None = None
        self._camera_status: dict[str, Any] = {}
        self.last_step = None

    def _discover_last_episode(self) -> int:
        """续接已有 episode 编号：服务重启后不从 1 重新开始（避免 meta.episode 重复）。"""
        import re
        max_n = 0
        for entry in self.root.glob("*_ep*"):
            match = re.search(r"_ep(\d+)$", entry.name)
            if match and entry.is_dir():
                max_n = max(max_n, int(match.group(1)))
        return max_n

    # -- 按键边沿检测 --
    def _edge(self, step, name: str) -> bool:
        pressed = bool(getattr(step, "buttons", {}).get(name, False))
        was = self._prev_buttons.get(name, False)
        self._prev_buttons[name] = pressed
        return pressed and not was

    # -- 生命周期 --
    def arm(self, step=None) -> bool:
        """开始一个新 episode；失败返回 False（不抛异常）。"""
        if self.armed:
            return True
        robot_pose = getattr(step, "robot_pose", ()) if step is not None else ()
        if len(robot_pose) != 6:
            self.printer("[collect] 无法开始录制：机器人位姿无效")
            return False
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self._wall()))
        self.episode_count += 1
        directory = self.root / f"{stamp}_ep{self.episode_count:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.disturbance = []
        self.events = [{"type": "arm", "t": self._clock() if step is None else None}]
        self._rows = 0
        # 先启动相机（预热），完成后再开始 episode 计时，避免预热时间被算进数据
        self._camera_status = self.cameras.start(directory)
        self._started_at = self._clock()
        if not self._camera_status.get("ok"):
            self.printer(f"[collect] 相机未就绪（继续录制机器人数据）：{self._camera_status}")
        self._csv_file = (directory / "robot.csv").open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(COLUMNS)
        self.armed = True
        self.printer(f"[collect] 开始录制 episode {self.episode_count}：{directory}")
        return True

    def step(self, step, *, gripper_position: int | None = None,
             gripper_state: int | None = None) -> str | None:
        """每个遥操作步调用；返回动作请求（目前只有 "save_home"）。"""
        self.last_step = step
        if self.armed:
            self._write_row(step, gripper_position, gripper_state)

        action: str | None = None
        if self._edge(step, BUTTON_SAVE):
            if not self.armed:
                self.arm(step)
            elif not bool(getattr(step, "servo", False)):
                self.events.append({"type": "save_request", "t": self._clock()})
                action = "save_home"
            else:
                self.printer("[collect] 机器人仍在运动：先松开 grip 再按 B 保存")

        if self._edge(step, BUTTON_DISTURB):
            if not self.armed:
                self.printer("[collect] 未在录制：摇杆标记无效")
            elif bool(getattr(step, "servo", False)):
                self.printer("[collect] 机器人仍在运动：先松开 grip 再做人工扰动标记")
            elif self.disturbance and self.disturbance[-1][1] is None:
                self.disturbance[-1][1] = self._clock()
                self.events.append({"type": "disturbance_end", "t": self._clock()})
                self.printer("[collect] 人工扰动窗口结束")
            else:
                self.disturbance.append([self._clock(), None])
                self.events.append({"type": "disturbance_start", "t": self._clock()})
                self.printer("[collect] 人工扰动窗口开始")
        return action

    def _write_row(self, step, gripper_position: int | None,
                   gripper_state: int | None) -> None:
        cams = self.cameras.latest()
        row = [
            f"{self._wall():.6f}", f"{self._clock():.6f}",
            getattr(step.state, "name", str(step.state)), str(int(bool(step.servo))),
            f"{getattr(step, 'grip_left', 0.0):.4f}", f"{getattr(step, 'grip_right', 0.0):.4f}",
            f"{getattr(step, 'trigger_left', 0.0):.4f}", f"{getattr(step, 'trigger_right', 0.0):.4f}",
            *_fmt(getattr(step, "robot_pose", ()), 6),
            *_fmt(getattr(step, "robot_joints", ()), 6),
            *_fmt(step.pose, 6),
            "" if gripper_position is None else str(int(gripper_position)),
            "" if gripper_state is None else str(int(gripper_state)),
            str(cams.get("global", {}).get("frame", "")),
            f"{cams.get('global', {}).get('t_mono', 0.0):.6f}",
            str(cams.get("wrist", {}).get("frame", "")),
            f"{cams.get('wrist', {}).get('t_mono', 0.0):.6f}",
        ]
        self._csv_writer.writerow(row)
        self._rows += 1

    def save(self, success: bool = True) -> Path | None:
        """结束当前 episode 并落盘；返回目录。"""
        if not self.armed:
            return None
        self.armed = False
        directory = self.directory
        camera_stats = self.cameras.stop()
        if self.disturbance and self.disturbance[-1][1] is None:
            self.disturbance[-1][1] = self._clock()
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
        ended = self._clock()
        meta = {
            "episode": self.episode_count,
            "task": self.task_en,
            "task_zh": self.task_zh,
            "started_at": self._started_at,
            "ended_at": ended,
            "duration_s": None if self._started_at is None else round(ended - self._started_at, 3),
            "rows": self._rows,
            "success": bool(success),
            "disturbance_windows": [[a, b] for a, b in self.disturbance],
            "camera": self._camera_status,
            "camera_stats": camera_stats,
            "columns": COLUMNS,
        }
        if directory is not None:
            (directory / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (directory / "annotations.json").write_text(
                json.dumps({"events": self.events, "disturbance_windows":
                            meta["disturbance_windows"]}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            self.printer(f"[collect] episode 已保存：{directory}")
        self.directory = None
        return directory

    def discard(self) -> Path | None:
        """丢弃当前 episode（移到 _discarded，仅用于调试排查）。"""
        if not self.armed:
            return None
        self.armed = False
        directory = self.directory
        self.directory = None
        self.cameras.stop()
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        if directory is not None:
            target = self.root / "_discarded" / directory.name
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                directory.rename(target)
            except OSError:
                pass
            self.printer(f"[collect] episode 已丢弃：{target}")
        return directory

    def close(self) -> None:
        if self.armed:
            self.save(success=False)
