"""遥操作数据录制：每个会话一个目录，CSV 记录机器人/目标/夹爪/PICO 手柄量。

目录结构::

    <base_dir>/<YYYYmmdd_HHMMSS>/
        meta.json    # 会话元数据（配置、机器人、夹爪参数、episode 数）
        robot.csv    # 主表：每个遥操作步一行

一个 episode 从 grip 按下（状态进入 TRACKING）开始计数，松开后结束；
``robot.csv`` 的 ``episode`` 列标记该行属于第几次示教，便于后续切分数据集。
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable

COLUMNS = [
    "t_wall", "t_mono", "episode", "state", "servo",
    "grip_left", "grip_right", "trigger_left", "trigger_right",
    "robot_x", "robot_y", "robot_z", "robot_rx", "robot_ry", "robot_rz",
    "q1", "q2", "q3", "q4", "q5", "q6",
    "target_x", "target_y", "target_z", "target_rx", "target_ry", "target_rz",
    "gripper_position", "gripper_state",
]


def _padded(values, length: int) -> list[str]:
    values = list(values or ())
    return [f"{float(v):.6f}" for v in values[:length]] + [""] * (length - len(values))


class SessionRecorder:
    """录制一个遥操作会话；:meth:`log` 每个主循环步调用一次。"""

    def __init__(self, base_dir: str | Path, *, meta: dict[str, Any] | None = None,
                 flush_every: int = 50,
                 wall_clock: Callable[[], float] = time.time,
                 mono_clock: Callable[[], float] = time.monotonic) -> None:
        self._wall_clock = wall_clock
        self._mono_clock = mono_clock
        self._flush_every = max(1, int(flush_every))
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self._wall_clock()))
        self.directory = Path(base_dir) / stamp
        self.directory.mkdir(parents=True, exist_ok=False)
        self.csv_path = self.directory / "robot.csv"
        self.meta_path = self.directory / "meta.json"
        self.episode = 0
        self._rows = 0
        self._prev_tracking = False
        self._meta = dict(meta or {})
        self._meta.update({
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self._wall_clock())),
            "columns": COLUMNS,
        })
        self._write_meta()
        self._file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(COLUMNS)

    def _write_meta(self) -> None:
        self.meta_path.write_text(
            json.dumps({**self._meta, "episodes": self.episode}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")

    def log(self, step, *, gripper_position: int | None = None,
            gripper_state: int | None = None) -> None:
        """记录一步；``step`` 为 ``TeleopStep``（缺省字段按空值记录）。"""
        state_name = getattr(step.state, "name", str(step.state))
        tracking = state_name == "TRACKING"
        if tracking and not self._prev_tracking:
            self.episode += 1
        self._prev_tracking = tracking
        row = [
            f"{self._wall_clock():.6f}", f"{self._mono_clock():.6f}",
            str(self.episode), state_name, str(int(bool(step.servo))),
            f"{getattr(step, 'grip_left', 0.0):.4f}", f"{getattr(step, 'grip_right', 0.0):.4f}",
            f"{getattr(step, 'trigger_left', 0.0):.4f}", f"{getattr(step, 'trigger_right', 0.0):.4f}",
            *_padded(getattr(step, "robot_pose", ()), 6),
            *_padded(getattr(step, "robot_joints", ()), 6),
            *_padded(step.pose, 6),
            "" if gripper_position is None else str(int(gripper_position)),
            "" if gripper_state is None else str(int(gripper_state)),
        ]
        self._writer.writerow(row)
        self._rows += 1
        if self._rows % self._flush_every == 0:
            self._flush()

    def _flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        if self._file.closed:
            return
        self._flush()
        self._file.close()
        self._write_meta()

    def __enter__(self) -> "SessionRecorder":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
