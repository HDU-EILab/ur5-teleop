"""配置：从 TOML 读取（Python 3.11+ 标准库 tomllib，无第三方依赖）。

配置文件默认在仓库根目录 ``config/ur5.toml``。
契约修改需 Lead 确认（见 docs/design.md §4）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import pycompat

pycompat.ensure_supported_python()   # 旧版 Python（<3.11）自动 re-exec 到新解释器

try:
    import tomllib
except ModuleNotFoundError:  # 3.11 以下且未成功 re-exec 时，尝试第三方 tomli
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError as exc:  # pragma: no cover - 由 pycompat 负责给出提示
        raise ModuleNotFoundError(
            "需要 Python ≥3.11（标准库 tomllib）或安装 tomli") from exc

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "ur5.toml"

VALID_SIDES = ("left", "right")


@dataclass(frozen=True)
class RobotConfig:
    ip: str = "192.168.1.100"
    protocol_version: int = 2          # RTDE 协议版本；CB3 老固件连不上时试 1
    rtde_frequency_hz: float = 125.0   # CB3 最大 125；e-Series 可到 500

    def __post_init__(self) -> None:
        if not self.ip:
            raise ValueError("robot.ip 不能为空")
        if self.protocol_version not in (1, 2, 3):
            raise ValueError(f"robot.protocol_version 必须是 1/2/3，收到 {self.protocol_version}")
        if not 1.0 <= self.rtde_frequency_hz <= 500.0:
            raise ValueError(f"robot.rtde_frequency_hz 必须在 [1, 500]，收到 {self.rtde_frequency_hz}")


@dataclass(frozen=True)
class InputConfig:
    udp_port: int = 9121
    side: str = "right"                # 用哪只手柄控制这只 UR5
    grip_threshold: float = 0.8        # 离合按下阈值
    grip_release_threshold: float = 0.6  # 离合松开阈值（回差，避免抖动）
    grip_release_debounce_s: float = 0.0   # grip 低于释放阈值需持续多久才真正松开（PICO 抖动过滤）
    grip_reengage_debounce_s: float = 0.0  # 未离合时 grip 达到软件阈值后持续多久自动续跟
    frame_timeout_s: float = 0.2       # 超过此时长没收到 PICO 帧即视为断流

    def __post_init__(self) -> None:
        if self.side not in VALID_SIDES:
            raise ValueError(f"input.side 必须是 {VALID_SIDES}，收到 {self.side!r}")
        if not 0 < self.udp_port < 65536:
            raise ValueError(f"input.udp_port 非法：{self.udp_port}")
        if not 0.0 < self.grip_threshold <= 1.0:
            raise ValueError(f"input.grip_threshold 必须在 (0, 1]，收到 {self.grip_threshold}")
        if self.grip_release_debounce_s < 0 or self.grip_reengage_debounce_s < 0:
            raise ValueError("input.grip_release_debounce_s / grip_reengage_debounce_s 不能为负")
        if not 0.0 <= self.grip_release_threshold < self.grip_threshold:
            raise ValueError(
                f"input.grip_release_threshold 必须在 [0, grip_threshold)，"
                f"收到 {self.grip_release_threshold}")
        if self.frame_timeout_s <= 0:
            raise ValueError("input.frame_timeout_s 必须 > 0")


@dataclass(frozen=True)
class TeleopConfig:
    scale: float = 1.0                 # 手柄位移 1:1 → 机器人位移；<1 缩小行程
    rot_scale: float = 1.0             # 手腕旋转灵敏度：<1 更不灵敏（0.7 = 打七折）
    mount_yaw_deg: float = 0.0         # UR 基座相对 PICO 前方的偏航角补偿
    max_pos_step_m: float = 0.005      # 主机侧每帧最大位移（0.5 m/s @ 100 Hz 帧率）
    max_rot_step_rad: float = 0.02     # 主机侧每帧最大转角

    def __post_init__(self) -> None:
        if not 0.0 < self.scale <= 2.0:
            raise ValueError(f"teleop.scale 必须在 (0, 2]，收到 {self.scale}")
        if not 0.0 < self.rot_scale <= 2.0:
            raise ValueError(f"teleop.rot_scale 必须在 (0, 2]，收到 {self.rot_scale}")
        if self.max_pos_step_m <= 0 or self.max_rot_step_rad <= 0:
            raise ValueError("teleop 步长限幅必须 > 0")


@dataclass(frozen=True)
class SafetyConfig:
    workspace_min: tuple[float, float, float] = (-0.8, -0.8, 0.0)
    workspace_max: tuple[float, float, float] = (0.8, 0.8, 1.2)
    deviation_pos_m: float = 0.08      # 目标与实测量偏差阈值（首次联调观察跟随误差后再收紧）
    deviation_rot_rad: float = 0.35    # ≈20°
    deviation_hold_s: float = 0.3      # 偏差持续多久判定故障
    comm_timeout_s: float = 0.25       # RTDE 状态帧超时（CB3 实测偶有 >100ms 停顿）

    def __post_init__(self) -> None:
        if len(self.workspace_min) != 3 or len(self.workspace_max) != 3:
            raise ValueError("safety.workspace_min/max 必须是 3 维")
        for lo, hi in zip(self.workspace_min, self.workspace_max):
            if not lo < hi:
                raise ValueError(
                    f"safety.workspace_min 必须逐轴小于 max，收到 {self.workspace_min} / "
                    f"{self.workspace_max}")
        if min(self.deviation_pos_m, self.deviation_rot_rad,
               self.deviation_hold_s, self.comm_timeout_s) <= 0:
            raise ValueError("safety 的看门狗阈值/时长必须 > 0")


@dataclass(frozen=True)
class GripperConfig:
    """大寰 PGC-300-60 电爪（RS485 / Modbus RTU）。位置 0-1000 = 闭合→全开。"""

    enabled: bool = False
    port: str = "/dev/ttyUSB0"
    baud: int = 115200
    slave_id: int = 1
    speed: int = 100                   # 0-100 %
    force: int = 20                    # 20-100 %
    stroke_mm: float = 60.0            # 行程（位置 0-1000 对应 0..stroke_mm mm，仅记录用）
    trigger_side: str = "right"        # 用哪只手的扳机控制
    invert_trigger: bool = False       # False: 扳机越按越闭合；True: 越按越张开
    trigger_mode: str = "analog"       # analog=随扳机行程连续；binary=过半即闭合、松开即全开
    release_debounce_s: float = 1.0    # binary 模式释放去抖：低电平持续这么久才张开（防丢包误开）
    rate_hz: float = 40.0              # 后台串口轮询/下发频率

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError(f"gripper.enabled 必须是布尔值，收到 {self.enabled!r}")
        if self.enabled and not self.port:
            raise ValueError("gripper.enabled=true 时必须配置 gripper.port")
        if not 1200 <= self.baud <= 921600:
            raise ValueError(f"gripper.baud 非法：{self.baud}")
        if not 1 <= self.slave_id <= 247:
            raise ValueError(f"gripper.slave_id 必须在 1..247，收到 {self.slave_id}")
        if not 0 <= self.speed <= 100:
            raise ValueError(f"gripper.speed 必须在 0..100，收到 {self.speed}")
        if not 20 <= self.force <= 100:
            raise ValueError(f"gripper.force 必须在 20..100，收到 {self.force}")
        if self.stroke_mm <= 0:
            raise ValueError("gripper.stroke_mm 必须 > 0")
        if self.trigger_side not in VALID_SIDES:
            raise ValueError(f"gripper.trigger_side 必须是 {VALID_SIDES}，收到 {self.trigger_side!r}")
        if self.trigger_mode not in ("analog", "binary"):
            raise ValueError(f"gripper.trigger_mode 必须是 analog/binary，收到 {self.trigger_mode!r}")
        if self.release_debounce_s < 0:
            raise ValueError(f"gripper.release_debounce_s 不能为负，收到 {self.release_debounce_s}")


@dataclass(frozen=True)
class RecordingConfig:
    """数据录制：每个会话写一个目录（robot.csv + meta.json）。"""

    enabled: bool = True
    dir: str = "data/sessions"         # 相对仓库根目录
    flush_every: int = 50              # 每 N 行 flush 一次

    def __post_init__(self) -> None:
        if not self.dir:
            raise ValueError("recording.dir 不能为空")
        if self.flush_every <= 0:
            raise ValueError(f"recording.flush_every 必须 > 0，收到 {self.flush_every}")


@dataclass(frozen=True)
class HomeConfig:
    """自动回 home。movej=控制器关节插值（推荐）；servo=旧的抬升→弧线→落下。"""

    mode: str = "movej"                # movej | servo
    joints_deg: tuple[float, ...] = (-180.0, -90.0, -90.0, -90.0, 90.0, -90.0)
    lift_m: float = 0.15               # 回 home 前相对当前高度抬升（避开桌面物体）
    speed_mps: float = 0.10            # 回 home 线速度
    countdown_s: float = 3.0           # 保存后的取消倒计时

    def __post_init__(self) -> None:
        if self.mode not in ("movej", "servo"):
            raise ValueError(f"home.mode 必须是 movej/servo，收到 {self.mode!r}")
        if len(self.joints_deg) != 6:
            raise ValueError(f"home.joints_deg 必须是 6 个角度，收到 {self.joints_deg}")
        if self.lift_m < 0:
            raise ValueError(f"home.lift_m 必须 ≥ 0，收到 {self.lift_m}")
        if self.speed_mps <= 0:
            raise ValueError(f"home.speed_mps 必须 > 0，收到 {self.speed_mps}")
        if self.countdown_s < 0:
            raise ValueError(f"home.countdown_s 必须 ≥ 0，收到 {self.countdown_s}")


@dataclass(frozen=True)
class CollectConfig:
    """Episode 采集与双相机参数。"""

    dir: str = "data/episodes"
    task_en: str = ("Put the red cylinder and the blue cube into the storage box. Leave all other objects outside the box.")
    task_zh: str = "将红色圆柱和蓝色方块放入收纳盒，其他无关物体留在盒外。"
    camera_backend: str = "opencv"           # camcap（原生 AVFoundation）| opencv（UVC）
    camcap_app: str = "~/Applications/camcap.app"
    global_camera_name: str = "global"
    wrist_camera_name: str = "wrist"
    camera_indices: tuple[int, int] = (0, 1)
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: float = 30.0

    def __post_init__(self) -> None:
        if not self.dir:
            raise ValueError("collect.dir 不能为空")
        if self.camera_backend not in ("camcap", "opencv"):
            raise ValueError(f"collect.camera_backend 必须是 camcap/opencv，收到 {self.camera_backend!r}")
        if len(self.camera_indices) != 2:
            raise ValueError(f"collect.camera_indices 必须是 2 个索引，收到 {self.camera_indices}")
        if self.camera_width <= 0 or self.camera_height <= 0 or self.camera_fps <= 0:
            raise ValueError("collect 相机分辨率/帧率必须 > 0")


@dataclass(frozen=True)
class Config:
    robot: RobotConfig
    input: InputConfig
    teleop: TeleopConfig
    safety: SafetyConfig
    source: Path | None = None
    gripper: GripperConfig = field(default_factory=GripperConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    home: HomeConfig = field(default_factory=HomeConfig)
    collect: CollectConfig = field(default_factory=CollectConfig)


def load(path: str | Path | None = None) -> Config:
    """读取 TOML 配置；缺省节/字段使用上面的默认值。"""
    source = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not source.is_file():
        raise FileNotFoundError(f"找不到配置文件：{source}")
    with source.open("rb") as fh:
        raw = tomllib.load(fh)

    def section(name: str) -> dict:
        value = raw.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"配置节 [{name}] 必须是表")
        return value

    def build(cls, name: str):
        values = dict(section(name))
        for key in ("workspace_min", "workspace_max"):
            if key in values:
                values[key] = tuple(float(v) for v in values[key])
        return cls(**values)

    return Config(
        robot=build(RobotConfig, "robot"),
        input=build(InputConfig, "input"),
        teleop=build(TeleopConfig, "teleop"),
        safety=build(SafetyConfig, "safety"),
        source=source,
        gripper=build(GripperConfig, "gripper"),
        recording=build(RecordingConfig, "recording"),
        home=build(HomeConfig, "home"),
        collect=build(CollectConfig, "collect"),
    )
