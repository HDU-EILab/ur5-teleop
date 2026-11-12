# 采集数据 → LeRobot 数据集导出

脚本：`tools/export_lerobot.py`（依赖：`numpy`、`opencv-python`、`pandas`、`pyarrow`，以及系统 `ffmpeg`）

## 输入

`data/episodes/<时间戳>_epNNN/`（由 `tools/teleop_supervisor.py` 采集）：

- `robot.csv`：100 Hz 实测/目标位姿、关节、夹爪位置、trigger、状态
- `videos/global.mp4`、`videos/wrist.mp4` + `*_timestamps.csv`
- `meta.json`：任务文本、成功标记、人工扰动窗口、相机帧统计
- `annotations.json`：按键/事件

## 输出（LeRobot v2.1）

| 字段 | 维度 | 含义 |
|---|---|---|
| `observation.state` | 13 | 实测 TCP 6 + 关节 6 + 夹爪开度 1（0=闭合，1=全开） |
| `action` | 7 | 目标 TCP 6 + 夹爪命令 1 |
| `observation.images.global` / `.wrist` | 720×1280×3 | 双相机，30 fps H.264 |

- 以全局相机帧时间戳为主时钟，每帧取最近机器人数据
- 两路相机帧数可能略有差异：统一截断到共同帧数，保证视频帧数 == 数据行数
- 含"人工扰动窗口"的 episode 默认在窗口处截断并切成前后两段（各段独立），
  避免把"人类操作/机器人等待"当作动作监督；`meta/cut_info.json` 记录映射关系
- 输出：`meta/{info,tasks,episodes,stats,cut_info,segment_tags}.json(l)`、
  `data/chunk-000/*.parquet`、`videos/chunk-000/observation.images.*/*.mp4`

## 用法

```bash
# 1) 导出
python tools/export_lerobot.py \
  --episodes-dir data/episodes \
  --out data/lerobot_ur5_pickplace

# 2) 校验（episode 数 / 视频帧数==行数 / 无 NaN / fps）
python tools/check_dataset.py --root data/lerobot_ur5_pickplace
```

若训练端使用 lerobot 0.4+（v3 格式），可用官方 `convert_dataset_v21_to_v30` 转换后再训练。

## 统计量

`meta/stats.json` 含 state/action 的 mean/std/min/max/q01/q99；图像特征为占位统计（训练归一化主要用 state/action）。
