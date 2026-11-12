# 设计说明

## 1. 总体数据流

```
VR 头显/手柄
   │  XRoboToolkit SDK（单客户端）
   ▼
pico_bridge/probe.py  ──UDP JSON:9121──►  ur5_teleop（主机）
                                             │
                              RTDE 125 Hz ◄──┤ 读 actual_TCP_pose / actual_q / 安全状态
                                             │
                          写 input_double_register_0..5
                          写 input_int_register_0（心跳）/ 1（cmd=1 跟随）
                                             ▼
                                 URScript: ur5_teleop.script
                                 get_inverse_kin + servoj（每周期限幅）
```

- 主机只发**目标 TCP 位姿**（基座系，m/rad），机器人端自行逆解并 `servoj`
- 机器人端每周期做位姿增量限幅、关节跳变拒绝、接触/阻塞保护，避免 C153 类保护停
- 夹爪走 Modbus RTU 独立串口，由主机直接控制

## 2. 状态机

`ur5_teleop/teleop.py`：

- **IDLE**：未离合。按 grip 时把当前手柄位姿与机器人实测位姿锁存为起点（相对映射）
- **TRACKING**：目标 = 锁存起点 + 手柄位移 × 平移倍率；姿态按旋转灵敏度缩放
- 松开 grip 立即回 IDLE（机器人保持最后目标，`servoj` 不推进）
- 安全层（`safety.py`）：工作空间裁剪、通信超时、偏差过大保持、故障 → FAULT

## 3. 数据采集

`ur5_teleop/collector.py` + `tools/teleop_supervisor.py`：

- B 短按开始/保存 episode；保存后倒计时回 home（期间再按 B 取消）
- 两路相机按时间戳与 100 Hz 机器人数据一起落盘：`robot.csv`、`videos/*.mp4`、`*_timestamps.csv`
- 右摇杆按下开始/结束标记"人工扰动窗口"，用于数据导出时截断

## 4. 回 home

`ur5_teleop/home.py`：默认 `movej` 到配置的关节 home（比 servoj 弧线更可靠），
回 home 失败会抛 `HomeError` 并保留当前程序状态。

## 5. 轨迹回放（可选）

`tools/play_trajectory.py` + `urscript/ur5_traj.script`：把离线关节轨迹按标称周期流式写给机器人端
`servoj`，用于复现固定动作；数据格式见 `cvpr-部署UR5/` 内的 manifest/校验文件。

## 6. 模块清单

| 模块 | 职责 |
|---|---|
| `ur_client.py` | RTDE 客户端（读写寄存器、Dashboard、脚本下发/程序状态） |
| `teleop.py` | 离合式遥操作状态机、目标位姿生成 |
| `tracking.py` / `target.py` / `frames.py` | 输入帧解析、位姿变换、相对映射 |
| `safety.py` | 工作空间/超时/偏差保护 |
| `cameras.py` | 双相机采集（OpenCV V4L2；macOS 可选 camcap） |
| `gripper.py` | 大寰夹爪 Modbus RTU 驱动 |
| `home.py` | movej 回 home |
| `collector.py` / `recording.py` | episode 录制与文件落盘 |
