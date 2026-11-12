# VR 输入层：XRoboToolkit PC Service + `probe.py`

把 VR 头显与两只手柄的 6DoF 位姿/按键，通过 UDP JSON 转发给本机的 `ur5_teleop`。
本层只负责"采集输入 + 转发"，不连接 UR 控制柜、不发送机械臂指令。

## 组成

| 文件 | 说明 |
|---|---|
| `probe.py` | 读取 XRoboToolkit SDK，向 `UDP 9121` 发送 JSON 帧 |
| `Dockerfile` | Ubuntu 22.04 + XRoboToolkit PC Service（可选，也可直接装在主机上） |
| `entrypoint.sh` | 容器入口：先起 PC Service，再 exec `probe.py` |
| `vendor/` | **不在本仓库**：PXREA SDK Python 绑定与 `libPXREARobotSDK.so`，需按官方文档自行获取 |

> 本层依赖 XRoboToolkit PC Service（官方提供）。SDK 为单客户端模式：
> 同一时间只允许一个进程初始化 SDK，调试时不要另开第二个客户端，否则会抢断数据流。

## 关键约定

- 输出端口：`UDP 9121`（与 `config/ur5.example.toml` 的 `input.udp_port` 一致）
- 帧格式（JSON）：
  ```json
  {"bridge_time_ns": 0,
   "head": {"position": [0,0,0], "rotation": [0,0,0,1]},
   "right_controller": {"position": [...], "rotation": [...], "grip": 0.0, "trigger": 0.0, "buttons": {}},
   "left_controller":  {...},
   "buttons": {}, "axis": {}}
  ```
- 单臂使用时，另一侧手柄休眠也可正常工作（`probe.py` 会做占位）

## 运行

```bash
# 先启动 XRoboToolkit PC Service（地址与端口按官方文档配置）
python3 pico_bridge/probe.py --host 127.0.0.1 --port 9121
```
