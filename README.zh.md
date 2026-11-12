# UR5 VR 遥操作

基于 VR 手柄的 UR5 机械臂遥操作与数据采集代码，支持 episode 采集与 LeRobot 数据集导出。

`VR 头显/手柄 → 主机 Python 进程 → UR RTDE 目标位姿 → 机器人端 URScript servoj`

[English](README.md) | [中文](README.zh.md)

## 硬件选型

| 部件 | 选型 |
|---|---|
| 机械臂 | Universal Robots **UR5**（CB3 / PolyScope 3.x 已验证；e-Series 兼容写法） |
| VR 头显 | **PICO 4 Ultra**（手柄），通过 XRoboToolkit PC Service 接入 |
| 相机 | 2 路 USB RGB 相机（1280×720 @30 fps）：一路全局、一路腕部 |
| 夹爪 | 大寰 **PGH-300-60** 平行夹爪（Modbus RTU / RS485） |
| 开发板 | NVIDIA **AGX Orin 64GB**（Ubuntu 20.04 / JetPack 5.1） |

其余环境信息（IP、工作空间、相机设备号、串口）都在 `config/ur5.toml` 里按现场配置，代码不写死。

## 目录结构

```
ur5_teleop/          核心库：RTDE 客户端、遥操作状态机、相机、夹爪、安全、采集
urscript/            机器人端程序：ur5_teleop.script（servoj 跟随）、ur5_traj.script（关节轨迹回放）
tools/               命令行工具：run_teleop、teleop_supervisor、ur_ctl、status、servo_check、export_lerobot ...
pico_bridge/         VR 输入层：XRoboToolkit PC Service + probe.py（UDP JSON 转发）
config/ur5.toml      示例配置（按现场修改）
tests/               单元测试（纯 Python，无需机器人）
docs/                设计、安全、数据导出说明
vendor/              UR 官方 RTDE Python 客户端（BSD-3-Clause）
```

## 安装（推荐 Miniconda）

需要 Python ≥ 3.11。**推荐使用 Miniconda（或其他 conda 发行版）创建独立虚拟环境**，
避免与系统 Python/其他项目的依赖冲突：

```bash
conda create -n ur5-teleop python=3.11 -y
conda activate ur5-teleop
pip install -r requirements.txt
cp config/ur5.toml config/ur5.local.toml   # 按现场修改；不要提交本地配置
```

<details>
<summary>备选：普通 venv</summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
</details>

## 配置要点

编辑 `config/ur5.local.toml`：

- `[robot] ip`：控制柜 IP
- `[safety] workspace_min/max`：**必须按现场实际工作空间改窄**，否则软件保护失去意义
- `[gripper] port`：USB-RS485 设备（Linux `/dev/ttyUSB0`，macOS `/dev/cu.usbserial-XXXX`）
- `[collect] camera_indices`：0=全局相机，1=腕部相机（按现场 OpenCV 索引调整）
- `[teleop] scale/rot_scale`：平移倍率与旋转灵敏度

## 快速开始

1. **启动 VR 输入层**（见 `pico_bridge/README.md`）：官方 XRoboToolkit PC Service + `probe.py`
   - `probe.py` 默认向 `UDP 9121` 发送手柄数据
2. **下发机器人程序**（首次/回 home 后）：
   ```bash
   python tools/ur_ctl.py --ip <ROBOT_IP> send-program --file urscript/ur5_teleop.script
   ```
3. **启动遥操作**：
   ```bash
   python tools/run_teleop.py --config config/ur5.local.toml
   ```
   - 按住 grip 进入跟随（离合式，松开即停）
   - 扳机控制夹爪开合
4. **采集数据**（可选）：
   ```bash
   python tools/teleop_supervisor.py --config config/ur5.local.toml
   ```
   - B 键开始/保存 episode，右摇杆按下标记人工扰动窗口
   - 数据在 `data/episodes/<时间戳>_epNNN/`

## 安全

上真机前务必：

1. 先跑 `python tools/status.py --ip <ROBOT_IP>` 确认安全状态/运行模式
2. 用 `tools/servo_check.py` 做 0 mm / 小步长伺服验证
3. 工作空间盒按现场改小；关节限位、步长限幅、接触保护都在机器人端脚本里
4. 保持急停按钮可达；出现保护停时按控制器提示解锁

详见 `docs/safety.md`。

## 测试

```bash
python -m unittest discover -s tests   # 213 项，纯 Python，无需机器人
```

## 数据导出

采集结果可用 `tools/export_lerobot.py` 导出为 LeRobot v2.1 数据集（双相机 + state/action），
用 `tools/check_dataset.py` 校验；详见 `docs/lerobot-export.md`。

## License

MIT（见 `LICENSE`）。`vendor/RTDE_Python_Client_Library` 为 Universal Robots 官方 RTDE Python 客户端，
BSD-3-Clause（见其 `RTDE_LICENSE`）。
