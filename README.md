# UR5 VR Teleoperation

VR-controller teleoperation for a UR5 robot arm, with data collection and LeRobot export.

`VR headset/controllers → host Python process → UR RTDE target pose → robot-side URScript servoj`

[English](README.md) | [中文](README.zh.md)

## Hardware selection

| Component | Selection |
|---|---|
| Robot arm | Universal Robots **UR5** (CB3 / PolyScope 3.x validated; e-Series compatible) |
| VR headset | **PICO 4 Ultra** (controllers) via XRoboToolkit PC Service |
| Cameras | 2× USB RGB cameras (1280×720 @30 fps): one global view, one wrist view |
| Gripper | Dahua **PGH-300-60** parallel jaw gripper (Modbus RTU / RS485) |
| Board | NVIDIA **AGX Orin 64GB** (Ubuntu 20.04 / JetPack 5.1) |

Everything else (IPs, workspace, camera device nodes, serial ports) is configured per site in
`config/ur5.toml`; the code does not hard-code it.

## Repository layout

```
ur5_teleop/          core library: RTDE client, teleop state machine, cameras, gripper, safety, collector
urscript/            robot-side programs: ur5_teleop.script (servoj tracking), ur5_traj.script (joint-trajectory replay)
tools/               CLI tools: run_teleop, teleop_supervisor, ur_ctl, status, servo_check, export_lerobot ...
pico_bridge/         VR input layer: XRoboToolkit PC Service + probe.py (UDP JSON forwarding)
config/ur5.toml      example configuration (edit per site)
tests/               unit tests (pure Python, no robot required)
docs/                design, safety, dataset export notes
vendor/              Universal Robots official RTDE Python client (BSD-3-Clause)
```

## Installation (Miniconda recommended)

Python ≥ 3.11 is required. **Miniconda (or an equivalent conda distribution) is strongly recommended**
so the teleoperation stack runs in an isolated environment:

```bash
conda create -n ur5-teleop python=3.11 -y
conda activate ur5-teleop
pip install -r requirements.txt
cp config/ur5.toml config/ur5.local.toml   # edit per site; do not commit local config
```

<details>
<summary>Alternative: plain venv</summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
</details>

## Configuration

`config/ur5.local.toml`:

- `[robot] ip`: controller IP
- `[safety] workspace_min/max`: **must be narrowed to the real workspace**, otherwise the software guard is meaningless
- `[gripper] port`: USB-RS485 adapter (Linux `/dev/ttyUSB0`, macOS `/dev/cu.usbserial-XXXX`)
- `[collect] camera_indices`: 0=global camera, 1=wrist camera (adjust to the site's OpenCV indices)
- `[teleop] scale/rot_scale`: translation scale and rotation sensitivity

## Quick start

1. **Start the VR input layer** (see `pico_bridge/README.md`): official XRoboToolkit PC Service + `probe.py`
   - `probe.py` sends controller frames to `UDP 9121` by default
2. **Load the robot-side program** (first run/after homing):
   ```bash
   python tools/ur_ctl.py --ip <ROBOT_IP> send-program --file urscript/ur5_teleop.script
   ```
3. **Start teleoperation**:
   ```bash
   python tools/run_teleop.py --config config/ur5.local.toml
   ```
   - Hold grip to engage (clutch style; release to stop)
   - Trigger controls the gripper
4. **Collect episodes** (optional):
   ```bash
   python tools/teleop_supervisor.py --config config/ur5.local.toml
   ```
   - `B` starts/saves an episode; press the right stick to mark human-intervention windows
   - Data goes to `data/episodes/<timestamp>_epNNN/`

## Safety

Before running on the real robot:

1. `python tools/status.py --ip <ROBOT_IP>`: verify safety state / robot mode
2. `tools/servo_check.py`: 0 mm / small-step servo verification
3. Narrow the workspace box; joint limits, step clamps and contact guards live in the robot-side script
4. Keep the e-stop within reach; unlock protective stops following controller prompts

See `docs/safety.md` for details.

## Tests

```bash
python -m unittest discover -s tests   # 213 tests, pure Python, no robot required
```

## Dataset export

`tools/export_lerobot.py` converts collected episodes into a LeRobot v2.1 dataset
(two cameras + state/action); `tools/check_dataset.py` validates it. See `docs/lerobot-export.md`.

## License

MIT (see `LICENSE`). `vendor/RTDE_Python_Client_Library` is the official Universal Robots RTDE
Python client, BSD-3-Clause (see its `RTDE_LICENSE`).
