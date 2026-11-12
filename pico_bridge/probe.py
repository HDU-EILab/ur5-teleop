"""Read real VR controller data only; never send robot commands."""

import json
import math
import os
import socket
import time

import xrobotoolkit_sdk as xrt

#: SDK 按键 getter → 帧字段名（2026-11-11 核对 pybind 导出）
BUTTON_GETTERS = {
    "a": "get_A_button",
    "b": "get_B_button",
    "x": "get_X_button",
    "y": "get_Y_button",
    "menu_left": "get_left_menu_button",
    "menu_right": "get_right_menu_button",
    "axis_left_click": "get_left_axis_click",
    "axis_right_click": "get_right_axis_click",
}


def read_buttons():
    """读取手柄按键；SDK 缺某个 getter 时按 False 处理（兼容旧版本）。"""
    result = {}
    for name, getter in BUTTON_GETTERS.items():
        fn = getattr(xrt, getter, None)
        try:
            result[name] = bool(fn()) if callable(fn) else False
        except Exception:  # noqa: BLE001 - SDK 偶发异常不影响主流程
            result[name] = False
    return result


def read_axis():
    """读取摇杆 (x, y)；无效时回 (0, 0)。"""
    result = {}
    for side, getter in (("left", "get_left_axis"), ("right", "get_right_axis")):
        fn = getattr(xrt, getter, None)
        try:
            value = fn() if callable(fn) else None
            if value is not None and len(value) >= 2:
                result[side] = [float(value[0]), float(value[1])]
            else:
                result[side] = [0.0, 0.0]
        except Exception:  # noqa: BLE001
            result[side] = [0.0, 0.0]
    return result


def pose_or_none(value):
    if value is None or len(value) < 7:
        return None
    pose = [float(item) for item in value[:7]]
    if not all(math.isfinite(item) for item in pose):
        return None
    if not 0.8 < math.sqrt(sum(item * item for item in pose[3:7])) < 1.2:
        return None
    if max(abs(item) for item in pose[:3]) < 1e-4 and max(abs(item) for item in pose[3:6]) < 1e-4 and abs(abs(pose[6]) - 1) < 1e-4:
        return None
    return pose


def main():
    receiver = os.environ.get("PICO_PREVIEW_HOST", "host.docker.internal")
    port = int(os.environ.get("PICO_PREVIEW_PORT", "9121"))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    xrt.init()
    last_stamp = 0
    last_print = 0.0
    print("Waiting for VR tracking frames", flush=True)
    try:
        while True:
            stamp = int(xrt.get_time_stamp_ns())
            if stamp <= 0 or stamp == last_stamp:
                time.sleep(0.01)
                continue
            last_stamp = stamp
            left = pose_or_none(xrt.get_left_controller_pose())
            right = pose_or_none(xrt.get_right_controller_pose())
            head = pose_or_none(xrt.get_headset_pose())
            # UR5 单臂遥操作只需要头显 + 右手柄：另一侧手柄休眠/未配对时不再整帧丢弃，
            # 用已有的一侧占位以保持 UDP JSON 帧格式不变（下游 side=right，不使用 left）。
            if head is None or (left is None and right is None):
                time.sleep(0.01)
                continue
            if left is None:
                left = right
            if right is None:
                right = left

            def grip_or_zero(value):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return 0.0

            frame = {
                "pico_time_ns": stamp,
                "bridge_time_ns": time.time_ns(),
                "left": left,
                "right": right,
                "head": head,
                "left_grip": grip_or_zero(xrt.get_left_grip()),
                "right_grip": grip_or_zero(xrt.get_right_grip()),
                "left_trigger": grip_or_zero(xrt.get_left_trigger()),
                "right_trigger": grip_or_zero(xrt.get_right_trigger()),
                "buttons": read_buttons(),
                "axis": read_axis(),
            }
            sock.sendto(json.dumps(frame).encode(), (receiver, port))
            now = time.monotonic()
            if now - last_print >= 1.0:
                print(json.dumps(frame), flush=True)
                last_print = now
            time.sleep(0.005)
    finally:
        xrt.close()
        sock.close()


if __name__ == "__main__":
    main()
