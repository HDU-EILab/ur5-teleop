"""双相机彩色采集（USB UVC 彩色相机；macOS 可选原生 AVFoundation 采集）。

macOS 上 librealsense 无法 claim 深度接口，但两台相机都作为 UVC 摄像头可见，
SOP 需要的是 RGB 图像（observation.images.global/wrist），因此这里用 OpenCV 采集彩色流。
无 cv2 或相机不可用时自动降级为"无相机"模式（记录 camera_status，不阻断遥操作）。
"""

from __future__ import annotations

import csv
import platform
import time
from pathlib import Path
from typing import Any, Callable

try:  # 可选依赖：缺省时降级
    import cv2  # type: ignore
except Exception:  # noqa: BLE001
    cv2 = None


def _capture_backend() -> int:
    """按平台选择 OpenCV 采集后端：Linux 用 V4L2，macOS 用 AVFoundation。"""
    if cv2 is None:
        return 0
    if platform.system() == "Linux":
        return getattr(cv2, "CAP_V4L2", 0)
    return getattr(cv2, "CAP_AVFOUNDATION", 0)


class CameraSet:
    """打开 UVC 相机（索引或 /dev/videoN 路径），episode 内写 mp4 + 帧时间戳 CSV。"""

    def __init__(self, indices: tuple[Any, ...] = (0, 1),
                 names: tuple[str, ...] = ("global", "wrist"),
                 width: int = 1280, height: int = 720, fps: float = 30.0) -> None:
        self.indices = indices
        self.names = names
        self.width = width
        self.height = height
        self.fps = float(fps)
        self._captures: dict[str, Any] = {}
        self._writers: dict[str, Any] = {}
        self._files: dict[str, Any] = {}
        self._counts = {name: 0 for name in names}
        self._latest_ts = {name: 0.0 for name in names}
        self._stop = False
        self._threads: list[Any] = []
        self.available = cv2 is not None
        self.error: str | None = None if cv2 is not None else "cv2 未安装"

    # -- 探测 --
    def probe(self) -> dict[str, Any]:
        """不启动录制，检查每个索引/设备能否打开并读到一帧。"""
        info: dict[str, Any] = {"opencv": cv2 is not None, "cameras": {}}
        if cv2 is None:
            return info
        backend = _capture_backend()
        for name, index in zip(self.names, self.indices):
            cap = cv2.VideoCapture(index, backend)
            entry: dict[str, Any] = {"index": index, "opened": bool(cap.isOpened())}
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                ok, frame = cap.read()
                entry["read"] = bool(ok)
                if ok and frame is not None:
                    entry["resolution"] = [int(frame.shape[1]), int(frame.shape[0])]
                entry["fps"] = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            cap.release()
            info["cameras"][name] = entry
        return info

    # -- 录制 --
    def start(self, episode_dir: Path, fps: float | None = None) -> dict[str, Any]:
        """开始写 mp4；返回状态字典（失败时仍返回，不抛异常）。"""
        status: dict[str, Any] = {"ok": False, "cameras": {}}
        if cv2 is None:
            status["error"] = "cv2 未安装"
            return status
        (episode_dir / "videos").mkdir(parents=True, exist_ok=True)
        self._stop = False
        self._threads = []
        rate = float(fps or self.fps)
        import threading

        backend = _capture_backend()
        for name, index in zip(self.names, self.indices):
            cap = cv2.VideoCapture(index, backend)
            entry: dict[str, Any] = {"index": index, "opened": bool(cap.isOpened())}
            if not cap.isOpened():
                status["cameras"][name] = entry
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, rate)
            path = episode_dir / "videos" / f"{name}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     rate, (self.width, self.height))
            csv_file = (episode_dir / "videos" / f"{name}_timestamps.csv").open(
                "w", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["frame", "t_mono", "t_wall"])
            self._captures[name] = cap
            self._writers[name] = writer
            self._files[name] = (csv_file, csv_writer)
            self._counts[name] = 0
            entry.update({"opened": True, "video": str(path)})
            status["cameras"][name] = entry

            def _loop(cam=cap, name_=name, wr=writer, csv_wr=csv_writer) -> None:
                while not self._stop:
                    ok, frame = cam.read()
                    if not ok or frame is None:
                        time.sleep(0.005)
                        continue
                    now = time.monotonic()
                    wr.write(frame)
                    csv_wr.writerow([self._counts[name_], f"{now:.6f}", f"{time.time():.6f}"])
                    self._counts[name_] += 1
                    self._latest_ts[name_] = now

            thread = threading.Thread(target=_loop, name=f"cam-{name}", daemon=True)
            thread.start()
            self._threads.append(thread)
        status["ok"] = all(c.get("opened") for c in status["cameras"].values()) \
            and bool(status["cameras"])
        return status

    def stop(self) -> dict[str, Any]:
        self._stop = True
        for thread in self._threads:
            thread.join(timeout=1.0)
        for name in list(self._writers):
            try:
                self._writers[name].release()
            except Exception:  # noqa: BLE001
                pass
            try:
                self._files[name][0].close()
            except Exception:  # noqa: BLE001
                pass
        for name in list(self._captures):
            try:
                self._captures[name].release()
            except Exception:  # noqa: BLE001
                pass
        self._captures.clear()
        self._writers.clear()
        self._files.clear()
        self._threads = []
        return {"frames": dict(self._counts), "latest_ts": dict(self._latest_ts)}

    def latest(self) -> dict[str, Any]:
        """当前各相机最新帧索引/时间戳（用于 episode CSV 同步列）。"""
        return {name: {"frame": self._counts.get(name, 0),
                       "t_mono": self._latest_ts.get(name, 0.0)} for name in self.names}


class CamcapCameraSet:
    """调用原生 `tools/camcap.app` 采集双相机（macOS 上用 AVFoundation 编码 H.264 mp4）。

    与 :class:`CameraSet` 接口一致（start/stop/latest/probe），供 Collector 使用。
    """

    def __init__(self, app_path: str | Path = "~/Applications/camcap.app",
                 width: int = 1280, height: int = 720, fps: float = 30.0,
                 global_name: str = "global",
                 wrist_name: str = "wrist",
                 start_timeout_s: float = 45.0) -> None:
        self.app_path = Path(app_path).expanduser()
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.global_name = global_name
        self.wrist_name = wrist_name
        self.start_timeout_s = float(start_timeout_s)
        self.available = self.app_path.is_dir()
        self.error: str | None = None if self.available else f"未找到 {self.app_path}"
        self._video_dir: Path | None = None
        self._status: dict[str, Any] = {}

    def _run(self, extra: list[str], target_dir: Path) -> None:
        import subprocess
        cmd = ["open", "-n", str(self.app_path), "--args", "--dir", str(target_dir),
               "--width", str(self.width), "--height", str(self.height), "--fps", f"{self.fps:g}",
               "--global-name", self.global_name, "--wrist-name", self.wrist_name, *extra]
        subprocess.run(cmd, check=False, timeout=30)

    def probe(self) -> dict[str, Any]:
        """不启动录制，列出相机设备（--list-formats 不需要相机权限）。"""
        import json
        import tempfile
        info: dict[str, Any] = {"backend": "camcap", "app": str(self.app_path),
                                "available": self.available}
        if not self.available:
            return info
        with tempfile.TemporaryDirectory() as tmp:
            self._run(["--list-formats"], Path(tmp))
            formats = Path(tmp) / "formats.json"
            deadline = time.time() + 15
            while time.time() < deadline and not formats.is_file():
                time.sleep(0.2)
            if formats.is_file():
                try:
                    payload = json.loads(formats.read_text(encoding="utf-8"))
                    info["permission"] = payload.get("permission")
                    info["devices"] = [d.get("name") for d in payload.get("devices", [])]
                except (OSError, ValueError) as exc:
                    info["error"] = str(exc)
            else:
                info["error"] = "formats.json 未生成"
        return info

    def start(self, episode_dir: Path, fps: float | None = None) -> dict[str, Any]:
        import json
        video_dir = Path(episode_dir) / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        self._video_dir = video_dir
        status: dict[str, Any] = {"ok": False, "backend": "camcap", "app": str(self.app_path),
                                  "dir": str(video_dir)}
        if not self.available:
            status["error"] = self.error
            return status
        try:
            self._run([], video_dir)
        except Exception as exc:  # noqa: BLE001
            status["error"] = f"启动失败：{exc}"
            return status
        status_file = video_dir / "status.json"
        deadline = time.time() + self.start_timeout_s
        payload: dict[str, Any] = {}
        while time.time() < deadline:
            if status_file.is_file():
                try:
                    payload = json.loads(status_file.read_text(encoding="utf-8"))
                    if "streams" in payload or "error" in payload:
                        break
                except (OSError, ValueError):
                    pass
            time.sleep(0.2)
        self._status = payload
        status["ok"] = bool(payload.get("ok"))
        status["cameras"] = payload.get("streams", [])
        if not status["ok"]:
            status["error"] = payload.get("error", "相机未就绪（首次使用需在系统弹窗允许 camcap 访问相机）")
        return status

    def latest(self) -> dict[str, Any]:
        import json
        if self._video_dir is None:
            return {}
        progress = self._video_dir / "progress.json"
        if not progress.is_file():
            return {}
        try:
            data = json.loads(progress.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        result = {}
        for name in ("global", "wrist"):
            entry = data.get(name) or {}
            result[name] = {"frame": int(entry.get("frames", 0)), "t_mono": float(entry.get("last", 0.0))}
        return result

    def stop(self) -> dict[str, Any]:
        import json
        if self._video_dir is None:
            return {"backend": "camcap", "frames": {}}
        (self._video_dir / "stop").touch()
        status_file = self._video_dir / "status.json"
        deadline = time.time() + 20
        payload: dict[str, Any] = {}
        while time.time() < deadline:
            try:
                payload = json.loads(status_file.read_text(encoding="utf-8"))
                if "stoppedAt" in payload:
                    break
            except (OSError, ValueError):
                pass
            time.sleep(0.2)
        frames = {k: int(v.get("frames", 0)) for k, v in (payload or {}).items()
                  if isinstance(v, dict) and "frames" in v}
        self._video_dir = None
        return {"backend": "camcap", "frames": frames, "ok": bool(payload.get("ok"))}
