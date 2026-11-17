"""双相机彩色采集（Intel RealSense D435i 的 UVC 彩色流；macOS 用 AVFoundation）。

macOS 上 librealsense 无法 claim 深度接口，但两台相机都作为 UVC 摄像头可见，
SOP 需要的是 RGB 图像（observation.images.global/wrist），因此这里用 OpenCV 采集彩色流。
无 cv2 或相机不可用时自动降级为"无相机"模式（记录 camera_status，不阻断遥操作）。
"""

from __future__ import annotations

import csv
import math
import platform
import threading
import time
from pathlib import Path
from typing import Any

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
    """Capture continuously, attaching per-episode video writers on demand."""

    def __init__(
        self,
        indices: tuple[Any, ...] = (0, 1),
        names: tuple[str, ...] = ("global", "wrist"),
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        persistent: bool = False,
    ) -> None:
        self.indices, self.names = indices, names
        self.width, self.height, self.fps = width, height, float(fps)
        self.persistent = persistent
        self._stream_fps = self.fps
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._captures: dict[str, Any] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._sinks: dict[str, Any] = {}
        self._errors: dict[str, str] = {}
        self._recording_errors: dict[str, str] = {}
        self._warmup: dict[str, Any] = {}
        self._last_seen: dict[str, float] = {}
        self._shapes: dict[str, Any] = {}
        self._reset_recording_frames()
        self._opened = False
        self._recording = False
        self._recording_started_at: float | None = None
        self.available = cv2 is not None
        self.error = None if self.available else "cv2 未安装"

    def _reset_recording_frames(self) -> None:
        self._first_ts = {name: None for name in self.names}
        self._latest_frames = {
            name: {"frame": -1, "t_mono": 0.0} for name in self.names
        }

    def _status(self) -> dict[str, Any]:
        with self._lock:
            cameras = {
                name: {
                    "index": index,
                    "opened": name in self._threads and self._threads[name].is_alive(),
                    "ready": bool(name in self._warmup and self._warmup[name].ready),
                    "last_frame_at": self._last_seen.get(name),
                    "read": name in self._last_seen,
                    "fps": self._stream_fps,
                    "resolution": list(self._shapes[name][1::-1])
                    if name in self._shapes
                    else None,
                }
                for name, index in zip(self.names, self.indices)
            }
            return {
                "ok": bool(cameras)
                and all(row["opened"] for row in cameras.values())
                and not self._errors,
                "ready": bool(cameras)
                and all(row["opened"] and row["ready"] for row in cameras.values())
                and not self._errors,
                "persistent": self.persistent,
                "opencv": cv2 is not None,
                "cameras": cameras,
                "errors": dict(self._errors),
            }

    def probe(self) -> dict[str, Any]:
        if self._opened:
            return self._status()
        info: dict[str, Any] = {"opencv": cv2 is not None, "cameras": {}}
        if cv2 is None:
            return info
        for name, index in zip(self.names, self.indices):
            capture = cv2.VideoCapture(index, _capture_backend())
            try:
                entry = {"index": index, "opened": bool(capture.isOpened())}
                if capture.isOpened():
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    capture.set(cv2.CAP_PROP_FPS, self.fps)
                    ok, frame = capture.read()
                    entry["read"] = bool(ok)
                    if ok and frame is not None:
                        entry["resolution"] = [int(frame.shape[1]), int(frame.shape[0])]
                    entry["fps"] = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                info["cameras"][name] = entry
            finally:
                capture.release()
        return info

    def open(self, fps: float | None = None) -> dict[str, Any]:
        """Open devices once and start draining their streams, including while idle."""
        rate = self.fps if fps is None else float(fps)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("Camera FPS must be positive and finite")
        if self._opened:
            if abs(rate - self._stream_fps) > 1e-6:
                raise ValueError("Recording FPS must match the opened camera stream")
            return self._status()
        self._stream_fps = rate
        if any(thread.is_alive() for thread in self._threads.values()):
            raise RuntimeError("Previous camera capture thread has not stopped")
        self._errors, self._warmup, self._last_seen, self._shapes = {}, {}, {}, {}
        if cv2 is None:
            self._errors = {name: "cv2 未安装" for name in self.names}
            return self._status()
        from .video_quality import FrameReadiness

        self._stop.clear()
        self._opened = True
        for name, index in zip(self.names, self.indices):
            capture = cv2.VideoCapture(index, _capture_backend())
            if not capture.isOpened():
                self._errors[name] = "Camera could not be opened"
                capture.release()
                continue
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            capture.set(cv2.CAP_PROP_FPS, self._stream_fps)
            self._captures[name] = capture
            self._warmup[name] = FrameReadiness(time.monotonic())
            thread = threading.Thread(
                target=self._capture,
                args=(name, capture),
                name=f"cam-{name}",
                daemon=True,
            )
            self._threads[name] = thread
            thread.start()
        return self._status()

    def wait_ready(self, timeout_s: float = 30.0) -> dict[str, Any]:
        """Wait during service startup, before the teleoperation loop starts."""
        deadline = time.monotonic() + timeout_s
        while True:
            status = self._status()
            if status["ready"] or status["errors"]:
                return status
            if time.monotonic() >= deadline:
                with self._lock:
                    for name in self.names:
                        if name not in self._warmup or not self._warmup[name].ready:
                            self._errors[name] = (
                                "Camera startup did not settle before the readiness timeout"
                            )
                return self._status()
            time.sleep(0.02)

    def _capture(self, name, capture) -> None:
        gate = self._warmup[name]
        last_frame = time.monotonic()
        try:
            while not self._stop.is_set():
                ok, frame = capture.read()
                now, wall = time.monotonic(), time.time()
                if self._stop.is_set():
                    break
                if not ok or frame is None:
                    if now - last_frame > 2:
                        raise RuntimeError(
                            "Camera has not delivered frames for 2 seconds"
                        )
                    time.sleep(0.005)
                    continue
                last_frame = now
                if frame.shape[:2] != (self.height, self.width):
                    raise ValueError(f"Unexpected capture resolution {frame.shape}")
                with self._lock:
                    self._last_seen[name] = now
                    self._shapes[name] = frame.shape
                if not gate.ready:
                    if not gate.accept(frame, now):
                        if now - gate.started_at > 30:
                            raise RuntimeError(
                                "Camera exposure/colour did not settle within 30 seconds"
                            )
                        continue
                    print(f"[camera] {name} 视频流已稳定", flush=True)
                with self._lock:
                    if not self._recording or now < self._recording_started_at:
                        continue
                    sink = self._sinks.get(name)
                    if sink is None:
                        continue
                    writer, _, stamps = sink
                    writer.write(frame)
                    frame_index = self._latest_frames[name]["frame"] + 1
                    stamps.writerow([frame_index, f"{now:.6f}", f"{wall:.6f}"])
                    if self._first_ts[name] is None:
                        self._first_ts[name] = now
                    # Publish a complete frame/time pair in one assignment. The
                    # control loop reads this snapshot without waiting for encoding.
                    self._latest_frames = {
                        **self._latest_frames,
                        name: {"frame": frame_index, "t_mono": now},
                    }
        except (RuntimeError, ValueError, OSError, csv.Error, cv2.error) as error:
            with self._lock:
                self._errors[name] = str(error)
            print(f"[camera] {name} 采集失败：{error}", flush=True)
        finally:
            capture.release()

    def start(self, episode_dir: Path, fps: float | None = None) -> dict[str, Any]:
        rate = self.fps if fps is None else float(fps)
        self.open(fps=rate)
        with self._lock:
            if self._recording:
                raise RuntimeError("Camera recording is already active")
            self._reset_recording_frames()
            self._recording_errors = {}
            videos = Path(episode_dir) / "videos"
            videos.mkdir(parents=True, exist_ok=True)
            try:
                for name in self.names:
                    if name not in self._captures or name in self._errors:
                        continue
                    path = videos / f"{name}.mp4"
                    writer = cv2.VideoWriter(
                        str(path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        rate,
                        (self.width, self.height),
                    )
                    if not writer.isOpened():
                        writer.release()
                        self._recording_errors[name] = (
                            "Video writer could not be opened"
                        )
                        continue
                    try:
                        stream = (videos / f"{name}_timestamps.csv").open(
                            "w", newline="", encoding="utf-8"
                        )
                    except OSError:
                        writer.release()
                        raise
                    stamps = csv.writer(stream)
                    stamps.writerow(["frame", "t_mono", "t_wall"])
                    self._sinks[name] = (writer, stream, stamps)
                self._recording_started_at = time.monotonic()
                self._recording = True
            except (OSError, cv2.error):
                self._finish_recording()
                raise
            status = self._status()
            status["ok"] = status["ok"] and not self._recording_errors
            for name in self._sinks:
                status["cameras"][name]["video"] = str(videos / f"{name}.mp4")
            status["recording_started_at"] = self._recording_started_at
            return status

    def _finish_recording(self) -> dict[str, Any]:
        with self._lock:
            self._recording = False
            for name, (writer, stream, _) in self._sinks.items():
                try:
                    writer.release()
                except (OSError, cv2.error) as error:
                    self._recording_errors[name] = str(error)
                try:
                    stream.close()
                except OSError as error:
                    self._recording_errors[name] = str(error)
            self._sinks.clear()
            errors = {**self._errors, **self._recording_errors}
            for name in self.names:
                if self._latest_frames[name]["frame"] < 0:
                    errors.setdefault(name, "No stable camera frames were recorded")
            return {
                "frames": {name: row["frame"] + 1 for name, row in self._latest_frames.items()},
                "latest_ts": {name: row["t_mono"] for name, row in self._latest_frames.items()},
                "errors": errors,
                "persistent": self.persistent,
                "recording_started_at": self._recording_started_at,
                "first_frame_delay_s": {
                    name: stamp - self._recording_started_at
                    if stamp is not None
                    else None
                    for name, stamp in self._first_ts.items()
                },
                "warmup": {
                    name: {"ready": gate.ready, "discarded_frames": gate.discarded}
                    for name, gate in self._warmup.items()
                },
            }

    def stop(self) -> dict[str, Any]:
        stats = self._finish_recording()
        if not self.persistent:
            self.close()
        return stats

    def close(self) -> None:
        self._finish_recording()
        self._stop.set()
        for name, thread in self._threads.items():
            thread.join(timeout=3.0)
            if thread.is_alive():
                raise RuntimeError(f"Camera capture thread did not stop: {name}")
        self._captures.clear()
        self._threads.clear()
        self._opened = False

    def latest(self) -> dict[str, Any]:
        # Each published snapshot is immutable, so video I/O never blocks callers.
        return {name: dict(row) for name, row in self._latest_frames.items()}


class CamcapCameraSet:
    """调用原生 `tools/camcap.app` 采集双相机（macOS 上用 AVFoundation 编码 H.264 mp4）。

    与 :class:`CameraSet` 接口一致（start/stop/latest/probe），供 Collector 使用。
    """

    def __init__(self, app_path: str | Path = "~/Applications/camcap.app",
                 width: int = 1280, height: int = 720, fps: float = 30.0,
                 global_name: str = "435 with RGB Module",
                 wrist_name: str = "435i",
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
