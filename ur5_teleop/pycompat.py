"""Python 版本兼容：低于 3.11 时自动 re-exec 到本机可用的新解释器。

项目用标准库 ``tomllib`` 读 TOML（Python 3.11+）。macOS 自带 ``/usr/bin/python3`` 仍是 3.9，
用户直接敲 ``python3 tools/xxx.py`` 时会导入失败；这里在导入 config 前先检查版本，
若太旧则在 PATH 和常见安装位置里找一个 ≥3.11 的解释器，用 ``os.execv`` 原地重启自己。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

MIN_VERSION = (3, 11)
_REEXEC_FLAG = "UR5_TELEOP_REEXEC"


def supported_in_place() -> bool:
    return sys.version_info >= MIN_VERSION


def _version_of(executable: str) -> tuple[int, int] | None:
    try:
        result = subprocess.run(
            [executable, "-c", "import sys;print('%d %d' % sys.version_info[:2])"],
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    parts = result.stdout.split()
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _candidates() -> list[str]:
    names = ("python3.13", "python3.12", "python3.11", "python3")
    found: list[str] = []
    for name in names:
        executable = shutil.which(name)
        if executable:
            found.append(executable)
    # PATH 里可能只指向系统旧版本；再检查常见安装目录
    for directory in (os.path.expanduser("~/.local/bin"), "/opt/homebrew/bin",
                      "/usr/local/bin", "/opt/local/bin"):
        for name in names:
            path = os.path.join(directory, name)
            if os.path.isfile(path) and os.access(path, os.X_OK):
                found.append(path)
    unique: list[str] = []
    seen: set[str] = set()
    for path in found:
        real = os.path.realpath(path)
        if real not in seen:
            seen.add(real)
            unique.append(path)
    return unique


def ensure_supported_python() -> None:
    """若当前解释器 <3.11，则尝试 re-exec；找不到可用解释器时给出明确错误。"""
    if supported_in_place():
        return
    if os.environ.get(_REEXEC_FLAG):
        raise RuntimeError(
            f"当前 Python {sys.version.split()[0]} 低于 3.11（缺少 tomllib），"
            "自动切换后仍不满足要求。请显式用 Python ≥3.11 运行，"
            "例如 PYTHON=/usr/bin/python3.11 python3 tools/run_teleop.py")
    script = sys.argv[0] if sys.argv and os.path.isfile(sys.argv[0]) else ""
    if not script:
        raise RuntimeError(
            f"当前 Python {sys.version.split()[0]} 低于 3.11（缺少 tomllib）。"
            "请用 Python ≥3.11 运行本工具（如 ~/.local/bin/python3）。")
    current = os.path.realpath(sys.executable)
    for executable in _candidates():
        if os.path.realpath(executable) == current:
            continue
        version = _version_of(executable)
        if version is None or version < MIN_VERSION:
            continue
        os.environ[_REEXEC_FLAG] = "1"
        os.execv(executable, [executable, *sys.argv])
    raise RuntimeError(
        f"当前 Python {sys.version.split()[0]} 低于 3.11（缺少 tomllib），"
        "且本机没找到 ≥3.11 的解释器。请安装 Python 3.11+ 或用 "
        "PYTHON=/path/to/python3 python3 tools/run_teleop.py 指定。")
