#!/usr/bin/env python3
"""快速清理失败 episode 的小工具（纯标准库，可在 AGX 的任意 python3 下运行）。

数据目录默认：<仓库根>/data/episodes
（也可 --root 指定；目录内 *_epNNN 为主数据集，_discarded/ 为已弃用，_extra/ 为额外样本）

常用：
  python3 tools/episode_manager.py list              # 列出最近 15 条（带健康状态）
  python3 tools/episode_manager.py list --all        # 列出全部
  python3 tools/episode_manager.py broken            # 只列异常/不完整 episode
  python3 tools/episode_manager.py discard --last 1  # 把最新一条移到 _discarded/（默认不硬删）
  python3 tools/episode_manager.py discard 1 3       # 删掉 list 里的第 1、3 条
  python3 tools/episode_manager.py discard --broken  # 把所有异常 episode 移到 _discarded/
  python3 tools/episode_manager.py discard --last 2 --hard --yes   # 永久删除最新 2 条
  python3 tools/episode_manager.py restore --last 1  # 从 _discarded 恢复最新一条
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "episodes"


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def main_episodes(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    eps = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith("_")]
    return sorted(eps, key=lambda p: p.stat().st_mtime, reverse=True)


def inspect(ep: Path) -> dict:
    """读取 meta/robot.csv，给出健康状态。"""
    info = {"name": ep.name, "mtime": ep.stat().st_mtime, "size": 0, "duration": None,
            "rows": None, "success": None, "windows": 0, "issues": []}
    for f in ep.rglob("*"):
        if f.is_file():
            info["size"] += f.stat().st_size
    meta_path = ep / "meta.json"
    if not meta_path.exists():
        info["issues"].append("缺 meta.json")
    else:
        try:
            meta = json.loads(meta_path.read_text())
            info["duration"] = meta.get("duration_s")
            info["rows"] = meta.get("rows")
            info["success"] = meta.get("success")
            info["windows"] = len(meta.get("disturbance_windows") or [])
        except Exception as exc:  # noqa: BLE001
            info["issues"].append(f"meta.json 解析失败: {exc}")
    robot_csv = ep / "robot.csv"
    if not robot_csv.exists():
        info["issues"].append("缺 robot.csv")
    else:
        try:
            with robot_csv.open(newline="", encoding="utf-8") as fh:
                info["rows"] = sum(1 for _ in csv.DictReader(fh))
        except Exception as exc:  # noqa: BLE001
            info["issues"].append(f"robot.csv 读取失败: {exc}")
    videos = ep / "videos"
    for cam in ("global", "wrist"):
        if not (videos / f"{cam}.mp4").exists():
            info["issues"].append(f"缺 videos/{cam}.mp4")
    if info["duration"] is not None and info["duration"] < 3.0:
        info["issues"].append(f"时长过短({info['duration']:.1f}s)")
    if info["rows"] is not None and info["rows"] < 50:
        info["issues"].append(f"行数过少({info['rows']})")
    if info["success"] is False:
        info["issues"].append("success=false")
    return info


def fmt_row(idx: int, info: dict) -> str:
    dur = f"{info['duration']:5.1f}s" if info["duration"] is not None else "  -  "
    rows = f"{info['rows']:5d}" if info["rows"] is not None else "  -  "
    flag = "OK" if not info["issues"] else "!! " + "; ".join(info["issues"])
    return (f"[{idx:2d}] {info['name']:<24s} {dur} {rows}行 "
            f"{human_size(info['size']):>9s} win={info['windows']}  {flag}")


def print_list(root: Path, show_all: bool, limit: int) -> list[Path]:
    eps = main_episodes(root)
    if not eps:
        print(f"（{root} 下没有主数据集 episode）")
        return []
    shown = eps if show_all else eps[:limit]
    print(f"数据目录: {root}")
    print(f"主数据集共 {len(eps)} 条，显示最近 {len(shown)} 条（按时间倒序）：")
    for i, ep in enumerate(shown, 1):
        print(fmt_row(i, inspect(ep)))
    return shown


def unique_target(dest_dir: Path, name: str) -> Path:
    dest = dest_dir / name
    n = 2
    while dest.exists():
        dest = dest_dir / f"{name}_dup{n}"
        n += 1
    return dest


def select(root: Path, args) -> list[Path]:
    chosen: list[Path] = []
    if args.broken:
        chosen.extend(ep for ep in main_episodes(root) if inspect(ep)["issues"])
    if args.last:
        chosen.extend(main_episodes(root)[: args.last])
    if args.names:
        for name in args.names:
            p = root / name
            chosen.append(p if p.is_dir() else None)  # type: ignore[arg-type]
    if args.indices:
        shown = main_episodes(root)[: max(args.indices)]
        for i in args.indices:
            if 1 <= i <= len(shown):
                chosen.append(shown[i - 1])
            else:
                print(f"  跳过无效编号 {i}（只有 {len(shown)} 条）")
    if args.all:
        chosen.extend(main_episodes(root))
    # 去重、剔除 None/不存在、排除 _ 目录
    out: list[Path] = []
    seen: set[str] = set()
    for p in chosen:
        if p is None or not p.is_dir() or p.name.startswith("_"):
            continue
        if p.name not in seen:
            seen.add(p.name)
            out.append(p)
    return out


def do_remove(root: Path, eps: list[Path], args) -> int:
    if not eps:
        print("没有匹配到要处理的 episode。")
        return 0
    dest_dir = root / "_discarded"
    if not args.hard:
        dest_dir.mkdir(exist_ok=True)
    print(f"{'将永久删除' if args.hard else '将移到 _discarded'} {len(eps)} 条：")
    for ep in eps:
        print(f"   - {ep.name}  ({human_size(inspect(ep)['size'])})")
    if args.dry_run:
        print("（--dry-run：未实际执行）")
        return 0
    if args.hard and not args.yes:
        print("永久删除需要加 --yes 确认，已取消。")
        return 1
    for ep in eps:
        if args.hard:
            shutil.rmtree(ep)
        else:
            target = unique_target(dest_dir, ep.name)
            shutil.move(str(ep), str(target))
    print(f"完成。主数据集剩余 {len(main_episodes(root))} 条。")
    return 0


def do_restore(root: Path, args) -> int:
    src = root / "_discarded"
    if not src.is_dir():
        print("没有 _discarded 目录。")
        return 1
    eps = sorted([d for d in src.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    if args.last:
        chosen = eps[: args.last]
    elif args.names:
        chosen = [src / n for n in args.names if (src / n).is_dir()]
    else:
        print("请用 --last N 或 --names 指定要恢复的 episode（目录名）。")
        return 1
    for ep in chosen:
        target = unique_target(root, ep.name)
        if args.dry_run:
            print(f"[dry-run] 恢复 {ep.name}")
        else:
            shutil.move(str(ep), str(target))
            print(f"已恢复 {ep.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="快速清理失败/异常 episode")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help=f"数据目录（默认 {DEFAULT_ROOT}）")
    sub = ap.add_subparsers(dest="cmd")

    p_list = sub.add_parser("list", help="列出 episode（默认最近 15 条）")
    p_list.add_argument("-n", "--limit", type=int, default=15)
    p_list.add_argument("--all", action="store_true")

    p_broken = sub.add_parser("broken", help="只列异常/不完整 episode")
    p_broken.add_argument("--all", action="store_true")
    p_broken.add_argument("-n", "--limit", type=int, default=15)

    p_disc = sub.add_parser("discard", help="弃用（默认移到 _discarded/）")
    p_disc.add_argument("indices", nargs="*", type=int, help="list 输出里的编号（最近=1）")
    p_disc.add_argument("--last", type=int, default=0, help="处理最近 N 条")
    p_disc.add_argument("--names", nargs="*", help="按目录名指定")
    p_disc.add_argument("--broken", action="store_true", help="处理所有异常 episode")
    p_disc.add_argument("--all", action="store_true", help="处理全部（危险）")
    p_disc.add_argument("--hard", action="store_true", help="永久删除（默认只是移到 _discarded）")
    p_disc.add_argument("--yes", action="store_true", help="硬删确认")
    p_disc.add_argument("--dry-run", action="store_true")

    p_restore = sub.add_parser("restore", help="从 _discarded 恢复")
    p_restore.add_argument("--last", type=int, default=0)
    p_restore.add_argument("--names", nargs="*")
    p_restore.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()
    root = args.root.expanduser().resolve()

    if args.cmd in (None, "list"):
        print_list(root, getattr(args, "all", False), getattr(args, "limit", 15))
        return 0
    if args.cmd == "broken":
        eps = main_episodes(root)
        broken = [ep for ep in eps if inspect(ep)["issues"]]
        if not broken:
            print("没有发现异常 episode。")
            return 0
        print(f"发现 {len(broken)} 条异常：")
        for i, ep in enumerate(broken[: getattr(args, "limit", 15)], 1):
            print(fmt_row(i, inspect(ep)))
        print("可用: python3 tools/episode_manager.py discard --broken 将它们移到 _discarded/")
        return 0
    if args.cmd == "discard":
        return do_remove(root, select(root, args), args)
    if args.cmd == "restore":
        return do_restore(root, args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
