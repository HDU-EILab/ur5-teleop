"""Withdraw the latest saved recording selected by the normal AGX configuration."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ur5_teleop.config import load  # noqa: E402
from ur5_teleop.recording_retraction import discard_last_recording  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/ur5-orin.toml")
    args = parser.parse_args()
    config = load(args.config).collect
    if config.format != "lerobot":
        raise ValueError('撤回命令需要 collect.format="lerobot"')
    os.nice(10)
    try:
        result = discard_last_recording(ROOT / config.raw_dir, ROOT / config.dir)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"撤回失败：{error}\n")
    print(f"已撤回：{Path(result['source']).name}")
    print(f"原始录制已移入：{result['archived_source']}")
    print(
        json.dumps(
            {
                "removed_episodes": result["removed_episodes"],
                "remaining_episodes": result["remaining_episodes"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
