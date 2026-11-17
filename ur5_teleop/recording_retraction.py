"""Retract one saved raw recording and its derived episodes under the writer lock."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
import uuid
from pathlib import Path

from .lerobot_recording import (
    dataset_lock,
    dataset_state,
    json_lines,
    read_job,
    read_json,
    write_batch_statistics,
    write_json,
)


def publish_generation(dataset, generation):
    dataset = Path(dataset)
    if generation is None:
        if dataset.is_symlink():
            dataset.unlink()
        return
    link = dataset.with_name("." + dataset.name + "." + uuid.uuid4().hex + ".link")
    try:
        link.symlink_to(
            os.path.relpath(generation, dataset.parent.resolve()), target_is_directory=True
        )
        link.replace(dataset)
    finally:
        if link.is_symlink():
            link.unlink()


def finish_withdrawal(path):
    record = read_json(path)
    dataset = Path(record["dataset"])
    current = str(dataset.resolve()) if dataset.is_symlink() else None
    if dataset.exists() and not dataset.is_symlink():
        raise ValueError(f"Expected a managed dataset link: {dataset}")
    if current not in {record["previous_generation"], record["next_generation"]}:
        raise RuntimeError("Dataset generation changed during withdrawal recovery")
    write_json(record["job_path"], record["discarded_job"])
    publish_generation(dataset, record["next_generation"])
    source, archived = Path(record["source"]), Path(record["archived_source"])
    archived.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        if archived.exists():
            raise FileExistsError(
                f"Both original and discarded recording exist: {source}"
            )
        source.rename(archived)
    elif not archived.is_dir():
        raise FileNotFoundError(
            f"Original and discarded recording are missing: {source}"
        )
    record.update(status="complete", completed_at=time.time())
    write_json(path, record)
    return record


def recover_withdrawals(dataset):
    """Caller holds dataset_lock. Prepared operations finish before another append."""
    completed = []
    for path in sorted((dataset_state(dataset) / "withdrawals").glob("*.json")):
        record = read_json(path)
        if record["status"] == "prepared":
            completed.append(finish_withdrawal(path))
    return completed


def generation_without_source(current, source_name, generations):
    """Reuse an exact previous generation, or rebuild all remaining episodes."""
    receipts = read_json(current / "meta/recording_sources.json")
    removed = set(receipts[source_name]["episodes"])
    remaining_receipts = {
        name: row for name, row in receipts.items() if name != source_name
    }
    if not remaining_receipts:
        return None, sorted(removed), 0
    for candidate in sorted(generations.iterdir()):
        metadata = candidate / "meta/recording_sources.json"
        if (
            candidate != current
            and metadata.is_file()
            and not candidate.name.startswith(".")
        ):
            if read_json(metadata) == remaining_receipts:
                info = read_json(candidate / "meta/info.json")
                return candidate.resolve(), sorted(removed), info["total_episodes"]

    import numpy as np
    import pandas as pd

    info = read_json(current / "meta/info.json")
    episodes = [
        row
        for row in json_lines(current / "meta/episodes.jsonl")
        if row["episode_index"] not in removed
    ]
    mapping = {row["episode_index"]: index for index, row in enumerate(episodes)}
    tasks_used = {task for row in episodes for task in row["tasks"]}
    old_tasks = [
        row
        for row in json_lines(current / "meta/tasks.jsonl")
        if row["task"] in tasks_used
    ]
    task_mapping = {row["task_index"]: index for index, row in enumerate(old_tasks)}
    tasks = [{**row, "task_index": index} for index, row in enumerate(old_tasks)]
    with tempfile.TemporaryDirectory(
        prefix=".building-withdrawal-", dir=generations
    ) as temporary:
        build = Path(temporary) / "dataset"
        (build / "meta").mkdir(parents=True)
        offset = 0
        for old_index, new_index in mapping.items():
            old_chunk, chunk = (
                old_index // info["chunks_size"],
                new_index // info["chunks_size"],
            )
            table = pd.read_parquet(
                current
                / info["data_path"].format(
                    episode_chunk=old_chunk, episode_index=old_index
                )
            )
            table["episode_index"] = np.full(len(table), new_index, dtype=np.int64)
            table["index"] = np.arange(offset, offset + len(table), dtype=np.int64)
            table["task_index"] = table["task_index"].map(task_mapping).astype(np.int64)
            target = build / info["data_path"].format(
                episode_chunk=chunk, episode_index=new_index
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(target, index=False)
            for camera in ["global", "wrist"]:
                video_key = f"observation.images.{camera}"
                source = current / info["video_path"].format(
                    episode_chunk=old_chunk,
                    episode_index=old_index,
                    video_key=video_key,
                )
                target = build / info["video_path"].format(
                    episode_chunk=chunk, episode_index=new_index, video_key=video_key
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, target)
            offset += len(table)
        for filename, rows in [
            (
                "episodes.jsonl",
                [
                    {**row, "episode_index": mapping[row["episode_index"]]}
                    for row in episodes
                ],
            ),
            ("tasks.jsonl", tasks),
        ]:
            (build / "meta" / filename).write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            )
        for name in ["cut_info.json", "segment_tags.json"]:
            rows = [
                {**row, "episode_index": mapping[row["episode_index"]]}
                for row in read_json(current / "meta" / name)
                if row["episode_index"] in mapping
            ]
            write_json(build / "meta" / name, rows)
        alignment = read_json(current / "meta/alignment.json")
        alignment["sources"] = [
            row for row in alignment["sources"] if row["source"] != source_name
        ]
        write_json(build / "meta/alignment.json", alignment)
        receipts = {
            name: {**row, "episodes": [mapping[index] for index in row["episodes"]]}
            for name, row in remaining_receipts.items()
        }
        write_json(build / "meta/recording_sources.json", receipts)
        count = len(episodes)
        info.update(
            total_episodes=count,
            total_frames=offset,
            total_tasks=len(tasks),
            total_videos=2 * count,
            total_chunks=math.ceil(count / info["chunks_size"]),
            splits={"train": f"0:{count}"},
        )
        write_json(build / "meta/info.json", info)
        write_batch_statistics(build)
        generation = generations / f"{count:08d}-{uuid.uuid4().hex}"
        build.replace(generation)
    return generation.resolve(), sorted(removed), count


def discard_last_recording(raw_root, dataset):
    raw_root, dataset = Path(raw_root).resolve(), Path(dataset).absolute()
    with dataset_lock(dataset) as state:
        recovered = recover_withdrawals(dataset)
        if recovered:
            return recovered[-1]
        with (state / "jobs.lock").open("a") as stream:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX)
            candidates = list(raw_root.glob("*/meta.json"))
            if not candidates:
                raise ValueError(f"没有可撤回的已保存录制：{raw_root}")
            metadata_path = max(
                candidates, key=lambda path: (path.stat().st_mtime_ns, path.parent.name)
            )
            source = metadata_path.parent
            archived = raw_root / "_discarded" / source.name
            if archived.exists():
                raise FileExistsError(f"废弃目录已存在：{archived}")
            metadata = read_json(metadata_path)
            requested_dataset = metadata.get("lerobot_export", {}).get("dataset")
            if requested_dataset and Path(requested_dataset).absolute() != dataset:
                raise ValueError("原始录制的目标数据集与当前配置不一致")
            generations = state / "generations"
            previous = dataset.resolve(strict=True) if dataset.is_symlink() else None
            if dataset.exists() and previous is None:
                raise ValueError("数据集必须使用采集器管理的版本链接")
            if previous and not previous.is_relative_to(generations.resolve()):
                raise ValueError("数据集链接位于采集器版本目录之外")
            next_generation, removed, remaining = previous, [], 0
            if previous:
                receipts = read_json(previous / "meta/recording_sources.json")
                remaining = read_json(previous / "meta/info.json")["total_episodes"]
                if source.name in receipts:
                    next_generation, removed, remaining = generation_without_source(
                        previous, source.name, generations
                    )
            identifier = hashlib.sha256(str(source).encode()).hexdigest()[:24]
            job_path = state / "jobs" / (identifier + ".json")
            job = (
                read_job(job_path)
                if job_path.exists()
                else {
                    "source": str(source),
                    "dataset": str(dataset),
                    "metadata_sha256": hashlib.sha256(
                        metadata_path.read_bytes()
                    ).hexdigest(),
                }
            )
            job.update(
                source=str(source),
                dataset=str(dataset),
                status="discarded",
                discarded_at=time.time(),
                archived_source=str(archived),
            )
            record = {
                "status": "prepared",
                "source": str(source),
                "archived_source": str(archived),
                "dataset": str(dataset),
                "previous_generation": str(previous) if previous else None,
                "next_generation": str(next_generation) if next_generation else None,
                "job_path": str(job_path),
                "discarded_job": job,
                "removed_episodes": removed,
                "remaining_episodes": remaining,
                "created_at": time.time(),
            }
            path = state / "withdrawals" / (uuid.uuid4().hex + ".json")
            write_json(path, record)
            return finish_withdrawal(path)
