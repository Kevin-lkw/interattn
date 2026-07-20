"""Run a resumable block-size/epsilon sweep on up to four exclusive GPUs."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..run import LONGBENCH_DATASETS
from ..run_all import LONGBENCH_EN_CODE_DATASETS


DEFAULT_OUTPUT_ROOT = Path(
    "llama/result/generate/condition_block_efficient_full_sweep"
)


@dataclass(frozen=True)
class SweepJob:
    block_size: int
    eps: float

    @property
    def name(self):
        return f"block={self.block_size}_eps={self.eps:g}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 2, 4, 5])
    parser.add_argument("--blocks", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--eps", nargs="+", type=float, default=[0.05, 0.1, 0.25, 0.5])
    parser.add_argument("--datasets", nargs="+", default=LONGBENCH_EN_CODE_DATASETS)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    return parser.parse_args()


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write_manifest_locked(path, manifest):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _update_manifest(path, manifest, lock, job_name=None, record=None):
    with lock:
        if job_name is not None:
            manifest["jobs"][job_name] = record
        _write_manifest_locked(path, manifest)


def _command(args, job):
    command = [
        sys.executable,
        "-m",
        "llama.script.analysis.condition_block_gen.longbench.run_all",
        "--model",
        args.model,
        "--device",
        "cuda:0",
        "--method",
        "condition_block_triton",
        "--condition-block-size",
        str(job.block_size),
        "--condition-eps",
        str(job.eps),
        "--datasets",
        *args.datasets,
        "--output-root",
        str(args.output_root),
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.max_new_tokens is not None:
        command.extend(("--max-new-tokens", str(args.max_new_tokens)))
    return command


def _efficient_environment(gpu):
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CONDITION_BLOCK_MIXED_SUMMARIES": "1",
            "CONDITION_BLOCK_K_BAR_DTYPE": "bfloat16",
            "CONDITION_BLOCK_POST_PREFILL_STATIC_CACHE": "1",
            "CONDITION_BLOCK_SKIP_STATS": "1",
        }
    )
    # Full LongBench contains many early-EOS tasks. CUDA graph would force all
    # of them to execute max_new_tokens, so eager decode is faster here.
    for name in (
        "CONDITION_BLOCK_CUDA_GRAPH",
        "CONDITION_BLOCK_TMA_BOUNDS",
        "CONDITION_BLOCK_DENSE_STAGE2",
        "CONDITION_BLOCK_COMPACT_SDPA_STAGE2",
        "CONDITION_BLOCK_LEGACY_STAGE2",
        "CONDITION_BLOCK_EAGER_SELECTION",
    ):
        environment.pop(name, None)
    return environment


def _worker(gpu, jobs, args, manifest, manifest_path, lock, failures):
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        log_path = args.log_dir / f"{job.name}.log"
        command = _command(args, job)
        running_record = {
            "block_size": job.block_size,
            "eps": job.eps,
            "gpu": gpu,
            "status": "running",
            "started_at": _utc_now(),
            "log": str(log_path),
        }
        _update_manifest(manifest_path, manifest, lock, job.name, running_record)
        print(f"START {job.name} on physical GPU {gpu}: {log_path}", flush=True)
        try:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n[{_utc_now()}] START gpu={gpu} command={command!r}\n")
                handle.flush()
                result = subprocess.run(
                    command,
                    env=_efficient_environment(gpu),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                handle.write(f"\n[{_utc_now()}] END returncode={result.returncode}\n")
            final_record = {
                **running_record,
                "status": "complete" if result.returncode == 0 else "failed",
                "returncode": int(result.returncode),
                "finished_at": _utc_now(),
            }
            _update_manifest(manifest_path, manifest, lock, job.name, final_record)
            if result.returncode != 0:
                with lock:
                    failures.append(job.name)
            print(
                f"END {job.name} on physical GPU {gpu}: returncode={result.returncode}",
                flush=True,
            )
        finally:
            jobs.task_done()


def main():
    args = parse_args()
    if not 1 <= len(args.gpus) <= 4:
        raise ValueError("--gpus must contain between one and four GPU IDs")
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("--gpus must not contain duplicates")
    unknown = sorted(set(args.datasets) - set(LONGBENCH_DATASETS))
    if unknown:
        raise ValueError(f"Unknown LongBench datasets: {unknown}")

    args.output_root = args.output_root.resolve()
    args.log_dir = (
        args.log_dir.resolve()
        if args.log_dir is not None
        else args.output_root / "sweep_logs"
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "sweep_manifest.json"

    job_queue = queue.Queue()
    # Start the denser/slower eps values first so dynamic GPU scheduling keeps
    # the tail short.
    for eps in sorted(set(args.eps)):
        for block_size in sorted(set(args.blocks)):
            job_queue.put(SweepJob(block_size=block_size, eps=eps))

    manifest = {
        "created_at": _utc_now(),
        "output_root": str(args.output_root),
        "model": args.model,
        "gpus": args.gpus,
        "blocks": sorted(set(args.blocks)),
        "eps": sorted(set(args.eps)),
        "datasets": args.datasets,
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "runtime": {
            "mixed_summaries": True,
            "k_bar_dtype": "bfloat16",
            "post_prefill_static_cache": True,
            "cuda_graph": False,
            "collect_stats": False,
            "tma_bounds": False,
        },
        "jobs": {},
    }
    lock = threading.Lock()
    failures = []
    _update_manifest(manifest_path, manifest, lock)
    workers = [
        threading.Thread(
            target=_worker,
            args=(gpu, job_queue, args, manifest, manifest_path, lock, failures),
            daemon=False,
        )
        for gpu in args.gpus
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if failures:
        raise RuntimeError(f"Sweep jobs failed: {sorted(failures)}")


if __name__ == "__main__":
    main()
