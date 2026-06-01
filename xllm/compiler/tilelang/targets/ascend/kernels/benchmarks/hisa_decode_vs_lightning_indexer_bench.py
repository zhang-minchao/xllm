#!/usr/bin/env python3

"""
Benchmark HISA decode (paged_block_sparse_mqa_attn_return_logits_kernel) baseline.

Uses npu_lightning_indexer with PA_BSND layout as the baseline implementation.
The HISA decode kernel computes sparse logits for paged KV cache indexing.

  - normal mode (--all or --case): run msprof profiling
  - worker mode: run warmup + measure loops without profiling (for msprof)

msprof orchestration: kernel_msprof_task_duration.py with appropriate
  --worker-cmd-template, or run this script directly with --all.
"""

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch_npu


# ── Fixed parameters ──
INDEX_N_HEADS = 32
INDEX_HEAD_DIM = 128
NUM_HEADS_KV = 1
BLOCK_SIZE = 128
SPARSE_COUNT = 64
SPARSE_MODE = 3
DEFAULT_WARMUP = 20
DEFAULT_ITERS = 100


@dataclass
class DecodeCase:
    name: str
    batch_size: int
    kv_lens: List[int]
    note: str


DECODE_CASES = [
    DecodeCase("decode_kv128k_b1", 1, [131072], "batch=1, 128K KV"),
    DecodeCase("decode_kv128k_b2", 2, [131072, 131072], "batch=2, 128K KV"),
    DecodeCase("decode_kv128k_b4", 4, [131072, 131072, 131072, 131072], "batch=4, 128K KV"),
    DecodeCase("decode_kv128k_b8", 8, [131072] * 8, "batch=8, 128K KV"),
    DecodeCase("decode_kv2k_b1", 1, [2048], "batch=1, 2K KV"),
    DecodeCase("decode_kv16k_b1", 1, [16384], "batch=1, 16K KV"),
    DecodeCase("decode_kv64k_b1", 1, [65536], "batch=1, 64K KV"),
    DecodeCase("decode_mixed_b4", 4, [2048, 16384, 65536, 131072], "batch=4, mixed KV"),
]


def build_decode_inputs(case: DecodeCase, device: str):
    B = case.batch_size
    max_kv = max(case.kv_lens)
    max_blocks_per_seq = (max_kv + BLOCK_SIZE - 1) // BLOCK_SIZE

    query = torch.randn(B, INDEX_N_HEADS, INDEX_HEAD_DIM, dtype=torch.float16, device=device)

    total_unique_blocks = max_blocks_per_seq + B
    key = torch.randn(
        total_unique_blocks, BLOCK_SIZE, NUM_HEADS_KV, INDEX_HEAD_DIM,
        dtype=torch.float16, device=device
    )

    block_table = torch.zeros(B, max_blocks_per_seq, dtype=torch.int32, device=device)
    for i in range(max_blocks_per_seq):
        block_table[:, i] = i % total_unique_blocks

    weights = torch.randn(B, INDEX_N_HEADS, dtype=torch.float16, device=device)

    actual_seq_lengths_query = torch.arange(1, B + 1, dtype=torch.int32, device=device)
    actual_seq_lengths_key = torch.tensor(case.kv_lens, dtype=torch.int32, device=device)

    return query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table


def run_decode(query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table):
    return torch_npu.npu_lightning_indexer(
        query, key, weights,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=SPARSE_COUNT,
        sparse_mode=SPARSE_MODE,
    )



def run_worker(args: argparse.Namespace) -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")

    torch.npu.set_device(args.device_id)
    device = f"npu:{args.device_id}"

    cases = DECODE_CASES
    if args.case:
        cases = [c for c in DECODE_CASES if c.name == args.case]
        if not cases:
            raise ValueError(f"Unknown decode case: {args.case}")

    for case in cases:
        query, key, weights, asq, ask, bt = build_decode_inputs(case, device)

        for _ in range(args.warmup):
            run_decode(query, key, weights, asq, ask, bt)
        torch.npu.synchronize()

        for _ in range(args.iters):
            run_decode(query, key, weights, asq, ask, bt)
        torch.npu.synchronize()


# ── msprof orchestration ──

MSPROF_OUTPUT_ROOT = Path("/tmp/msprof_hisa_decode")
WORKER_SCRIPT = Path(__file__).resolve()
KERNEL_PATTERN = "LightningIndexer"


def find_msprof_json(profile_dir):
    matches = sorted(Path(profile_dir).rglob("msprof_*.json"))
    if not matches:
        return None
    return str(matches[-1])


def extract_kernel_duration(json_path, iters):
    with open(json_path) as f:
        events = json.load(f)

    matched = [
        e for e in events
        if e.get("ph") == "X" and e.get("name") == KERNEL_PATTERN
    ]

    if not matched:
        all_x = [e for e in events if e.get("ph") == "X"]
        names = sorted(set(str(e.get("name", "")) for e in all_x))[:20]
        return {"error": "no matching events", "sample_names": names}

    durations = [float(e["dur"]) for e in matched]
    tail = durations[-iters:] if len(durations) >= iters else durations
    tail_sorted = sorted(tail)

    return {
        "total_events": len(durations),
        "measured_events": len(tail),
        "avg_us": sum(tail) / len(tail),
        "median_us": tail_sorted[len(tail_sorted) // 2],
        "min_us": min(tail),
        "max_us": max(tail),
        "p90_us": tail_sorted[int(len(tail_sorted) * 0.9)],
    }


def run_msprof_case(case_name, warmup, iters, device_id, keep_output):
    case_dir = MSPROF_OUTPUT_ROOT / case_name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    worker_cmd = (
        f"{sys.executable} {WORKER_SCRIPT} "
        f"--worker --case {case_name} --warmup {warmup} --iters {iters} "
        f"--device-id {device_id}"
    )
    msprof_cmd = f'msprof --output={case_dir} --application="{worker_cmd}"'

    result = subprocess.run(msprof_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-200:] if result.stderr else ''}")
        return None

    json_path = find_msprof_json(case_dir)
    if not json_path:
        return None

    stats = extract_kernel_duration(json_path, iters)

    if not keep_output:
        shutil.rmtree(case_dir)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HISA decode benchmark (msprof)")
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--keep-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.worker:
        run_worker(args)
        return

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")

    torch.npu.set_device(args.device_id)

    cases = DECODE_CASES
    if args.case:
        cases = [c for c in DECODE_CASES if c.name == args.case]
        if not cases:
            raise ValueError(f"Unknown decode case: {args.case}")
    elif not args.all:
        print("Specify --case <name> or --all")
        print(f"Available: {[c.name for c in DECODE_CASES]}")
        return

    MSPROF_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    header = "{:<30} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Case", "Avg(us)", "Median(us)", "Min(us)", "P90(us)", "Max(us)")
    print(header)
    print("-" * 90)

    for case in cases:
        sys.stdout.write(f"Running {case.name}... ")
        sys.stdout.flush()
        stats = run_msprof_case(
            case.name, args.warmup, args.iters, args.device_id, args.keep_output
        )
        if stats and "error" not in stats:
            print("{:<30} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}".format(
                case.name, stats["avg_us"], stats["median_us"],
                stats["min_us"], stats["p90_us"], stats["max_us"]))
        else:
            err = stats.get("error", "unknown") if stats else "msprof failed"
            print(f"{case.name:<30} ERROR: {err}")


if __name__ == "__main__":
    main()
