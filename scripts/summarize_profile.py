#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from typing import Dict, List


def load_events(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("traceEvents", [])


def top_k(events: List[Dict], keyword: str, k: int = 10) -> List[tuple]:
    agg = defaultdict(float)
    for ev in events:
        name = ev.get("name", "").lower()
        if keyword not in name:
            continue
        dur = float(ev.get("dur", 0.0)) / 1000.0  # microseconds to ms
        agg[ev.get("name", "unknown")] += dur
    return sorted(agg.items(), key=lambda x: x[1], reverse=True)[:k]


def estimate_peak_memory(events: List[Dict]) -> float:
    peak = 0.0
    for ev in events:
        args = ev.get("args", {})
        for key, val in args.items():
            if "bytes" in key.lower() and isinstance(val, (int, float)):
                peak = max(peak, float(val) / 1e6)
    return peak


def main():
    parser = argparse.ArgumentParser(description="Summarize torch.profiler chrome trace")
    parser.add_argument("path", nargs="?", default="./profiles/profile_rank0.json")
    args = parser.parse_args()
    events = load_events(args.path)
    cuda_ops = top_k(events, "cuda")
    comm_ops = top_k(events, "nccl") + top_k(events, "allreduce")
    dataloader = top_k(events, "dataloader")
    peak_mem = estimate_peak_memory(events)

    print("Top CUDA ops by time (ms):")
    for name, dur in cuda_ops:
        print(f"  {name}: {dur:.2f} ms")
    print("\nTop communication ops:")
    for name, dur in comm_ops:
        print(f"  {name}: {dur:.2f} ms")
    print("\nDataloader hotspots:")
    for name, dur in dataloader:
        print(f"  {name}: {dur:.2f} ms")
    print(f"\nEstimated peak memory (MB): {peak_mem:.1f}")


if __name__ == "__main__":
    main()
