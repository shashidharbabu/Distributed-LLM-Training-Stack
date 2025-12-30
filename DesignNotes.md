# Design Notes

## Fault handling & retries
- Launcher wraps the worker entrypoint in a retry loop with exponential backoff. Exceptions containing NCCL timeouts, transient connection resets, or CUDA OOM are treated as retriable.
- Graceful preemption: a SIGTERM sets an internal flag; the main loop saves a checkpoint (rank0) then exits so the process returns a retriable code.
- Non-retriable faults bubble up immediately to fail the job.

## Checkpoint atomicity & selection
- Checkpoints are written to `checkpoint_dir/step_<step>.tmp` then renamed, so partially written checkpoints are never exposed.
- `meta.json` and `state.pt` live in each checkpoint directory; a manifest (`manifest.json`) tracks the latest good step and recent history (bounded to 20 entries).
- On startup the loader prefers the manifest entry and falls back to scanning directories in descending step order, skipping missing/partial checkpoints.
- Validation lists every checkpoint and whether required files are present.

## Profiler interpretation
- `llmtrain profile` enables torch.profiler with a wait/warmup/active schedule and exports Chrome traces to `./profiles/profile_rank*.json`.
- Use `scripts/summarize_profile.py` to print hot CUDA ops, NCCL buckets, dataloader stalls, and a coarse peak memory estimate. Inspect the trace in Chrome for timeline-based debugging of communication overlap.
- Throughput knobs: gradient accumulation, bucket sizes (set via `TORCH_DISTRIBUTED_DEBUG` and NCCL envs), DataLoader workers/pinning, ZeRO stage, activation checkpointing, and mixed precision.
