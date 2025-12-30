import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist


def infer_default_backend() -> str:
    if torch.cuda.is_available():
        return "nccl"
    return "gloo"


def init_distributed(
    backend: Optional[str] = None,
    timeout_seconds: int = 900,
    init_method: Optional[str] = None,
) -> None:
    if dist.is_initialized():
        return
    backend = backend or infer_default_backend()
    timeout = torch.distributed.Timeout(timeout_seconds)
    init_method = init_method or os.environ.get("INIT_METHOD", "env://")
    dist.init_process_group(backend=backend, timeout=timeout, init_method=init_method)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def distributed_state() -> Dict[str, Any]:
    return {
        "rank": get_rank(),
        "world_size": get_world_size(),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", 1)),
        "node_rank": int(os.environ.get("NODE_RANK", 0)),
        "master_addr": os.environ.get("MASTER_ADDR", "localhost"),
        "master_port": os.environ.get("MASTER_PORT", "29500"),
    }


@contextmanager
def main_process_first():
    if not dist.is_available() or not dist.is_initialized():
        yield
        return
    is_main = is_main_process()
    try:
        if not is_main:
            barrier()
        yield
    finally:
        if is_main:
            barrier()
