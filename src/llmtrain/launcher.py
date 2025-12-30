import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch
import torch.multiprocessing as mp

from .faults import RetryPolicy, run_with_retries
from .metrics import setup_logging


@dataclass
class LaunchConfig:
    num_nodes: int = 1
    gpus_per_node: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    max_retries: int = 3
    base_backoff: float = 10.0
    max_backoff: float = 300.0
    backend: Optional[str] = None

    @property
    def world_size(self) -> int:
        return self.num_nodes * self.gpus_per_node


def launch(
    entrypoint: Callable[[int, Dict[str, Any]], None],
    config: LaunchConfig,
    worker_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    worker_kwargs = worker_kwargs or {}
    logger = setup_logging()
    retry_policy = RetryPolicy(
        max_retries=config.max_retries, base_backoff=config.base_backoff, max_backoff=config.max_backoff
    )

    def _run() -> None:
        if config.world_size > 1:
            mp.spawn(
                _wrap_worker,
                args=(entrypoint, config, worker_kwargs),
                nprocs=config.gpus_per_node,
                join=True,
            )
        else:
            _wrap_worker(0, entrypoint, config, worker_kwargs)

    run_with_retries(_run, retry_policy=retry_policy, logger=logger)


def _wrap_worker(local_rank: int, entrypoint: Callable, config: LaunchConfig, worker_kwargs: Dict[str, Any]) -> None:
    global_rank = config.node_rank * config.gpus_per_node + local_rank
    os.environ["RANK"] = str(global_rank)
    os.environ["WORLD_SIZE"] = str(config.world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(config.gpus_per_node)
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = str(config.master_port)
    if config.backend:
        os.environ["TORCH_BACKEND"] = config.backend
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    entrypoint(local_rank, worker_kwargs)
