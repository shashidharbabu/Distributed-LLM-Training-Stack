import os
import socket
import pytest
from typing import Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from llmtrain.checkpointing import CheckpointManager, load_latest_checkpoint
from llmtrain.model import ToyGPTConfig, ToyGPTModel


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", 0))
        except PermissionError:
            pytest.skip("Socket bind not permitted in this environment")
        return s.getsockname()[1]


def _train_worker(rank: int, world_size: int, port: int, checkpoint_dir: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    cfg = ToyGPTConfig(d_model=32, n_layers=1, n_heads=2, max_seq_len=16, vocab_size=32)
    model = ToyGPTModel(cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    manager = CheckpointManager(checkpoint_dir)

    start_step = 0
    ckpt = load_latest_checkpoint(manager, map_location="cpu")
    if ckpt:
        info, state = ckpt
        start_step = info.step + 1
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

    for step in range(start_step, start_step + 2):
        x = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
        loss = model(x, x)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        if rank == 0:
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            manager.save(step, state, loss=float(loss), is_main=True)
        dist.barrier()

    dist.destroy_process_group()


def test_resume_smoke(tmp_path):
    checkpoint_dir = tmp_path / "ckpts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    port = _find_free_port()

    def launch():
        mp.spawn(_train_worker, args=(2, port, str(checkpoint_dir)), nprocs=2, join=True)

    launch()
    manager = CheckpointManager(str(checkpoint_dir))
    first = manager.latest_checkpoint()
    assert first is not None
    assert first.step >= 1

    # second run should resume past previous step
    launch()
    second = manager.latest_checkpoint()
    assert second is not None
    assert second.step >= first.step + 1
