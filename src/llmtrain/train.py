import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .checkpointing import CheckpointManager, load_latest_checkpoint
from .data import DataConfig, create_dataloader
from .deepspeed_utils import initialize_engine, default_zero_config
from .distributed import barrier, distributed_state, get_rank, get_world_size, init_distributed, is_main_process
from .faults import GracefulTerminator
from .metrics import MetricsWriter, log_event, setup_logging
from .model import ToyGPTConfig, ToyGPTModel
from .profiler import ProfilerConfig, create_profiler


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    use_amp: bool = True


@dataclass
class TrainerConfig:
    max_steps: int = 100
    log_interval: int = 10
    save_interval: int = 50
    checkpoint_dir: str = "./checkpoints"
    metrics_dir: Optional[str] = None
    backend: Optional[str] = None
    use_deepspeed: bool = False
    deepspeed_zero_stage: int = 1
    profile: bool = False
    profile_dir: str = "./profiles"
    seed: int = 42
    resume: bool = True
    max_ckpts: int = 5
    lr_cooldown_steps: int = 0


@dataclass
class TrainRunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ToyGPTConfig = field(default_factory=ToyGPTConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_worker(local_rank: int, kwargs: Dict[str, Any]) -> None:
    config: TrainRunConfig = kwargs["config"]
    init_distributed(backend=config.trainer.backend)
    logger = setup_logging()
    log_event(logger, "startup", **distributed_state())
    set_seed(config.trainer.seed + get_rank())

    # build data/model
    dataloader = create_dataloader(config.data)
    data_iter = iter(dataloader)
    model = ToyGPTModel(config.model)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
        betas=config.optim.betas,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(config.trainer.max_steps, 1))
    scaler = GradScaler(enabled=config.optim.use_amp and torch.cuda.is_available())
    ds_engine = None
    if config.trainer.use_deepspeed:
        ds_conf = default_zero_config(config.trainer.deepspeed_zero_stage)
        ds_conf["train_batch_size"] = config.data.batch_size * get_world_size() * config.optim.grad_accum_steps
        ds_conf["gradient_accumulation_steps"] = config.optim.grad_accum_steps
        ds_engine, optimizer, scheduler = initialize_engine(model, optimizer, scheduler, ds_conf)
    else:
        if dist.is_initialized():
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank] if torch.cuda.is_available() else None,
                output_device=local_rank if torch.cuda.is_available() else None,
            )

    manager = CheckpointManager(config.trainer.checkpoint_dir)
    start_step = 0
    if config.trainer.resume:
        if ds_engine:
            latest = manager.latest_checkpoint()
            if latest:
                start_step = latest.step + 1
                ds_engine.load_checkpoint(latest.path)
                log_event(logger, "resume", checkpoint=latest.path, step=start_step)
        else:
            ckpt = load_latest_checkpoint(manager, map_location=device)
            if ckpt:
                info, state = ckpt
                start_step = info.step + 1
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
                if "scaler" in state and scaler.is_enabled():
                    scaler.load_state_dict(state["scaler"])
                log_event(logger, "resume", checkpoint=info.path, step=start_step)

    metrics = MetricsWriter(config.trainer.metrics_dir)
    terminator = GracefulTerminator()
    profiler = create_profiler(config.profiler)
    if profiler:
        profiler.__enter__()

    step = start_step
    model.train()
    tokens_per_step = config.data.batch_size * config.data.seq_len * get_world_size()
    start_wall = time.time()
    try:
        while step < config.trainer.max_steps:
            if terminator.should_terminate:
                if is_main_process():
                    _save_checkpoint(
                        manager,
                        step,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        ds_engine,
                        loss=None,
                    )
                    log_event(logger, "graceful_preemption", step=step)
                break

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss_total = 0.0
            optimizer.zero_grad(set_to_none=True)

            for micro in range(config.optim.grad_accum_steps):
                with autocast(enabled=scaler.is_enabled()):
                    loss = model(x, y)
                    loss = loss / config.optim.grad_accum_steps
                loss_total += loss.item()
                if ds_engine:
                    ds_engine.backward(loss)
                else:
                    loss.backward()

            if ds_engine:
                ds_engine.step()
            else:
                if config.optim.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()

            if profiler:
                profiler.step()

            if step % config.trainer.log_interval == 0:
                elapsed = time.time() - start_wall
                steps = step - start_step + 1
                steps_per_sec = steps / max(elapsed, 1e-6)
                tokens_sec = steps_per_sec * tokens_per_step
                grad_norm = _grad_norm(model)
                gpu_mem = (
                    torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
                )
                log_event(
                    logger,
                    "metrics",
                    step=step,
                    loss=loss_total,
                    steps_per_sec=steps_per_sec,
                    tokens_per_sec=tokens_sec,
                    grad_norm=grad_norm,
                    gpu_mem_gb=gpu_mem,
                )
                metrics.write_prometheus(
                    {
                        "loss": loss_total,
                        "steps_per_sec": steps_per_sec,
                        "tokens_per_sec": tokens_sec,
                        "grad_norm": grad_norm,
                        "gpu_mem_gb": gpu_mem,
                    }
                )

            if step % config.trainer.save_interval == 0 or step + 1 == config.trainer.max_steps:
                if is_main_process():
                    _save_checkpoint(
                        manager,
                        step,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        ds_engine,
                        loss=loss_total,
                    )
            step += 1
    finally:
        if profiler:
            profiler.__exit__(None, None, None)
    barrier()


def _grad_norm(model: torch.nn.Module) -> float:
    total_norm = 0.0
    parameters = [p for p in model.parameters() if p.grad is not None]
    for p in parameters:
        param_norm = p.grad.data.norm(2).item()
        total_norm += param_norm**2
    return total_norm ** 0.5 if parameters else 0.0


def _save_checkpoint(
    manager: CheckpointManager,
    step: int,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    scaler: GradScaler,
    ds_engine,
    loss: Optional[float],
) -> None:
    if ds_engine:
        path = manager.checkpoint_dir(step)
        ds_engine.save_checkpoint(path)
        manager.record_external(step, path, loss=loss)
        return
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if scaler.is_enabled():
        state["scaler"] = scaler.state_dict()
    manager.save(step, state, loss=loss)
