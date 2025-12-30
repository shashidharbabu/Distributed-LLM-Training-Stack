import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch.profiler import ProfilerActivity, profile, schedule

from .distributed import get_rank


@dataclass
class ProfilerConfig:
    enabled: bool = False
    wait: int = 2
    warmup: int = 2
    active: int = 5
    repeat: int = 1
    profile_dir: str = "./profiles"


def create_profiler(config: ProfilerConfig) -> Optional[profile]:
    if not config.enabled:
        return None
    os.makedirs(config.profile_dir, exist_ok=True)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    prof = profile(
        schedule=schedule(wait=config.wait, warmup=config.warmup, active=config.active, repeat=config.repeat),
        activities=activities,
        on_trace_ready=lambda p: _export_trace(p, config.profile_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )
    return prof


def _export_trace(prof: profile, profile_dir: str) -> None:
    rank = get_rank()
    path = os.path.join(profile_dir, f"profile_rank{rank}.json")
    prof.export_chrome_trace(path)
