from typing import Any, Dict, Optional, Tuple

import deepspeed
import torch.nn as nn


def default_zero_config(stage: int = 1) -> Dict[str, Any]:
    return {
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {"stage": stage},
        "bf16": {"enabled": False},
        "fp16": {"enabled": False},
    }


def initialize_engine(
    model: nn.Module,
    optimizer,
    lr_scheduler,
    ds_config: Optional[Dict[str, Any]],
) -> Tuple[deepspeed.DeepSpeedEngine, Any, Any]:
    ds_config = ds_config or default_zero_config()
    engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=ds_config,
    )
    return engine, optimizer, lr_scheduler
