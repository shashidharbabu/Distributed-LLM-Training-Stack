import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

import click
import yaml

from .checkpointing import CheckpointManager
from .launcher import LaunchConfig, launch
from .train import TrainRunConfig, train_worker


def _asdict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _asdict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _asdict(v) for k, v in obj.items()}
    return obj


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_dict(config: Any, updates: Dict[str, Any]) -> None:
    for key, value in updates.items():
        if not hasattr(config, key):
            continue
        current = getattr(config, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_dict(current, value)
        else:
            setattr(config, key, value)


def _apply_overrides(config: TrainRunConfig, overrides: List[str]) -> None:
    for ov in overrides:
        if "=" not in ov:
            continue
        key, raw = ov.split("=", 1)
        path = key.split(".")
        target = config
        for p in path[:-1]:
            target = getattr(target, p)
        field = path[-1]
        casted: Any = raw
        if raw.lower() in {"true", "false"}:
            casted = raw.lower() == "true"
        else:
            try:
                casted = int(raw)
            except ValueError:
                try:
                    casted = float(raw)
                except ValueError:
                    casted = raw
        setattr(target, field, casted)


@click.group()
def main() -> None:
    """llmtrain CLI."""


@main.command()
@click.option("--config", type=click.Path(exists=True), help="Path to YAML config.")
@click.option("--override", multiple=True, help="Override config values key=value (e.g. trainer.max_steps=10).")
@click.option("--gpus-per-node", type=int, default=1, show_default=True)
@click.option("--num-nodes", type=int, default=1, show_default=True)
@click.option("--node-rank", type=int, default=0, show_default=True)
@click.option("--master-addr", type=str, default="127.0.0.1")
@click.option("--master-port", type=int, default=29500)
@click.option("--max-retries", type=int, default=3, show_default=True)
@click.option("--backoff", type=float, default=10.0, show_default=True, help="Initial backoff seconds.")
@click.option("--max-backoff", type=float, default=300.0, show_default=True)
def run(config, override, gpus_per_node, num_nodes, node_rank, master_addr, master_port, max_retries, backoff, max_backoff) -> None:
    cfg = TrainRunConfig()
    updates = _load_yaml(config)
    _apply_dict(cfg, updates)
    _apply_overrides(cfg, list(override))
    launch_config = LaunchConfig(
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        node_rank=node_rank,
        master_addr=master_addr,
        master_port=master_port,
        max_retries=max_retries,
        base_backoff=backoff,
        max_backoff=max_backoff,
    )
    launch(train_worker, launch_config, {"config": cfg})


@main.command()
@click.option("--config", type=click.Path(exists=True), help="Path to YAML config.")
@click.option("--override", multiple=True, help="Override config values key=value.")
def profile(config, override) -> None:
    cfg = TrainRunConfig()
    updates = _load_yaml(config)
    _apply_dict(cfg, updates)
    cfg.profiler.enabled = True
    _apply_overrides(cfg, list(override))
    launch(train_worker, LaunchConfig(), {"config": cfg})


@main.command("validate-checkpoints")
@click.option("--checkpoint-dir", type=str, default="./checkpoints", show_default=True)
def validate_checkpoints(checkpoint_dir: str) -> None:
    manager = CheckpointManager(checkpoint_dir)
    results = manager.validate_all()
    click.echo(json.dumps(results, indent=2))


@main.command("k8s-render")
@click.option("--image", required=True, help="Container image for workers.")
@click.option("--gpus-per-node", type=int, default=4, show_default=True)
@click.option("--num-nodes", type=int, default=2, show_default=True)
@click.option("--checkpoint-pvc", type=str, default="llmtrain-pvc")
@click.option("--master-port", type=int, default=29500)
def k8s_render(image, gpus_per_node, num_nodes, checkpoint_pvc, master_port) -> None:
    template = {
        "image": image,
        "gpus_per_node": gpus_per_node,
        "num_nodes": num_nodes,
        "checkpoint_pvc": checkpoint_pvc,
        "master_port": master_port,
    }
    click.echo(json.dumps(template, indent=2))
