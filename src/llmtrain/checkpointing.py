import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from .distributed import barrier, get_rank, is_main_process


MANIFEST = "manifest.json"
STATE_FILE = "state.pt"
META_FILE = "meta.json"


@dataclass
class CheckpointInfo:
    step: int
    path: str
    loss: Optional[float] = None
    timestamp: float = 0.0
    ok: bool = True


def _atomic_write(path: str, data: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class CheckpointManager:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        _safe_makedirs(self.root_dir)
        self.manifest_path = os.path.join(self.root_dir, MANIFEST)

    def checkpoint_dir(self, step: int) -> str:
        return os.path.join(self.root_dir, f"step_{step:08d}")

    def _load_manifest(self) -> Dict[str, Any]:
        if not os.path.exists(self.manifest_path):
            return {"latest": None, "checkpoints": []}
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_manifest(self, manifest: Dict[str, Any]) -> None:
        _atomic_write(self.manifest_path, manifest)

    def _record_checkpoint(self, info: CheckpointInfo) -> None:
        manifest = self._load_manifest()
        entries: List[Dict[str, Any]] = manifest.get("checkpoints", [])
        entries.append(
            {
                "step": info.step,
                "path": info.path,
                "loss": info.loss,
                "timestamp": info.timestamp,
                "ok": info.ok,
            }
        )
        # keep only last 20 to avoid unbounded growth
        manifest["checkpoints"] = entries[-20:]
        manifest["latest"] = info.step if info.ok else manifest.get("latest")
        self._write_manifest(manifest)

    def save(
        self,
        step: int,
        state: Dict[str, Any],
        loss: Optional[float] = None,
        is_main: Optional[bool] = None,
    ) -> str:
        should_write = is_main if is_main is not None else is_main_process()
        # only main process writes to disk
        if not should_write:
            barrier()
            return ""
        ckpt_dir = self.checkpoint_dir(step)
        tmp_dir = f"{ckpt_dir}.tmp"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        _safe_makedirs(tmp_dir)
        state_path = os.path.join(tmp_dir, STATE_FILE)
        meta_path = os.path.join(tmp_dir, META_FILE)
        torch.save(state, state_path)
        _atomic_write(
            meta_path,
            {
                "step": step,
                "loss": loss,
                "timestamp": time.time(),
                "rank": get_rank(),
            },
        )
        os.replace(tmp_dir, ckpt_dir)
        info = CheckpointInfo(step=step, path=ckpt_dir, loss=loss, timestamp=time.time(), ok=True)
        self._record_checkpoint(info)
        barrier()
        return ckpt_dir

    def list_checkpoints(self) -> List[CheckpointInfo]:
        manifest = self._load_manifest()
        checkpoints: List[CheckpointInfo] = []
        for entry in manifest.get("checkpoints", []):
            checkpoints.append(
                CheckpointInfo(
                    step=int(entry["step"]),
                    path=entry["path"],
                    loss=entry.get("loss"),  # type: ignore[arg-type]
                    timestamp=float(entry.get("timestamp", 0.0)),
                    ok=bool(entry.get("ok", True)),
                )
            )
        return checkpoints

    def latest_checkpoint(self) -> Optional[CheckpointInfo]:
        manifest = self._load_manifest()
        latest = manifest.get("latest")
        if latest is not None:
            path = self.checkpoint_dir(int(latest))
            if self._is_valid_checkpoint(path):
                return CheckpointInfo(step=int(latest), path=path, ok=True)
        # fallback: scan directories
        candidates = self._scan_checkpoint_dirs()
        for step, path in candidates:
            if self._is_valid_checkpoint(path):
                return CheckpointInfo(step=step, path=path, ok=True)
        return None

    def _scan_checkpoint_dirs(self) -> List[Tuple[int, str]]:
        pattern = re.compile(r"step_(\d+)")
        candidates: List[Tuple[int, str]] = []
        for name in os.listdir(self.root_dir):
            match = pattern.match(name)
            if match:
                step = int(match.group(1))
                candidates.append((step, os.path.join(self.root_dir, name)))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates

    def _is_valid_checkpoint(self, path: str) -> bool:
        state = os.path.join(path, STATE_FILE)
        meta = os.path.join(path, META_FILE)
        has_state = os.path.isfile(state)
        has_meta = os.path.isfile(meta)
        # Accept DeepSpeed checkpoints that may not use STATE_FILE but still emit metadata
        return os.path.exists(path) and has_meta and (has_state or len(os.listdir(path)) > 0)

    def load(self, path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
        return torch.load(os.path.join(path, STATE_FILE), map_location=map_location)

    def validate_all(self) -> List[Tuple[str, bool, Optional[str]]]:
        results: List[Tuple[str, bool, Optional[str]]] = []
        for step, path in self._scan_checkpoint_dirs():
            ok = self._is_valid_checkpoint(path)
            reason = None if ok else "missing files"
            results.append((path, ok, reason))
        return results

    def record_external(self, step: int, path: str, loss: Optional[float] = None) -> None:
        """Record a checkpoint produced outside CheckpointManager (e.g., DeepSpeed)."""
        _safe_makedirs(path)
        meta_path = os.path.join(path, META_FILE)
        _atomic_write(
            meta_path,
            {
                "step": step,
                "loss": loss,
                "timestamp": time.time(),
                "rank": get_rank(),
            },
        )
        self._record_checkpoint(CheckpointInfo(step=step, path=path, loss=loss, timestamp=time.time(), ok=True))


def load_latest_checkpoint(
    manager: CheckpointManager, map_location: Optional[str] = None
) -> Optional[Tuple[CheckpointInfo, Dict[str, Any]]]:
    latest = manager.latest_checkpoint()
    if latest is None:
        return None
    try:
        state = manager.load(latest.path, map_location=map_location)
        return latest, state
    except Exception:
        # mark as bad and continue
        manifest = manager._load_manifest()
        manifest["checkpoints"] = [
            {**c, "ok": False} if int(c.get("step", -1)) == latest.step else c
            for c in manifest.get("checkpoints", [])
        ]
        manager._write_manifest(manifest)
        return None
