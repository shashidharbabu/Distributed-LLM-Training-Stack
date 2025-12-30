import json
import logging
import os
import time
from typing import Dict, Optional

from .distributed import get_rank


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("llmtrain")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log_event(logger: logging.Logger, event: str, **fields) -> None:
    payload = {"event": event, "ts": time.time(), "rank": get_rank(), **fields}
    logger.info(json.dumps(payload))


class MetricsWriter:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def write_prometheus(self, metrics: Dict[str, float]) -> None:
        if not self.output_dir:
            return
        lines = []
        for k, v in metrics.items():
            safe_key = k.replace(" ", "_")
            lines.append(f"# HELP {safe_key} {safe_key}")
            lines.append(f"# TYPE {safe_key} gauge")
            lines.append(f"{safe_key} {v}")
        path = os.path.join(self.output_dir, "metrics.prom")
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        os.replace(tmp, path)
