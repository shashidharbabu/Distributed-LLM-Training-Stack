import logging
import signal
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional


class FaultType(str, Enum):
    RETRIABLE = "retriable"
    NON_RETRIABLE = "non_retriable"


RETRIABLE_SIGNATURES = [
    "nccl",
    "connection reset",
    "timeout",
    "cuda error: out of memory",
    "runtimeerror: cuda error",
]


def classify_fault(exc: BaseException) -> FaultType:
    text = str(exc).lower()
    for sig in RETRIABLE_SIGNATURES:
        if sig in text:
            return FaultType.RETRIABLE
    return FaultType.NON_RETRIABLE


@dataclass
class RetryPolicy:
    max_retries: int = 3
    base_backoff: float = 10.0
    max_backoff: float = 300.0

    def backoff_for(self, attempt: int) -> float:
        # exponential backoff capped at max_backoff
        return min(self.max_backoff, self.base_backoff * (2 ** attempt))


def run_with_retries(
    fn: Callable[[], None], retry_policy: RetryPolicy, logger: Optional[logging.Logger] = None
) -> None:
    attempt = 0
    while True:
        try:
            fn()
            return
        except BaseException as exc:  # noqa: BLE001
            fault_type = classify_fault(exc)
            log = logger.info if logger else print
            if fault_type is FaultType.RETRIABLE and attempt < retry_policy.max_retries:
                delay = retry_policy.backoff_for(attempt)
                log(f"Retriable failure detected ({exc}); retrying in {delay:.1f}s (attempt {attempt+1})")
                time.sleep(delay)
                attempt += 1
                continue
            log(f"Non-retriable failure or retries exhausted: {exc}")
            raise


class GracefulTerminator:
    def __init__(self):
        self._terminate = threading.Event()
        self._install()

    def _install(self) -> None:
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame) -> None:  # type: ignore[override]
        _ = frame
        self._terminate.set()

    @property
    def should_terminate(self) -> bool:
        return self._terminate.is_set()
