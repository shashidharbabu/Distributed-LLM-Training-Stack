from dataclasses import dataclass
from typing import Iterator, Tuple

import torch


@dataclass
class DataConfig:
    vocab_size: int = 256
    seq_len: int = 128
    batch_size: int = 8
    num_workers: int = 2
    pin_memory: bool = True


class SyntheticTextDataset(torch.utils.data.IterableDataset):
    def __init__(self, vocab_size: int, seq_len: int, total_tokens: int = 1_000_000):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.total_tokens = total_tokens

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        generator = torch.Generator()
        # Seed differently per worker for variability
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            generator.manual_seed(worker_info.id + 1)
        while True:
            data = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long, generator=generator)
            # simple next-token prediction
            yield data, data


def create_dataloader(config: DataConfig) -> torch.utils.data.DataLoader:
    dataset = SyntheticTextDataset(config.vocab_size, config.seq_len)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
