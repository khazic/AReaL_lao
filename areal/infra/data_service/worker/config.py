from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataWorkerConfig:
    host: str = "0.0.0.0"
    port: int = 0
    rank: int = 0
    world_size: int = 1
    prefetch_batches: int = 2
    dataloader_num_workers: int = 4
    experiment_name: str = ""
    trial_name: str = ""
    fileroot: str = ""
