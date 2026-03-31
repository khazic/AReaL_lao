from __future__ import annotations

# pyright: reportMissingImports=false
from typing import Any

from pydantic import BaseModel, Field


class RegisterDatasetRequest(BaseModel):
    dataset_path: str
    dataset_type: str
    split: str = "train"
    tokenizer_or_processor_path: str = ""
    batch_size: int = 32
    seed: int = 42
    max_length: int | None = None
    shuffle: bool = True
    collate_mode: str = "identity"
    experiment_name: str = ""
    trial_name: str = ""
    fileroot: str = ""
    dataset_kwargs: dict[str, Any] = Field(default_factory=dict)


class RegisterDatasetResponse(BaseModel):
    api_key: str
    dataset_id: str
    steps_per_epoch: int
    dataset_size: int
    num_workers: int


class UnregisterDatasetRequest(BaseModel):
    dataset_id: str | None = None


class FetchBatchRequest(BaseModel):
    request_id: str


class FetchBatchResponse(BaseModel):
    batch_id: str
    epoch: int
    exhausted: bool
    data: Any


class EpochAdvanceRequest(BaseModel):
    epoch: int


class EpochAdvanceResponse(BaseModel):
    status: str = "ok"
    workers_reset: int = 0


class StateSaveResponse(BaseModel):
    path: str


class StateLoadRequest(BaseModel):
    path: str


class WorkerLoadDatasetRequest(BaseModel):
    dataset_id: str
    dataset_path: str
    dataset_type: str
    split: str = "train"
    tokenizer_or_processor_path: str = ""
    batch_size: int = 32
    seed: int = 42
    max_length: int | None = None
    shuffle: bool = True
    collate_mode: str = "identity"
    experiment_name: str = ""
    trial_name: str = ""
    fileroot: str = ""
    dataset_kwargs: dict[str, Any] = Field(default_factory=dict)


class WorkerUnloadDatasetRequest(BaseModel):
    dataset_id: str


class WorkerFetchBatchRequest(BaseModel):
    dataset_id: str


class WorkerEpochResetRequest(BaseModel):
    dataset_id: str
    epoch: int


class WorkerStateSaveRequest(BaseModel):
    dataset_id: str
    path: str


class WorkerStateLoadRequest(BaseModel):
    dataset_id: str
    path: str


class DatasetStatusResponse(BaseModel):
    dataset_id: str
    epoch: int
    exhausted: bool
    steps_per_epoch: int


class ServiceStatusResponse(BaseModel):
    status: str = "ok"
    datasets: list[DatasetStatusResponse] = Field(default_factory=list)
    num_workers: int = 0
