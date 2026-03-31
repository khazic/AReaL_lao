"""DataWorker FastAPI app with per-dataset prefetch and local tensor store."""

from __future__ import annotations

import pickle
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets.distributed import split_dataset_by_node
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response as RawResponse
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.dataset import get_custom_dataset
from areal.infra.data_service.types import (
    FetchBatchResponse,
    WorkerEpochResetRequest,
    WorkerLoadDatasetRequest,
    WorkerStateLoadRequest,
    WorkerStateSaveRequest,
    WorkerUnloadDatasetRequest,
)
from areal.infra.data_service.worker.config import DataWorkerConfig
from areal.infra.data_service.worker.prefetch import PrefetchBuffer
from areal.infra.data_service.worker.tensor_store import TensorStore
from areal.infra.rpc.serialization import serialize_value
from areal.utils import logging, seeding
from areal.utils.data import collate_samples_to_list
from areal.utils.hf_utils import load_hf_tokenizer

logger = logging.getLogger("DataWorker")


def _identity_collate(samples: list[Any]) -> list[Any]:
    return samples


@dataclass
class _DatasetState:
    dataset_id: str
    dataloader: Any
    data_iter: Iterator
    epoch: int
    exhausted: bool
    seed: int
    prefetch: Any
    tensor_shard_ids: list[str]


def create_worker_app(config: DataWorkerConfig) -> FastAPI:
    datasets: dict[str, _DatasetState] = {}
    idempotent_cache: dict[tuple[str, str], dict[str, Any]] = {}
    tensor_store = TensorStore()

    @asynccontextmanager
    async def lifespan(app: Any):
        app.state.config = config
        app.state.datasets = datasets
        app.state.fetch_cache = idempotent_cache
        app.state.tensor_store = tensor_store
        try:
            yield
        finally:
            for state in datasets.values():
                state.prefetch.stop()
                tensor_store.clear_for_dataset(state.tensor_shard_ids)
            datasets.clear()
            idempotent_cache.clear()
            tensor_store.clear()

    app = FastAPI(title="AReaL Data Worker", lifespan=lifespan)

    def _require_dataset(dataset_id: str) -> _DatasetState:
        state = datasets.get(dataset_id)
        if state is None:
            raise HTTPException(
                status_code=404, detail=f"Unknown dataset_id: {dataset_id}"
            )
        return state

    def _clear_cache_for_dataset(dataset_id: str) -> None:
        to_delete = [k for k in idempotent_cache if k[0] == dataset_id]
        for key in to_delete:
            del idempotent_cache[key]

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "rank": config.rank,
            "datasets": len(datasets),
            "tensor_shards": tensor_store.size,
        }

    @app.post("/datasets/load")
    async def load_dataset(body: WorkerLoadDatasetRequest):
        if body.dataset_id in datasets:
            raise HTTPException(
                status_code=409,
                detail=f"Dataset {body.dataset_id} is already loaded",
            )

        tokenizer = None
        if body.tokenizer_path:
            tokenizer = load_hf_tokenizer(body.tokenizer_path)

        processor = None
        if body.processor_path:
            from areal.utils.hf_utils import load_hf_processor_and_tokenizer

            processor, tokenizer = load_hf_processor_and_tokenizer(body.processor_path)

        seeding.set_random_seed(body.seed, key=f"data_worker_{config.rank}")

        from areal.api.cli_args import TrainDatasetConfig

        dataset_config = TrainDatasetConfig(
            path=body.dataset_path,
            type=body.dataset_type,
            batch_size=body.batch_size,
            max_length=body.max_length,
        )
        dataset = get_custom_dataset(
            split=body.split,
            dataset_config=dataset_config,
            tokenizer=tokenizer,
            processor=processor,
            **body.dataset_kwargs,
        )
        shard = split_dataset_by_node(
            dataset,
            rank=config.rank,
            world_size=config.world_size,
        )

        collate_fn = None
        if body.collate_mode == "identity":
            collate_fn = _identity_collate
        elif body.collate_mode == "tensor":
            collate_fn = collate_samples_to_list
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported collate_mode: {body.collate_mode}",
            )

        dataloader = StatefulDataLoader(
            shard,
            batch_size=body.batch_size,
            num_workers=config.dataloader_num_workers,
            shuffle=body.shuffle,
            collate_fn=collate_fn,
        )
        data_iter = iter(dataloader)

        prefetch = PrefetchBuffer(capacity=config.prefetch_batches)
        prefetch.start(data_iter, body.dataset_id)

        datasets[body.dataset_id] = _DatasetState(
            dataset_id=body.dataset_id,
            dataloader=dataloader,
            data_iter=data_iter,
            epoch=0,
            exhausted=False,
            seed=body.seed,
            prefetch=prefetch,
            tensor_shard_ids=[],
        )

        _clear_cache_for_dataset(body.dataset_id)
        return {
            "status": "ok",
            "dataset_size": len(shard),
            "steps_per_epoch": len(dataloader),
        }

    @app.post("/datasets/unload")
    async def unload_dataset(body: WorkerUnloadDatasetRequest):
        state = _require_dataset(body.dataset_id)
        state.prefetch.stop()
        tensor_store.clear_for_dataset(state.tensor_shard_ids)
        del datasets[body.dataset_id]
        _clear_cache_for_dataset(body.dataset_id)
        return {"status": "ok"}

    @app.post("/fetch_batch")
    async def fetch_batch(request: Request):
        payload = await request.json()
        dataset_id = payload.get("dataset_id")
        request_id = payload.get("request_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise HTTPException(status_code=400, detail="dataset_id is required")
        if not isinstance(request_id, str) or not request_id:
            raise HTTPException(status_code=400, detail="request_id is required")

        cache_key = (dataset_id, request_id)
        cached = idempotent_cache.get(cache_key)
        if cached is not None:
            return cached

        state = _require_dataset(dataset_id)
        data: Any = None
        if not state.exhausted:
            batch = await state.prefetch.get()
            if batch is None:
                state.exhausted = True
            else:
                data = serialize_value(batch)

        response = FetchBatchResponse(
            batch_id=request_id,
            epoch=state.epoch,
            exhausted=state.exhausted,
            data=data,
        ).model_dump()
        idempotent_cache[cache_key] = response
        return response

    @app.post("/epoch/reset")
    async def reset_epoch(body: WorkerEpochResetRequest):
        state = _require_dataset(body.dataset_id)
        state.prefetch.stop()

        seeding.set_random_seed(state.seed, key=f"data_worker_{config.rank}")
        state.epoch = body.epoch
        state.exhausted = False
        state.data_iter = iter(state.dataloader)
        state.prefetch.start(state.data_iter, state.dataset_id)

        _clear_cache_for_dataset(body.dataset_id)
        return {"status": "ok", "epoch": state.epoch}

    @app.post("/state/save")
    async def save_state(body: WorkerStateSaveRequest):
        state = _require_dataset(body.dataset_id)
        save_dir = Path(body.path)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"worker_{config.rank}.pkl"

        with save_path.open("wb") as f:
            pickle.dump(state.dataloader.state_dict(), f)

        return {"status": "ok", "path": str(save_path)}

    @app.post("/state/load")
    async def load_state(body: WorkerStateLoadRequest):
        state = _require_dataset(body.dataset_id)
        load_path = Path(body.path) / f"worker_{config.rank}.pkl"
        if not load_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"State file not found: {load_path}",
            )

        with load_path.open("rb") as f:
            state_dict = pickle.load(f)
        state.dataloader.load_state_dict(state_dict)

        state.prefetch.stop()
        state.exhausted = False
        state.data_iter = iter(state.dataloader)
        state.prefetch.start(state.data_iter, state.dataset_id)

        _clear_cache_for_dataset(body.dataset_id)
        return {"status": "ok", "path": str(load_path)}

    @app.get("/data/{shard_id}")
    async def fetch_data_shard(shard_id: str):
        data = tensor_store.get(shard_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"Shard {shard_id} not found")
        import json as _json

        data_bytes = _json.dumps(serialize_value(data)).encode()
        return RawResponse(content=data_bytes, media_type="application/json")

    @app.post("/data/batch")
    async def fetch_data_batch(request: Request):
        payload = await request.json()
        shard_ids = payload.get("shard_ids", [])
        if not isinstance(shard_ids, list) or not all(
            isinstance(sid, str) for sid in shard_ids
        ):
            raise HTTPException(
                status_code=400,
                detail="Expected JSON body with string list field 'shard_ids'",
            )

        data: list[Any] = []
        missing: list[str] = []
        for sid in shard_ids:
            item = tensor_store.get(sid)
            if item is None:
                missing.append(sid)
            else:
                data.append(item)

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Shards not found: {missing}",
            )

        import json as _json

        data_bytes = _json.dumps(serialize_value(data)).encode()
        return RawResponse(content=data_bytes, media_type="application/json")

    @app.delete("/data/clear")
    async def clear_data_shards():
        tensor_store.clear()
        for state in datasets.values():
            state.tensor_shard_ids.clear()
        return {"status": "ok", "tensor_shards": tensor_store.size}

    return app
