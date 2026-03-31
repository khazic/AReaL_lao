from __future__ import annotations

# pyright: reportMissingImports=false
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
from datasets import Dataset

from areal.infra.data_service.worker.app import create_worker_app
from areal.infra.data_service.worker.config import DataWorkerConfig

DATASET_ID = "test-train"


@pytest.fixture
def config() -> DataWorkerConfig:
    return DataWorkerConfig(
        host="127.0.0.1",
        port=0,
        rank=0,
        world_size=1,
        prefetch_batches=2,
        dataloader_num_workers=0,
    )


def _make_mock_dataset(n: int = 20) -> Dataset:
    return Dataset.from_dict(
        {
            "text": [f"sample_{i}" for i in range(n)],
            "label": list(range(n)),
        }
    )


def _load_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "dataset_path": "test/dataset",
        "dataset_type": "rl",
        "batch_size": 4,
        "seed": 42,
        "collate_mode": "identity",
        "shuffle": False,
    }
    payload.update(overrides)
    return payload


async def _fetch_batch(
    client: httpx.AsyncClient, payload: dict[str, object]
) -> httpx.Response:
    return await client.post("/fetch_batch", params={"request": "_"}, json=payload)


@pytest_asyncio.fixture
async def client(config: DataWorkerConfig):
    app = create_worker_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def loaded_client(config: DataWorkerConfig):
    with (
        patch("areal.infra.data_service.worker.app.get_custom_dataset") as mock_get,
        patch(
            "areal.infra.data_service.worker.app.split_dataset_by_node"
        ) as mock_split,
        patch(
            "areal.infra.data_service.worker.app.load_hf_processor_and_tokenizer"
        ) as mock_load,
    ):
        ds = _make_mock_dataset(20)
        mock_get.return_value = ds
        mock_split.return_value = ds
        mock_load.return_value = (None, None)

        app = create_worker_app(config)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/datasets/load", json=_load_payload())
            assert resp.status_code == 200
            yield c


@pytest.mark.asyncio
class TestWorkerHealth:
    async def test_health_returns_200(self, client: httpx.AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["datasets"] == 0

    async def test_health_shows_dataset_count(self, loaded_client: httpx.AsyncClient):
        resp = await loaded_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["datasets"] == 1


@pytest.mark.asyncio
class TestDatasetLoading:
    async def test_load_dataset_returns_steps_per_epoch(self, config: DataWorkerConfig):
        with (
            patch("areal.infra.data_service.worker.app.get_custom_dataset") as mock_get,
            patch(
                "areal.infra.data_service.worker.app.split_dataset_by_node"
            ) as mock_split,
        ):
            ds = _make_mock_dataset(20)
            mock_get.return_value = ds
            mock_split.return_value = ds

            app = create_worker_app(config)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as c:
                resp = await c.post("/datasets/load", json=_load_payload())

        assert resp.status_code == 200
        data = resp.json()
        assert data["steps_per_epoch"] > 0
        assert data["dataset_size"] == 20

    async def test_load_dataset_duplicate_409(self, loaded_client: httpx.AsyncClient):
        resp = await loaded_client.post("/datasets/load", json=_load_payload())
        assert resp.status_code == 409

    async def test_load_dataset_invalid_collate_mode_400(
        self, config: DataWorkerConfig
    ):
        with (
            patch("areal.infra.data_service.worker.app.get_custom_dataset") as mock_get,
            patch(
                "areal.infra.data_service.worker.app.split_dataset_by_node"
            ) as mock_split,
        ):
            ds = _make_mock_dataset(20)
            mock_get.return_value = ds
            mock_split.return_value = ds

            app = create_worker_app(config)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as c:
                resp = await c.post(
                    "/datasets/load",
                    json=_load_payload(collate_mode="invalid"),
                )

        assert resp.status_code == 400
        assert "Unsupported collate_mode" in resp.json()["detail"]

    async def test_unload_dataset_removes(self, loaded_client: httpx.AsyncClient):
        resp = await loaded_client.post(
            "/datasets/unload", json={"dataset_id": DATASET_ID}
        )
        assert resp.status_code == 200

        health = await loaded_client.get("/health")
        assert health.status_code == 200
        assert health.json()["datasets"] == 0

    async def test_unload_unknown_dataset_404(self, client: httpx.AsyncClient):
        resp = await client.post("/datasets/unload", json={"dataset_id": "unknown"})
        assert resp.status_code == 404


@pytest.mark.asyncio
class TestBatchFetching:
    async def test_fetch_batch_returns_data(self, loaded_client: httpx.AsyncClient):
        resp = await _fetch_batch(
            loaded_client,
            {"dataset_id": DATASET_ID, "request_id": "req-1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["batch_id"] == "req-1"
        assert data["exhausted"] is False
        assert data["data"] is not None

    async def test_fetch_batch_unknown_dataset_404(self, client: httpx.AsyncClient):
        resp = await _fetch_batch(
            client, {"dataset_id": "unknown", "request_id": "req-1"}
        )
        assert resp.status_code == 404

    async def test_fetch_batch_missing_request_id_400(
        self, loaded_client: httpx.AsyncClient
    ):
        resp = await _fetch_batch(loaded_client, {"dataset_id": DATASET_ID})
        assert resp.status_code == 400
        assert "request_id is required" in resp.json()["detail"]

    async def test_fetch_batch_idempotent_same_request_id(
        self, loaded_client: httpx.AsyncClient
    ):
        first = await loaded_client.post(
            "/fetch_batch",
            params={"request": "_"},
            json={"dataset_id": DATASET_ID, "request_id": "same-id"},
        )
        second = await loaded_client.post(
            "/fetch_batch",
            params={"request": "_"},
            json={"dataset_id": DATASET_ID, "request_id": "same-id"},
        )
        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json() == first.json()

    async def test_fetch_batch_exhaustion_signals_true(
        self, loaded_client: httpx.AsyncClient
    ):
        exhausted = False
        for idx in range(10):
            resp = await loaded_client.post(
                "/fetch_batch",
                params={"request": "_"},
                json={"dataset_id": DATASET_ID, "request_id": f"req-{idx}"},
            )
            assert resp.status_code == 200
            exhausted = resp.json()["exhausted"]
            if exhausted:
                break

        assert exhausted is True


@pytest.mark.asyncio
class TestEpochReset:
    async def test_epoch_reset_resets_iterator(self, loaded_client: httpx.AsyncClient):
        for idx in range(10):
            resp = await loaded_client.post(
                "/fetch_batch",
                params={"request": "_"},
                json={"dataset_id": DATASET_ID, "request_id": f"before-reset-{idx}"},
            )
            assert resp.status_code == 200
            if resp.json()["exhausted"]:
                break

        reset = await loaded_client.post(
            "/epoch/reset", json={"dataset_id": DATASET_ID, "epoch": 1}
        )
        assert reset.status_code == 200

        fetched = await loaded_client.post(
            "/fetch_batch",
            params={"request": "_"},
            json={"dataset_id": DATASET_ID, "request_id": "after-reset"},
        )
        assert fetched.status_code == 200
        body = fetched.json()
        assert body["epoch"] == 1
        assert body["exhausted"] is False
        assert body["data"] is not None

    async def test_epoch_reset_unknown_dataset_404(self, client: httpx.AsyncClient):
        resp = await client.post(
            "/epoch/reset", json={"dataset_id": "unknown", "epoch": 1}
        )
        assert resp.status_code == 404

    async def test_epoch_reset_clears_idempotent_cache(
        self, loaded_client: httpx.AsyncClient
    ):
        req = {"dataset_id": DATASET_ID, "request_id": "cached-id"}
        first = await loaded_client.post(
            "/fetch_batch", params={"request": "_"}, json=req
        )
        assert first.status_code == 200
        assert first.json()["epoch"] == 0

        reset = await loaded_client.post(
            "/epoch/reset", json={"dataset_id": DATASET_ID, "epoch": 3}
        )
        assert reset.status_code == 200

        second = await loaded_client.post(
            "/fetch_batch", params={"request": "_"}, json=req
        )
        assert second.status_code == 200
        assert second.json()["epoch"] == 3


@pytest.mark.asyncio
class TestStatePersistence:
    async def test_state_save_creates_file(
        self, loaded_client: httpx.AsyncClient, tmp_path: Path
    ):
        resp = await loaded_client.post(
            "/state/save", json={"dataset_id": DATASET_ID, "path": str(tmp_path)}
        )
        assert resp.status_code == 200
        out = resp.json()
        assert out["status"] == "ok"
        assert (tmp_path / "worker_0.pkl").exists()

    async def test_state_load_restores(
        self, loaded_client: httpx.AsyncClient, tmp_path: Path
    ):
        save = await loaded_client.post(
            "/state/save", json={"dataset_id": DATASET_ID, "path": str(tmp_path)}
        )
        assert save.status_code == 200

        load = await loaded_client.post(
            "/state/load", json={"dataset_id": DATASET_ID, "path": str(tmp_path)}
        )
        assert load.status_code == 200
        assert load.json()["status"] == "ok"

    async def test_state_load_missing_file_404(
        self, loaded_client: httpx.AsyncClient, tmp_path: Path
    ):
        missing = tmp_path / "does-not-exist"
        resp = await loaded_client.post(
            "/state/load", json={"dataset_id": DATASET_ID, "path": str(missing)}
        )
        assert resp.status_code == 404


@pytest.mark.asyncio
class TestTensorShardEndpoints:
    async def test_data_clear_returns_ok(self, client: httpx.AsyncClient):
        resp = await client.delete("/data/clear")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["tensor_shards"] == 0

    async def test_data_shard_not_found_404(self, client: httpx.AsyncClient):
        resp = await client.get("/data/nonexistent")
        assert resp.status_code == 404
