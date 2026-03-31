from __future__ import annotations

# pyright: reportMissingImports=false
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio

from areal.infra.data_service.gateway.app import create_gateway_app
from areal.infra.data_service.gateway.config import GatewayConfig

ADMIN_KEY = "test-admin-key"
WORKER_ADDR = "http://worker-1:8000"
WORKER_ADDR_2 = "http://worker-2:8000"
MODULE = "areal.infra.data_service.gateway.app"


@pytest.fixture
def config():
    return GatewayConfig(
        host="127.0.0.1",
        port=18090,
        admin_api_key=ADMIN_KEY,
        router_addr="http://mock-router:8091",
    )


@pytest_asyncio.fixture
async def client(config):
    app = create_gateway_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def admin_headers():
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


async def _register_dataset(client, dataset_id: str = "train-sample") -> dict:
    with (
        patch(
            f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
        ) as mock_workers,
        patch(
            f"{MODULE}._broadcast_to_workers", new_callable=AsyncMock
        ) as mock_broadcast,
    ):
        mock_workers.return_value = [WORKER_ADDR]
        mock_broadcast.return_value = [
            {
                "addr": WORKER_ADDR,
                "status": 200,
                "data": {"steps_per_epoch": 10, "dataset_size": 100},
            }
        ]
        resp = await client.post(
            "/v1/datasets/register",
            json={"dataset_id": dataset_id, "dataset_path": "/tmp/sample.jsonl"},
            headers=admin_headers(),
        )
    assert resp.status_code == 200
    return resp.json()


class TestGatewayHealth:
    @pytest.mark.asyncio
    async def test_health_no_auth_required(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_returns_router_addr(self, client, config):
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["router_addr"] == config.router_addr


class TestGatewayAuth:
    @pytest.mark.asyncio
    async def test_fetch_batch_no_auth_401(self, client):
        resp = await client.post("/v1/batches/next", json={})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_fetch_batch_bad_key_401(self, client):
        resp = await client.post(
            "/v1/batches/next",
            json={},
            headers={"Authorization": "Bearer unknown-key"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_admin_endpoint_with_dataset_key_403(self, client):
        resp = await client.post(
            "/v1/datasets/register",
            json={"dataset_id": "d1", "dataset_path": "/tmp/data.jsonl"},
            headers={"Authorization": "Bearer ds-not-admin"},
        )
        assert resp.status_code == 403


class TestDatasetRegistration:
    @pytest.mark.asyncio
    async def test_register_dataset_returns_api_key(self, client):
        with (
            patch(
                f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
            ) as mock_workers,
            patch(
                f"{MODULE}._broadcast_to_workers", new_callable=AsyncMock
            ) as mock_broadcast,
        ):
            mock_workers.return_value = [WORKER_ADDR, WORKER_ADDR_2]
            mock_broadcast.return_value = [
                {
                    "addr": WORKER_ADDR,
                    "status": 200,
                    "data": {"steps_per_epoch": 12, "dataset_size": 120},
                },
                {
                    "addr": WORKER_ADDR_2,
                    "status": 200,
                    "data": {"steps_per_epoch": 12, "dataset_size": 120},
                },
            ]

            resp = await client.post(
                "/v1/datasets/register",
                json={"dataset_id": "dataset-a", "dataset_path": "/tmp/a.jsonl"},
                headers=admin_headers(),
            )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["api_key"].startswith("ds-")
        assert payload["dataset_id"] == "dataset-a"
        assert payload["steps_per_epoch"] == 12

    @pytest.mark.asyncio
    async def test_register_then_fetch_uses_dataset_key(self, client):
        reg_payload = await _register_dataset(client, dataset_id="dataset-b")
        dataset_key = reg_payload["api_key"]

        mock_client = AsyncMock()
        mock_client.post.return_value = httpx.Response(
            200,
            json={"request_id": "req-1", "items": [{"id": 1}]},
        )
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_client

        with (
            patch(
                f"{MODULE}._query_router", new_callable=AsyncMock
            ) as mock_query_router,
            patch(f"{MODULE}.httpx.AsyncClient", return_value=mock_cm),
        ):
            mock_query_router.return_value = WORKER_ADDR
            resp = await client.post(
                "/v1/batches/next",
                json={"request_id": "req-1"},
                headers={"Authorization": f"Bearer {dataset_key}"},
            )

        assert resp.status_code == 200
        assert resp.json()["request_id"] == "req-1"
        mock_client.post.assert_awaited_once_with(
            f"{WORKER_ADDR}/fetch_batch",
            json={"dataset_id": "dataset-b", "request_id": "req-1"},
        )

    @pytest.mark.asyncio
    async def test_unregister_revokes_key(self, client):
        reg_payload = await _register_dataset(client, dataset_id="dataset-c")
        dataset_key = reg_payload["api_key"]

        with (
            patch(
                f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
            ) as mock_workers,
            patch(
                f"{MODULE}._broadcast_to_workers", new_callable=AsyncMock
            ) as mock_broadcast,
        ):
            mock_workers.return_value = [WORKER_ADDR]
            mock_broadcast.return_value = [
                {"addr": WORKER_ADDR, "status": 200, "data": {}}
            ]
            resp = await client.post(
                "/v1/datasets/unregister",
                json={"dataset_id": "dataset-c"},
                headers=admin_headers(),
            )
        assert resp.status_code == 200

        after_revoke = await client.post(
            "/v1/batches/next",
            json={"request_id": "r-2"},
            headers={"Authorization": f"Bearer {dataset_key}"},
        )
        assert after_revoke.status_code == 401


class TestBatchFetching:
    @pytest.mark.asyncio
    async def test_fetch_batch_routes_to_worker(self, client):
        reg_payload = await _register_dataset(client, dataset_id="dataset-d")
        dataset_key = reg_payload["api_key"]

        mock_client = AsyncMock()
        mock_client.post.return_value = httpx.Response(
            200, json={"request_id": "req-2"}
        )
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_client

        with (
            patch(
                f"{MODULE}._query_router", new_callable=AsyncMock
            ) as mock_query_router,
            patch(f"{MODULE}.httpx.AsyncClient", return_value=mock_cm),
        ):
            mock_query_router.return_value = WORKER_ADDR
            resp = await client.post(
                "/v1/batches/next",
                json={"request_id": "req-2"},
                headers={"Authorization": f"Bearer {dataset_key}"},
            )

        assert resp.status_code == 200
        mock_query_router.assert_awaited_once()
        mock_client.post.assert_awaited_once_with(
            f"{WORKER_ADDR}/fetch_batch",
            json={"dataset_id": "dataset-d", "request_id": "req-2"},
        )

    @pytest.mark.asyncio
    async def test_fetch_batch_returns_worker_response(self, client):
        reg_payload = await _register_dataset(client, dataset_id="dataset-e")
        dataset_key = reg_payload["api_key"]

        expected = {"request_id": "req-3", "batch": [{"x": 1}, {"x": 2}]}
        mock_client = AsyncMock()
        mock_client.post.return_value = httpx.Response(200, json=expected)
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_client

        with (
            patch(
                f"{MODULE}._query_router", new_callable=AsyncMock
            ) as mock_query_router,
            patch(f"{MODULE}.httpx.AsyncClient", return_value=mock_cm),
        ):
            mock_query_router.return_value = WORKER_ADDR
            resp = await client.post(
                "/v1/batches/next",
                json={"request_id": "req-3"},
                headers={"Authorization": f"Bearer {dataset_key}"},
            )

        assert resp.status_code == 200
        assert resp.json() == expected


class TestBroadcastEndpoints:
    @pytest.mark.asyncio
    async def test_epoch_advance_broadcasts_to_all_workers(self, client):
        reg_payload = await _register_dataset(client, dataset_id="dataset-f")
        dataset_key = reg_payload["api_key"]

        with (
            patch(
                f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
            ) as mock_workers,
            patch(
                f"{MODULE}._broadcast_to_workers", new_callable=AsyncMock
            ) as mock_broadcast,
        ):
            mock_workers.return_value = [WORKER_ADDR, WORKER_ADDR_2]
            mock_broadcast.return_value = [
                {"addr": WORKER_ADDR, "status": 200, "data": {}},
                {"addr": WORKER_ADDR_2, "status": 200, "data": {}},
            ]
            resp = await client.post(
                "/v1/epochs/advance",
                json={"epoch": 7},
                headers={"Authorization": f"Bearer {dataset_key}"},
            )

        assert resp.status_code == 200
        assert resp.json()["workers_reset"] == 2

    @pytest.mark.asyncio
    async def test_state_save_broadcasts_to_all_workers(self, client):
        reg_payload = await _register_dataset(client, dataset_id="dataset-g")
        dataset_key = reg_payload["api_key"]

        with (
            patch(
                f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
            ) as mock_workers,
            patch(
                f"{MODULE}._broadcast_to_workers", new_callable=AsyncMock
            ) as mock_broadcast,
        ):
            mock_workers.return_value = [WORKER_ADDR, WORKER_ADDR_2]
            mock_broadcast.return_value = [
                {"addr": WORKER_ADDR, "status": 200, "data": {}},
                {"addr": WORKER_ADDR_2, "status": 200, "data": {}},
            ]
            resp = await client.post(
                "/v1/state/save",
                json={"path": "/tmp/ckpt"},
                headers={"Authorization": f"Bearer {dataset_key}"},
            )

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "path": "/tmp/ckpt"}

    @pytest.mark.asyncio
    async def test_state_load_broadcasts_to_all_workers(self, client):
        reg_payload = await _register_dataset(client, dataset_id="dataset-h")
        dataset_key = reg_payload["api_key"]

        with (
            patch(
                f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
            ) as mock_workers,
            patch(
                f"{MODULE}._broadcast_to_workers", new_callable=AsyncMock
            ) as mock_broadcast,
        ):
            mock_workers.return_value = [WORKER_ADDR, WORKER_ADDR_2]
            mock_broadcast.return_value = [
                {"addr": WORKER_ADDR, "status": 200, "data": {}},
                {"addr": WORKER_ADDR_2, "status": 200, "data": {}},
            ]
            resp = await client.post(
                "/v1/state/load",
                json={"path": "/tmp/ckpt"},
                headers={"Authorization": f"Bearer {dataset_key}"},
            )

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestStatusAndWorkers:
    @pytest.mark.asyncio
    async def test_workers_returns_router_workers(self, client):
        with patch(
            f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
        ) as mock_workers:
            mock_workers.return_value = [WORKER_ADDR, WORKER_ADDR_2]
            resp = await client.get("/v1/workers", headers=admin_headers())

        assert resp.status_code == 200
        assert resp.json() == {
            "workers": [{"addr": WORKER_ADDR}, {"addr": WORKER_ADDR_2}]
        }

    @pytest.mark.asyncio
    async def test_status_returns_dataset_id(self, client):
        reg_payload = await _register_dataset(client, dataset_id="dataset-status")
        dataset_key = reg_payload["api_key"]

        with patch(
            f"{MODULE}._query_router", new_callable=AsyncMock
        ) as mock_query_router:
            mock_query_router.side_effect = RuntimeError("router unavailable")
            resp = await client.get(
                "/v1/status",
                headers={"Authorization": f"Bearer {dataset_key}"},
            )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "ok"
        assert payload["dataset_id"] == "dataset-status"


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_requires_admin_key(self, client):
        resp = await client.post(
            "/v1/shutdown",
            headers={"Authorization": "Bearer ds-not-admin"},
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_shutdown_returns_ok(self, client):
        with patch(
            f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
        ) as mock_workers:
            mock_workers.return_value = []
            resp = await client.post("/v1/shutdown", headers=admin_headers())
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
