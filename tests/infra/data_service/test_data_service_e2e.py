from __future__ import annotations

# pyright: reportMissingImports=false
import os
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from huggingface_hub import snapshot_download

from areal.infra.data_service.gateway.app import create_gateway_app
from areal.infra.data_service.gateway.config import GatewayConfig
from areal.infra.data_service.router.app import create_router_app
from areal.infra.data_service.router.config import RouterConfig
from areal.infra.data_service.worker.app import create_worker_app
from areal.infra.data_service.worker.config import DataWorkerConfig

pytestmark = pytest.mark.slow

ADMIN_KEY = "areal-data-admin"
BATCH_SIZE = 256


def _resolve_path(local: str, hf_id: str, repo_type: str = "dataset") -> str:
    if os.path.exists(local):
        return local
    return snapshot_download(repo_id=hf_id, repo_type=repo_type)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_uvicorn(
    app: object, host: str, port: int
) -> tuple[uvicorn.Server, threading.Thread]:
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread


def _wait_healthy(base_url: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{base_url}/health", timeout=1.0)
            if resp.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    raise TimeoutError(f"Service did not become healthy: {base_url}")


def _admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


def _dataset_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _register_dataset(
    client: httpx.Client,
    *,
    dataset_id: str,
    dataset_path: str,
    dataset_type: str = "rl",
    tokenizer_or_processor_path: str = "",
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    resp = client.post(
        "/v1/datasets/register",
        headers=_admin_headers(),
        json={
            "dataset_id": dataset_id,
            "dataset_path": dataset_path,
            "dataset_type": dataset_type,
            "split": "test",
            "batch_size": batch_size,
            "seed": 42,
            "collate_mode": "identity",
            "shuffle": False,
            "tokenizer_or_processor_path": tokenizer_or_processor_path,
        },
        timeout=120.0,
    )
    assert resp.status_code == 200, f"register failed: {resp.text}"
    payload = resp.json()
    assert payload["dataset_id"] == dataset_id
    return payload


def _unique_dataset_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def data_service_stack(tmp_path_factory: pytest.TempPathFactory):
    dataset_path = _resolve_path(
        "/storage/openpsi/data/gsm8k", "openai/gsm8k", repo_type="dataset"
    )
    geometry3k_path = _resolve_path(
        "/storage/openpsi/data/hiyouga__geometry3k/",
        "hiyouga/geometry3k",
        repo_type="dataset",
    )
    model_path = _resolve_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B",
        "Qwen/Qwen3-0.6B",
        repo_type="model",
    )
    vlm_model_path = _resolve_path(
        "/storage/openpsi/models/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        repo_type="model",
    )

    worker_port = _free_port()
    router_port = _free_port()
    gateway_port = _free_port()

    worker_addr = f"http://127.0.0.1:{worker_port}"
    router_addr = f"http://127.0.0.1:{router_port}"
    gateway_addr = f"http://127.0.0.1:{gateway_port}"
    state_dir = tmp_path_factory.mktemp("data-service-state")

    servers: list[tuple[uvicorn.Server, threading.Thread]] = []

    worker_app = create_worker_app(
        DataWorkerConfig(
            host="127.0.0.1",
            port=worker_port,
            rank=0,
            world_size=1,
            prefetch_batches=2,
            dataloader_num_workers=0,
        )
    )
    servers.append(_start_uvicorn(worker_app, "127.0.0.1", worker_port))
    _wait_healthy(worker_addr)

    router_app = create_router_app(
        RouterConfig(
            host="127.0.0.1",
            port=router_port,
            admin_api_key=ADMIN_KEY,
            poll_interval=0.2,
            routing_strategy="round_robin",
        )
    )
    servers.append(_start_uvicorn(router_app, "127.0.0.1", router_port))
    _wait_healthy(router_addr)

    register_resp = httpx.post(
        f"{router_addr}/register",
        headers=_admin_headers(),
        json={"worker_addr": worker_addr},
        timeout=3.0,
    )
    assert register_resp.status_code == 200

    gateway_app = create_gateway_app(
        GatewayConfig(
            host="127.0.0.1",
            port=gateway_port,
            admin_api_key=ADMIN_KEY,
            router_addr=router_addr,
        )
    )
    servers.append(_start_uvicorn(gateway_app, "127.0.0.1", gateway_port))
    _wait_healthy(gateway_addr)

    yield {
        "worker_addr": worker_addr,
        "router_addr": router_addr,
        "gateway_addr": gateway_addr,
        "state_dir": state_dir,
        "dataset_path": dataset_path,
        "geometry3k_path": geometry3k_path,
        "model_path": model_path,
        "vlm_model_path": vlm_model_path,
    }

    try:
        httpx.post(
            f"{gateway_addr}/v1/shutdown",
            headers=_admin_headers(),
            timeout=3.0,
        )
    except httpx.HTTPError:
        pass

    for server, thread in reversed(servers):
        server.should_exit = True
        thread.join(timeout=5)


@pytest.fixture
def gateway_client(data_service_stack: dict[str, Any]):
    gateway_addr = str(data_service_stack["gateway_addr"])
    with httpx.Client(base_url=gateway_addr, timeout=30.0) as client:
        yield client


class TestServiceHealth:
    def test_all_services_healthy(self, data_service_stack: dict[str, Any]):
        worker_addr = str(data_service_stack["worker_addr"])
        router_addr = str(data_service_stack["router_addr"])
        gateway_addr = str(data_service_stack["gateway_addr"])

        worker = httpx.get(f"{worker_addr}/health", timeout=3.0)
        router = httpx.get(f"{router_addr}/health", timeout=3.0)
        gateway = httpx.get(f"{gateway_addr}/health", timeout=3.0)

        assert worker.status_code == 200
        assert router.status_code == 200
        assert gateway.status_code == 200

    def test_router_shows_registered_worker(self, data_service_stack: dict[str, Any]):
        router_addr = str(data_service_stack["router_addr"])
        worker_addr = str(data_service_stack["worker_addr"])

        resp = httpx.get(
            f"{router_addr}/workers",
            headers=_admin_headers(),
            timeout=3.0,
        )
        assert resp.status_code == 200
        workers = resp.json()["workers"]
        assert workers == [{"addr": worker_addr, "healthy": True}]


class TestDatasetRegistration:
    def test_register_rl_dataset_returns_key(
        self, gateway_client: httpx.Client, data_service_stack: dict[str, Any]
    ):
        payload = _register_dataset(
            gateway_client,
            dataset_id=_unique_dataset_id("register-rl"),
            dataset_path=str(data_service_stack["dataset_path"]),
            dataset_type="rl",
        )
        assert str(payload["api_key"]).startswith("ds-")
        assert payload["dataset_size"] > 0

    def test_register_sft_dataset_returns_key(
        self, gateway_client: httpx.Client, data_service_stack: dict[str, Any]
    ):
        payload = _register_dataset(
            gateway_client,
            dataset_id=_unique_dataset_id("register-sft"),
            dataset_path=str(data_service_stack["dataset_path"]),
            dataset_type="sft",
            tokenizer_or_processor_path=str(data_service_stack["model_path"]),
        )
        assert str(payload["api_key"]).startswith("ds-")
        assert payload["dataset_size"] > 0

    def test_register_returns_valid_steps_per_epoch(
        self, gateway_client: httpx.Client, data_service_stack: dict[str, Any]
    ):
        payload = _register_dataset(
            gateway_client,
            dataset_id=_unique_dataset_id("register-steps"),
            dataset_path=str(data_service_stack["dataset_path"]),
        )
        dataset_size = payload["dataset_size"]
        expected_steps = (dataset_size + BATCH_SIZE - 1) // BATCH_SIZE
        assert payload["steps_per_epoch"] == expected_steps

    def test_register_geometry3k_rl_dataset(
        self, gateway_client: httpx.Client, data_service_stack: dict[str, Any]
    ):
        payload = _register_dataset(
            gateway_client,
            dataset_id=_unique_dataset_id("register-geo3k"),
            dataset_path=str(data_service_stack["geometry3k_path"]),
            dataset_type="rl",
            tokenizer_or_processor_path=str(data_service_stack["vlm_model_path"]),
        )
        assert str(payload["api_key"]).startswith("ds-")
        assert payload["dataset_size"] > 0


class TestBatchFetching:
    def test_fetch_batch_returns_data(
        self, gateway_client: httpx.Client, data_service_stack: dict[str, Any]
    ):
        reg = _register_dataset(
            gateway_client,
            dataset_id=_unique_dataset_id("fetch-one"),
            dataset_path=str(data_service_stack["dataset_path"]),
        )
        api_key = str(reg["api_key"])
        resp = gateway_client.post(
            "/v1/batches/next",
            headers=_dataset_headers(api_key),
            json={"request_id": "fetch-one-req-1"},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["batch_id"] == "fetch-one-req-1"
        assert payload["exhausted"] is False
        assert payload["data"]

    def test_fetch_multiple_batches(
        self, gateway_client: httpx.Client, data_service_stack: dict[str, Any]
    ):
        reg = _register_dataset(
            gateway_client,
            dataset_id=_unique_dataset_id("fetch-multi"),
            dataset_path=str(data_service_stack["dataset_path"]),
        )
        api_key = str(reg["api_key"])

        batches = []
        for idx in range(3):
            resp = gateway_client.post(
                "/v1/batches/next",
                headers=_dataset_headers(api_key),
                json={"request_id": f"fetch-multi-{idx}"},
            )
            assert resp.status_code == 200
            batches.append(resp.json()["data"])

        assert batches[0] != batches[1]
        assert batches[1] != batches[2]

    def test_fetch_batch_exhaustion(
        self, gateway_client: httpx.Client, data_service_stack: dict[str, Any]
    ):
        reg = _register_dataset(
            gateway_client,
            dataset_id=_unique_dataset_id("fetch-exhaust"),
            dataset_path=str(data_service_stack["dataset_path"]),
        )
        api_key = str(reg["api_key"])
        steps = int(reg["steps_per_epoch"])

        exhausted_at = -1
        for idx in range(steps + 2):
            resp = gateway_client.post(
                "/v1/batches/next",
                headers=_dataset_headers(api_key),
                json={"request_id": f"fetch-exhaust-{idx}"},
            )
            assert resp.status_code == 200
            if resp.json()["exhausted"]:
                exhausted_at = idx
                break

        assert exhausted_at == steps

    def test_idempotent_fetch(
        self, gateway_client: httpx.Client, data_service_stack: dict[str, Any]
    ):
        reg = _register_dataset(
            gateway_client,
            dataset_id=_unique_dataset_id("fetch-idempotent"),
            dataset_path=str(data_service_stack["dataset_path"]),
        )
        api_key = str(reg["api_key"])
        req = {"request_id": "same-request-id"}

        first = gateway_client.post(
            "/v1/batches/next",
            headers=_dataset_headers(api_key),
            json=req,
        )
        second = gateway_client.post(
            "/v1/batches/next",
            headers=_dataset_headers(api_key),
            json=req,
        )
        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json() == first.json()


class TestEpochReset:
    def test_epoch_reset_enables_more_fetches(
        self, gateway_client: httpx.Client, data_service_stack: dict[str, Any]
    ):
        reg = _register_dataset(
            gateway_client,
            dataset_id=_unique_dataset_id("epoch-reset"),
            dataset_path=str(data_service_stack["dataset_path"]),
        )
        api_key = str(reg["api_key"])
        steps = int(reg["steps_per_epoch"])

        exhausted = False
        for idx in range(steps + 2):
            resp = gateway_client.post(
                "/v1/batches/next",
                headers=_dataset_headers(api_key),
                json={"request_id": f"epoch-reset-before-{idx}"},
            )
            assert resp.status_code == 200
            exhausted = bool(resp.json()["exhausted"])
            if exhausted:
                break
        assert exhausted is True

        reset = gateway_client.post(
            "/v1/epochs/advance",
            headers=_dataset_headers(api_key),
            json={"epoch": 1},
        )
        assert reset.status_code == 200
        assert reset.json()["workers_reset"] == 1

        after = gateway_client.post(
            "/v1/batches/next",
            headers=_dataset_headers(api_key),
            json={"request_id": "epoch-reset-after"},
        )
        assert after.status_code == 200
        body = after.json()
        assert body["epoch"] == 1
        assert body["exhausted"] is False
        assert body["data"]


class TestStatePersistence:
    def test_state_save_and_load(
        self,
        gateway_client: httpx.Client,
        data_service_stack: dict[str, Any],
    ):
        reg = _register_dataset(
            gateway_client,
            dataset_id=_unique_dataset_id("state"),
            dataset_path=str(data_service_stack["dataset_path"]),
        )
        api_key = str(reg["api_key"])
        state_dir = Path(str(data_service_stack["state_dir"]))

        save = gateway_client.post(
            "/v1/state/save",
            headers=_dataset_headers(api_key),
            json={"path": str(state_dir)},
        )
        assert save.status_code == 200
        assert save.json()["status"] == "ok"
        assert (state_dir / "worker_0.pkl").exists()

        load = gateway_client.post(
            "/v1/state/load",
            headers=_dataset_headers(api_key),
            json={"path": str(state_dir)},
        )
        assert load.status_code == 200
        assert load.json()["status"] == "ok"


class TestDatasetUnregistration:
    def test_unregister_revokes_key(
        self, gateway_client: httpx.Client, data_service_stack: dict[str, Any]
    ):
        dataset_id = _unique_dataset_id("unregister")
        reg = _register_dataset(
            gateway_client,
            dataset_id=dataset_id,
            dataset_path=str(data_service_stack["dataset_path"]),
        )
        api_key = str(reg["api_key"])

        unreg = gateway_client.post(
            "/v1/datasets/unregister",
            headers=_admin_headers(),
            json={"dataset_id": dataset_id},
        )
        assert unreg.status_code == 200

        rejected = gateway_client.post(
            "/v1/batches/next",
            headers=_dataset_headers(api_key),
            json={"request_id": "revoked-key-request"},
        )
        assert rejected.status_code == 401


class TestFullLifecycle:
    @pytest.mark.parametrize("dataset_type", ["rl", "sft"])
    def test_complete_lifecycle(
        self,
        gateway_client: httpx.Client,
        data_service_stack: dict[str, Any],
        dataset_type: str,
    ):
        dataset_id = _unique_dataset_id(f"full-{dataset_type}")
        tokenizer_or_processor_path = (
            str(data_service_stack["model_path"]) if dataset_type == "sft" else ""
        )
        reg = _register_dataset(
            gateway_client,
            dataset_id=dataset_id,
            dataset_path=str(data_service_stack["dataset_path"]),
            dataset_type=dataset_type,
            tokenizer_or_processor_path=tokenizer_or_processor_path,
            batch_size=BATCH_SIZE,
        )
        api_key = str(reg["api_key"])
        steps = int(reg["steps_per_epoch"])
        state_dir = Path(str(data_service_stack["state_dir"])) / f"{dataset_id}-state"

        exhausted = False
        for idx in range(steps + 2):
            resp = gateway_client.post(
                "/v1/batches/next",
                headers=_dataset_headers(api_key),
                json={"request_id": f"full-fetch-{idx}"},
            )
            assert resp.status_code == 200
            exhausted = bool(resp.json()["exhausted"])
            if exhausted:
                break
        assert exhausted is True

        reset = gateway_client.post(
            "/v1/epochs/advance",
            headers=_dataset_headers(api_key),
            json={"epoch": 2},
        )
        assert reset.status_code == 200
        assert reset.json()["workers_reset"] == 1

        after_reset = gateway_client.post(
            "/v1/batches/next",
            headers=_dataset_headers(api_key),
            json={"request_id": "full-after-reset"},
        )
        assert after_reset.status_code == 200
        assert after_reset.json()["epoch"] == 2
        assert after_reset.json()["exhausted"] is False

        save = gateway_client.post(
            "/v1/state/save",
            headers=_dataset_headers(api_key),
            json={"path": str(state_dir)},
        )
        assert save.status_code == 200

        load = gateway_client.post(
            "/v1/state/load",
            headers=_dataset_headers(api_key),
            json={"path": str(state_dir)},
        )
        assert load.status_code == 200

        unreg = gateway_client.post(
            "/v1/datasets/unregister",
            headers=_admin_headers(),
            json={"dataset_id": dataset_id},
        )
        assert unreg.status_code == 200

        rejected = gateway_client.post(
            "/v1/batches/next",
            headers=_dataset_headers(api_key),
            json={"request_id": "full-after-unregister"},
        )
        assert rejected.status_code == 401

    def test_geometry3k_rl_lifecycle(
        self,
        gateway_client: httpx.Client,
        data_service_stack: dict[str, Any],
    ):
        dataset_id = _unique_dataset_id("full-geo3k")
        reg = _register_dataset(
            gateway_client,
            dataset_id=dataset_id,
            dataset_path=str(data_service_stack["geometry3k_path"]),
            dataset_type="rl",
            tokenizer_or_processor_path=str(data_service_stack["vlm_model_path"]),
            batch_size=BATCH_SIZE,
        )
        api_key = str(reg["api_key"])
        steps = int(reg["steps_per_epoch"])
        assert steps > 0

        resp = gateway_client.post(
            "/v1/batches/next",
            headers=_dataset_headers(api_key),
            json={"request_id": "geo3k-fetch-0"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]

        gateway_client.post(
            "/v1/datasets/unregister",
            headers=_admin_headers(),
            json={"dataset_id": dataset_id},
        )
