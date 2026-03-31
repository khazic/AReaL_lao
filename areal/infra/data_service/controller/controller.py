"""DataController — orchestrator for the distributed data loading service.

Manages the full lifecycle: create RPCGuard workers → fork DataWorkers,
Router, Gateway → register datasets → serve batches → shutdown.

Follows the same patterns as ``GatewayInferenceController``.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from areal.api.scheduler_api import Scheduler, Worker

from areal.infra.data_service.controller.config import DataServiceConfig
from areal.utils import logging
from areal.utils.network import format_hostport

logger = logging.getLogger("DataController")


class DataController:
    """Controller for the distributed data loading service.

    API follows ``TrainController`` / ``GatewayInferenceController`` patterns:
    ``__init__(config, scheduler)`` then ``initialize(role, ...)``.
    """

    _GUARD_SUFFIX = "-data"
    _ADMIN_API_KEY = "areal-data-admin"

    def __init__(
        self,
        config: DataServiceConfig,
        scheduler: Scheduler,
    ) -> None:
        self.config = config
        self.scheduler = scheduler

        self.workers: list[Worker] = []
        self._worker_role: str = ""

        self._gateway_addr: str = ""
        self._router_addr: str = ""
        self._worker_addrs: list[str] = []

        self._service_roles: list[str] = []
        self._forked_services: list[tuple[str, str, int]] = []

        self._admin_api_key: str = self._ADMIN_API_KEY

        self._datasets: dict[str, DatasetHandle] = {}

    # -- Initialize --------------------------------------------------------

    def initialize(
        self,
        role: str,
        dp_size: int = 1,
        **kwargs: Any,
    ) -> None:
        from areal.infra.utils.concurrent import run_async_task

        self._worker_role = role
        run_async_task(self._async_initialize, dp_size, **kwargs)

    async def _async_initialize(
        self,
        dp_size: int,
        **kwargs: Any,
    ) -> None:
        import requests

        from areal.api.cli_args import SchedulingSpec
        from areal.api.scheduler_api import Job

        cfg = self.config

        # ==================================================================
        # Step 0: Create RPCGuard workers (CPU-only)
        # ==================================================================
        guard_spec = SchedulingSpec(
            cpu=cfg.worker_cpu,
            mem=cfg.worker_mem,
            gpu=0,
        )
        guard_spec.cmd = "python -m areal.infra.data_service.guard"

        guard_role = f"{self._worker_role}{self._GUARD_SUFFIX}"

        guard_job = Job(
            replicas=dp_size,
            tasks=[guard_spec for _ in range(dp_size)],
            scheduling_strategy=cfg.scheduling_strategy,
            role=guard_role,
        )

        self.scheduler.create_workers(job=guard_job)
        self._service_roles.append(guard_role)
        guard_workers = self.scheduler.get_workers(role=guard_role)
        self.workers = guard_workers
        logger.info("RPCGuard workers ready: %s", [w.id for w in guard_workers])

        # ==================================================================
        # Step 1: Fork DataWorkers on each guard
        # ==================================================================
        for rank, worker in enumerate(guard_workers):
            guard_addr = (
                f"http://{format_hostport(worker.ip, int(worker.worker_ports[0]))}"
            )

            worker_cmd = [
                sys.executable,
                "-m",
                "areal.infra.data_service.worker",
                "--rank",
                str(rank),
                "--world-size",
                str(dp_size),
                "--prefetch-batches",
                str(cfg.prefetch_batches),
            ]

            worker_host, worker_port = self._fork_on_guard(
                guard_addr=guard_addr,
                role="data-worker",
                worker_index=rank,
                raw_cmd=worker_cmd,
            )
            self._worker_addrs.append(
                f"http://{format_hostport(worker_host, worker_port)}"
            )

        logger.info("DataWorkers: %s", self._worker_addrs)

        # ==================================================================
        # Step 2: Fork Router on guard 0
        # ==================================================================
        guard_addr_0 = f"http://{format_hostport(guard_workers[0].ip, int(guard_workers[0].worker_ports[0]))}"

        router_cmd = [
            sys.executable,
            "-m",
            "areal.infra.data_service.router",
            "--admin-api-key",
            self._admin_api_key,
            "--routing-strategy",
            cfg.routing_strategy,
        ]

        router_host, router_port = self._fork_on_guard(
            guard_addr=guard_addr_0,
            role="data-router",
            worker_index=0,
            raw_cmd=router_cmd,
        )
        self._router_addr = f"http://{format_hostport(router_host, router_port)}"
        logger.info("Router: %s", self._router_addr)

        # ==================================================================
        # Step 3: Fork Gateway on guard 0
        # ==================================================================
        gw_cmd = [
            sys.executable,
            "-m",
            "areal.infra.data_service.gateway",
            "--admin-api-key",
            self._admin_api_key,
            "--router-addr",
            self._router_addr,
            "--forward-timeout",
            str(60.0),
        ]

        gw_host, gw_port = self._fork_on_guard(
            guard_addr=guard_addr_0,
            role="data-gateway",
            worker_index=0,
            raw_cmd=gw_cmd,
        )
        self._gateway_addr = f"http://{format_hostport(gw_host, gw_port)}"
        logger.info("Gateway: %s", self._gateway_addr)

        # ==================================================================
        # Step 4: Register workers with Router
        # ==================================================================
        for worker_addr in self._worker_addrs:
            resp = requests.post(
                f"{self._router_addr}/register",
                json={"worker_addr": worker_addr},
                headers={"Authorization": f"Bearer {self._admin_api_key}"},
                timeout=5,
            )
            resp.raise_for_status()
            logger.info("Registered DataWorker %s in router", worker_addr)

        logger.info("DataController initialized with %d workers", dp_size)

    # -- Register / Unregister Datasets ------------------------------------

    def register_dataset(
        self,
        dataset_id: str,
        dataset_path: str,
        dataset_type: str,
        dataset_kwargs: dict[str, Any] | None = None,
        tokenizer_path: str = "",
        processor_path: str = "",
        batch_size: int = 32,
        split: str = "train",
        seed: int = 42,
        shuffle: bool = True,
        collate_mode: str = "identity",
        max_length: int = 2048,
        experiment_name: str = "",
        trial_name: str = "",
        fileroot: str = "",
        **kwargs: Any,
    ) -> DatasetHandle:
        """Register a dataset with the service.

        POST /v1/datasets/register on Gateway. Returns DatasetHandle that
        duck-types as StatefulDataLoader.
        """
        import requests

        payload = {
            "dataset_id": dataset_id,
            "dataset_path": dataset_path,
            "dataset_type": dataset_type,
            "split": split,
            "tokenizer_path": tokenizer_path,
            "processor_path": processor_path,
            "batch_size": batch_size,
            "seed": seed,
            "max_length": max_length,
            "shuffle": shuffle,
            "collate_mode": collate_mode,
            "experiment_name": experiment_name,
            "trial_name": trial_name,
            "fileroot": fileroot,
            "dataset_kwargs": dataset_kwargs or {},
        }

        resp = requests.post(
            f"{self._gateway_addr}/v1/datasets/register",
            json=payload,
            headers={"Authorization": f"Bearer {self._admin_api_key}"},
            timeout=self.config.setup_timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        handle = DatasetHandle(
            controller=self,
            api_key=data["api_key"],
            dataset_id=data["dataset_id"],
            batch_size=batch_size,
            steps_per_epoch=data["steps_per_epoch"],
            experiment_name=experiment_name,
            trial_name=trial_name,
            fileroot=fileroot,
        )

        self._datasets[data["api_key"]] = handle
        logger.info(
            "Registered dataset %s: steps_per_epoch=%d, workers=%d",
            dataset_id,
            data["steps_per_epoch"],
            data["num_workers"],
        )
        return handle

    def unregister_dataset(self, dataset_id: str) -> None:
        """Unregister a dataset from the service."""
        import requests

        resp = requests.post(
            f"{self._gateway_addr}/v1/datasets/unregister",
            json={"dataset_id": dataset_id},
            headers={"Authorization": f"Bearer {self._admin_api_key}"},
            timeout=30,
        )
        resp.raise_for_status()

        to_remove = [
            k for k, v in self._datasets.items() if v._dataset_id == dataset_id
        ]
        for k in to_remove:
            del self._datasets[k]

        logger.info("Unregistered dataset %s", dataset_id)

    # -- Batch cleanup -----------------------------------------------------

    def clear_batches(self) -> None:
        """Clear batch caches and tensor stores on all data workers.

        Called by trainers after each training step, alongside
        ``actor.clear_batches()``, to free memory held by the data
        service instead of relying on TTL-based eviction.
        """
        import requests

        for addr in self._worker_addrs:
            try:
                requests.delete(f"{addr}/data/clear", timeout=10)
            except Exception:
                logger.debug("Failed to clear batches on %s", addr)

    # -- Destroy -----------------------------------------------------------

    def destroy(self) -> None:
        """Shutdown service: unload all datasets, kill services, delete workers."""
        import requests

        # 1. POST /v1/shutdown to gateway (best-effort)
        if self._gateway_addr:
            try:
                requests.post(
                    f"{self._gateway_addr}/v1/shutdown",
                    headers={"Authorization": f"Bearer {self._admin_api_key}"},
                    timeout=10,
                )
            except Exception:
                logger.warning(
                    "Error sending shutdown to gateway: %s",
                    traceback.format_exc(),
                )

        # 2. Kill forked services in reverse order via guard /kill_forked_worker
        for guard_addr, role, worker_index in reversed(self._forked_services):
            try:
                self._kill_forked_service(guard_addr, role, worker_index)
            except Exception:
                logger.error(
                    "Error killing forked service %s/%d: %s",
                    role,
                    worker_index,
                    traceback.format_exc(),
                )
        self._forked_services.clear()

        # 3. Delete guard workers via scheduler
        for role in reversed(self._service_roles):
            try:
                self.scheduler.delete_workers(role=role)
                logger.info("Workers deleted for role: %s", role)
            except Exception:
                logger.error(
                    "Error deleting workers for role %s: %s",
                    role,
                    traceback.format_exc(),
                )

        self._service_roles.clear()
        self.workers.clear()
        self._worker_addrs.clear()
        self._router_addr = ""
        self._gateway_addr = ""
        self._datasets.clear()

    # -- Internal HTTP helpers ---------------------------------------------

    def _fork_on_guard(
        self,
        guard_addr: str,
        role: str,
        worker_index: int,
        raw_cmd: list[str],
        health_path: str = "/health",
    ) -> tuple[str, int]:
        """Fork a process on a RPCGuard worker via ``/fork`` with ``raw_cmd``.

        Returns ``(host, port)`` of the forked service and records the entry
        in ``_forked_services`` for cleanup.
        """
        import requests

        resp = requests.post(
            f"{guard_addr}/alloc_ports",
            json={"count": 1},
            timeout=30,
        )
        resp.raise_for_status()
        port_data = resp.json()
        host = port_data["host"]
        port = port_data["ports"][0]

        cmd = list(raw_cmd) + ["--host", host, "--port", str(port)]

        resp = requests.post(
            f"{guard_addr}/fork",
            json={
                "role": role,
                "worker_index": worker_index,
                "raw_cmd": cmd,
            },
            timeout=30,
        )
        resp.raise_for_status()

        self._forked_services.append((guard_addr, role, worker_index))

        addr = f"http://{format_hostport(host, port)}"
        self._wait_for_service(f"{addr}{health_path}", role)

        return host, port

    def _wait_for_service(
        self, url: str, name: str, timeout: float | None = None
    ) -> None:
        """Wait for a service to become healthy."""
        import requests

        timeout = timeout or self.config.setup_timeout
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                resp = requests.get(url, timeout=2)
                if resp.status_code == 200:
                    logger.info("%s is ready at %s", name, url)
                    return
            except requests.RequestException:
                pass
            time.sleep(0.1)
        raise TimeoutError(f"{name} did not become healthy at {url} within {timeout}s")

    def _kill_forked_service(
        self, guard_addr: str, role: str, worker_index: int
    ) -> None:
        import requests

        try:
            resp = requests.post(
                f"{guard_addr}/kill_forked_worker",
                json={"role": role, "worker_index": worker_index},
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info("Killed forked service %s/%d", role, worker_index)
            else:
                logger.warning(
                    "Failed to kill forked service %s/%d: %s",
                    role,
                    worker_index,
                    resp.text,
                )
        except requests.RequestException as exc:
            logger.error(
                "Error killing forked service %s/%d: %s", role, worker_index, exc
            )

    def _gateway_post(
        self,
        endpoint: str,
        api_key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Make a synchronous HTTP POST to the gateway with Bearer auth."""
        import requests

        url = f"{self._gateway_addr}{endpoint}"
        resp = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Gateway {endpoint} returned {resp.status_code}: {resp.text}"
            )
        return resp.json()


class DatasetHandle:
    """Handle for a registered dataset.

    Duck-type compatible with ``StatefulDataLoader``: implements
    ``__len__``, ``__iter__``, ``state_dict``, ``load_state_dict``,
    ``.batch_size``, and ``.sampler.set_epoch()``.
    """

    def __init__(
        self,
        controller: DataController,
        api_key: str,
        dataset_id: str,
        batch_size: int,
        steps_per_epoch: int,
        experiment_name: str,
        trial_name: str,
        fileroot: str,
    ):
        self._controller = controller
        self._api_key = api_key
        self._dataset_id = dataset_id
        self._batch_size = batch_size
        self._steps_per_epoch = steps_per_epoch
        self._epoch = 0

        # Save path following Saver convention
        from areal.utils.saver import Saver

        if experiment_name and trial_name and fileroot:
            self._save_root = os.path.join(
                Saver.get_save_root(experiment_name, trial_name, fileroot),
                "recover_info",
                "data_service",
                dataset_id,
            )
        else:
            self._save_root = ""

        # Sampler proxy for cycle_dataloader compatibility
        self.sampler = _SamplerProxy(self)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def __len__(self) -> int:
        return self._steps_per_epoch

    def __iter__(self):
        """Yield batches for one epoch via Gateway."""
        from areal.infra.rpc.serialization import deserialize_value

        for _ in range(self._steps_per_epoch):
            resp = self._controller._gateway_post(
                "/v1/batches/next",
                self._api_key,
                {"request_id": str(uuid.uuid4())},
            )
            yield deserialize_value(resp["data"])
        self._epoch += 1

    def state_dict(self) -> dict:
        """Save workers' state to shared FS, return path pointer."""
        if self._save_root:
            self._controller._gateway_post(
                "/v1/state/save",
                self._api_key,
                {"path": self._save_root},
            )
        return {
            "type": "data_service",
            "path": self._save_root,
            "dataset_id": self._dataset_id,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load workers' state from shared FS path."""
        if state.get("type") != "data_service":
            logger.warning("Incompatible dataloader state format, skipping")
            return
        path = state.get("path", "")
        if path:
            self._controller._gateway_post(
                "/v1/state/load",
                self._api_key,
                {"path": path},
            )


class _SamplerProxy:
    """Minimal sampler-like object for ``cycle_dataloader`` compatibility.

    Provides ``set_epoch()`` method that broadcasts epoch advance to all
    workers via the gateway.
    """

    def __init__(self, handle: DatasetHandle):
        self._handle = handle

    def set_epoch(self, epoch: int) -> None:
        self._handle._controller._gateway_post(
            "/v1/epochs/advance",
            self._handle._api_key,
            {"epoch": epoch},
        )
        self._handle._epoch = epoch
