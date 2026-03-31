from __future__ import annotations

import uuid
from typing import Any, cast
from unittest.mock import MagicMock, patch

from areal.api.cli_args import (
    DataServiceConfig,
    PPOConfig,
    SchedulingStrategy,
    SchedulingStrategyType,
    SFTConfig,
)
from areal.infra.data_service.controller.controller import (
    DataController,
    DatasetHandle,
    _SamplerProxy,
)
from areal.utils.data import cycle_dataloader


def _make_handle(steps_per_epoch: int = 3, batch_size: int = 4) -> DatasetHandle:
    controller = MagicMock()
    controller._gateway_post = MagicMock(
        return_value={
            "batch_id": str(uuid.uuid4()),
            "epoch": 0,
            "exhausted": False,
            "data": [{"text": "hello", "label": 1}],
        }
    )
    controller._gateway_addr = "http://fake-gateway:8090"

    return DatasetHandle(
        controller=controller,
        api_key="ds-test-key",
        dataset_id="test-train",
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        experiment_name="",
        trial_name="",
        fileroot="",
    )


class TestDataServiceConfig:
    def test_default_values(self):
        cfg = DataServiceConfig()

        assert cfg.worker_cpu == 4
        assert cfg.worker_mem == 16
        assert cfg.prefetch_batches == 2
        assert cfg.setup_timeout == 120.0
        assert cfg.routing_strategy == "round_robin"
        assert isinstance(cfg.scheduling_strategy, SchedulingStrategy)
        assert cfg.scheduling_strategy.type == "separation"
        assert cfg.scheduling_strategy.target is None

    def test_custom_values(self):
        cfg = DataServiceConfig(
            worker_cpu=8,
            worker_mem=64,
            prefetch_batches=5,
            setup_timeout=300.0,
            routing_strategy="sticky",
        )

        assert cfg.worker_cpu == 8
        assert cfg.worker_mem == 64
        assert cfg.prefetch_batches == 5
        assert cfg.setup_timeout == 300.0
        assert cfg.routing_strategy == "sticky"

    def test_scheduling_strategy_colocation(self):
        cfg = DataServiceConfig(
            scheduling_strategy=SchedulingStrategy(
                type=SchedulingStrategyType.colocation, target="rollout"
            ),
        )

        assert cfg.scheduling_strategy.type == SchedulingStrategyType.colocation
        assert cfg.scheduling_strategy.target == "rollout"
        assert cfg.scheduling_strategy.fork is True

    def test_scheduling_strategy_separation(self):
        cfg = DataServiceConfig(
            scheduling_strategy=SchedulingStrategy(type="separation"),
        )

        assert cfg.scheduling_strategy.type == "separation"
        assert cfg.scheduling_strategy.target is None

    def test_config_in_base_experiment_config(self):
        from areal.api.cli_args import TrainDatasetConfig

        cfg = TrainDatasetConfig(path="dummy", type="rl")
        assert hasattr(cfg, "data_service")
        assert cfg.data_service is None

    def test_config_in_ppo_config(self):
        cfg = PPOConfig(experiment_name="exp", trial_name="trial")
        assert hasattr(cfg.train_dataset, "data_service")
        assert cfg.train_dataset.data_service is None

    def test_config_in_sft_config(self):
        cfg = SFTConfig(experiment_name="exp", trial_name="trial")
        assert hasattr(cfg.train_dataset, "data_service")
        assert cfg.train_dataset.data_service is None

    def test_config_in_rw_config(self):
        from areal.api.cli_args import TrainDatasetConfig

        assert "data_service" in TrainDatasetConfig.__dataclass_fields__
        assert TrainDatasetConfig.__dataclass_fields__["data_service"].default is None


class TestDataControllerInit:
    def test_init_stores_config(self):
        cfg = DataServiceConfig()
        scheduler = MagicMock()

        controller = DataController(cfg, scheduler)

        assert controller.config is cfg

    def test_init_stores_scheduler(self):
        cfg = DataServiceConfig()
        scheduler = MagicMock()

        controller = DataController(cfg, scheduler)

        assert controller.scheduler is scheduler

    def test_init_empty_state(self):
        controller = DataController(DataServiceConfig(), MagicMock())

        assert controller.workers == []
        assert controller._gateway_addr == ""
        assert controller._datasets == {}


class TestDataControllerGatewayPost:
    def test_gateway_post_sends_bearer_auth(self):
        controller = DataController(DataServiceConfig(), MagicMock())
        controller._gateway_addr = "http://gateway"

        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"ok": True}

        with patch("requests.post", return_value=response) as mock_post:
            result = controller._gateway_post("/v1/test", "api-key", {"x": 1})

        assert result == {"ok": True}
        _, kwargs = mock_post.call_args
        assert kwargs["headers"] == {"Authorization": "Bearer api-key"}

    def test_gateway_post_sends_json_payload(self):
        controller = DataController(DataServiceConfig(), MagicMock())
        controller._gateway_addr = "http://gateway"

        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"ok": True}

        with patch("requests.post", return_value=response) as mock_post:
            controller._gateway_post("/v1/test", "api-key", {"payload": "value"})

        _, kwargs = mock_post.call_args
        assert kwargs["json"] == {"payload": "value"}

    def test_gateway_post_raises_on_error(self):
        controller = DataController(DataServiceConfig(), MagicMock())
        controller._gateway_addr = "http://gateway"

        response = MagicMock()
        response.status_code = 500
        response.text = "boom"

        with patch("requests.post", return_value=response):
            try:
                controller._gateway_post("/v1/test", "api-key", {})
            except RuntimeError as exc:
                assert "returned 500" in str(exc)
            else:
                raise AssertionError("Expected RuntimeError for gateway error response")


class TestDatasetHandle:
    def test_len_returns_steps_per_epoch(self):
        handle = _make_handle(steps_per_epoch=7)
        assert len(handle) == 7

    def test_batch_size_property(self):
        handle = _make_handle(batch_size=16)
        assert handle.batch_size == 16

    def test_iter_calls_gateway_for_each_batch(self):
        handle = _make_handle(steps_per_epoch=4)

        _ = list(handle)

        controller = cast(Any, handle._controller)
        assert controller._gateway_post.call_count == 4
        for call in controller._gateway_post.call_args_list:
            assert call.args[0] == "/v1/batches/next"

    def test_iter_deserializes_response_data(self):
        handle = _make_handle(steps_per_epoch=2)

        with patch(
            "areal.infra.rpc.serialization.deserialize_value",
            side_effect=lambda value: {"decoded": value},
        ) as mock_deserialize:
            batches = list(handle)

        assert len(batches) == 2
        assert all("decoded" in batch for batch in batches)
        assert mock_deserialize.call_count == 2

    def test_state_dict_returns_data_service_marker(self):
        handle = _make_handle()

        state = handle.state_dict()

        assert state["type"] == "data_service"
        assert state["dataset_id"] == "test-train"

    def test_state_dict_posts_to_gateway_save(self):
        handle = _make_handle()

        _ = handle.state_dict()

        controller = cast(Any, handle._controller)
        endpoints = [call.args[0] for call in controller._gateway_post.call_args_list]
        assert "/v1/state/save" not in endpoints

        handle_with_save_path = DatasetHandle(
            controller=controller,
            api_key="ds-test-key",
            dataset_id="test-train",
            batch_size=4,
            steps_per_epoch=1,
            experiment_name="exp",
            trial_name="trial",
            fileroot="/tmp",
        )
        _ = handle_with_save_path.state_dict()
        assert controller._gateway_post.call_args_list[-1].args[0] == "/v1/state/save"

    def test_load_state_dict_posts_to_gateway_load(self):
        handle = _make_handle()

        handle.load_state_dict({"type": "data_service", "path": "/tmp/recover"})

        controller = cast(Any, handle._controller)
        assert controller._gateway_post.call_args.args[0] == "/v1/state/load"
        assert controller._gateway_post.call_args.args[2] == {"path": "/tmp/recover"}

    def test_load_state_dict_skips_incompatible_format(self):
        handle = _make_handle()

        with patch(
            "areal.infra.data_service.controller.controller.logger.warning"
        ) as mock_warning:
            handle.load_state_dict({"type": "stateful_dataloader"})

        cast(Any, handle._controller)._gateway_post.assert_not_called()
        mock_warning.assert_called_once()

    def test_has_sampler_attribute(self):
        handle = _make_handle()
        assert hasattr(handle, "sampler")


class TestSamplerProxy:
    def test_set_epoch_calls_gateway(self):
        handle = _make_handle()
        sampler = _SamplerProxy(handle)

        sampler.set_epoch(5)

        cast(Any, handle._controller)._gateway_post.assert_called_once_with(
            "/v1/epochs/advance",
            "ds-test-key",
            {"epoch": 5},
        )

    def test_set_epoch_updates_handle_epoch(self):
        handle = _make_handle()
        sampler = _SamplerProxy(handle)

        sampler.set_epoch(3)

        assert handle._epoch == 3


class TestDatasetHandleDuckType:
    def test_cycle_dataloader_calls_set_epoch(self):
        handle = _make_handle(steps_per_epoch=1)

        _ = list(cycle_dataloader(handle, num_cycles=2))

        endpoints = [
            call.args[0]
            for call in cast(Any, handle._controller)._gateway_post.call_args_list
        ]
        epoch_advances = [ep for ep in endpoints if ep == "/v1/epochs/advance"]
        assert len(epoch_advances) == 2

    def test_cycle_dataloader_yields_batches(self):
        handle = _make_handle(steps_per_epoch=2)

        with patch(
            "areal.infra.rpc.serialization.deserialize_value",
            side_effect=lambda value: value,
        ):
            batches = list(cycle_dataloader(handle, num_cycles=2))

        assert len(batches) == 4
        assert all(isinstance(batch, list) for batch in batches)

    def test_dataset_handle_supports_state_dict_protocol(self):
        handle = _make_handle()

        state = handle.state_dict()
        handle.load_state_dict(state)

        assert state["type"] == "data_service"

    def test_dataset_handle_supports_len_protocol(self):
        handle = _make_handle(steps_per_epoch=5)
        assert len(handle) == 5
