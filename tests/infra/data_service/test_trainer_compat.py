from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock

from areal.api.cli_args import (
    DataServiceConfig,
    PPOConfig,
    RWConfig,
    SchedulingStrategy,
    SchedulingStrategyType,
    SFTConfig,
    TrainEngineConfig,
)
from areal.utils.data import cycle_dataloader


class _SamplerWithSetEpoch:
    def __init__(self):
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


class _FiniteDataloader:
    def __init__(self, batches_per_epoch: int = 2, with_sampler: bool = True):
        self.batches_per_epoch = batches_per_epoch
        if with_sampler:
            self.sampler = _SamplerWithSetEpoch()

    def __iter__(self):
        for i in range(self.batches_per_epoch):
            yield {"batch": i}


class _DataloaderWithSamplerNoSetEpoch:
    class _Sampler:
        pass

    def __init__(self):
        self.sampler = self._Sampler()

    def __iter__(self):
        yield {"batch": 0}


class TestCycleDataloaderCompat:
    def test_cycle_dataloader_with_stateful_dataloader(self):
        dataloader = _FiniteDataloader(batches_per_epoch=2, with_sampler=True)

        batches = list(cycle_dataloader(dataloader, num_cycles=2))

        assert len(batches) == 4
        assert dataloader.sampler.epochs == [0, 1]

    def test_cycle_dataloader_with_dataset_handle(self):
        dataloader = _FiniteDataloader(batches_per_epoch=3, with_sampler=True)

        batches = list(cycle_dataloader(dataloader, num_cycles=2))

        assert len(batches) == 6
        assert dataloader.sampler.epochs == [0, 1]

    def test_cycle_dataloader_without_sampler(self):
        dataloader = _FiniteDataloader(batches_per_epoch=2, with_sampler=False)

        batches = list(cycle_dataloader(dataloader, num_cycles=2))

        assert len(batches) == 4

    def test_cycle_dataloader_with_sampler_without_set_epoch(self):
        dataloader = _DataloaderWithSamplerNoSetEpoch()

        batches = list(cycle_dataloader(dataloader, num_cycles=2))

        assert len(batches) == 2

    def test_cycle_dataloader_num_cycles_limits_iterations(self):
        dataloader = _FiniteDataloader(batches_per_epoch=1, with_sampler=True)

        batches = list(cycle_dataloader(dataloader, num_cycles=2))

        assert len(batches) == 2
        assert dataloader.sampler.epochs == [0, 1]


class TestDataServiceConfigDefault:
    def test_ppo_config_data_service_enabled_by_default(self):
        cfg = PPOConfig(experiment_name="exp", trial_name="trial")
        assert isinstance(cfg.train_dataset.data_service, DataServiceConfig)

    def test_sft_config_data_service_enabled_by_default(self):
        cfg = SFTConfig(experiment_name="exp", trial_name="trial")
        assert isinstance(cfg.train_dataset.data_service, DataServiceConfig)

    def test_rw_config_data_service_enabled_by_default(self):
        cfg = RWConfig(
            experiment_name="exp",
            trial_name="trial",
            actor=TrainEngineConfig(is_critic=True),
        )
        assert isinstance(cfg.train_dataset.data_service, DataServiceConfig)

    def test_data_service_can_be_disabled(self):
        from areal.api.cli_args import TrainDatasetConfig

        ds_cfg = TrainDatasetConfig(path="dummy", type="rl", data_service=None)
        assert ds_cfg.data_service is None

    def test_ppo_config_with_data_service(self):
        from areal.api.cli_args import TrainDatasetConfig

        ds_cfg = TrainDatasetConfig(
            path="dummy", type="rl", data_service=DataServiceConfig()
        )
        cfg = PPOConfig(
            experiment_name="exp",
            trial_name="trial",
            train_dataset=ds_cfg,
        )

        assert cfg.train_dataset.data_service is not None

    def test_config_data_service_fields_accessible(self):
        from areal.api.cli_args import TrainDatasetConfig

        ds_cfg = TrainDatasetConfig(
            path="dummy",
            type="sft",
            data_service=DataServiceConfig(worker_cpu=12, prefetch_batches=7),
        )
        cfg = SFTConfig(
            experiment_name="exp",
            trial_name="trial",
            train_dataset=ds_cfg,
        )

        assert cfg.train_dataset.data_service is not None
        assert cfg.train_dataset.data_service.worker_cpu == 12
        assert cfg.train_dataset.data_service.prefetch_batches == 7
        assert cfg.train_dataset.data_service.worker_mem == 16

    def test_config_data_service_scheduling_strategy_default(self):
        ds_cfg = DataServiceConfig()

        assert isinstance(ds_cfg.scheduling_strategy, SchedulingStrategy)
        assert ds_cfg.scheduling_strategy.type == SchedulingStrategyType.colocation
        assert ds_cfg.scheduling_strategy.target == "actor"

    def test_config_data_service_scheduling_strategy_colocation(self):
        from areal.api.cli_args import TrainDatasetConfig

        ds_cfg = TrainDatasetConfig(
            path="dummy",
            type="rl",
            data_service=DataServiceConfig(
                scheduling_strategy=SchedulingStrategy(
                    type=SchedulingStrategyType.colocation, target="rollout"
                ),
            ),
        )
        cfg = PPOConfig(
            experiment_name="exp",
            trial_name="trial",
            train_dataset=ds_cfg,
        )

        sched = cfg.train_dataset.data_service.scheduling_strategy
        assert sched.type == SchedulingStrategyType.colocation
        assert sched.target == "rollout"
        assert sched.fork is True

    def test_config_data_service_scheduling_strategy_actor_target(self):
        from areal.api.cli_args import TrainDatasetConfig

        ds_cfg = TrainDatasetConfig(
            path="dummy",
            type="sft",
            data_service=DataServiceConfig(
                scheduling_strategy=SchedulingStrategy(
                    type=SchedulingStrategyType.colocation, target="actor"
                ),
            ),
        )
        cfg = SFTConfig(
            experiment_name="exp",
            trial_name="trial",
            train_dataset=ds_cfg,
        )

        sched = cfg.train_dataset.data_service.scheduling_strategy
        assert sched.type == SchedulingStrategyType.colocation
        assert sched.target == "actor"


class TestTrainerDataServicePath:
    def test_data_controller_importable(self):
        from areal.infra.data_service import DataController, DatasetHandle

        assert DataController is not None
        assert DatasetHandle is not None

    def test_data_controller_config_importable_from_cli_args(self):
        from areal.api.cli_args import DataServiceConfig

        assert DataServiceConfig is not None

    def test_dataset_handle_has_required_protocol(self):
        from areal.infra.data_service import DatasetHandle

        handle = DatasetHandle(
            controller=MagicMock(),
            api_key="key",
            dataset_id="train",
            batch_size=4,
            steps_per_epoch=2,
            experiment_name="",
            trial_name="",
            fileroot="",
        )

        assert hasattr(handle, "__len__")
        assert hasattr(handle, "__iter__")
        assert hasattr(handle, "state_dict")
        assert hasattr(handle, "load_state_dict")
        assert hasattr(handle, "batch_size")
        assert hasattr(handle, "sampler")


class TestRecoveryCompat:
    def test_recover_handler_dump_accepts_any_dataloader(self):
        from areal.utils.recover import RecoverHandler

        sig = inspect.signature(RecoverHandler.dump)
        annotation = sig.parameters["dataloader"].annotation

        assert annotation in (Any, "Any", inspect.Parameter.empty)

    def test_recover_handler_load_accepts_any_dataloader(self):
        from areal.utils.recover import RecoverHandler

        sig = inspect.signature(RecoverHandler.load)
        annotation = sig.parameters["dataloader"].annotation

        assert annotation in (Any, "Any", inspect.Parameter.empty)


class TestEmptyDataLoaderCompat:
    def test_empty_dataloader_still_works(self):
        from areal.trainer.rl_trainer import _EmptyDataLoader

        dataloader = _EmptyDataLoader(batch_size=2, steps_per_epoch=3)

        assert len(dataloader) == 3
        assert dataloader.state_dict() == {}
        dataloader.load_state_dict({"ignored": True})

    def test_empty_dataloader_compatible_with_cycle_dataloader(self):
        from areal.trainer.rl_trainer import _EmptyDataLoader

        dataloader = _EmptyDataLoader(batch_size=2, steps_per_epoch=3)
        generator = cycle_dataloader(dataloader)

        first = next(generator)
        second = next(generator)

        assert len(first) == 2
        assert len(second) == 2
        assert all(isinstance(x, dict) for x in first)
