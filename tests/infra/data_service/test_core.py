from __future__ import annotations

import asyncio
import uuid

import pytest

from areal.infra.data_service.worker.prefetch import PrefetchBuffer
from areal.infra.data_service.worker.tensor_store import TensorStore


class TestTensorStore:
    def test_put_returns_uuid(self):
        store = TensorStore()
        shard_id = store.put({"x": 1})
        assert isinstance(shard_id, str)
        assert uuid.UUID(shard_id)

    def test_put_get_roundtrip(self):
        store = TensorStore()
        payload = {"foo": [1, 2, 3]}
        shard_id = store.put(payload)
        assert store.get(shard_id) == payload

    def test_get_unknown_returns_none(self):
        store = TensorStore()
        assert store.get("missing-shard") is None

    def test_delete_existing(self):
        store = TensorStore()
        shard_id = store.put("payload")
        assert store.delete(shard_id) is True
        assert store.get(shard_id) is None

    def test_delete_nonexistent(self):
        store = TensorStore()
        assert store.delete("missing-shard") is False

    def test_clear_removes_all(self):
        store = TensorStore()
        store.put(1)
        store.put(2)
        store.put(3)
        assert store.size == 3
        store.clear()
        assert store.size == 0

    def test_clear_for_dataset_selective(self):
        store = TensorStore()
        shard_1 = store.put("a")
        shard_2 = store.put("b")
        shard_3 = store.put("c")
        store.clear_for_dataset([shard_1, shard_2])
        assert store.get(shard_1) is None
        assert store.get(shard_2) is None
        assert store.get(shard_3) == "c"

    def test_size_tracks_entries(self):
        store = TensorStore()
        shard_1 = store.put("a")
        shard_2 = store.put("b")
        assert store.size == 2
        assert store.delete(shard_1) is True
        assert store.size == 1
        assert store.delete(shard_2) is True
        assert store.size == 0


class TestPrefetchBuffer:
    @pytest.mark.asyncio
    async def test_start_and_get_returns_items(self):
        buffer = PrefetchBuffer(capacity=2)
        buffer.start(iter([1, 2, 3]), dataset_id="dataset-a")
        assert await buffer.get() == 1
        assert await buffer.get() == 2
        assert await buffer.get() == 3
        assert await buffer.get() is None
        buffer.stop()

    @pytest.mark.asyncio
    async def test_exhaustion_signals_none(self):
        buffer = PrefetchBuffer(capacity=2)
        buffer.start(iter([1]), dataset_id="dataset-a")
        assert await buffer.get() == 1
        assert await buffer.get() is None
        buffer.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        buffer = PrefetchBuffer(capacity=2)
        buffer.start(iter(range(100)), dataset_id="dataset-a")
        await asyncio.sleep(0)
        assert buffer.is_running is True
        buffer.stop()
        assert buffer.is_running is False

    @pytest.mark.asyncio
    async def test_stop_drains_queue(self):
        buffer = PrefetchBuffer(capacity=100)
        buffer.start(iter(range(100)), dataset_id="dataset-a")
        await asyncio.sleep(0.01)
        await buffer.get()
        await buffer.get()
        buffer.stop()
        assert buffer._queue.empty() is True

    @pytest.mark.asyncio
    async def test_capacity_limits_buffering(self):
        buffer = PrefetchBuffer(capacity=1)
        buffer.start(iter(range(10)), dataset_id="dataset-a")
        await asyncio.sleep(0.01)
        assert buffer._queue.maxsize == 1
        assert buffer._queue.qsize() <= 1
        buffer.stop()

    @pytest.mark.asyncio
    async def test_is_running_tracks_state(self):
        buffer = PrefetchBuffer(capacity=2)
        assert buffer.is_running is False
        buffer.start(iter(range(3)), dataset_id="dataset-a")
        await asyncio.sleep(0)
        assert buffer.is_running is True
        buffer.stop()
        assert buffer.is_running is False
