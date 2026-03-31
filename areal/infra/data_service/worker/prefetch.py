"""Per-dataset prefetch buffer using asyncio.Queue."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

from areal.utils import logging

logger = logging.getLogger("DataWorkerPrefetch")


class PrefetchBuffer:
    """Background async task that pre-fetches batches from a data iterator into a queue."""

    def __init__(self, capacity: int = 2):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=capacity)
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    def start(self, data_iter: Iterator, dataset_id: str):
        """Start background prefetch task."""
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(data_iter, dataset_id))

    async def _run(self, data_iter: Iterator, dataset_id: str):
        """Background task: continuously fetch from data_iter into queue."""
        try:
            while not self._stop_event.is_set():
                try:
                    batch = next(data_iter)
                    await self._queue.put(batch)
                except StopIteration:
                    await self._queue.put(None)  # Signal exhaustion
                    break
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Prefetch error for dataset %s", dataset_id)

    async def get(self) -> Any | None:
        """Pop next batch (blocks until available). Returns None if exhausted."""
        return await self._queue.get()

    def stop(self):
        """Stop the prefetch task."""
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            self._task = None
        # Drain the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()
