"""In-memory tensor storage for batch data staging."""

from __future__ import annotations

import uuid
from typing import Any

from areal.utils import logging

logger = logging.getLogger("DataWorkerTensorStore")


class TensorStore:
    # TODO: unify RTensor storage with this.
    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def put(self, data: Any) -> str:
        shard_id = str(uuid.uuid4())
        self._store[shard_id] = data
        return shard_id

    def get(self, shard_id: str) -> Any | None:
        return self._store.get(shard_id)

    def delete(self, shard_id: str) -> bool:
        return self._store.pop(shard_id, None) is not None

    def clear(self) -> None:
        count = len(self._store)
        self._store.clear()
        if count:
            logger.info("Cleared %d tensor shards", count)

    def clear_for_dataset(self, shard_ids: list[str]) -> None:
        for sid in shard_ids:
            self._store.pop(sid, None)

    @property
    def size(self) -> int:
        return len(self._store)
