from typing import cast

import pytest
import torch
import torch.distributed as dist

from areal.api.cli_args import MicroBatchSpec
from areal.engine.core.train_engine import compute_total_loss_weight
from areal.infra.controller.train_controller import (
    _dispatch_tensors,
    _pad_eval_batch,
)
from areal.utils.data import MicroBatchList, make_dummy_eval_item


def _make_item(idx: int, seqlen: int = 2) -> dict[str, object]:
    return {
        "input_ids": torch.full((1, seqlen), idx + 1, dtype=torch.long),
        "attention_mask": torch.ones((1, seqlen), dtype=torch.bool),
        "loss_mask": torch.ones((1, seqlen), dtype=torch.bool),
        "meta": {"id": idx},
    }


def _flatten_splits(splits: list[list[dict[str, object]]]) -> list[dict[str, object]]:
    return [item for group in splits for item in group]


def _count_dummies(items: list[dict[str, object]]) -> int:
    return sum(
        int(cast(torch.Tensor, item["attention_mask"]).sum().item() == 0)
        for item in items
    )


def test_pad_eval_batch_no_padding_when_divisible():
    """n=8, dp=4 → no dummies inserted."""
    items = [_make_item(i) for i in range(8)]
    (padded,) = _pad_eval_batch((items,), dp_size=4)
    assert len(padded) == 8
    assert _count_dummies(padded) == 0


def test_pad_eval_batch_pads_when_not_divisible():
    """n=7, dp=4 → 1 dummy padded, total=8."""
    items = [_make_item(i) for i in range(7)]
    (padded,) = _pad_eval_batch((items,), dp_size=4)
    assert len(padded) == 8
    assert _count_dummies(padded) == 1


def test_pad_eval_batch_pads_when_n_less_than_dp():
    """n=2, dp=4 → 2 dummies padded, total=4."""
    items = [_make_item(i) for i in range(2)]
    (padded,) = _pad_eval_batch((items,), dp_size=4)
    assert len(padded) == 4
    assert _count_dummies(padded) == 2


def test_dispatch_tensors_raises_when_not_divisible():
    """_dispatch_tensors itself still requires divisible input."""
    items = [_make_item(i) for i in range(7)]
    with pytest.raises(ValueError, match="divisible"):
        _dispatch_tensors(items, dp_size=4)


def test_pad_then_dispatch_end_to_end():
    """Full flow: pad → dispatch → all groups equal, dummies spread correctly."""
    items = [_make_item(i) for i in range(7)]
    (padded,) = _pad_eval_batch((items,), dp_size=4)
    splits, _ = _dispatch_tensors(padded, dp_size=4)
    assert all(len(group) == 2 for group in splits)
    assert _count_dummies(_flatten_splits(splits)) == 1


def test_make_dummy_eval_item_schema():
    template: dict[str, object] = {
        "input_ids": torch.tensor([[2, 3, 4]], dtype=torch.long),
        "attention_mask": torch.tensor([[True, True, False]], dtype=torch.bool),
        "loss_mask": torch.tensor([[1, 1, 0]], dtype=torch.int32),
        "multi_modal_input": [{"image": torch.tensor([1.0])}],
        "meta": {"tag": ["x"]},
    }

    dummy = make_dummy_eval_item(template)
    assert set(dummy.keys()) == set(template.keys())
    assert dummy["multi_modal_input"] == []
    torch.testing.assert_close(
        dummy["attention_mask"],
        torch.zeros((1, 1), dtype=torch.bool),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        dummy["loss_mask"],
        torch.zeros((1, 1), dtype=cast(torch.Tensor, template["loss_mask"]).dtype),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        dummy["input_ids"],
        torch.zeros((1, 1), dtype=cast(torch.Tensor, template["input_ids"]).dtype),
        rtol=0.0,
        atol=0.0,
    )
    cast(dict[str, list[str]], template["meta"])["tag"].append("y")
    assert dummy["meta"] == {"tag": ["x"]}


def test_compute_total_loss_weight_allows_local_zero(monkeypatch: pytest.MonkeyPatch):
    mb_list = MicroBatchList(
        data={},
        mb_spec=MicroBatchSpec(),
        mbs=[{"attention_mask": torch.zeros((1, 1), dtype=torch.bool)}],
        group_lens=[1],
    )

    def _mock_all_reduce(tensor: torch.Tensor, group: dist.ProcessGroup | None = None):
        del group
        tensor.add_(3.0)

    monkeypatch.setattr(dist, "all_reduce", _mock_all_reduce)

    total_weight = compute_total_loss_weight(
        mb_list=mb_list,
        loss_weight_fn=lambda _mb: torch.tensor(0.0),
        dp_group=cast(dist.ProcessGroup, object()),
    )

    torch.testing.assert_close(total_weight, torch.tensor(3.0), rtol=0.0, atol=0.0)
