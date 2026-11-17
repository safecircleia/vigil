import pytest
import torch

from vigil.merge_items import merge


def _items(conv, n=2):
    return [{"conv": conv, "ids": [1], "markers": [0], "qtype": 2, "target": [1.0, 0.0], "label": 0} for _ in range(n)]


def test_merge_concatenates_item_files(tmp_path):
    torch.save(_items("h:0") + _items("h:1"), tmp_path / "a.pt")
    torch.save(_items("mi:0", 3), tmp_path / "b.pt")
    assert merge([str(tmp_path / "a.pt"), str(tmp_path / "b.pt")], str(tmp_path / "out.pt")) == 7
    assert len(torch.load(tmp_path / "out.pt", weights_only=True)) == 7


def test_merge_refuses_files_that_share_conversation_ids(tmp_path):
    torch.save(_items("h:0"), tmp_path / "a.pt")
    torch.save(_items("h:0"), tmp_path / "b.pt")
    with pytest.raises(ValueError, match="collide"):
        merge([str(tmp_path / "a.pt"), str(tmp_path / "b.pt")], str(tmp_path / "out.pt"))
