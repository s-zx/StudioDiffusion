"""Unit tests for the manifest-driven ProductDataset.

These tests assume the team data bundle has been extracted at the repo root
(see share/TEAM_DATA_BUNDLE_README.txt). They run on CPU and finish in
seconds — no model weights or GPU needed.
"""
from pathlib import Path

import pytest
import torch

from adapters.ip_adapter.train import ProductDataset


PLATFORM_DIR = Path("data/platform_sets/shopify")
IMAGE_SIZE = 768


@pytest.fixture(scope="module")
def shopify_train():
    return ProductDataset(PLATFORM_DIR, split="train", image_size=IMAGE_SIZE)


@pytest.fixture(scope="module")
def shopify_val():
    return ProductDataset(PLATFORM_DIR, split="val", image_size=IMAGE_SIZE)


def test_train_split_loads(shopify_train):
    assert len(shopify_train) > 200, f"expected >200 train items, got {len(shopify_train)}"


def test_val_split_loads(shopify_val):
    assert len(shopify_val) > 50, f"expected >50 val items, got {len(shopify_val)}"


def test_item_shape_and_keys(shopify_train):
    item = shopify_train[0]
    assert set(item.keys()) == {"pixel_values", "clip_pixel_values", "caption"}
    assert isinstance(item["pixel_values"], torch.Tensor)
    assert item["pixel_values"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert item["clip_pixel_values"].shape == (3, 336, 336)
    assert isinstance(item["caption"], str) and len(item["caption"]) > 0


def test_resolves_primary_form(shopify_train):
    """At least one item should resolve via the <parent>_<basename> form
    (e.g. '08_08dcf907.jpg', 'shard_00004_000476.jpg')."""
    assert any("_" in it["path"].name for it in shopify_train.items), \
        "expected at least one primary-form (<parent>_<basename>) resolution"


def test_resolves_secondary_form(tmp_path):
    """Secondary (plain basename) resolution works when the primary form
    does not exist on disk — verified with a synthetic fixture."""
    platform_dir = tmp_path / "fake_platform"
    platform_dir.mkdir()
    manifests = tmp_path / "manifests"
    manifests.mkdir()

    # Create only the secondary (plain basename) form, not the primary form.
    img_file = platform_dir / "abc123.jpg"
    img_file.write_bytes(b"")  # empty placeholder

    (manifests / "fake_platform_train.csv").write_text(
        "image_path,category\n"
        "/some/parent/abc123.jpg,WIDGET\n"
    )

    ds = ProductDataset(platform_dir, split="train", image_size=IMAGE_SIZE)
    assert len(ds) == 1
    assert ds.items[0]["path"] == img_file
    assert "_" not in ds.items[0]["path"].name


def test_all_resolved_paths_exist(shopify_train, shopify_val):
    for ds in (shopify_train, shopify_val):
        for item in ds.items:
            assert item["path"].exists(), f"missing on disk: {item['path']}"
