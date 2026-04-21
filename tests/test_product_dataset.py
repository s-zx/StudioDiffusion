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
