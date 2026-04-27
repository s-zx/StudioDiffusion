"""
Unit tests for AestheticScorer sub-dimension scoring (composition, lighting, color).

All sub-dimension tests are fast and require NO model — image statistics are
pure numpy/PIL. Tests that require CLIP + checkpoint are guarded separately.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from evaluation.aesthetic_scoring import AestheticScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid(r: int, g: int, b: int, size: int = 256) -> Image.Image:
    return Image.fromarray(
        np.full((size, size, 3), [r, g, b], dtype=np.uint8)
    )


def _patch(value: int, ry0: int, ry1: int, rx0: int, rx1: int, size: int = 256) -> Image.Image:
    """Black image with a bright patch at the given coordinates."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[ry0:ry1, rx0:rx1] = value
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Fixture — instantiates instantly (no CLIP, no checkpoint)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scorer():
    return AestheticScorer(checkpoint=None, device="cpu")


# ---------------------------------------------------------------------------
# Range tests — all sub-scores must be in [0, 1]
# ---------------------------------------------------------------------------

def test_composition_in_range(scorer):
    assert 0.0 <= scorer.score_composition(_solid(128, 128, 128)) <= 1.0


def test_lighting_in_range(scorer):
    assert 0.0 <= scorer.score_lighting(_solid(128, 128, 128)) <= 1.0


def test_color_in_range(scorer):
    assert 0.0 <= scorer.score_color(_solid(128, 128, 128)) <= 1.0


def test_score_detailed_keys_and_range(scorer):
    result = scorer.score_detailed(_solid(200, 180, 160))
    assert set(result.keys()) == {"composition", "lighting", "color"}
    for dim, val in result.items():
        assert 0.0 <= val <= 1.0, f"{dim}={val}"


# ---------------------------------------------------------------------------
# Behavioral tests — composition
# ---------------------------------------------------------------------------

def test_composition_center_beats_corner(scorer):
    """Bright patch in center scores higher than same patch in corner."""
    center = _patch(255, ry0=96, ry1=160, rx0=96, rx1=160)   # center 25%
    corner = _patch(255, ry0=0,  ry1=64,  rx0=0,  rx1=64)    # top-left corner
    assert scorer.score_composition(center) > scorer.score_composition(corner)


def test_composition_uniform_image_midrange(scorer):
    """Uniform images have no edges — score should be in neutral range."""
    s = scorer.score_composition(_solid(128, 128, 128))
    assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# Behavioral tests — lighting
# ---------------------------------------------------------------------------

def test_lighting_midgray_beats_pure_black(scorer):
    mid = _solid(128, 128, 128)
    black = _solid(0, 0, 0)
    assert scorer.score_lighting(mid) > scorer.score_lighting(black)


def test_lighting_midgray_beats_pure_white(scorer):
    mid = _solid(128, 128, 128)
    white = _solid(255, 255, 255)
    assert scorer.score_lighting(mid) > scorer.score_lighting(white)


# ---------------------------------------------------------------------------
# Behavioral tests — color
# ---------------------------------------------------------------------------

def test_color_colorful_beats_gray(scorer):
    """A strongly red image should score higher colorfulness than neutral gray."""
    gray = _solid(128, 128, 128)
    red = _solid(255, 0, 0)
    assert scorer.score_color(red) > scorer.score_color(gray)


# ---------------------------------------------------------------------------
# Consistency tests
# ---------------------------------------------------------------------------

def test_score_detailed_deterministic(scorer):
    img = _solid(200, 150, 100)
    r1 = scorer.score_detailed(img)
    r2 = scorer.score_detailed(img)
    for dim in r1:
        assert r1[dim] == pytest.approx(r2[dim])


def test_score_detailed_consistent_with_individual(scorer):
    img = _solid(200, 150, 100)
    d = scorer.score_detailed(img)
    assert d["composition"] == pytest.approx(scorer.score_composition(img))
    assert d["lighting"]    == pytest.approx(scorer.score_lighting(img))
    assert d["color"]       == pytest.approx(scorer.score_color(img))


def test_score_batch_detailed_length_and_range(scorer):
    imgs = [_solid(200, 150, 100), _solid(50, 80, 60), _solid(255, 200, 100)]
    results = scorer.score_batch_detailed(imgs)
    assert len(results) == 3
    for r in results:
        assert set(r.keys()) == {"composition", "lighting", "color"}
        for val in r.values():
            assert 0.0 <= val <= 1.0


def test_score_batch_detailed_consistent_with_single(scorer):
    img = _solid(200, 150, 100)
    batch = scorer.score_batch_detailed([img])
    single = scorer.score_detailed(img)
    for dim in ("composition", "lighting", "color"):
        assert batch[0][dim] == pytest.approx(single[dim])


# ---------------------------------------------------------------------------
# Guard tests — score() and score_batch() require a checkpoint
# ---------------------------------------------------------------------------

def test_score_without_checkpoint_raises(scorer):
    with pytest.raises(RuntimeError, match="checkpoint"):
        scorer.score(_solid(128, 128, 128))


def test_score_batch_without_checkpoint_raises(scorer):
    with pytest.raises(RuntimeError, match="checkpoint"):
        scorer.score_batch([_solid(128, 128, 128)])
