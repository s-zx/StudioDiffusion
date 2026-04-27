"""
Unit tests for AestheticScorer sub-dimension scoring (composition, lighting, color).

Fast tests (no model): test _compute_subscore math via dummy unit vectors.
Model tests (loads CLIP ViT-L/14, scope=module): test the public API.
  - checkpoint=None is enough; the aesthetic predictor MLP is not needed
    for sub-dimension scoring.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from evaluation.aesthetic_scoring import AestheticScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_unit(d: int = 768) -> torch.Tensor:
    """Random unit vector (1, d)."""
    return F.normalize(torch.randn(1, d), dim=-1)


# ---------------------------------------------------------------------------
# Fast tests — _compute_subscore math, no model loading
# ---------------------------------------------------------------------------

def test_compute_subscore_in_range():
    s = AestheticScorer._compute_subscore(_rand_unit(), _rand_unit(), _rand_unit())
    assert 0.0 <= s <= 1.0


def test_compute_subscore_identical_prompts_is_half():
    """positive == negative → softmax → 0.5 exactly."""
    emb = _rand_unit()
    s = AestheticScorer._compute_subscore(_rand_unit(), emb, emb)
    assert s == pytest.approx(0.5)


def test_compute_subscore_aligned_positive_above_half():
    """Image collinear with positive prompt → score > 0.5."""
    feat = _rand_unit()
    s = AestheticScorer._compute_subscore(feat, feat.clone(), -feat)
    assert s > 0.5


def test_compute_subscore_aligned_negative_below_half():
    """Image collinear with negative prompt → score < 0.5."""
    feat = _rand_unit()
    s = AestheticScorer._compute_subscore(feat, -feat, feat.clone())
    assert s < 0.5


# ---------------------------------------------------------------------------
# Model tests — loads CLIP once per test session (scope=module)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scorer():
    """AestheticScorer with checkpoint=None (sub-dimension scoring only)."""
    return AestheticScorer(checkpoint=None, device="cpu")


@pytest.fixture
def white_image():
    return Image.fromarray(np.ones((256, 256, 3), dtype=np.uint8) * 255)


def test_score_composition_in_range(scorer, white_image):
    s = scorer.score_composition(white_image)
    assert 0.0 <= s <= 1.0


def test_score_lighting_in_range(scorer, white_image):
    s = scorer.score_lighting(white_image)
    assert 0.0 <= s <= 1.0


def test_score_color_in_range(scorer, white_image):
    s = scorer.score_color(white_image)
    assert 0.0 <= s <= 1.0


def test_score_detailed_keys_and_range(scorer, white_image):
    result = scorer.score_detailed(white_image)
    assert set(result.keys()) == {"composition", "lighting", "color"}
    for dim, val in result.items():
        assert 0.0 <= val <= 1.0, f"{dim}={val} out of [0,1]"


def test_score_detailed_deterministic(scorer, white_image):
    r1 = scorer.score_detailed(white_image)
    r2 = scorer.score_detailed(white_image)
    for dim in r1:
        assert r1[dim] == pytest.approx(r2[dim])


def test_score_detailed_consistent_with_individual(scorer, white_image):
    """score_detailed and score_* helpers must agree to float precision."""
    d = scorer.score_detailed(white_image)
    assert d["composition"] == pytest.approx(scorer.score_composition(white_image))
    assert d["lighting"] == pytest.approx(scorer.score_lighting(white_image))
    assert d["color"] == pytest.approx(scorer.score_color(white_image))


def test_score_without_checkpoint_raises(scorer, white_image):
    """score() must raise RuntimeError when checkpoint was not loaded."""
    with pytest.raises(RuntimeError, match="checkpoint"):
        scorer.score(white_image)


def test_score_batch_without_checkpoint_raises(scorer, white_image):
    with pytest.raises(RuntimeError, match="checkpoint"):
        scorer.score_batch([white_image])


def test_score_batch_detailed_keys_and_range(scorer, white_image):
    results = scorer.score_batch_detailed([white_image, white_image])
    assert len(results) == 2
    for r in results:
        assert set(r.keys()) == {"composition", "lighting", "color"}
        for val in r.values():
            assert 0.0 <= val <= 1.0


def test_score_batch_detailed_consistent_with_score_detailed(scorer, white_image):
    """Batch and single results must agree."""
    batch = scorer.score_batch_detailed([white_image])
    single = scorer.score_detailed(white_image)
    for dim in ("composition", "lighting", "color"):
        assert batch[0][dim] == pytest.approx(single[dim])
