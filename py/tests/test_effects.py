import numpy as np

from setdetect.synth_data.effects import (
    _make_blob_mask,
    make_card_cover,
    make_card_shading,
    make_hard_shadow,
    make_specular_highlight,
)


def test_make_specular_highlight_shape_and_determinism():
    card_img = np.zeros((64, 64, 4), dtype=np.uint8)
    layer1 = make_specular_highlight(card_img, rot_agl=0.0, prob=1.0, rng=np.random.default_rng(0))
    layer2 = make_specular_highlight(card_img, rot_agl=0.0, prob=1.0, rng=np.random.default_rng(0))
    assert layer1 is not None
    assert layer1.shape == (8, 8)
    np.testing.assert_array_equal(layer1, layer2)


def test_make_specular_highlight_none_when_prob_zero():
    card_img = np.zeros((64, 64, 4), dtype=np.uint8)
    assert make_specular_highlight(card_img, rot_agl=0.0, prob=0.0, rng=np.random.default_rng(0)) is None


def test_make_card_shading_shape_range_and_determinism():
    card_img = np.zeros((64, 64, 4), dtype=np.uint8)
    factor1 = make_card_shading(
        card_img,
        rot_agl=0.0,
        light_dir=45.0,
        prob=1.0,
        amp=0.3,
        type_="mixed",
        direction="global",
        rng=np.random.default_rng(1),
    )
    factor2 = make_card_shading(
        card_img,
        rot_agl=0.0,
        light_dir=45.0,
        prob=1.0,
        amp=0.3,
        type_="mixed",
        direction="global",
        rng=np.random.default_rng(1),
    )
    assert factor1 is not None
    assert factor1.shape == (8, 8, 3)
    assert np.all(factor1 <= 1.0)
    assert np.all(factor1 >= 0.0)
    np.testing.assert_array_equal(factor1, factor2)


def test_make_blob_mask_shape_and_range():
    mask = _make_blob_mask(40, 60, coverage=0.3, rng=np.random.default_rng(0))
    assert mask.shape == (40, 60)
    assert np.all(mask >= 0.0) and np.all(mask <= 1.0)


def test_make_hard_shadow_only_darkens():
    img = np.full((50, 50, 3), 200, dtype=np.uint8)
    original = img.copy()
    make_hard_shadow(img, prob=1.0, strength=0.5, coverage=0.3, edge_softness=0.01, rng=np.random.default_rng(0))
    assert img.shape == original.shape
    assert img.dtype == original.dtype
    assert np.all(img <= original)


def test_make_hard_shadow_noop_when_prob_zero():
    img = np.full((30, 30, 3), 128, dtype=np.uint8)
    original = img.copy()
    make_hard_shadow(img, prob=0.0, strength=0.5, coverage=0.3, edge_softness=0.01, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(img, original)


def test_make_card_cover_preserves_shape_and_alpha():
    card_img = np.full((32, 32, 4), 200, dtype=np.uint8)
    out = make_card_cover(card_img, prob=1.0, coverage=0.4, color_range=(20, 235), rng=np.random.default_rng(0))
    assert out.shape == card_img.shape
    assert out.dtype == card_img.dtype
    # alpha channel is untouched by design
    np.testing.assert_array_equal(out[..., 3], card_img[..., 3])


def test_make_card_cover_noop_when_prob_zero():
    card_img = np.full((32, 32, 4), 200, dtype=np.uint8)
    out = make_card_cover(card_img, prob=0.0, coverage=0.4, color_range=(20, 235), rng=np.random.default_rng(0))
    np.testing.assert_array_equal(out, card_img)
