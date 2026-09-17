import argparse

import pytest

from setdetect.synth_data.card_synth import CardAugConfig
from setdetect.synth_data.effects import (
    BlurConfig,
    CardCoverConfig,
    HardShadowConfig,
    ShadeConfig,
    ShadowConfig,
    SpecConfig,
)
from setdetect.ui.detection_args import (
    detection_kwargs_from_args,
    resolve_class_weights,
    resolve_corner_weights,
)
from setdetect.ui.synth_args import (
    add_blur_args,
    add_card_aug_args,
    add_card_cover_args,
    add_hard_shadow_args,
    add_shading_args,
    add_shadow_args,
    add_spec_args,
    blur_from_args,
    card_aug_from_args,
    card_cover_from_args,
    hard_shadow_from_args,
    shading_from_args,
    shadow_from_args,
    spec_from_args,
)


def test_resolve_corner_weights_missing_raises(tmp_path):
    with pytest.raises(SystemExit):
        resolve_corner_weights(tmp_path / "missing.onnx")


def test_resolve_class_weights_missing_raises(tmp_path):
    with pytest.raises(SystemExit):
        resolve_class_weights(tmp_path / "missing.onnx")


def test_resolve_corner_weights_happy_path(tmp_path):
    weights = tmp_path / "corners.onnx"
    weights.write_bytes(b"")
    assert resolve_corner_weights(weights) == weights


def test_detection_kwargs_from_args(tmp_path):
    corner_weights = tmp_path / "corners.onnx"
    class_weights = tmp_path / "class.onnx"
    corner_weights.write_bytes(b"")
    class_weights.write_bytes(b"")
    args = argparse.Namespace(
        det_corner_weights=corner_weights,
        det_class_weights=class_weights,
        det_device=0,
        det_conf=0.25,
        det_imgsz=640,
    )
    kwargs = detection_kwargs_from_args(args)
    assert kwargs == {
        "corner_weights": corner_weights,
        "class_weights": class_weights,
        "device_id": 0,
        "conf": 0.25,
        "imgsz": 640,
    }


@pytest.mark.parametrize(
    "add_args,from_args,expected_default",
    [
        (add_spec_args, spec_from_args, SpecConfig()),
        (add_shading_args, shading_from_args, ShadeConfig()),
        (add_shadow_args, shadow_from_args, ShadowConfig()),
        (add_hard_shadow_args, hard_shadow_from_args, HardShadowConfig()),
        (add_card_cover_args, card_cover_from_args, CardCoverConfig()),
        (add_card_aug_args, card_aug_from_args, CardAugConfig()),
        (add_blur_args, blur_from_args, BlurConfig()),
    ],
)
def test_synth_args_defaults_match_config_defaults(add_args, from_args, expected_default):
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args([])
    assert from_args(args) == expected_default
