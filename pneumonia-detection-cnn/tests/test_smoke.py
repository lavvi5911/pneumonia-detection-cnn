"""
tests/test_smoke.py
====================
Lightweight smoke tests that don't require the dataset or a trained model —
they verify the architectures build, compile, and produce correctly-shaped
output on random tensors. Run with:

    pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src import model_builder


@pytest.mark.parametrize("architecture", ["custom_cnn", "efficientnet", "densenet", "resnet"])
def test_model_builds_and_predicts(architecture):
    model = model_builder.get_model(architecture)
    dummy_batch = np.random.rand(2, *config.INPUT_SHAPE).astype("float32")
    preds = model.predict(dummy_batch, verbose=0)
    assert preds.shape == (2, 1)
    assert np.all((preds >= 0) & (preds <= 1)), "Sigmoid output must be in [0, 1]"


def test_custom_cnn_param_count_reasonable():
    model = model_builder.get_model("custom_cnn")
    n_params = model.count_params()
    assert 500_000 < n_params < 50_000_000


def test_transfer_model_has_backbone_handle():
    model = model_builder.build_transfer_model("efficientnet", freeze_backbone=True)
    assert hasattr(model, "backbone")
    assert model.backbone.trainable is False


def test_fine_tune_unfreezes_top_layers():
    model = model_builder.build_transfer_model("efficientnet", freeze_backbone=True)
    model = model_builder.unfreeze_for_fine_tuning(model, fine_tune_at_layer=100)
    trainable_layers = [l for l in model.backbone.layers if l.trainable]
    assert len(trainable_layers) > 0


def test_squeeze_excite_preserves_shape():
    inp = tf.keras.Input(shape=(16, 16, 32))
    out = model_builder.squeeze_excite_block(inp, name="test_se")
    m = tf.keras.Model(inp, out)
    dummy = np.random.rand(1, 16, 16, 32).astype("float32")
    result = m.predict(dummy, verbose=0)
    assert result.shape == (1, 16, 16, 32)
