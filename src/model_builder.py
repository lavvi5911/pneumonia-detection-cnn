"""
model_builder.py
=================
Defines every architecture the project can train:

1. `build_custom_cnn`   — a from-scratch CNN with residual blocks,
   squeeze-and-excitation attention, and global pooling. This is the
   "advanced" hand-built network showcasing CNN internals end-to-end.
2. `build_transfer_model` — EfficientNetB0 / DenseNet121 / ResNet50V2
   backbones with a custom classification head, used for best raw
   accuracy via ImageNet-pretrained features.

`get_model(architecture)` is the single factory used by train.py.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model

import config

# `tf.keras.saving.register_keras_serializable` isn't reliably populated on the
# lazy-loaded `tf.keras` alias across all TF/Keras versions. The standalone
# `keras` package (bundled with TF 2.16+) exposes it consistently, so we prefer
# that and fall back to the older `tf.keras.utils` location for older TF builds.
try:
    import keras as _keras
    register_keras_serializable = _keras.saving.register_keras_serializable
except (ImportError, AttributeError):
    register_keras_serializable = tf.keras.utils.register_keras_serializable


# ------------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------------
def squeeze_excite_block(x, ratio: int = 16, name: str = "se"):
    """Squeeze-and-Excitation: lets the network learn to re-weight channels
    by their global importance — a lightweight form of channel attention.
    """
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    se = layers.Dense(max(channels // ratio, 8), activation="relu", name=f"{name}_fc1")(se)
    se = layers.Dense(channels, activation="sigmoid", name=f"{name}_fc2")(se)
    se = layers.Reshape((1, 1, channels), name=f"{name}_reshape")(se)
    return layers.Multiply(name=f"{name}_scale")([x, se])


def residual_conv_block(x, filters: int, stride: int = 1, name: str = "res"):
    """A pre-activation residual block: BN -> ReLU -> Conv, twice, with a
    projection shortcut when the shape changes, followed by SE attention.
    """
    shortcut = x

    y = layers.BatchNormalization(name=f"{name}_bn1")(x)
    y = layers.Activation("relu", name=f"{name}_relu1")(y)
    y = layers.Conv2D(
        filters, 3, strides=stride, padding="same",
        kernel_regularizer=regularizers.l2(config.L2_REG),
        use_bias=False, name=f"{name}_conv1",
    )(y)

    y = layers.BatchNormalization(name=f"{name}_bn2")(y)
    y = layers.Activation("relu", name=f"{name}_relu2")(y)
    y = layers.Conv2D(
        filters, 3, strides=1, padding="same",
        kernel_regularizer=regularizers.l2(config.L2_REG),
        use_bias=False, name=f"{name}_conv2",
    )(y)

    y = squeeze_excite_block(y, name=f"{name}_se")

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, 1, strides=stride, padding="same",
            kernel_regularizer=regularizers.l2(config.L2_REG),
            use_bias=False, name=f"{name}_shortcut_conv",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_shortcut_bn")(shortcut)

    return layers.Add(name=f"{name}_add")([shortcut, y])


# ------------------------------------------------------------------
# 1. Custom CNN
# ------------------------------------------------------------------
def build_custom_cnn(input_shape=config.INPUT_SHAPE, name: str = "PulmoNet") -> Model:
    """A from-scratch residual CNN with squeeze-excite attention,
    purpose-built for grayscale-like chest radiograph classification.

    Stem -> 4 residual stages (with downsampling) -> GAP -> Dense head.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Stem
    x = layers.Conv2D(32, 7, strides=2, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="stem_pool")(x)

    # Residual stages: (filters, num_blocks, stride_of_first_block)
    stage_config = [(64, 2, 1), (128, 2, 2), (256, 3, 2), (512, 3, 2)]
    for stage_idx, (filters, num_blocks, first_stride) in enumerate(stage_config, start=1):
        for block_idx in range(num_blocks):
            stride = first_stride if block_idx == 0 else 1
            x = residual_conv_block(x, filters, stride=stride, name=f"stage{stage_idx}_block{block_idx}")

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(config.DROPOUT_RATE, name="head_dropout1")(x)
    x = layers.Dense(
        256, activation="relu", kernel_regularizer=regularizers.l2(config.L2_REG), name="head_dense1"
    )(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(config.DROPOUT_RATE, name="head_dropout2")(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32", name="pneumonia_probability")(x)

    return Model(inputs, outputs, name=name)


# ------------------------------------------------------------------
# 2. Transfer learning backbones
# ------------------------------------------------------------------
_BACKBONE_FACTORY = {
    "efficientnet": lambda input_shape: tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=input_shape
    ),
    "densenet": lambda input_shape: tf.keras.applications.DenseNet121(
        include_top=False, weights="imagenet", input_shape=input_shape
    ),
    "resnet": lambda input_shape: tf.keras.applications.ResNet50V2(
        include_top=False, weights="imagenet", input_shape=input_shape
    ),
}

_PREPROCESS_MODE = {
    "efficientnet": "efficientnet",  # EfficientNet's Keras impl normalizes internally (no-op here)
    "densenet": "torch",             # scale to [0,1] then normalize by ImageNet mean/std
    "resnet": "tf",                  # scale to [-1, 1]
}


@register_keras_serializable(package="pulmovision", name="BackbonePreprocess")
class BackbonePreprocess(layers.Layer):
    """Self-contained, serializable replacement for wrapping a library
    `preprocess_input` function in a `Lambda` layer.

    Wrapping an external function (e.g. `tf.keras.applications.efficientnet.
    preprocess_input`) in `Lambda` fails to reload from a saved .keras file on
    some Keras versions, because Keras can't always resolve where that function
    lives at load time. Reimplementing the (very simple) preprocessing formulas
    directly in a registered custom layer avoids that failure mode entirely —
    the layer is decorated with `register_keras_serializable`, so `load_model`
    can always reconstruct it as long as this module has been imported.
    """

    def __init__(self, mode: str, **kwargs):
        super().__init__(**kwargs)
        if mode not in ("efficientnet", "torch", "tf"):
            raise ValueError(f"Unknown preprocess mode '{mode}'")
        self.mode = mode
        # ImageNet channel mean/std (RGB), expressed in [0, 255] pixel range, for "torch" mode.
        self._mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32) * 255.0
        self._std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32) * 255.0

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        if self.mode == "efficientnet":
            return x
        if self.mode == "torch":
            return (x - self._mean) / self._std
        return (x / 127.5) - 1.0  # "tf" mode (ResNetV2)

    def get_config(self):
        config_dict = super().get_config()
        config_dict.update({"mode": self.mode})
        return config_dict


def build_transfer_model(
    architecture: str = "efficientnet",
    input_shape=config.INPUT_SHAPE,
    freeze_backbone: bool = True,
) -> Model:
    """ImageNet-pretrained backbone + custom dense head for binary classification.

    Our tf.data pipeline already scales images to [0, 1]; we therefore feed the
    backbone-specific preprocessing inside the graph so a single SavedModel
    stays self-contained (no separate preprocessing step needed at inference time).
    """
    if architecture not in _BACKBONE_FACTORY:
        raise ValueError(f"Unknown architecture '{architecture}'. Choose from {list(_BACKBONE_FACTORY)}")

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = layers.Rescaling(255.0, name="to_0_255")(inputs)  # undo the [0,1] scaling from data_loader
    x = BackbonePreprocess(mode=_PREPROCESS_MODE[architecture], name="backbone_preprocess")(x)

    backbone = _BACKBONE_FACTORY[architecture](input_shape)
    backbone._name = f"{architecture}_backbone"
    backbone.trainable = not freeze_backbone

    x = backbone(x, training=False if freeze_backbone else None)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(config.DROPOUT_RATE, name="head_dropout1")(x)
    x = layers.Dense(
        256, activation="relu", kernel_regularizer=regularizers.l2(config.L2_REG), name="head_dense1"
    )(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(config.DROPOUT_RATE, name="head_dropout2")(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32", name="pneumonia_probability")(x)

    model = Model(inputs, outputs, name=f"PulmoNet_{architecture}")
    model.backbone = backbone  # convenient handle for fine-tuning later
    return model


def unfreeze_for_fine_tuning(model: Model, fine_tune_at_layer: int = config.FINE_TUNE_AT_LAYER) -> Model:
    """Unfreezes backbone layers from `fine_tune_at_layer` onward, keeping the
    earlier (more generic, edge/texture-level) layers frozen. Batch-norm layers
    are always kept frozen to preserve stable running statistics.
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise ValueError("Model has no `.backbone` attribute — was it built with build_transfer_model()?")

    backbone.trainable = True
    for i, layer in enumerate(backbone.layers):
        if i < fine_tune_at_layer or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return model


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------
def get_model(architecture: str = config.MODEL_ARCHITECTURE, **kwargs) -> Model:
    if architecture == "custom_cnn":
        return build_custom_cnn(**kwargs)
    return build_transfer_model(architecture=architecture, **kwargs)


if __name__ == "__main__":
    m = get_model("custom_cnn")
    m.summary()
    print(f"\nTotal params: {m.count_params():,}")
