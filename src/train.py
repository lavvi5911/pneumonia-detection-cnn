"""
train.py
========
End-to-end training entry point.

Usage:
    python -m src.train --architecture efficientnet
    python -m src.train --architecture custom_cnn --epochs 40

For transfer-learning architectures this runs a two-phase schedule:
  Phase 1 (head warm-up):   backbone frozen, train the new classification head.
  Phase 2 (fine-tuning):    unfreeze the top backbone layers, train end-to-end
                             at a much lower learning rate.

For `custom_cnn` it just runs a single training phase since there is no
pretrained backbone to warm up around.

Produces:
  models/best_model.keras     (best val_auc checkpoint)
  models/final_model.keras    (last epoch, for reference)
  artifacts/history.json      (full training history, both phases concatenated)
"""

from __future__ import annotations

import argparse
import time

import tensorflow as tf

import config
from src import data_loader, model_builder, utils


def build_callbacks(phase_name: str):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.BEST_MODEL_PATH),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LR,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(config.ARTIFACTS_DIR / f"training_log_{phase_name}.csv")),
        tf.keras.callbacks.TensorBoard(log_dir=str(config.ARTIFACTS_DIR / "tensorboard" / phase_name)),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=config.LABEL_SMOOTHING),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def merge_histories(*histories: dict) -> dict:
    merged = {}
    for h in histories:
        for k, v in h.items():
            merged.setdefault(k, []).extend(v)
    return merged


def main(architecture: str, epochs_head: int, epochs_fine_tune: int, batch_size: int):
    utils.set_global_seed(config.SEED)
    utils.configure_gpu_memory_growth()
    if config.USE_MIXED_PRECISION:
        utils.enable_mixed_precision()

    print(f"[INFO] Architecture: {architecture}")
    print("[INFO] Class counts —",
          "train:", data_loader.dataset_class_counts("train"),
          "val:", data_loader.dataset_class_counts("val"),
          "test:", data_loader.dataset_class_counts("test"))

    train_ds, n_train = data_loader.make_dataset("train", batch_size=batch_size, shuffle=True, augment=True)
    val_ds, n_val = data_loader.make_dataset("val", batch_size=batch_size, shuffle=False, augment=False)

    class_weight = data_loader.compute_class_weights() if config.USE_CLASS_WEIGHTS else None
    print(f"[INFO] Class weights: {class_weight}")

    all_histories = []
    t0 = time.time()

    if architecture == "custom_cnn":
        model = model_builder.get_model("custom_cnn")
        model = compile_model(model, config.LEARNING_RATE_HEAD)
        model.summary()
        hist = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_head + epochs_fine_tune,
            class_weight=class_weight,
            callbacks=build_callbacks("custom_cnn"),
        )
        all_histories.append(hist.history)
    else:
        # ---- Phase 1: head warm-up, backbone frozen ----
        model = model_builder.get_model(architecture, freeze_backbone=True)
        model = compile_model(model, config.LEARNING_RATE_HEAD)
        model.summary()
        print("\n[PHASE 1] Training classification head (backbone frozen)...")
        hist1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_head,
            class_weight=class_weight,
            callbacks=build_callbacks(f"{architecture}_phase1_head"),
        )
        all_histories.append(hist1.history)

        # ---- Phase 2: fine-tune top backbone layers ----
        print("\n[PHASE 2] Fine-tuning top backbone layers...")
        model = model_builder.unfreeze_for_fine_tuning(model, config.FINE_TUNE_AT_LAYER)
        model = compile_model(model, config.LEARNING_RATE_FINE_TUNE)
        hist2 = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=hist1.epoch[-1] + 1 if hist1.epoch else 0,
            epochs=epochs_head + epochs_fine_tune,
            class_weight=class_weight,
            callbacks=build_callbacks(f"{architecture}_phase2_finetune"),
        )
        all_histories.append(hist2.history)

    elapsed = time.time() - t0
    print(f"\n[INFO] Training finished in {elapsed / 60:.1f} minutes.")

    model.save(config.FINAL_MODEL_PATH)
    print(f"[INFO] Final model saved to {config.FINAL_MODEL_PATH}")
    print(f"[INFO] Best model (by val_auc) saved to {config.BEST_MODEL_PATH}")

    merged_history = merge_histories(*all_histories)
    merged_history["architecture"] = architecture
    merged_history["n_train"] = n_train
    merged_history["n_val"] = n_val
    merged_history["training_time_seconds"] = elapsed
    utils.save_json(merged_history, config.HISTORY_PATH)
    print(f"[INFO] Training history saved to {config.HISTORY_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the pneumonia detection CNN.")
    parser.add_argument("--architecture", default=config.MODEL_ARCHITECTURE,
                         choices=["custom_cnn", "efficientnet", "densenet", "resnet"])
    parser.add_argument("--epochs_head", type=int, default=config.EPOCHS_HEAD)
    parser.add_argument("--epochs_fine_tune", type=int, default=config.EPOCHS_FINE_TUNE)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    main(args.architecture, args.epochs_head, args.epochs_fine_tune, args.batch_size)
