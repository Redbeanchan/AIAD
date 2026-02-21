"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.14
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50


def _decode_mask(mask_path: tf.Tensor, img_size: int) -> tf.Tensor:
    m_bytes = tf.io.read_file(mask_path)
    m = tf.image.decode_image(m_bytes, channels=1, expand_animations=False)
    m = tf.image.resize(m, (img_size, img_size), method="nearest")
    m = tf.cast(m, tf.float32)

    m_max = tf.reduce_max(m)
    m = tf.cond(
        m_max <= 1.0,
        lambda: tf.cast(m > 0.5, tf.float32),
        lambda: tf.cast(m > 127.0, tf.float32),
    )
    return m


def _decode_pair(
    img_path: tf.Tensor,
    mask_path: tf.Tensor,
    img_size: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    img_bytes = tf.io.read_file(img_path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32) / 255.0

    m = _decode_mask(mask_path, img_size)
    return img, m


def _augment(img: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    flip = tf.random.uniform([]) > 0.5
    img = tf.cond(flip, lambda: tf.image.flip_left_right(img), lambda: img)
    mask = tf.cond(flip, lambda: tf.image.flip_left_right(mask), lambda: mask)

    img = tf.image.random_brightness(img, 0.2)
    noise = tf.random.normal(tf.shape(img), stddev=0.05)
    img = tf.clip_by_value(img + noise, 0.0, 1.0)
    return img, mask


def make_tf_dataset(
    df: pd.DataFrame,
    img_size: int,
    batch_size: int,
    training: bool,
    shuffle_buffer: int,
) -> tf.data.Dataset:
    img_paths = df["images"].astype(str).tolist()
    msk_paths = df["masks"].astype(str).tolist()

    ds = tf.data.Dataset.from_tensor_slices((img_paths, msk_paths))

    if training:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    autotune = tf.data.AUTOTUNE
    ds = ds.map(lambda ip, mp: _decode_pair(ip, mp, img_size), num_parallel_calls=autotune)

    if training:
        ds = ds.map(_augment, num_parallel_calls=autotune)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)  # stable RAM usage
    return ds


def unet_resnet50(
    input_shape: Tuple[int, int, int],
    C: int = 32,
    num_classes: int = 1,
) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)
    base = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)

    c1 = base.get_layer("conv1_relu").output
    c2 = base.get_layer("conv2_block3_out").output
    c3 = base.get_layer("conv3_block4_out").output
    c4 = base.get_layer("conv4_block6_out").output
    b = base.get_layer("conv5_block3_out").output

    u6 = layers.Conv2DTranspose(8 * C, 2, strides=2, padding="same")(b)
    u6 = layers.Concatenate()([u6, c4])
    x = layers.Conv2D(8 * C, 3, activation="relu", padding="same")(u6)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(8 * C, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)

    u7 = layers.Conv2DTranspose(4 * C, 2, strides=2, padding="same")(x)
    u7 = layers.Concatenate()([u7, c3])
    x = layers.Conv2D(4 * C, 3, activation="relu", padding="same")(u7)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(4 * C, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)

    u8 = layers.Conv2DTranspose(2 * C, 2, strides=2, padding="same")(x)
    u8 = layers.Concatenate()([u8, c2])
    x = layers.Conv2D(2 * C, 3, activation="relu", padding="same")(u8)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(2 * C, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)

    u9 = layers.Conv2DTranspose(C, 2, strides=2, padding="same")(x)
    u9 = layers.Concatenate()([u9, c1])
    x = layers.Conv2D(C, 3, activation="relu", padding="same")(u9)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(C, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)

    u10 = layers.Conv2DTranspose(C, 2, strides=2, padding="same")(x)
    x = layers.Conv2D(C, 3, activation="relu", padding="same")(u10)
    x = layers.BatchNormalization()(x)

    if num_classes == 1:
        outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)
    else:
        outputs = layers.Conv2D(num_classes, 1, activation="softmax")(x)

    return models.Model(inputs, outputs)


def train_segmentation_model(
    train_manifest: pd.DataFrame,
    val_manifest: pd.DataFrame,
    img_size: int,
    batch_size: int,
    shuffle_buffer: int,
    base_channels: int,
    lr: float,
    epochs: int,
) -> Tuple[tf.keras.Model, Dict[str, list]]:
    train_ds = make_tf_dataset(
        train_manifest,
        img_size=img_size,
        batch_size=batch_size,
        training=True,
        shuffle_buffer=shuffle_buffer,
    )
    val_ds = make_tf_dataset(
        val_manifest,
        img_size=img_size,
        batch_size=batch_size,
        training=False,
        shuffle_buffer=shuffle_buffer,
    )

    model = unet_resnet50(input_shape=(img_size, img_size, 3), C=base_channels, num_classes=1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    return model, hist.history


def save_model(model: tf.keras.Model, model_output_path: str) -> str:
    out = Path(model_output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    return str(out)