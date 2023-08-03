import shutil

import numpy as np
import os
import time
from pathlib import Path

import tensorflow as tf

root_dir = Path("/training/")
image_size = (180, 180)
batch_size = 32
epochs = 20
model_name = "my_model"


def make_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    # Image augmentation block
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
        ]
    )
    x = data_augmentation(inputs)

    # Entry block
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = tf.keras.layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = tf.keras.layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(units, activation=activation)(x)
    return tf.keras.Model(inputs, outputs)


def train_data():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(root_dir),
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(root_dir),
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)

    model = make_model(input_shape=image_size + (3,), num_classes=2)
    tf.keras.utils.plot_model(model, show_shapes=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(f"{model_name}_{{epoch}}.h5"),
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
    )


def clean_images():
    move_to = root_dir.parent / "bad_data"  # needs to be not in the training directory
    move_to.mkdir(exist_ok=True)
    moved = 0
    for directory in root_dir.glob("*"):
        if directory.name == move_to.name:
            continue
        if directory.is_dir():
            for i, file in enumerate(directory.glob("*")):
                if not file.name.lower().endswith(("jpg", "jpeg")) or not tf.compat.as_bytes("JFIF") in file.open("rb").read(10):
                    shutil.move(file, move_to / file.name)
                    moved += 1
        print("moved unclean data", moved, "from", directory)


def move_images():
    model = tf.keras.models.load_model(f"{model_name}_{epochs}.h5")
    cartoon_dir = Path("/cartoon/")
    real_dir = Path("/real/")

    real, cartoon, unknown = 0, 0, 0

    for file in Path("/unsorted/").glob("*.[jpg][jpeg][png]"):

        img = tf.keras.preprocessing.image.load_img(
            str(file), target_size=image_size
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = model.predict(img_array)
        score = predictions[0][0]
        if score > 0.98:
            real += 1
            shutil.move(file, real_dir / file.name)
        elif score < 0.02:
            cartoon += 1
            shutil.move(file, cartoon_dir / file.name)
        else:
            unknown += 1
            print(f"Could not figure out {file} as it was {score * 100}")
    print(f"Moved {real} to real and {cartoon} to cartoon, {unknown} were unmoved")


if __name__ == '__main__':
    clean_images()
    train_data()
    move_images()
