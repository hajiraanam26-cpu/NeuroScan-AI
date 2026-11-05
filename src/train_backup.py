import json
from pathlib import Path

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from .preprocess import TRAIN_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE, SEED

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "neuroscan_mobilenet.keras"

EPOCHS_HEAD = 15
EPOCHS_FINE = 10
NUM_CLASSES = 4

AUTOTUNE = tf.data.AUTOTUNE


# Load datasets
def load_datasets():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    class_names = train_ds.class_names

    if len(class_names) != NUM_CLASSES:
        raise ValueError("Expected exactly four MRI classes.")

    if val_ds.class_names != class_names or test_ds.class_names != class_names:
        raise ValueError("Dataset class names do not match.")

    return (
        train_ds.prefetch(AUTOTUNE),
        val_ds.prefetch(AUTOTUNE),
        test_ds.prefetch(AUTOTUNE),
        class_names
    )


# Build MobileNetV2 model
def build_model():

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.05, 0.05)
    ])

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))

    x = augmentation(inputs)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


# Training
def train_model():

    train_ds, val_ds, test_ds, class_names = load_datasets()

    print("\nMRI Classes:", class_names)

    model, base_model = build_model()

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            verbose=1
        )
    ]

    print("\nPHASE 1: Training classification head")

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks
    )

    print("\nPHASE 2: Fine-tuning MobileNetV2")

    base_model.trainable = True

    # Unfreeze only the final 30 layers.
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Keep batch normalization layers frozen.
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE,
        callbacks=callbacks
    )

    # Load best checkpoint
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(MODEL_DIR / "class_names.json", "w") as file:
        json.dump(class_names, file)

    print("\nFINAL TEST EVALUATION")

    test_loss, test_accuracy = model.evaluate(test_ds)

    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Predictions
    y_true = np.concatenate([
        labels.numpy().argmax(axis=1)
        for _, labels in test_ds
    ])

    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    print("\nClassification Report:\n")

    report = classification_report(
        y_true,
        y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0
    )

    print(report)

    with open(MODEL_DIR / "classification_report.txt", "w") as file:
        file.write(report)

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=range(len(class_names))
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(cmap="Blues", xticks_rotation=45)
    plt.title("NeuroScan AI - MobileNetV2 Confusion Matrix")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png")
    plt.show()

    # Training history
    acc = history1.history["accuracy"] + history2.history["accuracy"]
    val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]

    plt.figure(figsize=(10, 5))
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("NeuroScan AI - MobileNetV2 Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "training_history.png")
    plt.show()

    print("\nMobileNetV2 training completed!")
    print("Best model saved at:", MODEL_PATH)


if __name__ == "__main__":
    train_model()