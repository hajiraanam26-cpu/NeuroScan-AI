
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from .preprocess import TRAIN_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE, SEED


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

PHASE1_PATH = MODEL_DIR / "neuroscan_focal_phase1.keras"
PHASE2_PATH = MODEL_DIR / "neuroscan_focal_phase2.keras"
FINAL_PATH = MODEL_DIR / "neuroscan_mobilenet_improved.keras"

NUM_CLASSES = 4
EPOCHS_HEAD = 12
EPOCHS_FINE = 8

AUTOTUNE = tf.data.AUTOTUNE


# --------------------------------------------------
# LOAD DATASETS
# --------------------------------------------------

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
        raise ValueError("Expected exactly four classes.")

    if val_ds.class_names != class_names:
        raise ValueError("Validation class order mismatch.")

    if test_ds.class_names != class_names:
        raise ValueError("Testing class order mismatch.")

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


# --------------------------------------------------
# BUILD EXPERIMENTAL MODEL
# --------------------------------------------------

def build_model():

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.04),
        tf.keras.layers.RandomZoom(0.08),
        tf.keras.layers.RandomTranslation(0.04, 0.04),
        tf.keras.layers.RandomContrast(0.10)
    ], name="MRI_augmentation")

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

    x = tf.keras.layers.Dropout(0.35)(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(0.35)(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0005
        ),
        loss=tf.keras.losses.CategoricalFocalCrossentropy(
            gamma=2.0
        ),
        metrics=["accuracy"]
    )

    return model, base_model


# --------------------------------------------------
# CALLBACKS
# --------------------------------------------------

def get_callbacks(checkpoint_path):

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
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
            min_lr=1e-7,
            verbose=1
        )
    ]


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

def train_model():

    tf.keras.utils.set_random_seed(SEED)

    train_ds, val_ds, test_ds, class_names = load_datasets()

    print("\nClasses:", class_names)

    model, base_model = build_model()

    print("\nPHASE 1: Training classification head")

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=get_callbacks(PHASE1_PATH)
    )

    print("\nPHASE 2: Fine-tuning MobileNetV2")

    base_model.trainable = True

    for layer in base_model.layers[:-40]:
        layer.trainable = False

    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.00001
        ),
        loss=tf.keras.losses.CategoricalFocalCrossentropy(
            gamma=2.0
        ),
        metrics=["accuracy"]
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE,
        callbacks=get_callbacks(PHASE2_PATH)
    )

    # --------------------------------------------------
    # SELECT MODEL USING VALIDATION LOSS
    # --------------------------------------------------

    print("\nComparing validation performance...")

    phase1_model = tf.keras.models.load_model(PHASE1_PATH)
    phase2_model = tf.keras.models.load_model(PHASE2_PATH)

    phase1_loss = phase1_model.evaluate(
        val_ds, verbose=0
    )[0]

    phase2_loss = phase2_model.evaluate(
        val_ds, verbose=0
    )[0]

    print(f"Phase 1 validation loss: {phase1_loss:.4f}")
    print(f"Phase 2 validation loss: {phase2_loss:.4f}")

    if phase2_loss < phase1_loss:
        best_model = phase2_model
        selected_phase = "Fine-tuned model"
    else:
        best_model = phase1_model
        selected_phase = "Classification-head model"

    best_model.save(FINAL_PATH)

    print("\nSelected:", selected_phase)
    print("Saved:", FINAL_PATH)

    with open(MODEL_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f)

    # --------------------------------------------------
    # FINAL TEST EVALUATION
    # --------------------------------------------------

    print("\nFINAL TEST EVALUATION")

    y_true = np.concatenate([
        labels.numpy().argmax(axis=1)
        for _, labels in test_ds
    ])

    probabilities = best_model.predict(
        test_ds,
        verbose=1
    )

    y_pred = np.argmax(probabilities, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        labels=range(NUM_CLASSES),
        target_names=class_names,
        zero_division=0
    )

    print("\nClassification Report:\n")
    print(report)

    with open(
        MODEL_DIR / "improved_classification_report.txt",
        "w"
    ) as f:
        f.write(report)

    # Metrics
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "selected_phase": selected_phase
    }

    with open(
        MODEL_DIR / "improved_metrics.json",
        "w"
    ) as f:
        json.dump(metrics, f, indent=4)

    print("\nOverall Metrics:")
    print(json.dumps(metrics, indent=4))

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=range(NUM_CLASSES)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(cmap="Blues", xticks_rotation=45)

    plt.title("NeuroScan AI - Focal Loss Model")
    plt.tight_layout()

    plt.savefig(
        MODEL_DIR / "improved_confusion_matrix.png",
        dpi=150
    )

    plt.close()

    # Training history
    acc = (
        history1.history["accuracy"]
        + history2.history["accuracy"]
    )

    val_acc = (
        history1.history["val_accuracy"]
        + history2.history["val_accuracy"]
    )

    plt.figure(figsize=(10, 5))

    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("NeuroScan AI - Improved Training History")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        MODEL_DIR / "improved_training_history.png",
        dpi=150
    )

    plt.close()

    print("\nExperimental training completed!")


if __name__ == "__main__":
    train_model()
