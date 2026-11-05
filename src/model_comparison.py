import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from .preprocess import TEST_DIR, IMAGE_SIZE, BATCH_SIZE


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODELS = {
    "CNN": MODEL_DIR / "neuroscan_cnn.keras",
    "MobileNetV2": MODEL_DIR / "neuroscan_mobilenet.keras",
    "Improved MobileNetV2": MODEL_DIR / "neuroscan_mobilenet_improved.keras"
}


def load_test_dataset():

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    return test_ds.prefetch(tf.data.AUTOTUNE)


def evaluate_model(model_name, model_path, test_ds):

    print(f"\nEvaluating: {model_name}")

    model = tf.keras.models.load_model(model_path)

    y_true = np.concatenate([
        labels.numpy().argmax(axis=1)
        for _, labels in test_ds
    ])

    probabilities = model.predict(
        test_ds,
        verbose=0
    )

    y_pred = np.argmax(probabilities, axis=1)

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    }


def main():

    test_ds = load_test_dataset()

    results = []

    for model_name, model_path in MODELS.items():

        if not model_path.exists():
            print(f"\nSkipping {model_name}")
            print(f"Model not found: {model_path}")
            continue

        result = evaluate_model(
            model_name,
            model_path,
            test_ds
        )

        results.append(result)

    if not results:
        raise RuntimeError("No trained models were found.")

    df = pd.DataFrame(results)

    print("\n==============================")
    print("NEUROSCAN AI MODEL COMPARISON")
    print("==============================\n")

    print(
        df.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1 Score": "{:.4f}".format
            }
        )
    )

    output_path = MODEL_DIR / "model_comparison.csv"

    df.to_csv(
        output_path,
        index=False
    )

    print("\nComparison saved to:")
    print(output_path)


if __name__ == "__main__":
    main()