
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "neuroscan_mobilenet.keras"
TEST_DIR = BASE_DIR / "data" / "classification" / "Testing"
OUTPUT_DIR = BASE_DIR / "models"

# Load model and class names
model = load_model(MODEL_PATH)

with open(OUTPUT_DIR / "class_names.json") as file:
    class_names = json.load(file)


def generate_gradcam(img_array):

    # Locate nested MobileNetV2 model
    base_model = next(
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model)
        and any(
            sublayer.name == "out_relu"
            for sublayer in layer.layers
        )
    )

    # Find MobileNetV2 position in outer model
    base_index = model.layers.index(base_model)

    # Use final convolutional activation
    last_conv = base_model.get_layer("out_relu")

    # Build an internal feature extraction model
    feature_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[
            last_conv.output,
            base_model.output
        ]
    )

    # Prepare image for MobileNetV2
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(
        tf.cast(img_array, tf.float32)
    )

    with tf.GradientTape() as tape:

        # Extract convolutional maps and final features
        conv_maps, features = feature_model(
            preprocessed,
            training=False
        )

        # Pass features through original classification head
        x = features

        for layer in model.layers[base_index + 1:]:
            x = layer(x, training=False)

        predictions = x

        predicted_class = tf.argmax(
            predictions[0]
        )

        class_score = predictions[:, predicted_class]

    # Calculate gradients
    gradients = tape.gradient(class_score, conv_maps)

    if gradients is None:
        raise RuntimeError(
            "Could not calculate Grad-CAM gradients."
        )

    # Global average pooling of gradients
    weights = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    # Weighted combination of feature maps
    conv_maps = conv_maps[0]

    heatmap = tf.reduce_sum(
        conv_maps * weights,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy(), predictions.numpy()[0]


def explain_image(image_path):

    # Load MRI image
    img = load_img(
        image_path,
        target_size=(224, 224)
    )

    img_array = img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    # Generate explanation
    heatmap, predictions = generate_gradcam(img_batch)

    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class] * 100

    # Resize heatmap to original image dimensions
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        (224, 224)
    ).numpy().squeeze()

    # Display results
    plt.figure(figsize=(13, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original MRI")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_resized, cmap="jet")
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(
        heatmap_resized,
        cmap="jet",
        alpha=0.4
    )
    plt.title("MRI + Grad-CAM")
    plt.axis("off")

    plt.suptitle(
        f"Prediction: {class_names[predicted_class]} "
        f"({confidence:.2f}%)"
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "gradcam_example.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    print("\nPrediction:", class_names[predicted_class])
    print(f"Confidence: {confidence:.2f}%")
    print("Grad-CAM saved:", output_path)


if __name__ == "__main__":

    sample_class = TEST_DIR / class_names[0]

    sample_images = sorted([
        path for path in sample_class.iterdir()
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    if not sample_images:
        raise FileNotFoundError(
            f"No images found in {sample_class}"
        )

    explain_image(sample_images[0])
