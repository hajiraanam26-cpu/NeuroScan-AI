
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "neuroscan_mobilenet.keras"
CLASS_PATH = BASE_DIR / "models" / "class_names.json"
COMPARISON_PATH = BASE_DIR / "models" / "model_comparison.csv"

IMAGE_SIZE = (224, 224)


# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 3. CUSTOM STYLING
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
}

section[data-testid="stSidebar"] {
    background-color: #111c30;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #38bdf8;
}

.subtitle {
    color: #94a3b8;
    font-size: 16px;
}

.metric-card {
    background: #17243a;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #263b58;
}

.result-title {
    color: #38bdf8;
    font-size: 25px;
    font-weight: bold;
}

div.stButton > button {
    background-color: #0284c7;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_neuroscan_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Train the model first."
        )

    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():

    if not CLASS_PATH.exists():
        raise FileNotFoundError(
            "class_names.json not found."
        )

    with open(CLASS_PATH, "r") as file:
        return json.load(file)


@st.cache_data
def load_model_comparison():

    if not COMPARISON_PATH.exists():
        raise FileNotFoundError(
            "Model comparison CSV not found. "
            "Run: python -m src.model_comparison"
        )

    df = pd.read_csv(COMPARISON_PATH)

    required_columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing CSV columns: {missing_columns}"
        )

    return df


try:
    model = load_neuroscan_model()
    class_names = load_class_names()

except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()


# ==========================================
# 5. IMAGE PREPROCESSING
# ==========================================

def preprocess_image(uploaded_image):

    image = uploaded_image.convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_batch = np.expand_dims(
        image_array,
        axis=0
    )

    return image, image_batch


# ==========================================
# 6. GRAD-CAM EXPLAINABILITY
# ==========================================

def generate_gradcam(img_array):

    # Locate nested MobileNetV2
    base_model = next(
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model)
        and any(
            sublayer.name == "out_relu"
            for sublayer in layer.layers
        )
    )

    base_index = model.layers.index(base_model)

    last_conv = base_model.get_layer("out_relu")

    # Feature extraction model
    feature_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[
            last_conv.output,
            base_model.output
        ]
    )

    # MobileNetV2 preprocessing
    preprocessed = (
        tf.keras.applications.mobilenet_v2.preprocess_input(
            tf.convert_to_tensor(img_array)
        )
    )

    with tf.GradientTape() as tape:

        conv_maps, features = feature_model(
            preprocessed,
            training=False
        )

        # Pass extracted features through classifier
        x = features

        for layer in model.layers[base_index + 1:]:
            x = layer(x, training=False)

        predictions = x

        predicted_class = tf.argmax(predictions[0])

        class_score = predictions[:, predicted_class]

    gradients = tape.gradient(
        class_score,
        conv_maps
    )

    if gradients is None:
        raise RuntimeError("Grad-CAM gradients unavailable.")

    weights = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_maps = conv_maps[0]

    heatmap = tf.reduce_sum(
        conv_maps * weights,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        IMAGE_SIZE
    )

    return heatmap.numpy().squeeze(), predictions.numpy()[0]


# ==========================================
# 7. SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:

    st.markdown("## 🧠 NeuroScan AI")

    st.caption("AI-Powered Brain MRI Classification")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "MRI Classification",
            "Model Performance",
            "About the Project"
        ]
    )

    st.divider()

    st.markdown("### Model Information")

    st.write("Architecture: MobileNetV2")
    st.write("Image Size: 224 × 224")
    st.write("Classes: 4")
    st.write("Explainability: Grad-CAM")

    st.divider()

    st.caption("Educational AI research prototype")


# ==========================================
# 8. DASHBOARD
# ==========================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">🧠 NeuroScan AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Deep Learning Based Brain MRI Classification '
        'and Visual Explainability'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("AI Architecture", "MobileNetV2")
    col2.metric("MRI Categories", "4")
    col3.metric("Input Resolution", "224 × 224")
    col4.metric("Test Accuracy", "86.44%")

    st.caption(
        "Accuracy is measured on this project's held-out test dataset. "
        "It does not guarantee performance on new images."
    )

    st.divider()

    st.subheader("Supported MRI Categories")

    categories = {
        "Glioma": "A category of brain tumour.",
        "Meningioma": "A category of tumour arising from the meninges.",
        "No Tumor": "Images labelled as no tumour in the dataset.",
        "Pituitary": "A category of pituitary-region tumour."
    }

    cols = st.columns(4)

    for col, (name, description) in zip(
        cols,
        categories.items()
    ):

        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>{name}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    st.info(
        "Navigate to MRI Classification to upload an image "
        "and generate a prediction with Grad-CAM visualization."
    )


# ==========================================
# 9. MRI CLASSIFICATION
# ==========================================

elif page == "MRI Classification":

    st.markdown(
        '<div class="main-title">MRI Classification</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a brain MRI image to generate an AI classification "
        "and visual explanation."
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Brain MRI Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image, image_batch = preprocess_image(
            Image.open(uploaded_file)
        )

        st.subheader("Uploaded MRI Image")

        st.image(
            image,
            caption="Uploaded MRI Scan",
            width=350
        )

        if st.button(
            "🔍 Analyze MRI Image",
            use_container_width=True
        ):

            with st.spinner("Analyzing MRI image..."):

                try:

                    heatmap, predictions = generate_gradcam(
                        image_batch
                    )

                    predicted_index = np.argmax(predictions)

                    predicted_class = class_names[
                        predicted_index
                    ]

                    confidence = (
                        float(predictions[predicted_index]) * 100
                    )

                    # Prediction results
                    st.divider()

                    st.subheader("AI Prediction Results")

                    col1, col2 = st.columns(2)

                    with col1:

                        st.markdown(
                            f'<div class="result-title">'
                            f'{predicted_class}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                        st.metric(
                            "Model Confidence",
                            f"{confidence:.2f}%"
                        )

                    with col2:

                        st.write("Prediction Probability")

                        st.progress(
                            float(np.clip(
                                predictions[predicted_index],
                                0,
                                1
                            ))
                        )

                    # Class probabilities
                    st.divider()

                    st.subheader("Class Probability Distribution")

                    probability_df = pd.DataFrame({
                        "Category": class_names,
                        "Probability": predictions * 100
                    })

                    probability_df = probability_df.sort_values(
                        "Probability",
                        ascending=False
                    )

                    st.bar_chart(
                        probability_df.set_index("Category")
                    )

                    # Grad-CAM
                    st.divider()

                    st.subheader(
                        "Grad-CAM Visual Explanation"
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.image(
                            image,
                            caption="Original MRI"
                        )

                    with col2:

                        st.image(
                            heatmap,
                            caption="Grad-CAM Heatmap",
                            clamp=True
                        )

                    with col3:

                        import matplotlib.pyplot as plt

                        fig, ax = plt.subplots(
                            figsize=(5, 5)
                        )

                        ax.imshow(image)

                        ax.imshow(
                            heatmap,
                            cmap="jet",
                            alpha=0.4
                        )

                        ax.axis("off")

                        ax.set_title("MRI + Grad-CAM")

                        st.pyplot(
                            fig,
                            clear_figure=True
                        )

                        plt.close(fig)

                    st.success(
                        "Image analysis completed successfully."
                    )

                    st.warning(
                        "This prediction is generated by an "
                        "experimental AI model. It is not a medical "
                        "diagnosis and must not be used to guide "
                        "clinical decisions."
                    )

                except Exception as error:

                    st.error(
                        f"Prediction failed: {error}"
                    )


# ==========================================
# 10. MODEL PERFORMANCE DASHBOARD
# ==========================================

elif page == "Model Performance":

    st.markdown(
        '<div class="main-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Evaluation and comparison of deep learning models'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    try:

        comparison_df = load_model_comparison()

        # Identify best model using macro F1-score
        best_index = comparison_df["F1 Score"].idxmax()

        best_model = comparison_df.loc[
            best_index
        ]

        st.subheader("Best Performing Model")

        st.success(
            f"🏆 {best_model['Model']} "
            f"has the highest macro F1-score."
        )

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Best Accuracy",
            f"{best_model['Accuracy'] * 100:.2f}%"
        )

        col2.metric(
            "Best Precision",
            f"{best_model['Precision'] * 100:.2f}%"
        )

        col3.metric(
            "Best Recall",
            f"{best_model['Recall'] * 100:.2f}%"
        )

        col4.metric(
            "Best Macro F1",
            f"{best_model['F1 Score'] * 100:.2f}%"
        )

        st.divider()

        # Model comparison table
        st.subheader("Model Comparison Results")

        display_df = comparison_df.copy()

        metric_columns = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]

        for column in metric_columns:
            display_df[column] = (
                display_df[column] * 100
            ).round(2)

        st.dataframe(
            display_df.rename(columns={
                "Accuracy": "Accuracy (%)",
                "Precision": "Precision (%)",
                "Recall": "Recall (%)",
                "F1 Score": "Macro F1 (%)"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "All values are percentages. Precision, recall and F1 "
            "are macro-averaged across the four MRI categories."
        )

        st.divider()

        # Accuracy comparison
        st.subheader("Accuracy Comparison")

        accuracy_chart = comparison_df[
            ["Model", "Accuracy"]
        ].set_index("Model")

        st.bar_chart(
            accuracy_chart,
            y="Accuracy",
            y_label="Accuracy",
            x_label="Model"
        )

        # Precision, Recall and F1
        st.subheader("Detailed Model Metrics")

        detailed_chart = comparison_df[
            ["Model", "Precision", "Recall", "F1 Score"]
        ].set_index("Model")

        st.bar_chart(
            detailed_chart,
            y=["Precision", "Recall", "F1 Score"],
            y_label="Score",
            x_label="Model"
        )

        st.divider()

        # Best model explanation
        st.subheader("Model Selection Analysis")

        st.write(
            f"**Selected model: {best_model['Model']}**"
        )

        st.write(
            f"Its macro F1-score is "
            f"**{best_model['F1 Score'] * 100:.2f}%**, "
            f"with accuracy of "
            f"**{best_model['Accuracy'] * 100:.2f}%**."
        )

        st.write(
            "The CNN baseline and experimental focal-loss "
            "MobileNetV2 were also evaluated. The original "
            "MobileNetV2 achieved the strongest overall "
            "performance among these three evaluated models."
        )

        st.divider()

        # Download comparison report
        st.subheader("Download Evaluation Report")

        csv_data = comparison_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇ Download Model Comparison CSV",
            data=csv_data,
            file_name="neuroscan_model_comparison.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.divider()

        st.info(
            "Evaluation results are based on this project's "
            "1,600-image test dataset. The test set has already "
            "been used for model comparison; further model tuning "
            "should use a separate validation set."
        )

    except Exception as error:

        st.error(
            f"Unable to load model comparison results: {error}"
        )

        st.info(
            "Check that models/model_comparison.csv exists. "
            "If it does not, run: python -m src.model_comparison"
        )


# ==========================================
# 11. ABOUT THE PROJECT
# ==========================================

elif page == "About the Project":

    st.markdown(
        '<div class="main-title">About NeuroScan AI</div>',
        unsafe_allow_html=True
    )

    st.write(
        "NeuroScan AI is an educational deep learning project "
        "designed to explore brain MRI image classification."
    )

    st.subheader("Technology Stack")

    technologies = [
        "Python",
        "TensorFlow / Keras",
        "MobileNetV2",
        "Streamlit",
        "NumPy",
        "Pandas",
        "Matplotlib",
        "Scikit-learn",
        "Grad-CAM"
    ]

    for technology in technologies:
        st.write(f"• {technology}")

    st.divider()

    st.subheader("Project Workflow")

    st.markdown("""
    1. MRI dataset collection
    2. Image preprocessing
    3. CNN baseline development
    4. MobileNetV2 transfer learning
    5. Experimental focal-loss training
    6. Model evaluation and comparison
    7. Grad-CAM explainability
    8. Interactive Streamlit application
    """)

    st.divider()

    st.subheader("Medical Disclaimer")

    st.warning(
        "NeuroScan AI is intended strictly for educational "
        "and research purposes. It is not clinically validated, "
        "does not replace radiologists, and must not be used "
        "for diagnosis or treatment decisions."
    )
