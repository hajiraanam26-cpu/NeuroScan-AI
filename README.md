# 🧠 NeuroScan AI
### AI-Powered Brain MRI Classification and Explainability

NeuroScan AI is a deep learning project that classifies brain MRI images into four categories using transfer learning and convolutional neural networks. It also provides visual explanations using Grad-CAM.

## 🚀 Key Features

- Brain MRI classification into four categories:
  - Glioma
  - Meningioma
  - No Tumor
  - Pituitary
- Transfer learning using MobileNetV2.
- CNN baseline model for comparison.
- Experimental model using focal loss.
- Grad-CAM visual explanations.
- Interactive Streamlit web application.
- Model performance comparison dashboard.
- Automatically generated PDF evaluation report.

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- MobileNetV2
- Scikit-learn
- NumPy and Pandas
- Matplotlib
- Streamlit
- Grad-CAM
- ReportLab

## 📊 Model Performance

Models were evaluated on the same held-out test dataset containing 1,600 MRI images.

| Model | Accuracy | Precision | Recall | Macro F1 |
|---|---:|---:|---:|---:|
| CNN | 63.56% | 68.45% | 63.56% | 61.23% |
| MobileNetV2 | **86.44%** | **87.51%** | **86.44%** | **86.07%** |
| Improved MobileNetV2 | 84.75% | 85.63% | 84.75% | 84.26% |

**Best-performing model:** MobileNetV2.

## 🧠 Explainable AI

NeuroScan AI uses Grad-CAM to generate heatmap overlays that highlight image regions influencing the model's prediction.

These visualizations help demonstrate model behavior but do not establish tumor location or provide medical evidence.

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd NeuroScan-AI
```

### 2. Create a virtual environment

```powershell
py -3.11 -m venv .venv
```

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📁 Dataset Structure

The project uses a four-category brain MRI image dataset.

```text
data/
└── classification/
    ├── Training/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    └── Testing/
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/
```

## 📄 Evaluation Report

A PDF evaluation report is available at:

`reports/neuroscan_evaluation_report.pdf`

It includes the model comparison, evaluation results, interpretation, and limitations.

## ⚠️ Limitations and Medical Disclaimer

- This is an educational AI/ML project, not a clinically validated diagnostic system.
- Predictions may be incorrect.
- The dataset and reported results do not establish real-world clinical performance.
- Grad-CAM visualizations are model explanations, not verified tumor segmentations.
- This application must not be used for medical diagnosis or treatment decisions.

## 👩‍💻 Project Goal

The goal of NeuroScan AI is to explore deep learning, transfer learning, medical image classification, model evaluation, and explainable AI through an interactive application.

**Built as an educational AI and Machine Learning project.**