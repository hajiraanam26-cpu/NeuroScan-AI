
from pathlib import Path

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "models" / "model_comparison.csv"

REPORT_DIR = BASE_DIR / "reports"

REPORT_PATH = REPORT_DIR / "neuroscan_evaluation_report.pdf"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD MODEL COMPARISON DATA
# ==========================================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Comparison CSV not found: {CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

required_columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

if not all(column in df.columns for column in required_columns):
    raise ValueError("The comparison CSV has unexpected columns.")


# ==========================================
# 3. IDENTIFY BEST MODEL
# ==========================================

best_index = df["F1 Score"].idxmax()

best_model = df.loc[best_index]

# Convert metrics to percentages for display
metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

# ==========================================
# 4. PDF DOCUMENT SETUP
# ==========================================

document = SimpleDocTemplate(
    str(REPORT_PATH),
    pagesize=A4,
    rightMargin=45,
    leftMargin=45,
    topMargin=45,
    bottomMargin=45
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "ReportTitle",
    parent=styles["Title"],
    fontSize=24,
    leading=30,
    textColor=colors.HexColor("#075985"),
    alignment=TA_CENTER,
    spaceAfter=10
)

subtitle_style = ParagraphStyle(
    "ReportSubtitle",
    parent=styles["Normal"],
    fontSize=11,
    leading=16,
    textColor=colors.HexColor("#475569"),
    alignment=TA_CENTER,
    spaceAfter=20
)

heading_style = ParagraphStyle(
    "SectionHeading",
    parent=styles["Heading2"],
    fontSize=15,
    leading=20,
    textColor=colors.HexColor("#075985"),
    spaceBefore=14,
    spaceAfter=8
)

body_style = ParagraphStyle(
    "ReportBody",
    parent=styles["BodyText"],
    fontSize=9.5,
    leading=15,
    spaceAfter=8
)

disclaimer_style = ParagraphStyle(
    "Disclaimer",
    parent=styles["BodyText"],
    fontSize=9,
    leading=14,
    textColor=colors.HexColor("#7f1d1d"),
    backColor=colors.HexColor("#fee2e2"),
    borderPadding=9,
    spaceBefore=8
)


# ==========================================
# 5. BUILD REPORT CONTENT
# ==========================================

story = []

# Title
story.append(
    Paragraph(
        "NEUROSCAN AI",
        title_style
    )
)

story.append(
    Paragraph(
        "Brain MRI Classification — Model Evaluation Report",
        subtitle_style
    )
)

story.append(
    HRFlowable(
        width="100%",
        thickness=1,
        color=colors.HexColor("#38bdf8")
    )
)

story.append(Spacer(1, 15))


# Project overview
story.append(
    Paragraph(
        "1. Project Overview",
        heading_style
    )
)

story.append(
    Paragraph(
        "NeuroScan AI is an educational deep learning project "
        "developed to explore brain MRI image classification "
        "using convolutional neural networks and transfer learning.",
        body_style
    )
)

story.append(
    Paragraph(
        "The system evaluates four dataset categories: glioma, "
        "meningioma, no tumor and pituitary.",
        body_style
    )
)


# Dataset details
story.append(
    Paragraph(
        "2. Evaluation Dataset",
        heading_style
    )
)

dataset_data = [
    ["Parameter", "Details"],
    ["Evaluation images", "1,600"],
    ["Number of categories", "4"],
    ["Images per category", "400"],
    ["Input resolution", "224 × 224"],
    ["Evaluation approach", "Held-out test dataset"]
]

dataset_table = Table(
    dataset_data,
    colWidths=[2.4 * inch, 3.4 * inch]
)

dataset_table.setStyle(
    TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#075985")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f1f5f9")),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ])
)

story.append(dataset_table)


# Model comparison
story.append(
    Paragraph(
        "3. Model Performance Comparison",
        heading_style
    )
)

table_data = [
    [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "Macro F1"
    ]
]

for _, row in df.iterrows():

    table_data.append([
        str(row["Model"]),
        f"{row['Accuracy'] * 100:.2f}%",
        f"{row['Precision'] * 100:.2f}%",
        f"{row['Recall'] * 100:.2f}%",
        f"{row['F1 Score'] * 100:.2f}%"
    ])

comparison_table = Table(
    table_data,
    colWidths=[
        1.55 * inch,
        1.05 * inch,
        1.05 * inch,
        1.05 * inch,
        1.05 * inch
    ],
    repeatRows=1
)

comparison_table.setStyle(
    TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#075985")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f1f5f9")),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, best_index + 1), (-1, best_index + 1),
         colors.HexColor("#dbeafe")),
        ("FONTNAME", (0, best_index + 1), (-1, best_index + 1),
         "Helvetica-Bold")
    ])
)

story.append(comparison_table)


# Best model
story.append(
    Paragraph(
        "4. Best Performing Model",
        heading_style
    )
)

story.append(
    Paragraph(
        f"<b>{best_model['Model']}</b> achieved the highest "
        f"macro F1-score of "
        f"<b>{best_model['F1 Score'] * 100:.2f}%</b> "
        f"and accuracy of "
        f"<b>{best_model['Accuracy'] * 100:.2f}%</b> "
        "among the evaluated models.",
        body_style
    )
)


# Interpretation
story.append(
    Paragraph(
        "5. Evaluation Interpretation",
        heading_style
    )
)

story.append(
    Paragraph(
        "The results show that transfer learning with MobileNetV2 "
        "performed better overall than the CNN baseline and the "
        "experimental focal-loss MobileNetV2 model on this test set. "
        "The comparison is based on the reported metrics and does "
        "not establish clinical effectiveness.",
        body_style
    )
)


# Limitations
story.append(
    Paragraph(
        "6. Limitations",
        heading_style
    )
)

story.append(
    Paragraph(
        "The dataset is limited to four image categories. "
        "Performance on external datasets, different MRI scanners, "
        "image acquisition protocols and real clinical populations "
        "has not been established. The test dataset has already "
        "been used for model comparison, so further model tuning "
        "should use a separate validation dataset.",
        body_style
    )
)


# Medical disclaimer
story.append(
    Paragraph(
        "7. Medical Disclaimer",
        heading_style
    )
)

story.append(
    Paragraph(
        "<b>NeuroScan AI is an educational and research prototype "
        "only. It is not clinically validated, does not replace "
        "radiologists or medical professionals, and must not be "
        "used for diagnosis, treatment or clinical decision-making.</b>",
        disclaimer_style
    )
)


# ==========================================
# 6. FOOTER
# ==========================================

def add_footer(canvas, doc):

    canvas.saveState()

    canvas.setFont("Helvetica", 8)

    canvas.setFillColor(
        colors.HexColor("#64748b")
    )

    canvas.drawString(
        45,
        25,
        "NeuroScan AI | Educational Research Project"
    )

    canvas.drawRightString(
        A4[0] - 45,
        25,
        f"Page {doc.page}"
    )

    canvas.restoreState()


# ==========================================
# 7. GENERATE PDF
# ==========================================

document.build(
    story,
    onFirstPage=add_footer,
    onLaterPages=add_footer
)

print("\n===================================")
print("NEUROSCAN AI REPORT GENERATED")
print("===================================")
print(f"Report saved to:\n{REPORT_PATH}")
print("===================================")
