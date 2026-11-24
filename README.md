# ICU Multimodal Mortality Prediction

### Multimodal Fusion of Clinical Notes and Structured Data for ICU Mortality Prediction

An educational deep learning project that explores ICU mortality risk prediction using two complementary data modalities: structured physiological data and unstructured clinical notes.

The project implements a **Gated Multimodal Fusion Architecture** that combines information from both modalities to estimate mortality risk.

> **Disclaimer:** This is an educational research prototype trained using synthetic ICU data. Its predictions are not clinically validated and must never be used for actual medical diagnosis or patient-care decisions.

---

## Overview

Predicting mortality risk in Intensive Care Units (ICUs) requires analysing multiple types of patient information.

Traditional approaches often focus on structured clinical measurements, while valuable information is also present in unstructured clinical notes.

This project addresses this challenge through a multimodal deep learning architecture that integrates:

- Structured physiological measurements and laboratory results.
- Unstructured clinical notes.
- Deep learning-based feature extraction.
- Gated multimodal fusion.
- Binary mortality risk prediction.

The model uses information from the first 24 hours of ICU admission to estimate the mortality outcome.

## Key Features

- **Structured Data Processing:** Handles physiological measurements, laboratory values and categorical patient information.
- **Clinical Text Processing:** Processes clinical notes using medical language representations.
- **Dual-Branch Architecture:** Uses separate encoders for structured and textual information.
- **Gated Multimodal Fusion:** Learns how to combine information from both branches.
- **Mortality Prediction:** Produces a probability score and binary prediction.
- **Risk Categorization:** Categorizes predictions into Low, Moderate and High demonstration risk levels.
- **Model Comparison:** Compares structured-only, text-only and multimodal approaches.
- **Performance Evaluation:** Supports metrics such as Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC and Brier Score.

## Model Architecture

The architecture consists of two parallel branches.

### 1. Structured Data Branch

Processes numerical and categorical patient information.

- Numerical feature preprocessing.
- Missing-value handling.
- Missingness indicators.
- Categorical feature encoding.
- Feature normalization.
- Neural network-based feature extraction.

The structured encoder generates a 24-dimensional latent representation.

### 2. Clinical Text Branch

Processes unstructured clinical notes.

Primary text representation:

- Bio_ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`)
- WordPiece tokenization.
- CLS token representation.
- Projection into a 24-dimensional latent representation.

An offline TF-IDF and Truncated SVD-based alternative is also described in the project methodology.

### 3. Gated Multimodal Fusion

The two 24-dimensional representations are concatenated into a 48-dimensional vector.

A sigmoid-based gating mechanism learns the contribution of the combined features before the fused representation is passed through the final prediction layer.

The model produces a mortality probability using a sigmoid output.

**Architecture flow:**

Structured Data → Structured Encoder ─┐

Clinical Notes → Text Encoder ────────┤

                                      ↓

                             Feature Concatenation

                                      ↓

                              Gated Fusion Layer

                                      ↓

                              Fused Representation

                                      ↓

                             Binary Classification

                                      ↓

                            Mortality Probability

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Clinical Language Model | Bio_ClinicalBERT |
| Text Processing | Transformers, Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Application Interface | Streamlit |
| Testing | Pytest |

## Project Structure

```text
icu-multimodal-mortality-prediction/
│
├── app/
│   └── app.py
│
├── data/
│   └── synthetic/
│
├── docs/
│
├── notebooks/
│
├── outputs/
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── structured_encoder.py
│   ├── text_encoder.py
│   ├── fusion_model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Dataset

The project methodology uses a synthetically generated ICU cohort.

- 2,500 ICU stays configured for the synthetic cohort.
- Patient-level training, validation and testing split.
- 70% Training.
- 15% Validation.
- 15% Testing.
- Random seed: 42.

The dataset is synthetic and does not contain real patient records.

## Model Evaluation

Three approaches are considered:

1. Structured-only model.
2. Text-only model.
3. Gated multimodal fusion model.

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Brier Score
- Confusion Matrix
- Specificity

The project report presents comparative results across these three approaches.

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/icu-multimodal-mortality-prediction.git
```

### 2. Navigate into the project

```bash
cd icu-multimodal-mortality-prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch the application

```bash
streamlit run app/app.py
```

The application should open in your browser at:

`http://localhost:8501`

Note: The application may require model artifacts to be generated or downloaded before predictions can run. Follow the project's training and configuration requirements if prompted.

## Future Scope

- Evaluation on appropriately authorized real-world clinical datasets.
- Improved model calibration and external validation.
- Explainable AI for understanding model predictions.
- Additional clinical modalities.
- Improved generalization across different ICU populations.
- More comprehensive model monitoring and evaluation.

## Project Objective

The primary objective is to investigate whether combining structured physiological information with unstructured clinical narratives through gated multimodal fusion can improve mortality prediction compared with using either modality independently.

---

**Project Type:** Academic Research / Deep Learning / Healthcare AI

**Application:** ICU Mortality Risk Prediction

**Model:** Gated Multimodal Fusion Network

**Dataset:** Synthetic ICU Data

**Status:** Educational Research Prototype
