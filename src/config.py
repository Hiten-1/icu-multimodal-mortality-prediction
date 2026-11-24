"""Configuration settings for the Multimodal ICU Mortality Prediction project."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "synthetic"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
SCREENSHOTS_DIR = OUTPUT_DIR / "screenshots"
MODELS_DIR = OUTPUT_DIR / "models"


@dataclass
class ProjectConfig:
    """Centralized configuration for reproducibility and pipeline control."""

    random_seed: int = 42
    n_patients: int = 1800
    n_stays: int = 2500
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Structured clinical features (first 24 hours of ICU stay)
    numeric_features: List[str] = field(
        default_factory=lambda: [
            "age",
            "heart_rate_mean",
            "heart_rate_max",
            "sbp_mean",
            "sbp_min",
            "dbp_mean",
            "resp_rate_mean",
            "spo2_mean",
            "spo2_min",
            "temperature_c",
            "wbc_count",
            "hemoglobin",
            "platelets",
            "creatinine",
            "bun",
            "lactate",
            "sodium",
            "potassium",
            "bicarbonate",
            "glucose",
            "gcs_total",
        ]
    )

    categorical_features: List[str] = field(
        default_factory=lambda: [
            "gender",
            "admission_type",
            "first_careunit",
        ]
    )

    # Text & Transformer settings
    pretrained_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    max_text_length: int = 256
    text_vocab_size: int = 1200
    text_embed_dim: int = 64

    # Neural network & training hyperparameters
    structured_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    text_hidden_dim: int = 32
    fusion_hidden_dim: int = 24
    dropout_rate: float = 0.25
    learning_rate: float = 0.002
    weight_decay: float = 0.015
    batch_size: int = 64
    epochs: int = 45
    early_stopping_patience: int = 12
    decision_threshold: float = 0.50
    pos_weight_cap: float = 6.0

    # Execution backend: "auto", "pytorch", or "numpy"
    backend: str = "auto"
