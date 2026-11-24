"""Integration tests for end-to-end prediction and evaluation metrics."""

import tempfile
import unittest
from pathlib import Path
import numpy as np

from data.synthetic.generate_synthetic_data import generate_synthetic_cohort
from src.config import ProjectConfig
from src.evaluate import compute_classification_metrics
from src.predict import classify_risk_tier, predict_patient_mortality
from src.train import train_all_models


class TestPredictionPipeline(unittest.TestCase):
    def test_risk_tier_classification(self):
        self.assertIn("Low", classify_risk_tier(0.12))
        self.assertIn("Moderate", classify_risk_tier(0.38))
        self.assertIn("High", classify_risk_tier(0.76))

    def test_metrics_computation(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.7, 0.4, 0.9])
        m = compute_classification_metrics(y_true, y_prob, threshold=0.5)
        self.assertAlmostEqual(m["accuracy"], 1.0)
        self.assertAlmostEqual(m["roc_auc"], 1.0)
        self.assertAlmostEqual(m["specificity"], 1.0)
        self.assertIn("brier_score", m)

    def test_end_to_end_train_and_predict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "mini_cohort.csv"
            generate_synthetic_cohort(
                n_patients=150, n_stays=200, seed=42, output_path=csv_path
            )
            cfg = ProjectConfig(
                n_patients=150, n_stays=200, epochs=8, text_embed_dim=16
            )
            train_all_models(
                csv_path=csv_path,
                config=cfg,
                models_dir=Path(tmpdir),
                metrics_dir=Path(tmpdir),
            )
            sample = {
                "subject_id": "TEST-01",
                "age": 68.0,
                "gender": "F",
                "admission_type": "EMERGENCY",
                "first_careunit": "MICU",
                "heart_rate_mean": 110.0,
                "heart_rate_max": 128.0,
                "sbp_mean": 92.0,
                "sbp_min": 78.0,
                "dbp_mean": 54.0,
                "resp_rate_mean": 24.0,
                "spo2_mean": 92.0,
                "spo2_min": 87.0,
                "temperature_c": 38.0,
                "wbc_count": 15.0,
                "hemoglobin": 10.2,
                "platelets": 140.0,
                "creatinine": 2.2,
                "bun": 36.0,
                "lactate": 3.8,
                "sodium": 137.0,
                "potassium": 4.8,
                "bicarbonate": 18.0,
                "glucose": 180.0,
                "gcs_total": 11,
                "clinical_note": "ICU 24h Progress Note: septic shock with lactic acidosis.",
            }
            res = predict_patient_mortality(
                sample, bundle_path=Path(tmpdir) / "multimodal_bundle.pkl"
            )
            self.assertIn("predicted_mortality_probability", res)
            self.assertGreaterEqual(res["predicted_mortality_probability"], 0.0)
            self.assertLessEqual(res["predicted_mortality_probability"], 1.0)


if __name__ == "__main__":
    unittest.main()
