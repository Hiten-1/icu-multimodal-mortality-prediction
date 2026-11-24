"""Unit tests for data loading, patient-level splitting, and preprocessing."""

import unittest
import numpy as np

from data.synthetic.generate_synthetic_data import generate_synthetic_cohort
from src.config import ProjectConfig
from src.data_loader import patient_level_split
from src.preprocessing import (
    ClinicalTextVectorizer,
    StructuredPreprocessor,
    clean_clinical_note,
)


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.cfg = ProjectConfig(n_patients=120, n_stays=180, random_seed=42)
        self.df = generate_synthetic_cohort(
            n_patients=self.cfg.n_patients,
            n_stays=self.cfg.n_stays,
            seed=self.cfg.random_seed,
        )

    def test_patient_level_split_zero_leakage(self):
        tr, va, te = patient_level_split(self.df, seed=42)
        tr_ids = set(tr["subject_id"])
        va_ids = set(va["subject_id"])
        te_ids = set(te["subject_id"])
        self.assertTrue(tr_ids.isdisjoint(va_ids))
        self.assertTrue(tr_ids.isdisjoint(te_ids))
        self.assertTrue(va_ids.isdisjoint(te_ids))
        self.assertEqual(len(tr) + len(va) + len(te), len(self.df))

    def test_structured_preprocessor_no_nans(self):
        tr, va, _ = patient_level_split(self.df, seed=42)
        prep = StructuredPreprocessor(self.cfg)
        x_tr = prep.fit_transform(tr)
        x_va = prep.transform(va)
        self.assertFalse(np.isnan(x_tr).any())
        self.assertFalse(np.isnan(x_va).any())
        self.assertEqual(x_tr.shape[1], x_va.shape[1])
        # 21 numerics + 21 missingness flags + 10 one-hot categoricals = 52
        self.assertEqual(x_tr.shape[1], 52)

    def test_clinical_text_cleaning_and_vectorization(self):
        raw = "ICU Note [**2026-01-01**]: HR 110, SBP 85, SpO2 89%, Cr 2.4."
        cleaned = clean_clinical_note(raw)
        self.assertIn("deid token", cleaned)
        self.assertIn("heart rate", cleaned)
        self.assertIn("systolic blood pressure", cleaned)
        self.assertEqual(clean_clinical_note(None), "")  # type: ignore[arg-type]

        vec = ClinicalTextVectorizer(max_features=200, embed_dim=16)
        x_t = vec.fit_transform(self.df["clinical_note"].tolist())
        self.assertEqual(x_t.shape[0], len(self.df))
        self.assertEqual(x_t.shape[1], 16)
        self.assertFalse(np.isnan(x_t).any())


if __name__ == "__main__":
    unittest.main()
