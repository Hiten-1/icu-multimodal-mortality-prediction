"""Dataset loading and patient-level splitting utilities."""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from data.synthetic.generate_synthetic_data import generate_synthetic_cohort
from src.config import DATA_DIR, ProjectConfig

REQUIRED_COLUMNS = {
    "subject_id",
    "stay_id",
    "clinical_note",
    "hospital_expire_flag",
}


def load_icu_dataset(
    csv_path: Path | None = None,
    config: ProjectConfig | None = None,
) -> pd.DataFrame:
    """Load the ICU cohort CSV, generating synthetic demonstration data if missing."""
    cfg = config or ProjectConfig()
    path = csv_path or (DATA_DIR / "synthetic_icu_cohort.csv")
    if not path.exists():
        df = generate_synthetic_cohort(
            n_patients=cfg.n_patients,
            n_stays=cfg.n_stays,
            seed=cfg.random_seed,
            output_path=path,
        )
    else:
        df = pd.read_csv(path)

    missing_req = REQUIRED_COLUMNS - set(df.columns)
    if missing_req:
        raise ValueError(f"Dataset missing required columns: {sorted(missing_req)}")

    df["clinical_note"] = df["clinical_note"].fillna("").astype(str)
    df["hospital_expire_flag"] = df["hospital_expire_flag"].astype(int)
    return df


def patient_level_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by unique subject_id to prevent patient-level data leakage."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    patient_outcomes = (
        df.groupby("subject_id")["hospital_expire_flag"].max().reset_index()
    )

    pos_patients = patient_outcomes.loc[
        patient_outcomes["hospital_expire_flag"] == 1, "subject_id"
    ].to_numpy()
    neg_patients = patient_outcomes.loc[
        patient_outcomes["hospital_expire_flag"] == 0, "subject_id"
    ].to_numpy()

    rng.shuffle(pos_patients)
    rng.shuffle(neg_patients)

    def _partition(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(arr)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        return arr[:n_train], arr[n_train : n_train + n_val], arr[n_train + n_val :]

    p_tr, p_va, p_te = _partition(pos_patients)
    n_tr, n_va, n_te = _partition(neg_patients)

    train_ids = set(np.concatenate([p_tr, n_tr]))
    val_ids = set(np.concatenate([p_va, n_va]))
    test_ids = set(np.concatenate([p_te, n_te]))

    # Verify zero overlap across splits
    assert train_ids.isdisjoint(val_ids), "Patient leakage between train and val!"
    assert train_ids.isdisjoint(test_ids), "Patient leakage between train and test!"
    assert val_ids.isdisjoint(test_ids), "Patient leakage between val and test!"

    train_df = df[df["subject_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["subject_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["subject_id"].isin(test_ids)].reset_index(drop=True)

    return train_df, val_df, test_df
