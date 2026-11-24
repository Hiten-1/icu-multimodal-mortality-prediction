"""Structured clinical data and unstructured clinical note preprocessing."""

import re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.config import ProjectConfig

ABBREVIATION_MAP = {
    r"\bhr\b": "heart rate",
    r"\bsbp\b": "systolic blood pressure",
    r"\bdbp\b": "diastolic blood pressure",
    r"\brr\b": "respiratory rate",
    r"\bspo2\b": "oxygen saturation",
    r"\bwbc\b": "white blood cell count",
    r"\bcr\b": "creatinine",
    r"\bgcs\b": "glasgow coma scale",
    r"\bicu\b": "intensive care unit",
}


def clean_clinical_note(text: str) -> str:
    """Clean and normalize unstructured clinical note text."""
    if not isinstance(text, str):
        return ""
    cleaned = text.lower()
    # Replace MIMIC-style de-identification brackets [**...**] with neutral token
    cleaned = re.sub(r"\[\*\*.*?\*\*\]", " deid_token ", cleaned)
    for pattern, replacement in ABBREVIATION_MAP.items():
        cleaned = re.sub(pattern, replacement, cleaned)
    cleaned = re.sub(r"[^a-z0-9\s\.\-/]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class StructuredPreprocessor:
    """Fit-transform preprocessor for structured clinical data without data leakage."""

    def __init__(self, config: ProjectConfig | None = None):
        self.config = config or ProjectConfig()
        self.medians_: Dict[str, float] = {}
        self.means_: np.ndarray | None = None
        self.stds_: np.ndarray | None = None
        self.cat_columns_: List[str] = []
        self.feature_names_: List[str] = []
        self.fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> "StructuredPreprocessor":
        """Compute imputation medians, categorical levels, and z-score statistics on train set."""
        num_cols = self.config.numeric_features
        for col in num_cols:
            median_val = float(df[col].median()) if col in df.columns else 0.0
            self.medians_[col] = 0.0 if np.isnan(median_val) else median_val

        num_matrix, missing_flags, num_names = self._extract_numeric_and_flags(df)
        cat_df = pd.get_dummies(
            df[self.config.categorical_features].fillna("UNKNOWN"),
            prefix=self.config.categorical_features,
            dtype=float,
        )
        self.cat_columns_ = list(cat_df.columns)

        full_matrix = np.hstack([num_matrix, missing_flags, cat_df.to_numpy(dtype=float)])
        self.means_ = full_matrix.mean(axis=0)
        stds = full_matrix.std(axis=0)
        self.stds_ = np.where(stds < 1e-6, 1.0, stds)
        self.feature_names_ = num_names + self.cat_columns_
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform DataFrame into standardized numerical feature matrix."""
        if not self.fitted_:
            raise RuntimeError("StructuredPreprocessor must be fitted before transform().")

        num_matrix, missing_flags, _ = self._extract_numeric_and_flags(df)
        cat_df = pd.get_dummies(
            df[self.config.categorical_features].fillna("UNKNOWN"),
            prefix=self.config.categorical_features,
            dtype=float,
        )
        cat_df = cat_df.reindex(columns=self.cat_columns_, fill_value=0.0)
        full_matrix = np.hstack([num_matrix, missing_flags, cat_df.to_numpy(dtype=float)])
        standardized = (full_matrix - self.means_) / self.stds_
        return standardized.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def _extract_numeric_and_flags(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        num_cols = self.config.numeric_features
        values = np.zeros((len(df), len(num_cols)), dtype=np.float64)
        flags = np.zeros((len(df), len(num_cols)), dtype=np.float64)
        names: List[str] = []

        for idx, col in enumerate(num_cols):
            series = (
                pd.to_numeric(df[col], errors="coerce")
                if col in df.columns
                else pd.Series(np.nan, index=df.index)
            )
            is_missing = series.isna().to_numpy(dtype=np.float64)
            flags[:, idx] = is_missing
            values[:, idx] = series.fillna(self.medians_[col]).to_numpy(dtype=np.float64)
            names.append(col)

        for col in num_cols:
            names.append(f"{col}_missing")

        return values, flags, names


class ClinicalTextVectorizer:
    """TF-IDF + SVD dense text representation for lightweight/offline execution."""

    def __init__(self, max_features: int = 1200, embed_dim: int = 64):
        self.max_features = max_features
        self.embed_dim = embed_dim
        self.vocab_: Dict[str, int] = {}
        self.idf_: np.ndarray | None = None
        self.proj_: np.ndarray | None = None
        self.fitted_: bool = False

    @staticmethod
    def tokenize(text: str) -> List[str]:
        cleaned = clean_clinical_note(text)
        tokens = cleaned.split()
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        return tokens + bigrams

    def fit(self, texts: List[str]) -> "ClinicalTextVectorizer":
        doc_freq: Dict[str, int] = {}
        tokenized_docs = [self.tokenize(t) for t in texts]
        n_docs = max(len(tokenized_docs), 1)

        for tokens in tokenized_docs:
            for tok in set(tokens):
                doc_freq[tok] = doc_freq.get(tok, 0) + 1

        # Keep most informative terms appearing in at least 2 documents
        filtered = [(tok, df_cnt) for tok, df_cnt in doc_freq.items() if df_cnt >= 2]
        filtered.sort(key=lambda x: (-x[1], x[0]))
        selected = filtered[: self.max_features]

        self.vocab_ = {tok: idx for idx, (tok, _) in enumerate(selected)}
        v_size = max(len(self.vocab_), 1)
        self.idf_ = np.ones(v_size, dtype=np.float64)
        for tok, idx in self.vocab_.items():
            self.idf_[idx] = np.log((1.0 + n_docs) / (1.0 + doc_freq[tok])) + 1.0

        tfidf = self._build_tfidf(tokenized_docs)
        # Truncated SVD via NumPy SVD on TF-IDF matrix
        k = min(self.embed_dim, tfidf.shape[0], tfidf.shape[1])
        if k > 0 and tfidf.shape[1] > 0:
            _, _, vt = np.linalg.svd(tfidf, full_matrices=False)
            self.proj_ = vt[:k].T
        else:
            self.proj_ = np.eye(v_size, self.embed_dim, dtype=np.float64)

        self.fitted_ = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("ClinicalTextVectorizer must be fitted before transform().")
        tokenized_docs = [self.tokenize(t) for t in texts]
        tfidf = self._build_tfidf(tokenized_docs)
        dense = tfidf @ self.proj_
        # L2 normalize rows
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return (dense / norms).astype(np.float32)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)

    def _build_tfidf(self, tokenized_docs: List[List[str]]) -> np.ndarray:
        v_size = max(len(self.vocab_), 1)
        mat = np.zeros((len(tokenized_docs), v_size), dtype=np.float64)
        for row_idx, tokens in enumerate(tokenized_docs):
            if not tokens:
                continue
            counts: Dict[int, int] = {}
            for tok in tokens:
                if tok in self.vocab_:
                    col = self.vocab_[tok]
                    counts[col] = counts.get(col, 0) + 1
            total = sum(counts.values())
            if total > 0:
                for col, cnt in counts.items():
                    mat[row_idx, col] = (cnt / total) * self.idf_[col]
        return mat
