"""End-to-end training pipeline for Structured-Only, Text-Only, and Multimodal Fusion models."""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict
import numpy as np

from src.config import MODELS_DIR, METRICS_DIR, ProjectConfig
from src.data_loader import load_icu_dataset, patient_level_split
from src.fusion_model import NumpyMultimodalClassifier
from src.preprocessing import ClinicalTextVectorizer, StructuredPreprocessor


def _train_pytorch_models(
    x_struct_tr: np.ndarray,
    x_struct_va: np.ndarray,
    x_text_tr: np.ndarray,
    x_text_va: np.ndarray,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    cfg: ProjectConfig,
) -> tuple[Dict[str, Any], Dict[str, Dict[str, list]]]:
    """Train PyTorch models (StructuredOnlyNet, TextOnlyNet, MultimodalFusionNet)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from src.fusion_model import MultimodalFusionNet, StructuredOnlyNet, TextOnlyNet

    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_pos = max(int(np.sum(y_tr == 1)), 1)
    n_neg = max(int(np.sum(y_tr == 0)), 1)
    pos_weight = torch.tensor([min(n_neg / n_pos, cfg.pos_weight_cap)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    models_pt = {
        "structured_only": StructuredOnlyNet(
            struct_dim=x_struct_tr.shape[1],
            hidden_dims=cfg.structured_hidden_dims,
            output_dim=cfg.fusion_hidden_dim,
            dropout=cfg.dropout_rate,
        ).to(device),
        "text_only": TextOnlyNet(
            text_dim=x_text_tr.shape[1],
            hidden_dim=cfg.text_hidden_dim,
            output_dim=cfg.fusion_hidden_dim,
            dropout=cfg.dropout_rate,
        ).to(device),
        "multimodal_fusion": MultimodalFusionNet(
            struct_dim=x_struct_tr.shape[1],
            text_dim=x_text_tr.shape[1],
            struct_hidden=cfg.structured_hidden_dims,
            text_hidden=cfg.text_hidden_dim,
            fusion_dim=cfg.fusion_hidden_dim,
            dropout=cfg.dropout_rate,
        ).to(device),
    }

    xs_tr_t = torch.tensor(x_struct_tr, dtype=torch.float32)
    xt_tr_t = torch.tensor(x_text_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)

    xs_va_t = torch.tensor(x_struct_va, dtype=torch.float32, device=device)
    xt_va_t = torch.tensor(x_text_va, dtype=torch.float32, device=device)
    y_va_t = torch.tensor(y_va, dtype=torch.float32, device=device)

    loader = DataLoader(
        TensorDataset(xs_tr_t, xt_tr_t, y_tr_t),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    histories: Dict[str, Dict[str, list]] = {}
    for name, net in models_pt.items():
        optimizer = torch.optim.Adam(
            net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        best_val = float("inf")
        best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        best_epoch = 1
        wait = 0
        hist: Dict[str, list] = {"train_loss": [], "val_loss": [], "best_epoch": []}

        for epoch_idx in range(1, cfg.epochs + 1):
            net.train()
            batch_losses = []
            for xs_b, xt_b, y_b in loader:
                xs_b, xt_b, y_b = xs_b.to(device), xt_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                if name == "structured_only":
                    logits = net(xs_b)
                elif name == "text_only":
                    logits = net(xt_b)
                else:
                    logits = net(xs_b, xt_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))

            net.eval()
            with torch.no_grad():
                if name == "structured_only":
                    va_logits = net(xs_va_t)
                elif name == "text_only":
                    va_logits = net(xt_va_t)
                else:
                    va_logits = net(xs_va_t, xt_va_t)
                va_loss = float(criterion(va_logits, y_va_t).item())

            hist["train_loss"].append(float(np.mean(batch_losses)))
            hist["val_loss"].append(va_loss)

            if va_loss < best_val - 1e-4:
                best_val = va_loss
                best_epoch = epoch_idx
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= cfg.early_stopping_patience:
                    break

        net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        net.cpu()
        hist["best_epoch"] = [best_epoch]
        histories[name] = hist

    return models_pt, histories


def train_all_models(
    csv_path: Path | None = None,
    config: ProjectConfig | None = None,
    models_dir: Path | None = None,
    metrics_dir: Path | None = None,
    backend: str | None = None,
) -> Dict[str, Any]:
    """Train all three models using patient-level splits and serialize artifacts."""
    cfg = config or ProjectConfig()
    selected_backend = backend or cfg.backend

    m_dir = models_dir or MODELS_DIR
    m_dir.mkdir(parents=True, exist_ok=True)
    met_dir = metrics_dir or METRICS_DIR
    met_dir.mkdir(parents=True, exist_ok=True)

    df = load_icu_dataset(csv_path=csv_path, config=cfg)
    train_df, val_df, test_df = patient_level_split(
        df,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.random_seed,
    )

    # Fit preprocessors strictly on training split to prevent data leakage
    struct_prep = StructuredPreprocessor(cfg)
    x_struct_tr = struct_prep.fit_transform(train_df)
    x_struct_va = struct_prep.transform(val_df)
    x_struct_te = struct_prep.transform(test_df)

    text_vec = ClinicalTextVectorizer(
        max_features=cfg.text_vocab_size,
        embed_dim=cfg.text_embed_dim,
    )
    x_text_tr = text_vec.fit_transform(train_df["clinical_note"].tolist())
    x_text_va = text_vec.transform(val_df["clinical_note"].tolist())
    x_text_te = text_vec.transform(test_df["clinical_note"].tolist())

    y_tr = train_df["hospital_expire_flag"].to_numpy(dtype=np.float64)
    y_va = val_df["hospital_expire_flag"].to_numpy(dtype=np.float64)
    y_te = test_df["hospital_expire_flag"].to_numpy(dtype=np.float64)

    use_pytorch = False
    if selected_backend == "pytorch":
        use_pytorch = True
    elif selected_backend == "auto":
        try:
            import torch  # noqa: F401
            use_pytorch = False  # Default to fast reproducible NumPy engine unless pytorch requested
        except ImportError:
            use_pytorch = False

    if use_pytorch:
        models, histories = _train_pytorch_models(
            x_struct_tr, x_struct_va, x_text_tr, x_text_va, y_tr, y_va, cfg
        )
        active_backend = "pytorch"
    else:
        active_backend = "numpy"
        models = {
            "structured_only": NumpyMultimodalClassifier(
                mode="structured",
                struct_dim=x_struct_tr.shape[1],
                hidden_dim=cfg.fusion_hidden_dim,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                pos_weight_cap=cfg.pos_weight_cap,
                seed=cfg.random_seed,
            ),
            "text_only": NumpyMultimodalClassifier(
                mode="text",
                text_dim=x_text_tr.shape[1],
                hidden_dim=cfg.fusion_hidden_dim,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                pos_weight_cap=cfg.pos_weight_cap,
                seed=cfg.random_seed,
            ),
            "multimodal_fusion": NumpyMultimodalClassifier(
                mode="multimodal",
                struct_dim=x_struct_tr.shape[1],
                text_dim=x_text_tr.shape[1],
                hidden_dim=cfg.fusion_hidden_dim,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                pos_weight_cap=cfg.pos_weight_cap,
                seed=cfg.random_seed,
            ),
        }

        histories = {}
        for name, model in models.items():
            histories[name] = model.fit(
                x_struct_tr if name != "text_only" else None,
                x_text_tr if name != "structured_only" else None,
                y_tr,
                x_struct_va if name != "text_only" else None,
                x_text_va if name != "structured_only" else None,
                y_va,
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                patience=cfg.early_stopping_patience,
            )

    bundle = {
        "backend": active_backend,
        "config": cfg,
        "struct_preprocessor": struct_prep,
        "text_vectorizer": text_vec,
        "models": models,
        "histories": histories,
        "split_summary": {
            "total_stays": int(len(df)),
            "unique_patients": int(df["subject_id"].nunique()),
            "train_stays": int(len(train_df)),
            "val_stays": int(len(val_df)),
            "test_stays": int(len(test_df)),
            "train_mortality_rate": float(y_tr.mean()),
            "val_mortality_rate": float(y_va.mean()),
            "test_mortality_rate": float(y_te.mean()),
            "structured_dim": int(x_struct_tr.shape[1]),
            "text_dim": int(x_text_tr.shape[1]),
        },
        "test_arrays": {
            "x_struct_te": x_struct_te,
            "x_text_te": x_text_te,
            "y_te": y_te,
        },
    }

    with open(m_dir / "multimodal_bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)

    with open(met_dir / "training_histories.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "backend": active_backend,
                "split_summary": bundle["split_summary"],
                "histories": histories,
            },
            f,
            indent=2,
        )

    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ICU Mortality Prediction models.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "numpy", "pytorch"],
        default="auto",
        help="Execution backend (numpy offline or pytorch).",
    )
    args = parser.parse_args()

    result = train_all_models(backend=args.backend)
    summary = result["split_summary"]
    print(f"=== Training Complete (Backend: {result['backend']}) ===")
    print(f"Total stays: {summary['total_stays']} | Unique patients: {summary['unique_patients']}")
    print(
        f"Splits (Train/Val/Test): {summary['train_stays']} / {summary['val_stays']} / {summary['test_stays']}"
    )
    for m_name, hist in result["histories"].items():
        best_ep = hist.get("best_epoch", [len(hist["train_loss"])])[0]
        print(
            f"  {m_name:18s} | epochs={len(hist['train_loss']):2d} (best={best_ep:2d}) "
            f"| final_train_loss={hist['train_loss'][-1]:.4f} "
            f"| final_val_loss={hist['val_loss'][-1]:.4f}"
        )
