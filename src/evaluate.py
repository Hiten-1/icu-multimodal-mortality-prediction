"""Model evaluation metrics and black-and-white publication visualizations."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[int, int, int, int]:
    """Return (TN, FP, FN, TP) for binary labels."""
    y_t = y_true.astype(int)
    y_p = y_pred.astype(int)
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    return tn, fp, fn, tp


def compute_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute FPR, TPR, and trapezoidal ROC-AUC."""
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    n_pos = max(float(np.sum(y_true == 1)), 1.0)
    n_neg = max(float(np.sum(y_true == 0)), 1.0)

    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    tpr = np.concatenate([[0.0], tps / n_pos, [1.0]])
    fpr = np.concatenate([[0.0], fps / n_neg, [1.0]])
    auc = float(getattr(np, "trapezoid", getattr(np, "trapz", None))(tpr, fpr))
    return fpr, tpr, auc


def compute_pr_curve(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute Precision, Recall, and PR-AUC."""
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    n_pos = max(float(np.sum(y_true == 1)), 1.0)

    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    precision = tps / np.maximum(tps + fps, 1.0)
    recall = tps / n_pos
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    pr_auc = float(getattr(np, "trapezoid", getattr(np, "trapz", None))(precision, recall))
    return recall, precision, pr_auc


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Compute comprehensive binary classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = compute_confusion_matrix(y_true, y_pred)
    total = max(tn + fp + fn + tp, 1)
    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    brier = float(np.mean((y_prob - y_true) ** 2))
    fpr, tpr, roc_auc = compute_roc_curve(y_true, y_prob)
    rec_curve, prec_curve, pr_auc = compute_pr_curve(y_true, y_prob)

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "curves": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "pr_recall": rec_curve.tolist(),
            "pr_precision": prec_curve.tolist(),
        },
    }


def _configure_bw_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": "#000000",
            "axes.labelcolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "text.color": "#000000",
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
        }
    )


def generate_evaluation_plots(
    metrics_by_model: Dict[str, Dict[str, Any]],
    histories: Dict[str, Dict[str, list]],
    output_dir: Path = FIGURES_DIR,
) -> None:
    """Generate clean black-and-white high-DPI evaluation charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_bw_style()

    labels_map = {
        "structured_only": "Structured-Only",
        "text_only": "Clinical-Text-Only",
        "multimodal_fusion": "Multimodal Fusion",
    }
    line_styles = {
        "structured_only": ("--", "0.35", 1.8),
        "text_only": (":", "0.55", 2.0),
        "multimodal_fusion": ("-", "0.0", 2.4),
    }

    # 1. Confusion Matrix (Multimodal Fusion)
    cm_data = metrics_by_model["multimodal_fusion"]["confusion_matrix"]
    cm_arr = np.array([[cm_data["tn"], cm_data["fp"]], [cm_data["fn"], cm_data["tp"]]])
    fig, ax = plt.subplots(figsize=(5.5, 4.4), dpi=300)
    im = ax.imshow(cm_arr, cmap="Greys", vmin=0, vmax=max(cm_arr.max() * 1.15, 1))
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted: Survived (0)", "Predicted: Expired (1)"])
    ax.set_yticklabels(["Actual: Survived (0)", "Actual: Expired (1)"])
    ax.set_title(
        "Confusion Matrix: Multimodal Fusion Model (Test Set)",
        fontweight="bold",
        pad=12,
    )
    for i in range(2):
        for j in range(2):
            val = cm_arr[i, j]
            txt_color = "#FFFFFF" if val > cm_arr.max() * 0.55 else "#000000"
            ax.text(
                j,
                i,
                f"{val}",
                ha="center",
                va="center",
                color=txt_color,
                fontsize=13,
                fontweight="bold",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_confusion_matrix.png", dpi=300)
    plt.close(fig)

    # 2. ROC Curves
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=300)
    for key, m in metrics_by_model.items():
        ls, col, lw = line_styles[key]
        ax.plot(
            m["curves"]["fpr"],
            m["curves"]["tpr"],
            linestyle=ls,
            color=col,
            linewidth=lw,
            label=f"{labels_map[key]} (AUC = {m['roc_auc']:.3f})",
        )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="-.",
        color="0.7",
        linewidth=1.0,
        label="Random Baseline (AUC = 0.500)",
    )
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve Comparison Across Modalities", fontweight="bold", pad=10)
    ax.grid(True, linestyle=":", color="0.82")
    ax.legend(loc="lower right", frameon=True, edgecolor="#000000")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_roc_curves.png", dpi=300)
    plt.close(fig)

    # 3. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=300)
    for key, m in metrics_by_model.items():
        ls, col, lw = line_styles[key]
        ax.plot(
            m["curves"]["pr_recall"],
            m["curves"]["pr_precision"],
            linestyle=ls,
            color=col,
            linewidth=lw,
            label=f"{labels_map[key]} (PR-AUC = {m['pr_auc']:.3f})",
        )
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title("Precision-Recall Curve Comparison", fontweight="bold", pad=10)
    ax.grid(True, linestyle=":", color="0.82")
    ax.legend(loc="upper right", frameon=True, edgecolor="#000000")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_pr_curves.png", dpi=300)
    plt.close(fig)

    # 4. Training and Validation Loss Curve
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300)
    mm_hist = histories["multimodal_fusion"]
    epochs = np.arange(1, len(mm_hist["train_loss"]) + 1)
    ax.plot(
        epochs,
        mm_hist["train_loss"],
        color="0.0",
        linestyle="-",
        linewidth=2.0,
        label="Training Loss (Multimodal)",
    )
    ax.plot(
        epochs,
        mm_hist["val_loss"],
        color="0.45",
        linestyle="--",
        linewidth=2.0,
        label="Validation Loss (Multimodal)",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted Binary Cross-Entropy Loss")
    ax.set_title("Training and Validation Loss Convergence", fontweight="bold", pad=10)
    ax.grid(True, linestyle=":", color="0.82")
    ax.legend(loc="upper right", frameon=True, edgecolor="#000000")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_loss_curves.png", dpi=300)
    plt.close(fig)

    # 5. Model Comparison Grouped Bar Chart
    metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    display_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x = np.arange(len(metric_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=300)

    bars_cfg = [
        ("structured_only", "0.75", "//"),
        ("text_only", "0.50", ".."),
        ("multimodal_fusion", "0.18", ""),
    ]
    for idx, (key, shade, hatch) in enumerate(bars_cfg):
        vals = [metrics_by_model[key][mn] for mn in metric_names]
        rects = ax.bar(
            x + (idx - 1) * width,
            vals,
            width,
            label=labels_map[key],
            color=shade,
            edgecolor="#000000",
            hatch=hatch,
        )
        for r in rects:
            h = r.get_height()
            ax.text(
                r.get_x() + r.get_width() / 2,
                h + 0.015,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.set_ylabel("Score")
    ax.set_title(
        "Test-Set Performance Comparison Across Approaches",
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, axis="y", linestyle=":", color="0.82")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=True,
        edgecolor="#000000",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "fig_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_bundle(bundle_path: Path | None = None) -> Dict[str, Any]:
    """Load trained bundle, evaluate on held-out test set, and save metrics & plots."""
    path = bundle_path or (MODELS_DIR / "multimodal_bundle.pkl")
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    cfg = bundle["config"]
    backend = bundle.get("backend", "numpy")
    models = bundle["models"]
    x_struct_te = bundle["test_arrays"]["x_struct_te"]
    x_text_te = bundle["test_arrays"]["x_text_te"]
    y_te = bundle["test_arrays"]["y_te"]

    metrics_by_model: Dict[str, Dict[str, Any]] = {}
    for name, model in models.items():
        if backend == "pytorch":
            import torch

            model.eval()
            with torch.no_grad():
                xs_t = torch.tensor(x_struct_te, dtype=torch.float32)
                xt_t = torch.tensor(x_text_te, dtype=torch.float32)
                if name == "structured_only":
                    logits = model(xs_t)
                elif name == "text_only":
                    logits = model(xt_t)
                else:
                    logits = model(xs_t, xt_t)
                probs = torch.sigmoid(logits).cpu().numpy()
        else:
            probs = model.predict_proba(
                x_struct=x_struct_te if name != "text_only" else None,
                x_text=x_text_te if name != "structured_only" else None,
            )

        metrics_by_model[name] = compute_classification_metrics(
            y_te, probs, threshold=cfg.decision_threshold
        )

    generate_evaluation_plots(metrics_by_model, bundle["histories"], output_dir=FIGURES_DIR)

    summary_metrics = {
        k: {mk: mv for mk, mv in v.items() if mk != "curves"}
        for k, v in metrics_by_model.items()
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
    with open(METRICS_DIR / "evaluation_curves.json", "w", encoding="utf-8") as f:
        json.dump(metrics_by_model, f, indent=2)

    return summary_metrics


if __name__ == "__main__":
    results = evaluate_bundle()
    print("=== Test-Set Evaluation Results (Synthetic Demonstration Cohort) ===")
    for model_name, m in results.items():
        cm = m["confusion_matrix"]
        print(
            f"{model_name:18s} | Acc={m['accuracy']:.4f} | Prec={m['precision']:.4f} "
            f"| Rec={m['recall']:.4f} | F1={m['f1_score']:.4f} "
            f"| ROC-AUC={m['roc_auc']:.4f} | PR-AUC={m['pr_auc']:.4f} "
            f"| Brier={m['brier_score']:.4f} "
            f"| CM=[TN:{cm['tn']}, FP:{cm['fp']}, FN:{cm['fn']}, TP:{cm['tp']}]"
        )
