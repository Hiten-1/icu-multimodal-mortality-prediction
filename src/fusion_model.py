"""Multimodal fusion neural architecture and executable training/inference engine.

Provides:
1. PyTorch implementations (`StructuredOnlyNet`, `TextOnlyNet`, `MultimodalFusionNet`)
   for GPU/PyTorch environments.
2. Self-contained NumPy neural network implementations (`NumpyMultimodalClassifier`)
   for reproducible CPU execution when PyTorch is not installed or offline mode is selected.
"""

from typing import Dict, List, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    from src.structured_encoder import StructuredEncoder

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:

    class StructuredOnlyNet(nn.Module):
        def __init__(
            self,
            struct_dim: int,
            hidden_dims: List[int] | None = None,
            output_dim: int = 24,
            dropout: float = 0.25,
        ):
            super().__init__()
            self.encoder = StructuredEncoder(
                struct_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                dropout=dropout,
            )
            self.head = nn.Linear(self.encoder.output_dim, 1)

        def forward(self, x_struct: "torch.Tensor") -> "torch.Tensor":
            z = self.encoder(x_struct)
            return self.head(z).squeeze(-1)

    class TextOnlyNet(nn.Module):
        def __init__(
            self,
            text_dim: int,
            hidden_dim: int = 32,
            output_dim: int = 24,
            dropout: float = 0.25,
        ):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.ReLU(),
            )
            self.head = nn.Linear(output_dim, 1)

        def forward(self, x_text: "torch.Tensor") -> "torch.Tensor":
            z = self.encoder(x_text)
            return self.head(z).squeeze(-1)

    class MultimodalFusionNet(nn.Module):
        """Gated multimodal fusion network combining 24-dim structured and 24-dim text vectors."""

        def __init__(
            self,
            struct_dim: int,
            text_dim: int,
            struct_hidden: List[int] | None = None,
            text_hidden: int = 32,
            fusion_dim: int = 24,
            dropout: float = 0.25,
        ):
            super().__init__()
            self.struct_encoder = StructuredEncoder(
                struct_dim,
                hidden_dims=struct_hidden,
                output_dim=fusion_dim,
                dropout=dropout,
            )
            self.text_encoder = nn.Sequential(
                nn.Linear(text_dim, text_hidden),
                nn.BatchNorm1d(text_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(text_hidden, fusion_dim),
                nn.ReLU(),
            )
            combined_dim = self.struct_encoder.output_dim + fusion_dim  # 24 + 24 = 48
            # Gating mechanism learns modality weighting per patient
            self.gate = nn.Sequential(
                nn.Linear(combined_dim, combined_dim),
                nn.Sigmoid(),
            )
            self.fusion_mlp = nn.Sequential(
                nn.Linear(combined_dim, fusion_dim),
                nn.BatchNorm1d(fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, 1),
            )

        def forward_with_gate(
            self, x_struct: "torch.Tensor", x_text: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            h_s = self.struct_encoder(x_struct)
            h_t = self.text_encoder(x_text)
            concat = torch.cat([h_s, h_t], dim=1)
            gate_weights = self.gate(concat)
            gated = concat * gate_weights
            logits = self.fusion_mlp(gated).squeeze(-1)
            return logits, gate_weights

        def forward(
            self, x_struct: "torch.Tensor", x_text: "torch.Tensor"
        ) -> "torch.Tensor":
            logits, _ = self.forward_with_gate(x_struct, x_text)
            return logits


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z_clipped = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


class NumpyMultimodalClassifier:
    """Reproducible NumPy neural classifier supporting structured, text, and gated fusion modes.

    Implements a 2-branch neural architecture (24-dim structured + 24-dim text ->
    48-dim sigmoid gate -> 24-dim fused representation -> sigmoid output) with
    class-weighted binary cross-entropy loss, L2 weight decay, and Adam optimization.
    """

    def __init__(
        self,
        mode: str = "multimodal",
        struct_dim: int = 0,
        text_dim: int = 0,
        hidden_dim: int = 24,
        lr: float = 0.002,
        weight_decay: float = 0.015,
        pos_weight_cap: float = 6.0,
        seed: int = 42,
    ):
        if mode not in {"structured", "text", "multimodal"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode
        self.struct_dim = struct_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.pos_weight_cap = pos_weight_cap
        self.seed = seed
        self.params: Dict[str, np.ndarray] = {}
        self._init_weights()

    def _init_weights(self) -> None:
        rng = np.random.default_rng(self.seed)

        def _xavier(in_d: int, out_d: int) -> np.ndarray:
            scale = np.sqrt(2.0 / max(in_d + out_d, 1))
            return rng.normal(0.0, scale, size=(in_d, out_d)).astype(np.float64)

        if self.mode in {"structured", "multimodal"}:
            self.params["Ws1"] = _xavier(self.struct_dim, self.hidden_dim)
            self.params["bs1"] = np.zeros(self.hidden_dim, dtype=np.float64)

        if self.mode in {"text", "multimodal"}:
            self.params["Wt1"] = _xavier(self.text_dim, self.hidden_dim)
            self.params["bt1"] = np.zeros(self.hidden_dim, dtype=np.float64)

        if self.mode == "multimodal":
            comb = self.hidden_dim * 2
            self.params["Wg"] = _xavier(comb, comb)
            self.params["bg"] = np.zeros(comb, dtype=np.float64)
            self.params["Wf"] = _xavier(comb, self.hidden_dim)
            self.params["bf"] = np.zeros(self.hidden_dim, dtype=np.float64)
            head_in = self.hidden_dim
        else:
            head_in = self.hidden_dim

        self.params["Wo"] = _xavier(head_in, 1)
        self.params["bo"] = np.zeros(1, dtype=np.float64)

    def _forward(
        self,
        x_struct: np.ndarray | None,
        x_text: np.ndarray | None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        cache: Dict[str, np.ndarray] = {}
        if self.mode == "structured":
            assert x_struct is not None
            zs = x_struct @ self.params["Ws1"] + self.params["bs1"]
            hs = _relu(zs)
            cache.update({"xs": x_struct, "zs": zs, "hs": hs})
            feat = hs
        elif self.mode == "text":
            assert x_text is not None
            zt = x_text @ self.params["Wt1"] + self.params["bt1"]
            ht = _relu(zt)
            cache.update({"xt": x_text, "zt": zt, "ht": ht})
            feat = ht
        else:
            assert x_struct is not None and x_text is not None
            zs = x_struct @ self.params["Ws1"] + self.params["bs1"]
            hs = _relu(zs)
            zt = x_text @ self.params["Wt1"] + self.params["bt1"]
            ht = _relu(zt)
            concat = np.hstack([hs, ht])
            gate = _sigmoid(concat @ self.params["Wg"] + self.params["bg"])
            gated = concat * gate
            zf = gated @ self.params["Wf"] + self.params["bf"]
            hf = _relu(zf)
            cache.update(
                {
                    "xs": x_struct,
                    "zs": zs,
                    "hs": hs,
                    "xt": x_text,
                    "zt": zt,
                    "ht": ht,
                    "concat": concat,
                    "gate": gate,
                    "gated": gated,
                    "zf": zf,
                    "hf": hf,
                }
            )
            feat = hf

        logits = (feat @ self.params["Wo"] + self.params["bo"]).ravel()
        probs = _sigmoid(logits)
        cache["feat"] = feat
        cache["probs"] = probs
        return probs, cache

    def compute_loss(
        self,
        probs: np.ndarray,
        y: np.ndarray,
        pos_weight: float = 1.0,
    ) -> float:
        eps = 1e-7
        p = np.clip(probs, eps, 1.0 - eps)
        losses = -(pos_weight * y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        l2 = 0.5 * self.weight_decay * sum(
            np.sum(v * v) for k, v in self.params.items() if k.startswith("W")
        )
        return float(np.mean(losses) + l2)

    def _backward(
        self,
        y: np.ndarray,
        cache: Dict[str, np.ndarray],
        pos_weight: float,
    ) -> Dict[str, np.ndarray]:
        n = max(len(y), 1)
        probs = cache["probs"]
        weights = np.where(y == 1, pos_weight, 1.0)
        d_logits = ((probs - y) * weights / n)[:, None]

        grads: Dict[str, np.ndarray] = {}
        feat = cache["feat"]
        grads["Wo"] = feat.T @ d_logits + self.weight_decay * self.params["Wo"]
        grads["bo"] = np.sum(d_logits, axis=0)
        d_feat = d_logits @ self.params["Wo"].T

        if self.mode == "structured":
            d_zs = d_feat * (cache["zs"] > 0)
            grads["Ws1"] = cache["xs"].T @ d_zs + self.weight_decay * self.params["Ws1"]
            grads["bs1"] = np.sum(d_zs, axis=0)
        elif self.mode == "text":
            d_zt = d_feat * (cache["zt"] > 0)
            grads["Wt1"] = cache["xt"].T @ d_zt + self.weight_decay * self.params["Wt1"]
            grads["bt1"] = np.sum(d_zt, axis=0)
        else:
            d_zf = d_feat * (cache["zf"] > 0)
            grads["Wf"] = cache["gated"].T @ d_zf + self.weight_decay * self.params["Wf"]
            grads["bf"] = np.sum(d_zf, axis=0)
            d_gated = d_zf @ self.params["Wf"].T

            gate = cache["gate"]
            concat = cache["concat"]
            d_concat_direct = d_gated * gate
            d_gate_pre = d_gated * concat * gate * (1.0 - gate)
            grads["Wg"] = concat.T @ d_gate_pre + self.weight_decay * self.params["Wg"]
            grads["bg"] = np.sum(d_gate_pre, axis=0)
            d_concat = d_concat_direct + d_gate_pre @ self.params["Wg"].T

            d_hs = d_concat[:, : self.hidden_dim]
            d_ht = d_concat[:, self.hidden_dim :]
            d_zs = d_hs * (cache["zs"] > 0)
            d_zt = d_ht * (cache["zt"] > 0)
            grads["Ws1"] = cache["xs"].T @ d_zs + self.weight_decay * self.params["Ws1"]
            grads["bs1"] = np.sum(d_zs, axis=0)
            grads["Wt1"] = cache["xt"].T @ d_zt + self.weight_decay * self.params["Wt1"]
            grads["bt1"] = np.sum(d_zt, axis=0)

        return grads

    def fit(
        self,
        x_struct_train: np.ndarray | None,
        x_text_train: np.ndarray | None,
        y_train: np.ndarray,
        x_struct_val: np.ndarray | None,
        x_text_val: np.ndarray | None,
        y_val: np.ndarray,
        epochs: int = 45,
        batch_size: int = 64,
        patience: int = 12,
    ) -> Dict[str, List[float]]:
        rng = np.random.default_rng(self.seed)
        n_pos = max(int(np.sum(y_train == 1)), 1)
        n_neg = max(int(np.sum(y_train == 0)), 1)
        pos_weight = min(n_neg / n_pos, self.pos_weight_cap)

        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "best_epoch": [],
        }
        best_val_loss = float("inf")
        best_epoch = 1
        best_params = {k: val.copy() for k, val in self.params.items()}
        wait = 0
        n_samples = len(y_train)

        for epoch_idx in range(1, epochs + 1):
            perm = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                idx = perm[start : start + batch_size]
                xs_b = x_struct_train[idx] if x_struct_train is not None else None
                xt_b = x_text_train[idx] if x_text_train is not None else None
                y_b = y_train[idx]
                _, cache = self._forward(xs_b, xt_b)
                grads = self._backward(y_b, cache, pos_weight)
                step += 1
                for k in self.params:
                    m[k] = beta1 * m[k] + (1.0 - beta1) * grads[k]
                    v[k] = beta2 * v[k] + (1.0 - beta2) * (grads[k] ** 2)
                    m_hat = m[k] / (1.0 - beta1 ** step)
                    v_hat = v[k] / (1.0 - beta2 ** step)
                    self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

            tr_probs, _ = self._forward(x_struct_train, x_text_train)
            va_probs, _ = self._forward(x_struct_val, x_text_val)
            tr_loss = self.compute_loss(tr_probs, y_train, pos_weight)
            va_loss = self.compute_loss(va_probs, y_val, pos_weight)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)

            if va_loss < best_val_loss - 1e-4:
                best_val_loss = va_loss
                best_epoch = epoch_idx
                best_params = {k: val.copy() for k, val in self.params.items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        self.params = best_params
        history["best_epoch"] = [best_epoch]
        return history

    def predict_proba(
        self,
        x_struct: np.ndarray | None = None,
        x_text: np.ndarray | None = None,
    ) -> np.ndarray:
        probs, _ = self._forward(x_struct, x_text)
        return probs

    def predict_with_gate(
        self,
        x_struct: np.ndarray | None = None,
        x_text: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        probs, cache = self._forward(x_struct, x_text)
        return probs, cache.get("gate")

    def predict(
        self,
        x_struct: np.ndarray | None = None,
        x_text: np.ndarray | None = None,
        threshold: float = 0.50,
    ) -> np.ndarray:
        return (self.predict_proba(x_struct, x_text) >= threshold).astype(int)
