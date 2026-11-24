"""Unstructured clinical note encoder (HuggingFace Bio_ClinicalBERT + offline fallback)."""

from typing import List

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

if TRANSFORMERS_AVAILABLE:

    class ClinicalTextEncoder(nn.Module):
        """Transformer-based clinical note encoder with 32-dim hidden -> 24-dim latent projection."""

        def __init__(
            self,
            model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
            hidden_dim: int = 32,
            output_dim: int = 24,
            dropout: float = 0.25,
            freeze_backbone: bool = True,
        ):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(model_name)
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            hidden_size = self.backbone.config.hidden_size
            self.projection = nn.Sequential(
                nn.Linear(hidden_size, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.ReLU(),
            )
            self.output_dim = output_dim

        def forward(
            self,
            input_ids: "torch.Tensor",
            attention_mask: "torch.Tensor",
        ) -> "torch.Tensor":
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            return self.projection(cls_embedding)

    def tokenize_clinical_notes(
        notes: List[str],
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        max_length: int = 256,
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer(
            notes,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

else:

    class ClinicalTextEncoder:  # type: ignore[no-redef]
        """Descriptor for ClinicalTextEncoder when Transformers is unavailable offline."""

        def __init__(
            self,
            model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
            hidden_dim: int = 32,
            output_dim: int = 24,
            dropout: float = 0.25,
            freeze_backbone: bool = True,
        ):
            self.model_name = model_name
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.dropout = dropout
            self.freeze_backbone = freeze_backbone
