# Architecture Specification & Report Reconciliation

## Dual-Branch Sigmoid-Gated Multimodal Fusion
1. **Branch A (Structured Encoder)**:
   - Input: 21 numerical features + 21 binary missingness indicators + 10 one-hot categorical columns (`gender`: 2, `admission_type`: 3, `first_careunit`: 5) = **52 dimensions**.
   - Encoder: `[64, 32]` hidden layers with ReLU & Dropout (`0.25`), projecting to a **24-dimensional** latent vector $h_s$.
2. **Branch B (Clinical Text Encoder)**:
   - Cleaning: Lowercasing, `[**...**]` -> `deid token`, abbreviation expansion.
   - Primary: `emilyalsentzer/Bio_ClinicalBERT` (`max_length=256`, 768-dim `[CLS]`) -> `32` hidden -> **24-dim** $h_t$.
   - Offline: Unigram/bigram TF-IDF (`1200` vocab) + `64`-dim SVD -> `32` hidden -> **24-dim** $h_t$.
3. **Branch C (Gated Fusion)**:
   - $h_{cat} = [h_s ; h_t] \in \mathbb{R}^{48}$
   - $g = \sigma(W_g h_{cat} + b_g) \in (0, 1)^{48}$
   - $h_{gated} = g \odot h_{cat}$
   - $h_f = \text{ReLU}(W_f h_{gated} + b_f) \in \mathbb{R}^{24}$
   - $\hat{y} = \sigma(W_o h_f + b_o) \in [0, 1]$

## Documented Report Inconsistency Corrections
1. **Structured Input Dimension (Prose "45" vs. Actual "52")**: The report prose in Sections 3.1.1 and 5.1 states a 45-dimensional structured vector, whereas the report's own `StructuredPreprocessor` code stacks 21 numerics + 21 missingness flags + 10 one-hot categoricals = **52 dimensions**. The implementation preserves the exact `StructuredPreprocessor` logic (52 dimensions) without dropping features.
2. **PyTorch Encoder Output Dimensions**: In the report's `src/fusion_model.py` listing, `MultimodalFusionNet` used `32 + 32 = 64` combined dimensions, whereas Section 3.1.3 and `src/train.py` (`hidden_dim=cfg.fusion_hidden_dim=24`) specify `24 + 24 = 48` combined dimensions projected to `24`. Both the PyTorch and NumPy implementations are aligned to **24-dim** branch outputs -> **48-dim** gated fusion -> **24-dim** fused representation.
