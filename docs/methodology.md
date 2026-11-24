# Methodology — Project 21

## Problem Formulation
Predict in-hospital mortality (`hospital_expire_flag` in `{0, 1}`) beyond the first 24 hours of ICU admission (`t > 24h`) using strictly first-24-hour (`t in [0h, 24h]`) structured EHR measurements and clinical progress notes.

## Leakage Prevention
1. **Temporal Cutoff**: All features are restricted to the first 24h window.
2. **Patient-Level Splitting**: `src/data_loader.py` groups stays by `subject_id` and partitions unique patients into 70% train / 15% validation / 15% test stratified by maximum patient outcome, asserting zero `subject_id` overlap.
3. **Train-Only Preprocessing**: `StructuredPreprocessor` and `ClinicalTextVectorizer` fit medians, categorical vocabularies, Z-score means/stds, TF-IDF IDF weights, and SVD projections strictly on the training split.

## Training Configuration
- **Random Seed**: `42`
- **Optimizer**: Adam (`lr = 0.002`, `weight_decay = 0.015`)
- **Loss**: Class-weighted Binary Cross-Entropy with `pos_weight = min(N_neg / N_pos, 6.0)`
- **Batch Size**: `64`
- **Max Epochs**: `45` with Early Stopping (`patience = 12` on validation loss)
- **Decision Threshold**: `0.50`
- **Educational Risk Tiers**: Low (`< 0.25`), Moderate (`0.25 to < 0.55`), High (`>= 0.55`).
