# Synthetic ICU Demonstration Data

This directory contains the synthetic cohort generator (`synthetic/generate_synthetic_data.py`) used for reproducible, privacy-preserving demonstration of Project 21.

- **No Real Patient Data**: No MIMIC-IV, MIMIC-IV-Note, or protected health information (PHI) is stored or distributed in this repository.
- **Fictional Records**: All generated stays (`2,500` stays across `1,800` configured fictional patient IDs, seed `42`) are completely synthetic.
- **Temporal Cutoff**: All features represent the first 24 hours of ICU admission (`t in [0h, 24h]`), and `hospital_expire_flag` represents in-hospital mortality beyond the 24-hour window.
