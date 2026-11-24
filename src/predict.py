"""Single-patient mortality risk prediction interface (educational prototype)."""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict
import pandas as pd

from src.config import MODELS_DIR


def classify_risk_tier(prob: float) -> str:
    """Map predicted probability to an educational demonstration risk tier."""
    if prob < 0.25:
        return "Low Risk (Demonstration Tier)"
    if prob < 0.55:
        return "Moderate Risk (Demonstration Tier)"
    return "High Risk (Demonstration Tier)"


def predict_patient_mortality(
    patient_record: Dict[str, Any],
    bundle_path: Path | None = None,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Run multimodal inference for a single fictional/anonymized patient record."""
    if not isinstance(patient_record, dict):
        raise ValueError("patient_record must be a dictionary.")

    path = bundle_path or (MODELS_DIR / "multimodal_bundle.pkl")
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model bundle not found at {path}. Run `python -m src.train` first."
        )

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    backend = bundle.get("backend", "numpy")
    struct_prep = bundle["struct_preprocessor"]
    text_vec = bundle["text_vectorizer"]
    model = bundle["models"]["multimodal_fusion"]

    df_single = pd.DataFrame([patient_record])
    x_struct = struct_prep.transform(df_single)
    x_text = text_vec.transform([str(patient_record.get("clinical_note", ""))])

    struct_gate_mean = None
    text_gate_mean = None

    if backend == "pytorch":
        import torch

        model.eval()
        with torch.no_grad():
            xs_t = torch.tensor(x_struct, dtype=torch.float32)
            xt_t = torch.tensor(x_text, dtype=torch.float32)
            logits, gate_weights = model.forward_with_gate(xs_t, xt_t)
            prob = float(torch.sigmoid(logits).item())
            gw = gate_weights.cpu().numpy()[0]
            half = len(gw) // 2
            struct_gate_mean = round(float(gw[:half].mean()), 4)
            text_gate_mean = round(float(gw[half:].mean()), 4)
    else:
        if hasattr(model, "predict_with_gate"):
            probs, gate_weights = model.predict_with_gate(x_struct=x_struct, x_text=x_text)
            prob = float(probs[0])
            if gate_weights is not None:
                gw = gate_weights[0]
                half = len(gw) // 2
                struct_gate_mean = round(float(gw[:half].mean()), 4)
                text_gate_mean = round(float(gw[half:].mean()), 4)
        else:
            prob = float(model.predict_proba(x_struct=x_struct, x_text=x_text)[0])

    pred_class = int(prob >= threshold)
    return {
        "patient_id": patient_record.get("subject_id", "DEMO-001"),
        "predicted_mortality_probability": round(prob, 4),
        "predicted_class": pred_class,
        "predicted_label": (
            "High Mortality Risk (1)" if pred_class == 1 else "Survival Expected (0)"
        ),
        "risk_tier": classify_risk_tier(prob),
        "structured_gate_mean": struct_gate_mean,
        "text_gate_mean": text_gate_mean,
        "disclaimer": (
            "Educational decision-support prototype only. Output is a synthetic model "
            "probability and NOT a clinically validated medical risk score."
        ),
    }


if __name__ == "__main__":
    sample_patient = {
        "subject_id": "DEMO-ICU-9042",
        "age": 72.0,
        "gender": "M",
        "admission_type": "EMERGENCY",
        "first_careunit": "MICU",
        "heart_rate_mean": 114.0,
        "heart_rate_max": 136.0,
        "sbp_mean": 88.0,
        "sbp_min": 74.0,
        "dbp_mean": 52.0,
        "resp_rate_mean": 26.0,
        "spo2_mean": 90.0,
        "spo2_min": 84.0,
        "temperature_c": 38.4,
        "wbc_count": 18.2,
        "hemoglobin": 9.4,
        "platelets": 112.0,
        "creatinine": 2.9,
        "bun": 48.0,
        "lactate": 4.8,
        "sodium": 135.0,
        "potassium": 5.1,
        "bicarbonate": 16.0,
        "glucose": 210.0,
        "gcs_total": 10,
        "clinical_note": (
            "ICU 24h Progress Note (MICU): 72-year-old male emergency admission "
            "presenting with septic shock, worsening lactic acidosis, and refractory hypotension. "
            "Patient remains hemodynamically unstable on norepinephrine infusion. "
            "Oliguric renal failure noted; guarded prognosis."
        ),
    }

    parser = argparse.ArgumentParser(description="Predict ICU mortality risk for a sample patient.")
    parser.add_argument("--json-input", type=Path, default=None)
    args = parser.parse_args()
    record = json.loads(args.json_input.read_text()) if args.json_input else sample_patient
    output = predict_patient_mortality(record)
    print(json.dumps(output, indent=2))
