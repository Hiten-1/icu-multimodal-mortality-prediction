"""Synthetic ICU cohort generator for educational multimodal demonstration.

Generates realistic, entirely fictional patient stays modeled on the schema
of MIMIC-IV and MIMIC-IV-Note without using any real patient records.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic_cohort(
    n_patients: int = 1800,
    n_stays: int = 2500,
    seed: int = 42,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Generate a synthetic multimodal ICU dataset with patient-level IDs."""
    rng = np.random.default_rng(seed)

    # Assign stays to unique patients so patient-level splitting can be tested
    patient_ids = np.arange(100001, 100001 + n_patients)
    stay_subject_ids = rng.choice(patient_ids, size=n_stays, replace=True)
    stay_ids = np.arange(300001, 300001 + n_stays)

    # Latent physiological severity per patient + stay-level acuity
    patient_frailty = {pid: rng.normal(0.0, 0.7) for pid in patient_ids}
    frailty = np.array([patient_frailty[pid] for pid in stay_subject_ids])
    acuity = rng.normal(0.0, 1.0, size=n_stays)
    severity = 0.55 * frailty + 0.85 * acuity

    # Demographics
    age = np.clip(62 + 12 * frailty + rng.normal(0, 11, n_stays), 18, 91).round(1)
    gender = rng.choice(["M", "F"], size=n_stays, p=[0.55, 0.45])
    admission_type = rng.choice(
        ["EMERGENCY", "URGENT", "ELECTIVE"], size=n_stays, p=[0.72, 0.16, 0.12]
    )
    first_careunit = rng.choice(
        ["MICU", "SICU", "CCU", "CVICU", "TSICU"],
        size=n_stays,
        p=[0.36, 0.22, 0.16, 0.14, 0.12],
    )

    # First-24h vital signs & laboratory measurements correlated with severity
    heart_rate_mean = np.clip(84 + 10 * severity + rng.normal(0, 12, n_stays), 42, 165).round(1)
    heart_rate_max = np.clip(heart_rate_mean + rng.uniform(6, 28, n_stays), 50, 190).round(1)
    sbp_mean = np.clip(118 - 11 * severity + rng.normal(0, 14, n_stays), 62, 195).round(1)
    sbp_min = np.clip(sbp_mean - rng.uniform(8, 30, n_stays), 45, 165).round(1)
    dbp_mean = np.clip(65 - 6 * severity + rng.normal(0, 9, n_stays), 32, 115).round(1)
    resp_rate_mean = np.clip(19 + 3.2 * severity + rng.normal(0, 3.5, n_stays), 10, 40).round(1)
    spo2_mean = np.clip(
        96.5 - 1.8 * np.maximum(severity, 0) + rng.normal(0, 1.8, n_stays), 72, 100
    ).round(1)
    spo2_min = np.clip(spo2_mean - rng.uniform(1, 9, n_stays), 60, 100).round(1)
    temperature_c = np.clip(36.9 + 0.35 * severity + rng.normal(0, 0.65, n_stays), 33.5, 40.8).round(2)

    wbc_count = np.clip(10.5 + 3.2 * severity + rng.normal(0, 4.0, n_stays), 1.0, 42.0).round(2)
    hemoglobin = np.clip(11.4 - 0.7 * severity + rng.normal(0, 1.7, n_stays), 5.5, 17.5).round(2)
    platelets = np.clip(215 - 28 * severity + rng.normal(0, 65, n_stays), 18, 650).round(1)
    creatinine = np.clip(
        1.25 + 0.55 * np.maximum(severity, -0.5) + rng.exponential(0.35, n_stays), 0.4, 9.5
    ).round(2)
    bun = np.clip(
        22 + 9 * np.maximum(severity, -0.5) + rng.normal(0, 8, n_stays), 5, 120
    ).round(1)
    lactate = np.clip(
        1.6 + 1.15 * np.maximum(severity, -0.3) + rng.exponential(0.45, n_stays), 0.5, 14.0
    ).round(2)
    sodium = np.clip(138.5 + rng.normal(0, 4.2, n_stays), 120, 160).round(1)
    potassium = np.clip(4.2 + 0.25 * severity + rng.normal(0, 0.55, n_stays), 2.5, 7.2).round(2)
    bicarbonate = np.clip(23.5 - 2.4 * severity + rng.normal(0, 3.2, n_stays), 9, 36).round(1)
    glucose = np.clip(
        138 + 18 * np.maximum(severity, 0) + rng.normal(0, 38, n_stays), 45, 480
    ).round(1)
    gcs_total = np.clip(
        np.round(14.0 - 1.9 * np.maximum(severity, 0) + rng.normal(0, 1.5, n_stays)), 3, 15
    ).astype(int)

    # Mortality probability driven by both structured physiology and clinical note findings
    note_risk_latent = rng.normal(0.0, 0.9, size=n_stays)
    logit = (
        -3.15
        + 0.025 * (age - 65)
        + 0.55 * np.maximum(lactate - 2.0, 0)
        + 0.35 * np.maximum(creatinine - 1.5, 0)
        + 0.032 * np.maximum(95 - sbp_min, 0)
        + 0.14 * np.maximum(13 - gcs_total, 0)
        + 0.08 * np.maximum(92 - spo2_min, 0)
        + 0.95 * note_risk_latent
        + 0.30 * (admission_type == "EMERGENCY").astype(float)
    )
    prob_mortality = 1.0 / (1.0 + np.exp(-logit))
    hospital_expire_flag = rng.binomial(1, prob_mortality)

    # Generate realistic synthetic clinical notes from the first 24h window
    notes = []
    for i in range(n_stays):
        is_high_text_risk = note_risk_latent[i] > 0.65 or (
            hospital_expire_flag[i] == 1 and rng.random() < 0.72
        )
        is_moderate_risk = (0.0 < note_risk_latent[i] <= 0.65) and not is_high_text_risk

        unit = first_careunit[i]
        adm = admission_type[i].lower()
        sex_word = "male" if gender[i] == "M" else "female"

        if is_high_text_risk:
            chief = rng.choice(
                [
                    "septic shock with worsening lactic acidosis and refractory hypotension",
                    "acute hypoxemic respiratory failure requiring mechanical ventilation",
                    "cardiogenic shock with acute kidney injury and altered mental status",
                    "severe pneumonia with multi-organ dysfunction and vasopressor dependence",
                ]
            )
            course = rng.choice(
                [
                    "Patient remains hemodynamically unstable on norepinephrine infusion. Metabolic acidosis persists with rising lactate.",
                    "Required emergent endotracheal intubation for worsening hypoxia. Oliguric renal failure noted; nephrology consulted.",
                    "Mental status declining (GCS reduced). Broad-spectrum antibiotics initiated; guarded prognosis discussed with family.",
                    "Persistent tachycardia and hypotension despite fluid resuscitation. Serial troponins and lactate trending upward.",
                ]
            )
        elif is_moderate_risk:
            chief = rng.choice(
                [
                    "acute decompensated heart failure with pulmonary edema",
                    "community-acquired pneumonia and mild hyponatremia",
                    "post-operative monitoring following abdominal exploration",
                    "diabetic ketoacidosis with moderate dehydration",
                ]
            )
            course = rng.choice(
                [
                    "Oxygenation improved on supplemental nasal cannula. Electrolytes being repleted per ICU protocol.",
                    "Hemodynamics stable off vasopressors. Continue close telemetry and serial laboratory monitoring.",
                    "Responding gradually to diuresis and medical therapy. Pain controlled; awaiting morning labs.",
                    "Mild confusion at admission now improving. Vital signs stable over past 12 hours.",
                ]
            )
        else:
            chief = rng.choice(
                [
                    "elective post-operative observation after cardiac catheterization",
                    "uncomplicated atrial fibrillation with controlled ventricular rate",
                    "mild gastrointestinal bleed, hemodynamically stable",
                    "overnight airway monitoring following minor procedure",
                ]
            )
            course = rng.choice(
                [
                    "Patient comfortable, alert and oriented x3. Normal urine output and stable vital signs.",
                    "No acute distress. Tolerating oral intake; candidate for step-down floor transfer tomorrow.",
                    "Sinus rhythm restored; labs within acceptable limits. Routine supportive care continued.",
                    "Afebrile, normotensive, breathing comfortably on room air. Plan for morning discharge from ICU.",
                ]
            )

        note_text = (
            f"ICU 24h Progress Note ({unit}): {int(age[i])}-year-old {sex_word} "
            f"({adm} admission) presenting with {chief}. "
            f"Vitals: HR {heart_rate_mean[i]} bpm, SBP {sbp_mean[i]} mmHg, "
            f"RR {resp_rate_mean[i]}/min, SpO2 {spo2_mean[i]}%. "
            f"Labs: Lactate {lactate[i]} mmol/L, Cr {creatinine[i]} mg/dL, "
            f"WBC {wbc_count[i]} K/uL. Assessment/Plan: {course}"
        )
        notes.append(note_text)

    df = pd.DataFrame(
        {
            "subject_id": stay_subject_ids,
            "stay_id": stay_ids,
            "age": age,
            "gender": gender,
            "admission_type": admission_type,
            "first_careunit": first_careunit,
            "heart_rate_mean": heart_rate_mean,
            "heart_rate_max": heart_rate_max,
            "sbp_mean": sbp_mean,
            "sbp_min": sbp_min,
            "dbp_mean": dbp_mean,
            "resp_rate_mean": resp_rate_mean,
            "spo2_mean": spo2_mean,
            "spo2_min": spo2_min,
            "temperature_c": temperature_c,
            "wbc_count": wbc_count,
            "hemoglobin": hemoglobin,
            "platelets": platelets,
            "creatinine": creatinine,
            "bun": bun,
            "lactate": lactate,
            "sodium": sodium,
            "potassium": potassium,
            "bicarbonate": bicarbonate,
            "glucose": glucose,
            "gcs_total": gcs_total,
            "clinical_note": notes,
            "hospital_expire_flag": hospital_expire_flag,
        }
    )

    # Inject realistic missingness (~6% in selected lab/vital columns)
    missing_cols = [
        "lactate",
        "bicarbonate",
        "platelets",
        "temperature_c",
        "spo2_min",
        "glucose",
    ]
    for col in missing_cols:
        mask = rng.random(n_stays) < 0.06
        df.loc[mask, col] = np.nan

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic ICU multimodal dataset.")
    parser.add_argument("--n-patients", type=int, default=1800)
    parser.add_argument("--n-stays", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "synthetic_icu_cohort.csv",
    )
    args = parser.parse_args()
    cohort = generate_synthetic_cohort(
        n_patients=args.n_patients,
        n_stays=args.n_stays,
        seed=args.seed,
        output_path=args.output,
    )
    pos_rate = cohort["hospital_expire_flag"].mean() * 100
    print(f"Generated {len(cohort)} ICU stays ({cohort['subject_id'].nunique()} unique fictional patients) -> {args.output}")
    print(f"In-hospital mortality rate: {pos_rate:.2f}% ({cohort['hospital_expire_flag'].sum()} positive)")
