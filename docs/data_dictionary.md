# Data Dictionary — Synthetic ICU Cohort

All records are fictional and modeled on the first 24 hours of ICU admission (`t in [0h, 24h]`).

## Identifiers & Target
- `subject_id`: Fictional patient identifier (`100001`–`101800`)
- `stay_id`: Fictional ICU stay identifier (`300001`–`302500`)
- `hospital_expire_flag`: Binary in-hospital mortality target (`0` = Survived, `1` = Expired beyond 24h)

## 21 Numerical Features
`age`, `heart_rate_mean`, `heart_rate_max`, `sbp_mean`, `sbp_min`, `dbp_mean`, `resp_rate_mean`, `spo2_mean`, `spo2_min`, `temperature_c`, `wbc_count`, `hemoglobin`, `platelets`, `creatinine`, `bun`, `lactate`, `sodium`, `potassium`, `bicarbonate`, `glucose`, `gcs_total`

## 3 Categorical Features
- `gender`: `M`, `F`
- `admission_type`: `EMERGENCY`, `URGENT`, `ELECTIVE`
- `first_careunit`: `MICU`, `SICU`, `CCU`, `CVICU`, `TSICU`

## Unstructured Text
- `clinical_note`: Fictional 24-hour ICU progress note containing chief complaint, vitals, labs, and assessment/plan.
