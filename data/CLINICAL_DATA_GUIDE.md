# Clinical Dataset Guidelines for Machine Learning

## 1. Ethical & Academic Standards

> **IMPORTANT:**
> 1. Clinical labels (`clinical_label`) must originate from **qualified medical evaluations** (e.g., radiographic Kellgren-Lawrence grade $\ge 2$, ACR clinical criteria, or physician confirmation).
> 2. **Never derive clinical labels from the sensor readings** itself; doing so causes tautological circular reasoning.
> 3. Synthetic or demonstration datasets must **never** be used for clinical claims. They exist strictly for verifying software execution.

---

## 2. Dataset Columns & Structure

When preparing a curated dataset for `ml/train_model.py`, use the following columns (see `data/clinical_dataset_template.csv`):

| Column | Type | Description |
|---|---|---|
| `participant_id` | String | Pseudonymous participant identifier (e.g. `patient_001`). Used for group-aware splitting. |
| `session_id` | String | Unique walking session identifier (e.g. `sess_001`). |
| `clinical_label` | Integer | **0** = Asymptomatic / Healthy Control, **1** = Clinically Diagnosed Knee Osteoarthritis. |
| `age` | Integer | Participant age in years. |
| `heel_peak` | Float | Maximum pressure ADC reading at the heel. |
| `midfoot_peak` | Float | Maximum pressure ADC reading at the midfoot. |
| `forefoot_peak` | Float | Maximum pressure ADC reading at the forefoot. |
| `heel_mean` | Float | Average heel pressure over the trial. |
| `midfoot_mean` | Float | Average midfoot pressure over the trial. |
| `forefoot_mean` | Float | Average forefoot pressure over the trial. |
| `heel_std` | Float | Standard deviation of heel pressure. |
| `midfoot_std` | Float | Standard deviation of midfoot pressure. |
| `forefoot_std` | Float | Standard deviation of forefoot pressure. |
| `acc_mag_mean` | Float | Mean magnitude of 3-axis acceleration ($g$). |
| `acc_mag_peak` | Float | Peak magnitude of 3-axis acceleration ($g$). |
| `acceleration_variability` | Float | Standard deviation of acceleration magnitude. |
| `gyro_mag_mean` | Float | Mean magnitude of angular velocity ($^\circ/\text{s}$). |
| `gyro_mag_peak` | Float | Peak magnitude of angular velocity ($^\circ/\text{s}$). |
| `gyro_variability` | Float | Standard deviation of angular velocity. |
| `stance_duration_seconds` | Float | Total weight-bearing duration during the walking trial. |
| `sampling_frequency_hz` | Float | Verified acquisition sampling frequency (target: ~50 Hz). |
| `notes` | String | Diagnostic notes (e.g., Kellgren-Lawrence grade, joint affected). |

---

## 3. Participant-Level Splitting Principle

In gait analysis, individuals possess unique biomechanical walking signatures. If multiple walking trials from the same person are split across the training and testing sets, the model will learn to recognize the *person* rather than the *disease pattern* (data leakage).

`ml/train_model.py` automatically implements `GroupShuffleSplit` on `participant_id`, guaranteeing:
$$\text{Train Participants} \cap \text{Test Participants} = \emptyset$$

---

## 4. Minimum Sample Size Recommendations

For an academic feasibility prototype:
- **Minimum:** 15–20 participants (at least 7–10 in each class) with 3–5 trials each.
- **Balanced Ratio:** Aim for roughly equal proportions of controls and confirmed OA subjects.
- **Reporting:** Always report Sensitivity, Specificity, and Confusion Matrix alongside Accuracy.

