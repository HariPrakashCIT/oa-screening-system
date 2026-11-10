"""
anomaly_detector.py — Interpretable statistical anomaly and change detection engine.

Simple explanation
------------------
Rather than using opaque black-box models that claim to diagnose disease,
this engine uses robust statistical methods (Robust Z-scores, Rolling Medians,
and Median Absolute Deviations) to identify when a patient's day-to-day
measurements have significantly changed from their own personal baseline.

Outputs three distinct categories:
  - Category 1: Within personal baseline (typical day-to-day fluctuation)
  - Category 2: Change to monitor (significant isolated or repeated deviation)
  - Category 3: Insufficient data (baseline not established yet)

Important Healthcare Guardrail:
A deviation is NEVER automatically reported as 'OA deterioration'.
Changes can be influenced by footwear, walking speed, tiredness, or sensor positioning.
"""

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.baseline_engine import BaselineEngine, PatientBaseline

logger = logging.getLogger(__name__)

# Standard z-score threshold for flagging significant change
DEFAULT_Z_THRESHOLD = 2.0


@dataclass
class MetricDeviation:
    metric: str
    current_value: float
    baseline_median: float
    baseline_std: float
    deviation_percent: float
    z_score: float
    status: str  # 'within_baseline' | 'change_to_monitor' | 'insufficient_data'
    clinical_note: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnomalyReport:
    patient_id: str
    overall_category: str  # 'Within personal baseline' | 'Change to monitor' | 'Insufficient data'
    baseline_status: str
    timestamp_utc: str
    deviations_detected: List[MetricDeviation] = field(default_factory=list)
    multi_variable_pattern: Optional[str] = None
    summary_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "overall_category": self.overall_category,
            "baseline_status": self.baseline_status,
            "timestamp_utc": self.timestamp_utc,
            "deviations_detected": [d.to_dict() for d in self.deviations_detected],
            "multi_variable_pattern": self.multi_variable_pattern,
            "summary_message": self.summary_message,
        }


class AnomalyDetector:
    """
    Evaluates new gait, activity, nutrition, and symptom observations
    against the patient's personal baseline.
    """

    def __init__(
        self,
        z_threshold: float = DEFAULT_Z_THRESHOLD,
        baseline_engine: Optional[BaselineEngine] = None,
    ):
        self.z_threshold = z_threshold
        self.baseline_engine = baseline_engine or BaselineEngine()

    def evaluate_metric(
        self,
        metric_name: str,
        current_value: Optional[float],
        baseline_stats: Optional[Dict[str, Any]],
        higher_is_worse: bool = True,
    ) -> MetricDeviation:
        """
        Evaluates a single metric using robust Z-score calculation.
        """
        if current_value is None or baseline_stats is None or baseline_stats.get("count", 0) < 3:
            return MetricDeviation(
                metric=metric_name,
                current_value=0.0 if current_value is None else float(current_value),
                baseline_median=0.0,
                baseline_std=0.0,
                deviation_percent=0.0,
                z_score=0.0,
                status="insufficient_data",
                clinical_note="Insufficient personal baseline data to establish regular range.",
            )

        med = float(baseline_stats.get("median", 0.0))
        mad = float(baseline_stats.get("mad", 0.0))
        std = float(baseline_stats.get("std", 1.0))

        # Robust Z-Score: using MAD if non-zero, otherwise standard dev
        if mad > 1e-4:
            # 1.4826 normalizes MAD to match standard normal distribution
            z = (current_value - med) / (mad * 1.4826)
        elif std > 1e-4:
            z = (current_value - med) / std
        else:
            z = 0.0

        diff = current_value - med
        pct = (diff / med * 100.0) if abs(med) > 1e-4 else 0.0

        if abs(z) >= self.z_threshold:
            status = "change_to_monitor"
            direction = "increase" if diff > 0 else "decrease"
            note = f"Observed {direction} ({abs(pct):.1f}%) from personal median (Z = {z:+.2f})."
        else:
            status = "within_baseline"
            note = f"Consistent with personal baseline (Z = {z:+.2f})."

        return MetricDeviation(
            metric=metric_name,
            current_value=round(float(current_value), 2),
            baseline_median=round(float(med), 2),
            baseline_std=round(float(std), 2),
            deviation_percent=round(float(pct), 1),
            z_score=round(float(z), 2),
            status=status,
            clinical_note=note,
        )

    def evaluate_patient_status(
        self,
        patient_id: str,
        current_daily_log: Optional[Dict[str, Any]] = None,
        latest_gait_features: Optional[Dict[str, Any]] = None,
        patient_baseline: Optional[PatientBaseline] = None,
    ) -> AnomalyReport:
        """
        Comprehensive evaluation across gait, physical activity, nutrition, and symptoms.
        """
        baseline = patient_baseline or self.baseline_engine.get_baseline(patient_id)
        b_status = baseline.baseline_status if baseline else "insufficient_data"

        deviations: List[MetricDeviation] = []

        # 1. Activity checks (Steps & Duration)
        if current_daily_log:
            steps_cur = current_daily_log.get("steps_count")
            steps_base = baseline.activity_baseline.get("steps_count") if baseline else None
            dev_steps = self.evaluate_metric("steps_count", steps_cur, steps_base, higher_is_worse=False)
            deviations.append(dev_steps)

            # Symptoms checks
            pain_cur = current_daily_log.get("pain_score")
            pain_base = baseline.symptom_baseline.get("pain_score") if baseline else None
            deviations.append(self.evaluate_metric("pain_score", pain_cur, pain_base, higher_is_worse=True))

            stiff_cur = current_daily_log.get("stiffness_minutes")
            stiff_base = baseline.symptom_baseline.get("stiffness_minutes") if baseline else None
            deviations.append(self.evaluate_metric("stiffness_minutes", stiff_cur, stiff_base, higher_is_worse=True))

            # Nutrition checks
            prot_cur = current_daily_log.get("protein_intake_g")
            prot_target = current_daily_log.get("protein_target_g", 60.0)
            if prot_cur is not None and prot_target > 0:
                prot_pct = ((prot_cur - prot_target) / prot_target) * 100.0
                p_status = "change_to_monitor" if prot_cur < (prot_target * 0.75) else "within_baseline"
                deviations.append(MetricDeviation(
                    metric="protein_intake_g",
                    current_value=round(float(prot_cur), 1),
                    baseline_median=round(float(prot_target), 1),
                    baseline_std=0.0,
                    deviation_percent=round(float(prot_pct), 1),
                    z_score=0.0,
                    status=p_status,
                    clinical_note=f"Intake {prot_cur:.1f}g vs target {prot_target:.1f}g ({prot_pct:+.1f}%).",
                ))

        # 2. Gait checks
        if latest_gait_features and baseline and baseline.gait_baseline:
            for g_metric in ("heel_peak", "forefoot_peak", "pressure_variability", "stance_duration_seconds"):
                cur_val = latest_gait_features.get(g_metric)
                base_st = baseline.gait_baseline.get(g_metric)
                deviations.append(self.evaluate_metric(g_metric, cur_val, base_st))

        # 3. Categorize overall condition
        flagged_changes = [d for d in deviations if d.status == "change_to_monitor"]

        if b_status == "insufficient_data" and not flagged_changes:
            overall = "Insufficient data"
            summary = "Baseline data is currently being established. Continue regular daily tracking."
        elif flagged_changes:
            overall = "Change to monitor"
            flag_names = ", ".join(d.metric.replace("_", " ") for d in flagged_changes)
            summary = f"Detected changes to monitor in: {flag_names}. Check symptom context."
        else:
            overall = "Within personal baseline"
            summary = "All recorded gait, activity, and symptom measurements remain within regular personal range."

        # 4. Multi-variable pattern detection
        # e.g., steps decreased + pain increased + stance duration altered
        pattern_desc = None
        has_step_drop = any(d.metric == "steps_count" and d.z_score <= -self.z_threshold for d in deviations)
        has_pain_rise = any(d.metric == "pain_score" and d.z_score >= self.z_threshold for d in deviations)
        has_gait_change = any(d.metric in ("heel_peak", "stance_duration_seconds") and d.status == "change_to_monitor" for d in deviations)

        if has_step_drop and has_pain_rise:
            pattern_desc = "Reduced physical activity co-occurring with elevated pain score."
        elif has_gait_change and has_pain_rise:
            pattern_desc = "Altered foot loading dynamics co-occurring with increased joint discomfort."
        elif has_step_drop and has_gait_change:
            pattern_desc = "Concurrently reduced activity and foot loading distribution changes."

        return AnomalyReport(
            patient_id=patient_id,
            overall_category=overall,
            baseline_status=b_status,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            deviations_detected=deviations,
            multi_variable_pattern=pattern_desc,
            summary_message=summary,
        )

