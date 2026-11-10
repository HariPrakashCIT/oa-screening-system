"""
trend_analyzer.py — Longitudinal trend and exploratory correlation analysis for OA monitoring.

Simple explanation
------------------
Osteoarthritis symptoms fluctuate over weeks and months based on activity,
therapeutic exercise, and joint load.
This module performs:
  1. 7-day and 14-day rolling activity and symptom trends.
  2. Spearman rank correlation analysis across paired observations (e.g. Steps vs. Pain, Exercise vs. Stiffness).
  3. Adherence rates (therapeutic exercise completion and nutrition target achievement).

IMPORTANT ACADEMIC & MEDICAL DISCLAIMER:
Statistical correlations are strictly labeled as exploratory associations.
They do NOT prove medical causation (e.g. 'walking caused pain').
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

MIN_CORRELATION_PAIRS = 14


@dataclass
class CorrelationFinding:
    variable_x: str
    variable_y: str
    spearman_rho: float
    p_value: float
    n_observations: int
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variable_x": self.variable_x,
            "variable_y": self.variable_y,
            "spearman_rho": round(float(self.spearman_rho), 3),
            "p_value": round(float(self.p_value), 4),
            "n_observations": int(self.n_observations),
            "interpretation": self.interpretation,
        }


class TrendAnalyzer:
    """
    Analyzes multi-week trends and associations in OA patient data.
    """

    def __init__(self, min_correlation_pairs: int = MIN_CORRELATION_PAIRS):
        self.min_correlation_pairs = min_correlation_pairs

    def compute_weekly_summary(self, df_history: pd.DataFrame) -> Dict[str, Any]:
        """
        Summarizes the most recent 7 days of activity, nutrition, and symptoms.
        """
        if df_history.empty:
            return {"status": "insufficient_data", "message": "No historical data logged yet."}

        # Last 7 records
        recent = df_history.tail(7).copy()
        n_days = len(recent)

        avg_steps = float(recent["steps_count"].mean()) if "steps_count" in recent.columns else 0.0
        avg_pain = float(recent["pain_score"].mean()) if "pain_score" in recent.columns else 0.0
        avg_stiffness = float(recent["stiffness_minutes"].mean()) if "stiffness_minutes" in recent.columns else 0.0
        avg_protein = float(recent["protein_intake_g"].mean()) if "protein_intake_g" in recent.columns else 0.0

        # Adherence percentages
        exercise_days = int(recent["exercise_completed"].sum()) if "exercise_completed" in recent.columns else 0
        exercise_rate_pct = round((exercise_days / n_days) * 100.0, 1) if n_days > 0 else 0.0

        target_g = float(recent["protein_target_g"].iloc[-1]) if "protein_target_g" in recent.columns else 60.0
        met_protein_days = int((recent["protein_intake_g"] >= target_g).sum()) if "protein_intake_g" in recent.columns else 0
        protein_adherence_pct = round((met_protein_days / n_days) * 100.0, 1) if n_days > 0 else 0.0

        return {
            "status": "success",
            "days_analyzed": n_days,
            "avg_daily_steps": round(avg_steps, 0),
            "avg_pain_score": round(avg_pain, 1),
            "avg_stiffness_minutes": round(avg_stiffness, 1),
            "avg_daily_protein_g": round(avg_protein, 1),
            "exercise_completion_rate_pct": exercise_rate_pct,
            "protein_adherence_rate_pct": protein_adherence_pct,
        }

    def compute_exploratory_correlations(self, df_history: pd.DataFrame) -> List[CorrelationFinding]:
        """
        Calculates Spearman rank correlations between paired daily variables.
        Requires at least min_correlation_pairs (default 14) observations.
        """
        findings: List[CorrelationFinding] = []
        if len(df_history) < self.min_correlation_pairs:
            return findings

        pairs = [
            ("steps_count", "pain_score", "Daily step volume vs. self-reported pain"),
            ("steps_count", "stiffness_minutes", "Physical activity vs. morning stiffness duration"),
            ("pain_score", "fatigue_score", "Pain severity vs. general fatigue"),
            ("protein_intake_g", "pain_score", "Protein consumption vs. symptom reporting"),
        ]

        for var_x, var_y, desc in pairs:
            if var_x in df_history.columns and var_y in df_history.columns:
                sub = df_history[[var_x, var_y]].dropna()
                if len(sub) >= self.min_correlation_pairs:
                    x_vals = sub[var_x].values
                    y_vals = sub[var_y].values

                    # Check variance (avoid zero-division in constant series)
                    if np.std(x_vals) > 1e-4 and np.std(y_vals) > 1e-4:
                        res = spearmanr(x_vals, y_vals)
                        rho = float(res.statistic)
                        p_val = float(res.pvalue)

                        # Formulate readable academic interpretation
                        if abs(rho) < 0.2:
                            strength = "negligible"
                        elif abs(rho) < 0.4:
                            strength = "weak"
                        elif abs(rho) < 0.7:
                            strength = "moderate"
                        else:
                            strength = "strong"

                        direction = "positive" if rho > 0 else "inverse"
                        interp = (
                            f"Observed {strength} {direction} exploratory association (rho = {rho:+.2f}, p = {p_val:.3f}). "
                            "Notice: Association does not imply clinical causation."
                        )

                        findings.append(CorrelationFinding(
                            variable_x=var_x,
                            variable_y=var_y,
                            spearman_rho=rho,
                            p_value=p_val,
                            n_observations=len(sub),
                            interpretation=interp,
                        ))

        return findings

