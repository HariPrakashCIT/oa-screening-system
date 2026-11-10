"""
patient_monitoring.py — Unified coordinator for personalized osteoarthritis monitoring.

Provides a unified high-level interface that ties together:
  - Daily Lifestyle Logs (Nutrition, Activity, Symptoms)
  - Gait Biomechanical Sessions
  - Personalized Baseline Engine
  - Robust Anomaly & Change Detection
  - Multi-Week Trend & Exploratory Association Analysis
  - Gemini AI Plain-English Explanation Generator
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from app.ai_report_generator import AIReportGenerator
from app.anomaly_detector import AnomalyDetector, AnomalyReport
from app.baseline_engine import BaselineEngine, PatientBaseline
from app.daily_log_service import DailyLog, DailyLogService
from app.feature_extractor import FeatureExtractor
from app.trend_analyzer import TrendAnalyzer

logger = logging.getLogger(__name__)


class PatientMonitoringCoordinator:
    """
    Central orchestration facade used by the dashboard and background tasks.
    """

    def __init__(
        self,
        daily_log_service: Optional[DailyLogService] = None,
        baseline_engine: Optional[BaselineEngine] = None,
        anomaly_detector: Optional[AnomalyDetector] = None,
        trend_analyzer: Optional[TrendAnalyzer] = None,
        ai_report_generator: Optional[AIReportGenerator] = None,
    ):
        self.log_service = daily_log_service or DailyLogService()
        self.baseline_engine = baseline_engine or BaselineEngine()
        self.anomaly_detector = anomaly_detector or AnomalyDetector(baseline_engine=self.baseline_engine)
        self.trend_analyzer = trend_analyzer or TrendAnalyzer()
        self.ai_generator = ai_report_generator or AIReportGenerator()

    def run_daily_monitoring(
        self,
        patient_id: str,
        today_date_str: Optional[str] = None,
        upload_to_firebase: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs the complete daily evaluation pipeline:
          1. Fetches today's daily log (if logged).
          2. Fetches latest gait features.
          3. Evaluates personal baseline.
          4. Runs robust statistical anomaly detection.
          5. Computes multi-week trend summary.
          6. Generates understandable patient explanation via Gemini AI.
        """
        today = today_date_str or datetime.now().strftime("%Y-%m-%d")

        # 1. Daily log
        log = self.log_service.get_log(patient_id, today)
        daily_log_dict = log.to_dict() if log else None

        # 2. Latest gait features
        history_df = self.log_service.get_history_dataframe(patient_id, limit_days=30)
        baseline = self.baseline_engine.compute_baseline(patient_id)

        # Look for latest gait session
        latest_gait = None
        gait_list = self.baseline_engine._load_patient_gait_features(patient_id)
        if gait_list:
            latest_gait = sorted(gait_list, key=lambda x: x.get("timestamp_ms", 0), reverse=True)[0]

        # 3. Anomaly detection
        anomaly_report = self.anomaly_detector.evaluate_patient_status(
            patient_id=patient_id,
            current_daily_log=daily_log_dict,
            latest_gait_features=latest_gait,
            patient_baseline=baseline,
        )

        # 4. Weekly trend analysis
        weekly_summary = self.trend_analyzer.compute_weekly_summary(history_df)
        correlations = self.trend_analyzer.compute_exploratory_correlations(history_df)

        # 5. Gemini AI Report
        ai_insight = self.ai_generator.generate_patient_report(
            patient_id=patient_id,
            anomaly_report=anomaly_report.to_dict(),
            daily_log=daily_log_dict,
            latest_gait_features=latest_gait,
            weekly_trend=weekly_summary,
            upload_to_firebase=upload_to_firebase,
        )

        return {
            "patient_id": patient_id,
            "date": today,
            "daily_log": daily_log_dict,
            "baseline": baseline.to_dict(),
            "anomaly_report": anomaly_report.to_dict(),
            "weekly_summary": weekly_summary,
            "correlations": [c.to_dict() for c in correlations],
            "ai_insight": ai_insight,
        }

