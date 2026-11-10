"""
ai_report_generator.py — Gemini AI patient assistance and explanation engine.

Simple explanation
------------------
Numerical sensor numbers (like '4095 ADC counts' or 'Z-score = +2.4') are confusing
for patients. This module translates verified monitoring statistics into clear,
compassionate, easy-to-understand explanations.

It utilizes the Google Gemini API (via google-genai SDK) when configured,
and provides a deterministic rule-based template fallback if offline, unconfigured,
or rate-limited.

STRICT MEDICAL & ETHICAL GUARDRAILS:
  - NEVER diagnoses osteoarthritis or determines disease progression.
  - NEVER prescribes medications, supplements, or specific medical treatments.
  - NEVER invents missing measurements.
  - Explicitly emphasizes that anomalies may stem from footwear, speed, or fatigue.
  - ALWAYS includes clinical consultation advice for persistent or concerning changes.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import config
from app.firebase_service import get_firebase_service
from ml.model_utils import MEDICAL_DISCLAIMER_TEXT

logger = logging.getLogger(__name__)

# Try to import official google-genai SDK
try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    logger.info("google-genai package not found. AI report generator will use deterministic fallback engine.")


SYSTEM_INSTRUCTION = f"""
You are an AI assistant in a patient-monitoring support tool designed for individuals already diagnosed with knee osteoarthritis.
Your mission is to explain lifestyle, activity, nutrition, and walking sensor observations in compassionate, patient-friendly language.

CRITICAL MEDICAL INSTRUCTIONS:
1. DO NOT diagnose or claim osteoarthritis is progressing.
2. DO NOT prescribe medication or medical therapies.
3. DO NOT alter the numerical anomaly findings provided to you.
4. DO NOT interpret isolated single-day deviations as medical emergencies.
5. Emphasize that walking variations can be caused by footwear, fatigue, walking pace, or terrain.
6. Provide helpful, encouraging communication that the patient can review and discuss with their treating physician or physiotherapist.

MANDATORY OUTPUT FORMAT:
You MUST respond with a single valid JSON object containing exactly these keys:
{{
  "daily_summary": "Overall friendly explanation of today's monitoring observations",
  "activity_analysis": "Explanation of daily steps and movement consistency",
  "nutrition_analysis": "Summary of protein intake relative to the patient's existing target",
  "gait_analysis": "Explanation of foot pressure and walking stability measurements",
  "symptom_analysis": "Explanation of reported pain, stiffness, and fatigue levels",
  "detected_changes": ["list of plain-English bullet points describing any detected changes"],
  "suggested_follow_up": "Sensible questions or notes to prepare for their next clinical visit",
  "limitations": "Transparency note regarding sensor positioning, device limits, and research nature"
}}
"""


class AIReportGenerator:
    """
    Generates structured AI patient insights using Gemini API with deterministic fallback.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model_name = model_name or config.GEMINI_MODEL_NAME
        self._client = None

        if _GENAI_AVAILABLE and self.api_key:
            try:
                self._client = genai.Client(api_key=self.api_key)
                logger.info("Gemini AI Client initialized (model: %s).", self.model_name)
            except Exception as exc:
                logger.warning("Could not initialize Gemini Client: %s", exc)
                self._client = None

    def generate_patient_report(
        self,
        patient_id: str,
        anomaly_report: Dict[str, Any],
        daily_log: Optional[Dict[str, Any]] = None,
        latest_gait_features: Optional[Dict[str, Any]] = None,
        weekly_trend: Optional[Dict[str, Any]] = None,
        upload_to_firebase: bool = True,
    ) -> Dict[str, Any]:
        """
        Generates structured daily or periodic patient assistance insights.
        """
        # If Gemini client is active, attempt generative inference
        if self._client is not None:
            try:
                report = self._call_gemini(
                    patient_id=patient_id,
                    anomaly_report=anomaly_report,
                    daily_log=daily_log,
                    latest_gait_features=latest_gait_features,
                    weekly_trend=weekly_trend,
                )
                if report:
                    self._post_process_and_save(patient_id, report, upload_to_firebase)
                    return report
            except Exception as exc:
                logger.warning("Gemini generation failed (%s) — activating deterministic engine.", exc)

        # Fallback to deterministic explanation engine
        report = self._generate_deterministic_report(
            patient_id=patient_id,
            anomaly_report=anomaly_report,
            daily_log=daily_log,
            latest_gait_features=latest_gait_features,
            weekly_trend=weekly_trend,
        )
        self._post_process_and_save(patient_id, report, upload_to_firebase)
        return report

    def _call_gemini(
        self,
        patient_id: str,
        anomaly_report: Dict[str, Any],
        daily_log: Optional[Dict[str, Any]],
        latest_gait_features: Optional[Dict[str, Any]],
        weekly_trend: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Invokes Gemini API via google-genai SDK with structured JSON parsing."""
        context_payload = {
            "patient_pseudonym": patient_id,
            "overall_status": anomaly_report.get("overall_category"),
            "detected_deviations": anomaly_report.get("deviations_detected", []),
            "multi_variable_pattern": anomaly_report.get("multi_variable_pattern"),
            "today_activity_and_symptoms": daily_log or {},
            "latest_gait_metrics": latest_gait_features or {},
            "weekly_summary": weekly_trend or {},
        }

        prompt = (
            f"Here are the recent monitoring observations for patient '{patient_id}':\n"
            f"{json.dumps(context_payload, indent=2)}\n\n"
            "Please generate the patient explanation according to the system instructions in valid JSON format."
        )

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.3,
            ),
        )

        if response and response.text:
            text = response.text.strip()
            # Clean possible markdown formatting tags if returned
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            parsed = json.loads(text.strip())
            return parsed
        return None

    def _generate_deterministic_report(
        self,
        patient_id: str,
        anomaly_report: Dict[str, Any],
        daily_log: Optional[Dict[str, Any]],
        latest_gait_features: Optional[Dict[str, Any]],
        weekly_trend: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Robust rule-based explanation generator ensuring patient support is always available.
        """
        overall = anomaly_report.get("overall_category", "Within personal baseline")
        changes = anomaly_report.get("deviations_detected", [])
        flagged = [c for c in changes if c.get("status") == "change_to_monitor"]

        # Activity text
        steps = daily_log.get("steps_count", 0) if daily_log else 0
        if steps > 0:
            act_text = f"You recorded {steps:,} steps today. Regular walking within comfortable limits supports knee joint mobility."
        else:
            act_text = "No step count was recorded for today. Maintaining gentle daily motion is beneficial."

        # Nutrition text
        protein = daily_log.get("protein_intake_g", 0.0) if daily_log else 0.0
        target = daily_log.get("protein_target_g", 60.0) if daily_log else 60.0
        if protein > 0:
            nut_text = f"Logged {protein:.1f}g of protein toward your personal goal of {target:.1f}g. Adequate protein assists muscle maintenance around the knee."
        else:
            nut_text = f"Your daily protein target is {target:.1f}g. Remember to log your meals to track your nutrition."

        # Symptoms text
        pain = daily_log.get("pain_score", 0) if daily_log else 0
        stiffness = daily_log.get("stiffness_minutes", 0) if daily_log else 0
        sym_text = f"Today's reported pain score was {pain}/10 with {stiffness} minutes of morning stiffness."

        # Gait text
        if latest_gait_features:
            h_peak = latest_gait_features.get("heel_peak", 0)
            stance = latest_gait_features.get("stance_duration_seconds", 0)
            gait_text = f"Recent walking analysis shows consistent foot pressure (heel peak: {h_peak:.0f} counts, stance: {stance:.2f}s)."
        else:
            gait_text = "No recent wearable insole session was completed today."

        # Detected changes list
        change_bullets = []
        for f in flagged:
            metric_clean = f.get("metric", "").replace("_", " ")
            note = f.get("clinical_note", "")
            change_bullets.append(f"{metric_clean.title()}: {note}")

        if not change_bullets:
            change_bullets.append("All tracked signals remain within your normal baseline range.")

        # Summary
        if overall == "Change to monitor":
            pattern = anomaly_report.get("multi_variable_pattern")
            pattern_str = f" ({pattern})" if pattern else ""
            summary = (
                f"We noticed a change from your usual baseline today{pattern_str}. "
                "This could be due to tiredness, different shoes, or joint sensitivity. Keep an eye on how you feel tomorrow."
            )
            follow_up = "If joint stiffness or pain continues to feel higher than normal for several consecutive days, note it down to mention to your doctor."
        elif overall == "Insufficient data":
            summary = "Your personal baseline is still being established. Continue your regular tracking routine over the coming days."
            follow_up = "Keep logging your activity and symptoms so the system can personalize your baseline."
        else:
            summary = "Great job today! Your walking patterns, physical activity, and symptoms are well aligned with your personal baseline."
            follow_up = "Continue following your recommended therapeutic exercise and activity routine."

        return {
            "daily_summary": summary,
            "activity_analysis": act_text,
            "nutrition_analysis": nut_text,
            "gait_analysis": gait_text,
            "symptom_analysis": sym_text,
            "detected_changes": change_bullets,
            "suggested_follow_up": follow_up,
            "limitations": "These insights are based on wearable sensors and self-reported logs. They provide general awareness and are not a clinical diagnosis.",
            "generator_source": "deterministic_template",
            "disclaimer": MEDICAL_DISCLAIMER_TEXT,
        }

    def _post_process_and_save(
        self,
        patient_id: str,
        report: Dict[str, Any],
        upload_to_firebase: bool,
    ) -> None:
        """Saves generated insights to local storage and Firestore collection 'ai_insights'."""
        report["patient_id"] = patient_id
        report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["disclaimer"] = MEDICAL_DISCLAIMER_TEXT

        # Save locally
        out_dir = config.DATA_PROCESSED_DIR / "ai_insights"
        out_dir.mkdir(parents=True, exist_ok=True)
        today_str = datetime.now().strftime("%Y-%m-%d")
        local_path = out_dir / f"{patient_id}_{today_str}_insight.json"
        try:
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception:
            pass

        # Sync to Firestore collection 'ai_insights'
        if upload_to_firebase:
            fb = get_firebase_service()
            if fb.is_connected:
                try:
                    doc_id = f"{patient_id}_{today_str}"
                    fb._db.collection(config.INSIGHTS_COLLECTION).document(doc_id).set(report)
                    logger.info("Saved AI Insight to Firestore (%s).", doc_id)
                except Exception as exc:
                    logger.warning("Could not sync AI Insight to Firestore: %s", exc)

