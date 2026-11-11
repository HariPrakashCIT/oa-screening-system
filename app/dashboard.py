"""
dashboard.py — Streamlit Dashboard for AI-Powered Osteoarthritis Patient Assistance and Monitoring.

Redesigned with the Orange & Black Fitness-Tech Visual Identity:
- Background: #101010
- Surfaces: #1A1A1A / #222222
- Primary Accent: #FF6B00 (Orange)
- Secondary Accent: #FF8A24 (Warm Orange)
- Typography: High-contrast white (#FFFFFF) and muted grey (#9E9E9E)

Full Navigation Modules:
  1. Dashboard Overview        - Daily KPIs, anomaly status, baseline status, activity charts.
  2. Patient Profile           - Complete patient record & step-based onboarding wizard.
  3. Medical Reports           - Upload and view clinical imaging, X-rays, lab records.
  4. Nutrition Tracking        - Daily food diary, protein goal progress bar, weekly trends.
  5. Physical Activity         - Daily step volume, walking duration, baseline comparison.
  6. Gait Monitoring           - Live ESP32 / demo insole pressure meters, recording, trial inspection.
  7. Symptom Journal           - Pain score (0-10), morning stiffness, swelling, exercise tracking.
  8. AI Insights               - Baseline deviations, Gemini AI summaries, physician questions.
  9. System Settings           - Hardware ports, cloud sync, Gemini API configuration.

Run with:
    streamlit run app/dashboard.py
"""

import json
import hmac
from html import escape as html_escape
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False
    print("Streamlit is not installed. Run: pip install streamlit matplotlib")

from app.config import config
from app.daily_log_service import DailyLog, DailyLogService, MealEntry, calculate_daily_protein
from app.data_processor import DataProcessor
from app.feature_extractor import FeatureExtractor
from app.firebase_service import get_firebase_service
from app.patient_monitoring import PatientMonitoringCoordinator
from app.patient_profile_service import PatientProfileService, calculate_bmi
from app.serial_receiver import list_serial_ports
from app.session_manager import (
    SessionManager,
    generate_session_id,
    list_participants,
    next_participant_id,
)
from app.ui_theme import apply_theme, create_dark_figure, create_light_figure
from ml.model_utils import MEDICAL_DISCLAIMER_TEXT
from ml.predict import find_latest_model, predict

logger = logging.getLogger(__name__)


# ===========================================================================
# PAGE SETUP & TOP BAR
# ===========================================================================

def setup_page():
    """Initialise Streamlit page layout and inject the Orange & Black theme."""
    if not _STREAMLIT_AVAILABLE:
        return
    st.set_page_config(
        page_title="KINETIX OA — Patient Monitoring",
        page_icon="🦵",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()


def render_topbar(title: str, category: str, patient_id: str):
    """Render a modern top application header with patient context and live date."""
    profile_svc = PatientProfileService()
    profile = profile_svc.get_profile(patient_id)
    kl_info = profile.get("kl_grade", "Grade 2")
    side_info = profile.get("affected_side", "Right")
    today_disp = datetime.now().strftime("%a, %b %d, %Y")

    st.markdown(
        f"""
        <div class="k-topbar">
            <div>
                <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; color:#FF8A24; font-weight:600;">
                    {category}
                </div>
                <h1 class="k-topbar-title">{title}</h1>
            </div>
            <div style="display:flex; align-items:center; gap:12px;">
                <div class="k-topbar-patient">
                    <span class="k-patient-dot"></span>
                    <strong style="color:#FFFFFF;">{patient_id}</strong>
                    <span style="color:#7A7A7A;">|</span>
                    <span style="color:#A3A3A3;">{side_info} Knee • {kl_info}</span>
                </div>
                <div style="color:#888888; font-size:0.82rem; font-weight:500;">
                    {today_disp}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# SIDEBAR
# ===========================================================================

def render_sidebar():
    """Render the sleek dark sidebar with brand logo, patient selector, and navigation."""
    # Brand Header
    st.sidebar.markdown(
        """
        <div class="k-brand-badge">
            <div class="k-brand-icon">&#9889;</div>
            <div>
                <h2 class="k-brand-title">KINETIX <span style="color:#FF6B00;">OA</span></h2>
                <div class="k-brand-sub">Clinical Gait & Monitoring</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Medical Disclaimer notice
    st.sidebar.markdown(
        """
        <div class="k-disclaimer">
            <strong>&#9888; Research Prototype:</strong><br>
            Designed for continuous lifestyle and gait monitoring of diagnosed knee OA patients.
            Does not replace medical diagnosis or clinical consultation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Patient Selector
    profile_svc = PatientProfileService()
    all_patients = profile_svc.list_all_patients()
    if not all_patients:
        all_patients = ["patient_001"]

    active_patient = st.sidebar.selectbox(
        "Active Patient Profile",
        all_patients,
        index=0,
        help="Select a patient to load their personalized baseline, nutrition, and walking trials.",
    )

    st.sidebar.markdown("---")

    # Primary Navigation
    page = st.sidebar.radio(
        "Navigation Menu",
        [
            "1. Dashboard Overview",
            "2. Patient Profile",
            "3. Medical Reports",
            "4. Nutrition Tracking",
            "5. Physical Activity",
            "6. Gait Monitoring",
            "7. Symptom Journal",
            "8. AI Insights",
            "9. Settings",
        ],
    )

    st.sidebar.markdown("---")

    # Live System Health Badges
    st.sidebar.markdown(
        "<div style='font-size:0.75rem; text-transform:uppercase; color:#777; font-weight:600; margin-bottom:8px;'>System Health</div>",
        unsafe_allow_html=True,
    )

    fb = get_firebase_service()
    fb_label = "Cloud Synced" if fb.is_connected else "Local Storage"
    fb_class = "k-pill-green" if fb.is_connected else "k-pill-neutral"
    st.sidebar.markdown(
        f"<div style='margin-bottom:6px;'><span class='k-pill {fb_class}'>● Firestore: {fb_label}</span></div>",
        unsafe_allow_html=True,
    )

    latest_m = find_latest_model()
    ml_label = f"Ready ({latest_m.stem})" if latest_m else "No Model"
    ml_class = "k-pill-orange" if latest_m else "k-pill-neutral"
    st.sidebar.markdown(
        f"<div style='margin-bottom:6px;'><span class='k-pill {ml_class}'>● ML Engine: {ml_label}</span></div>",
        unsafe_allow_html=True,
    )

    gemini_key = config.GEMINI_API_KEY
    ai_label = "Gemini 2.5 Flash" if gemini_key else "Deterministic Engine"
    ai_class = "k-pill-green" if gemini_key else "k-pill-neutral"
    st.sidebar.markdown(
        f"<div style='margin-bottom:6px;'><span class='k-pill {ai_class}'>● AI Engine: {ai_label}</span></div>",
        unsafe_allow_html=True,
    )

    ports = list_serial_ports()
    port_label = ports[0] if ports else "Demo Mode (~50Hz)"
    st.sidebar.markdown(
        f"<div style='margin-bottom:6px;'><span class='k-pill k-pill-orange'>● Hardware: {port_label}</span></div>",
        unsafe_allow_html=True,
    )

    return page, active_patient


# ===========================================================================
# PAGE 1: DASHBOARD OVERVIEW
# ===========================================================================

def page_patient_overview(patient_id: str):
    """A clean overview of recent activity, symptoms, and clinical context."""
    st.markdown('<div class="k-dashboard-shell"></div>', unsafe_allow_html=True)
    render_topbar("Daily Overview", "Monitoring Dashboard", patient_id)

    profile = PatientProfileService().get_profile(patient_id)
    safe_patient_id = html_escape(str(patient_id))
    safe_display_name = html_escape(str(profile.get("display_name", patient_id)))
    safe_kl_grade = html_escape(str(profile.get("kl_grade", "Grade 2")))
    safe_affected_side = html_escape(str(profile.get("affected_side", "Right")))
    safe_mobility_notes = html_escape(str(profile.get("mobility_limitations", "No mobility notes recorded.")))
    today_str = datetime.now().strftime("%Y-%m-%d")
    coordinator = PatientMonitoringCoordinator()
    log_service = coordinator.log_service
    today_log = log_service.get_log(patient_id, today_str)
    status_eval = coordinator.run_daily_monitoring(
        patient_id, today_date_str=today_str, upload_to_firebase=False
    )
    history_df = log_service.get_history_dataframe(patient_id, limit_days=14)

    steps_count = today_log.steps_count if today_log else 0
    step_goal = profile.get("daily_step_goal", 5000) or 5000
    step_pct = min(int(steps_count / step_goal * 100), 100)
    protein_val = today_log.protein_intake_g if today_log else 0.0
    protein_goal = profile.get("daily_protein_goal_g", 65.0) or 65.0
    protein_pct = min(int(protein_val / protein_goal * 100), 100) if protein_goal else 0
    pain_val = today_log.pain_score if today_log else 0
    pain_label = "Low" if pain_val <= 3 else ("Moderate" if pain_val <= 6 else "Elevated")
    exercise_done = bool(today_log.exercise_completed) if today_log else False

    st.markdown(
        """
        <div class="k-page-intro">
            <div class="k-breadcrumb">Dashboard <span>/</span> Overview</div>
            <p>Track activity, symptoms, and daily progress in one place.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    anomaly = status_eval.get("anomaly_report", {})
    anomaly_summary = html_escape(str(anomaly.get("summary_message", "Log regular activity and symptoms to build a personal trend.")))
    category = anomaly.get("overall_category", "Within personal baseline")
    if category == "Change to monitor":
        alert_class, alert_title, alert_icon = "k-alert-monitor", "Change to monitor", "!"
    elif category == "Insufficient data":
        alert_class, alert_title, alert_icon = "k-alert-progress", "Building your baseline", "i"
    else:
        alert_class, alert_title, alert_icon = "k-alert-baseline", "Within personal baseline", "✓"
    st.markdown(
        f"""
        <div class="{alert_class} k-dashboard-alert">
            <span class="k-alert-icon">{alert_icon}</span>
            <div><strong>{alert_title}</strong><p>{anomaly_summary}</p></div>
            <span class="k-pill {'k-pill-red' if category == 'Change to monitor' else 'k-pill-green' if category != 'Insufficient data' else 'k-pill-neutral'}">{'Review' if category == 'Change to monitor' else 'Monitoring'}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI cards follow the compact three-card rhythm in the supplied dashboard reference.
    kpi_cols = st.columns(3)
    metrics = [
        ("Daily steps", f"{steps_count:,}", "steps", f"Goal {step_goal:,} · {step_pct}% complete", step_pct, "#7462D3"),
        ("Knee discomfort", str(pain_val), "/ 10", f"{pain_label} today", max(0, 100 - pain_val * 10), "#D98972"),
        ("Protein today", f"{protein_val:.0f}", "g", f"Goal {protein_goal:.0f}g · {protein_pct}% reached", protein_pct, "#3AA681"),
    ]
    for col, (label, value, unit, sub, progress, accent) in zip(kpi_cols, metrics):
        with col:
            st.markdown(
                f"""
                <div class="k-kpi-card" style="--k-accent:{accent}">
                    <div class="k-kpi-heading"><span class="k-kpi-label">{label}</span><span class="k-kpi-mark"></span></div>
                    <div class="k-kpi-value">{value}<span class="k-kpi-unit">{unit}</span></div>
                    <div class="k-kpi-sub">{sub}</div>
                    <div class="k-kpi-bar"><div class="k-kpi-bar-fill" style="width:{progress}%;"></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    chart_col, detail_col = st.columns([1.65, 1], gap="large")
    with chart_col:
        st.markdown(
            '<div class="k-section-heading"><div><h3>Activity trend</h3><p>Daily steps over the last 14 days</p></div><span class="k-range-chip">Last 14 days</span></div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True, key="steps_trend_card"):
            if not history_df.empty and "steps_count" in history_df.columns:
                step_data = history_df[["date", "steps_count"]].copy()
                step_data["date"] = pd.to_datetime(step_data["date"]).dt.strftime("%b %d")
                step_data = step_data.set_index("date")
                if _MATPLOTLIB_AVAILABLE:
                    fig, ax = create_light_figure(figsize=(9, 3.2))
                    ax.plot(step_data.index, step_data["steps_count"], color="#7462D3", marker="o", linewidth=2.4, markersize=4, label="Steps")
                    ax.fill_between(range(len(step_data)), step_data["steps_count"].to_numpy(), alpha=.10, color="#7462D3")
                    ax.axhline(step_goal, color="#B3B7C5", linestyle="--", linewidth=1.0, label=f"Daily goal ({step_goal:,})")
                    ax.set_ylabel("Steps", color="#707587", fontsize=9)
                    ax.legend(facecolor="#FFFFFF", edgecolor="#E9EAF0", labelcolor="#5F6475", fontsize=8, frameon=False)
                    plt.xticks(rotation=0, ha="center")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.line_chart(step_data)
            else:
                st.markdown('<div class="k-empty-state">No activity records yet. Add a daily step count in Physical Activity to start this trend.</div>', unsafe_allow_html=True)

    with detail_col:
        base = status_eval.get("baseline", {})
        baseline_status = base.get("baseline_status", "provisional").title()
        observation_count = base.get("valid_observation_count", 0)
        gait_files = sorted(
            list(config.DATA_RAW_DIR.glob(f"{patient_id}_*.csv")) + list(config.DATA_DEMO_DIR.glob(f"{patient_id}_*.csv")),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        st.markdown(
            f"""
            <div class="k-section-heading"><div><h3>Patient overview</h3><p>Profile and monitoring status</p></div></div>
            <div class="k-profile-card">
                <div class="k-profile-card-top"><div class="k-avatar">{safe_display_name[:1].upper()}</div><div><strong>{safe_display_name}</strong><span>{safe_patient_id}</span></div><span class="k-pill k-pill-orange">{safe_kl_grade}</span></div>
                <div class="k-profile-grid"><div><span>Affected knee</span><strong>{safe_affected_side}</strong></div><div><span>Baseline</span><strong>{baseline_status}</strong></div><div><span>Sessions</span><strong>{observation_count}</strong></div><div><span>Rehab today</span><strong>{'Done' if exercise_done else 'Pending'}</strong></div></div>
                <div class="k-profile-note">{safe_mobility_notes}</div>
            </div>
            <div class="k-gait-status"><span class="k-gait-status-dot"></span><div><strong>Wearable insole</strong><span>{'Latest trial: ' + datetime.fromtimestamp(gait_files[0].stat().st_mtime).strftime('%b %d, %Y') if gait_files else 'No gait trial recorded yet'}</span></div><span class="k-gait-frequency">50 Hz</span></div>
            """,
            unsafe_allow_html=True,
        )

    lower_chart, lower_detail = st.columns([1.25, 1], gap="large")
    with lower_chart:
        st.markdown(
            '<div class="k-section-heading"><div><h3>Nutrition consistency</h3><p>Protein intake compared with daily goal</p></div></div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True, key="nutrition_trend_card"):
            if not history_df.empty and "protein_intake_g" in history_df.columns:
                protein_data = history_df[["date", "protein_intake_g"]].copy()
                protein_data["date"] = pd.to_datetime(protein_data["date"]).dt.strftime("%b %d")
                protein_data = protein_data.set_index("date")
                if _MATPLOTLIB_AVAILABLE:
                    fig, ax = create_light_figure(figsize=(8, 2.7))
                    ax.plot(protein_data.index, protein_data["protein_intake_g"], color="#3AA681", marker="o", linewidth=2.2, markersize=4, label="Protein")
                    ax.fill_between(range(len(protein_data)), protein_data["protein_intake_g"].to_numpy(), alpha=.10, color="#3AA681")
                    ax.axhline(protein_goal, color="#B3B7C5", linestyle="--", linewidth=1.0, label=f"Daily goal ({protein_goal:.0f}g)")
                    ax.set_ylabel("Grams", color="#707587", fontsize=9)
                    ax.legend(facecolor="#FFFFFF", edgecolor="#E9EAF0", labelcolor="#5F6475", fontsize=8, frameon=False)
                    plt.xticks(rotation=0, ha="center")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.line_chart(protein_data)
            else:
                st.markdown('<div class="k-empty-state">No nutrition records yet. Log meals to see daily protein here.</div>', unsafe_allow_html=True)

    with lower_detail:
        st.markdown(
            """
            <div class="k-section-heading"><div><h3>Care snapshot</h3><p>Useful context for today</p></div></div>
            <div class="k-care-card">
                <div class="k-care-line"><span>Exercise plan</span><strong>{exercise}</strong></div>
                <div class="k-care-line"><span>Body mass index</span><strong>{bmi}</strong></div>
                <div class="k-care-line"><span>Severity grade</span><strong>{grade}</strong></div>
            </div>
            """.format(
                exercise="Completed" if exercise_done else "Not logged",
                bmi=html_escape(str(profile.get("bmi", "—"))),
                grade=safe_kl_grade,
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="k-section-heading k-recent-heading"><div><h3>Recent monitoring</h3><p>Your latest recorded activity, nutrition, and symptoms</p></div><span class="k-range-chip">Recent records</span></div>',
        unsafe_allow_html=True,
    )
    if not history_df.empty:
        recent = history_df.copy().sort_values("date", ascending=False).head(7)
        recent["date"] = pd.to_datetime(recent["date"]).dt.strftime("%b %d, %Y")
        columns = [column for column in ("date", "steps_count", "protein_intake_g", "pain_score") if column in recent.columns]
        recent = recent[columns].rename(columns={
            "date": "Date", "steps_count": "Steps", "protein_intake_g": "Protein (g)", "pain_score": "Pain / 10"
        })
        st.dataframe(recent, hide_index=True, use_container_width=True)
    else:
        st.markdown('<div class="k-empty-state">There are no daily records to show yet.</div>', unsafe_allow_html=True)
    st.caption("Research prototype · Monitoring support only; this dashboard does not provide a medical diagnosis.")


# ===========================================================================
# PAGE 2: PATIENT PROFILE & ONBOARDING WIZARD
# ===========================================================================

def page_patient_profile(patient_id: str):
    """Full-page profile viewer and comprehensive step-based patient onboarding."""
    render_topbar("Patient Profile & Clinical Onboarding", "Profile Management", patient_id)

    profile_svc = PatientProfileService()
    profile = profile_svc.get_profile(patient_id)

    tab_view, tab_wizard = st.tabs(["👤 Active Clinical Profile", "📝 Patient Onboarding Wizard"])

    # -----------------------------------------------------------------------
    # TAB 1: Structured Profile View & Quick Edit
    # -----------------------------------------------------------------------
    with tab_view:
        st.markdown(
            f"""
            <div class="k-card">
                <div class="k-card-header">
                    <div>
                        <h2 class="k-card-title" style="font-size:1.4rem;">{profile.get('display_name', patient_id)}</h2>
                        <div class="k-card-subtitle">ID: {patient_id} • Knee Osteoarthritis Clinical Cohort</div>
                    </div>
                    <span class="k-pill k-pill-orange">{profile.get('kl_grade', 'Grade 2')}</span>
                </div>
                <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:16px; margin-top:14px;">
                    <div><span style="color:#888; font-size:0.8rem;">AGE</span><br><strong style="font-size:1.1rem;">{profile.get('age', 58)} yrs</strong></div>
                    <div><span style="color:#888; font-size:0.8rem;">GENDER</span><br><strong style="font-size:1.1rem;">{profile.get('gender', 'Female')}</strong></div>
                    <div><span style="color:#888; font-size:0.8rem;">HEIGHT / WEIGHT</span><br><strong style="font-size:1.1rem;">{profile.get('height_cm', 165.0)} cm / {profile.get('weight_kg', 72.0)} kg</strong></div>
                    <div><span style="color:#888; font-size:0.8rem;">BMI</span><br><strong style="font-size:1.1rem; color:#FF8A24;">{profile.get('bmi', 26.4)}</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("edit_profile_form"):
            st.subheader("Edit Profile & Target Settings")
            c1, c2, c3 = st.columns(3)
            with c1:
                disp_name = st.text_input("Display Name", value=profile.get("display_name", patient_id))
                age_val = st.number_input("Age (Years)", min_value=18, max_value=110, value=int(profile.get("age", 58)))
                gender_val = st.selectbox("Gender", ["Female", "Male", "Other"], index=0 if profile.get("gender") == "Female" else 1)
            with c2:
                height_val = st.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=float(profile.get("height_cm", 165.0)))
                weight_val = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=float(profile.get("weight_kg", 72.0)))
                step_goal_val = st.number_input("Daily Step Goal", min_value=1000, max_value=30000, value=int(profile.get("daily_step_goal", 5000)), step=500)
            with c3:
                prot_goal_val = st.number_input("Daily Protein Goal (g)", min_value=30.0, max_value=250.0, value=float(profile.get("daily_protein_goal_g", 65.0)), step=5.0)
                joint_val = st.selectbox("Affected Joint", ["Knee", "Hip", "Ankle"], index=0)
                side_val = st.selectbox("Affected Side", ["Right", "Left", "Bilateral"], index=0 if profile.get("affected_side") == "Right" else 1)

            st.markdown("---")
            c_notes1, c_notes2 = st.columns(2)
            with c_notes1:
                meds_val = st.text_area("Current Medications & Supplements", value=profile.get("current_medications", ""))
                mobility_val = st.text_area("Mobility Limitations / Symptom Triggers", value=profile.get("mobility_limitations", ""))
            with c_notes2:
                regimen_val = st.text_area("Physiotherapy & Exercise Regimen", value=profile.get("exercise_regimen", ""))
                clinical_notes_val = st.text_area("Clinical Notes & Imaging Summary", value=profile.get("clinical_notes", ""))

            save_profile_btn = st.form_submit_button("💾 Save Profile Changes")

            if save_profile_btn:
                profile.update({
                    "display_name": disp_name,
                    "age": int(age_val),
                    "gender": gender_val,
                    "height_cm": float(height_val),
                    "weight_kg": float(weight_val),
                    "daily_step_goal": int(step_goal_val),
                    "daily_protein_goal_g": float(prot_goal_val),
                    "oa_joint": joint_val,
                    "affected_side": side_val,
                    "current_medications": meds_val,
                    "mobility_limitations": mobility_val,
                    "exercise_regimen": regimen_val,
                    "clinical_notes": clinical_notes_val,
                })
                profile_svc.save_profile(profile)
                st.success(f"Profile for '{patient_id}' saved successfully to local storage and cloud!")
                st.rerun()

    # -----------------------------------------------------------------------
    # TAB 2: Full Patient Onboarding Wizard
    # -----------------------------------------------------------------------
    with tab_wizard:
        st.markdown(
            """
            <div class="k-card">
                <h3 class="k-card-title">&#128640; Comprehensive Patient Onboarding Wizard</h3>
                <p style="color:#A3A3A3; font-size:0.88rem; margin:4px 0 0 0;">
                    Follow the guided steps below to enroll a new knee osteoarthritis participant.
                    All baseline metrics and clinical history will be configured.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("onboarding_wizard_form"):
            st.markdown("#### **Step 1: Patient Identity & Demographics**")
            w_c1, w_c2, w_c3 = st.columns(3)
            with w_c1:
                suggested_id = next_participant_id()
                new_id = st.text_input("Pseudonymous Patient ID", value=suggested_id)
                new_name = st.text_input("Patient Display Name", value=f"Participant {suggested_id.split('_')[-1]}")
            with w_c2:
                new_age = st.number_input("Age (Years)", min_value=18, max_value=110, value=60)
                new_gender = st.selectbox("Biological Sex", ["Female", "Male", "Other"])
            with w_c3:
                new_height = st.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=168.0)
                new_weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=74.0)

            st.markdown("---")
            st.markdown("#### **Step 2: Knee Osteoarthritis Diagnosis**")
            w_d1, w_d2, w_d3 = st.columns(3)
            with w_d1:
                new_side = st.selectbox("Affected Limb / Knee", ["Right Leg / Knee", "Left Leg / Knee", "Bilateral / Both"])
                new_kl = st.selectbox("Kellgren-Lawrence Severity", [
                    "Grade 1 (Doubtful)",
                    "Grade 2 (Mild)",
                    "Grade 3 (Moderate)",
                    "Grade 4 (Severe)",
                ], index=1)
            with w_d2:
                new_diag_year = st.number_input("Year of OA Diagnosis", min_value=1980, max_value=2026, value=2024)
                new_duration = st.number_input("Symptom Duration (Months)", min_value=1, max_value=360, value=12)
            with w_d3:
                new_limitations = st.text_area("Mobility Limitations", "Morning stiffness > 20 min, pain on stairs, prolonged standing discomfort")

            st.markdown("---")
            st.markdown("#### **Step 3: Goals, Nutrition & Regimen**")
            w_g1, w_g2, w_g3 = st.columns(3)
            with w_g1:
                new_steps_target = st.number_input("Daily Step Target", min_value=1000, max_value=25000, value=5000, step=500)
            with w_g2:
                new_prot_target = st.number_input("Daily Protein Goal (grams)", min_value=30.0, max_value=250.0, value=65.0, step=5.0)
            with w_g3:
                new_meds = st.text_input("Active Medications", "Acetaminophen, Glucosamine, Topical Gel")

            new_notes = st.text_area("Physician Notes & Clinical Background", "Patient enrolled for bilateral insole gait analysis study. Radiographs show medial compartment joint space narrowing.")

            complete_btn = st.form_submit_button("🏁 Complete Onboarding & Register Patient")

            if complete_btn:
                new_profile = {
                    "patient_id": new_id,
                    "display_name": new_name,
                    "age": int(new_age),
                    "gender": new_gender,
                    "height_cm": float(new_height),
                    "weight_kg": float(new_weight),
                    "bmi": calculate_bmi(new_height, new_weight),
                    "oa_joint": "Knee",
                    "affected_side": new_side,
                    "kl_grade": new_kl,
                    "diagnosis_year": int(new_diag_year),
                    "symptoms_duration_months": int(new_duration),
                    "mobility_limitations": new_limitations,
                    "current_medications": new_meds,
                    "daily_step_goal": int(new_steps_target),
                    "daily_protein_goal_g": float(new_prot_target),
                    "exercise_regimen": "Standard knee joint stabilization",
                    "clinical_notes": new_notes,
                    "onboarding_completed": True,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                profile_svc.save_profile(new_profile)
                st.success(f"🎉 Patient '{new_id}' successfully onboarded and registered!")
                st.info("You can now select this patient in the sidebar to record sessions and track recovery.")


# ===========================================================================
# PAGE 3: MEDICAL REPORTS & DOCUMENTS
# ===========================================================================

def page_medical_reports(patient_id: str):
    """Upload and view diagnostic medical imaging, laboratory tests, and clinical documents."""
    render_topbar("Medical Records & Imaging", "Diagnostic Vault", patient_id)

    profile_svc = PatientProfileService()

    c_upload, c_list = st.columns([1, 1.4])

    with c_upload:
        st.markdown(
            """
            <div class="k-card">
                <div class="k-card-header">
                    <h3 class="k-card-title">&#128196; Upload Medical Document</h3>
                </div>
                <p style="color:#A3A3A3; font-size:0.85rem;">
                    Upload radiographs, MRI reports, or lab panels for patient record correlation.
                </p>
            """,
            unsafe_allow_html=True,
        )

        with st.form("upload_report_form"):
            uploaded_file = st.file_uploader(
                "Select File (PDF, PNG, JPG)",
                type=["pdf", "png", "jpg", "jpeg", "txt"],
                help="Maximum file size 25MB.",
            )
            report_type = st.selectbox(
                "Document Category",
                [
                    "Knee Radiograph (X-Ray)",
                    "MRI Imaging Report",
                    "Synovial Fluid / Lab Panel",
                    "Orthopedic Consultation Note",
                    "Physiotherapy Discharge Note",
                    "Prescription Record",
                ],
            )
            clinical_note = st.text_area(
                "Clinical Findings / Summary Notes",
                placeholder="e.g. Bilateral AP standing view shows medial joint space loss, subchondral sclerosis.",
            )
            submit_upload = st.form_submit_button("📤 Upload Document")

            if submit_upload and uploaded_file is not None:
                res = profile_svc.save_report(
                    patient_id=patient_id,
                    uploaded_file=uploaded_file,
                    report_type=report_type,
                    clinical_note=clinical_note,
                )
                if res:
                    st.success(f"File '{uploaded_file.name}' saved to patient vault!")
                    st.rerun()
                else:
                    st.error("Failed to save document.")

        st.markdown("</div>", unsafe_allow_html=True)

    with c_list:
        st.markdown(
            """
            <div class="k-card">
                <div class="k-card-header">
                    <h3 class="k-card-title">&#128451; Patient Document Repository</h3>
                </div>
            """,
            unsafe_allow_html=True,
        )

        reports = profile_svc.list_reports(patient_id)
        if reports:
            for rep in reports:
                f_size_kb = round(rep.get("size_bytes", 0) / 1024, 1)
                st.markdown(
                    f"""
                    <div style="background:#141414; border:1px solid #2D2D2D; border-radius:10px; padding:12px 16px; margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                            <strong style="color:#FF8A24; font-size:0.95rem;">{rep.get('report_type')}</strong>
                            <span class="k-pill k-pill-neutral">{f_size_kb} KB</span>
                        </div>
                        <div style="font-size:0.85rem; color:#FFFFFF; margin-bottom:4px;">
                            📄 {rep.get('original_name')}
                        </div>
                        <div style="font-size:0.8rem; color:#888;">
                            Uploaded: {rep.get('uploaded_at', '')[:10]}
                        </div>
                        <div style="font-size:0.85rem; color:#CCC; margin-top:6px; font-style:italic;">
                            "{rep.get('clinical_note', 'No clinical notes provided.')}"
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No medical documents uploaded for this patient yet.")

        st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================================
# PAGE 4: NUTRITION TRACKING
# ===========================================================================

def page_nutrition_tracking(patient_id: str):
    """Premium nutrition tracking: daily food diary, protein summation, and weekly trend graphs."""
    render_topbar("Nutrition & Protein Tracking", "Lifestyle & Dietary Monitoring", patient_id)

    log_svc = DailyLogService()
    today_str = datetime.now().strftime("%Y-%m-%d")
    current_log = log_svc.get_log(patient_id, today_str) or DailyLog(patient_id=patient_id, date=today_str)

    profile_svc = PatientProfileService()
    profile = profile_svc.get_profile(patient_id)
    target_goal = profile.get("daily_protein_goal_g", 65.0)

    total_prot = current_log.protein_intake_g
    pct_goal = int((total_prot / target_goal) * 100) if target_goal > 0 else 0

    # Headline Stats
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="k-kpi-card">
                <div class="k-kpi-label">Today's Protein Intake</div>
                <div class="k-kpi-value">{total_prot:.1f} <span class="k-kpi-unit">g</span></div>
                <div class="k-kpi-sub">Target: {target_goal:.0f}g • {pct_goal}% achieved</div>
                <div class="k-kpi-bar"><div class="k-kpi-bar-fill" style="width: min({pct_goal}%, 100%);"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        meals_logged = len(current_log.meals)
        st.markdown(
            f"""
            <div class="k-kpi-card">
                <div class="k-kpi-label">Meals / Items Logged</div>
                <div class="k-kpi-value">{meals_logged} <span class="k-kpi-unit">entries</span></div>
                <div class="k-kpi-sub">Nutritional muscle support tracking</div>
                <div class="k-kpi-bar"><div class="k-kpi-bar-fill" style="width: min({meals_logged * 25}%, 100%);"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        adherence_badge = "Target Met &#10003;" if total_prot >= target_goal else "In Progress"
        st.markdown(
            f"""
            <div class="k-kpi-card">
                <div class="k-kpi-label">Daily Goal Status</div>
                <div class="k-kpi-value" style="font-size:1.6rem;">{adherence_badge}</div>
                <div class="k-kpi-sub">Supports periarticular muscle mass</div>
                <div class="k-kpi-bar"><div class="k-kpi-bar-fill" style="width: min({pct_goal}%, 100%);"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    col_form, col_diary = st.columns([1, 1.4])

    with col_form:
        st.markdown(
            """
            <div class="k-card">
                <div class="k-card-header">
                    <h3 class="k-card-title">&#10133; Log Meal / Food Item</h3>
                </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Add Presets
        st.markdown("<span style='font-size:0.8rem; color:#888;'>Quick Add Common High-Protein Items:</span>", unsafe_allow_html=True)
        q_cols = st.columns(2)
        if q_cols[0].button("+ 2 Boiled Eggs (12g)", use_container_width=True):
            current_log.meals.append(MealEntry("2 Boiled Eggs", 12.0, "2 whole eggs").to_dict())
            current_log.protein_intake_g = calculate_daily_protein(current_log.meals)
            log_svc.save_log(current_log)
            st.rerun()
        if q_cols[1].button("+ Lentil Soup (15g)", use_container_width=True):
            current_log.meals.append(MealEntry("Lentil Soup / Dal", 15.0, "1 bowl (250ml)").to_dict())
            current_log.protein_intake_g = calculate_daily_protein(current_log.meals)
            log_svc.save_log(current_log)
            st.rerun()

        with st.form("custom_meal_form"):
            food_name = st.text_input("Food Item Name", placeholder="e.g. Grilled Chicken, Tofu, Greek Yogurt")
            quantity = st.text_input("Portion / Serving Size", value="1 serving")
            protein_g = st.number_input("Protein Content (grams)", min_value=0.0, max_value=200.0, value=20.0, step=1.0)
            user_target = st.number_input("Daily Protein Goal (grams)", min_value=30.0, max_value=250.0, value=float(target_goal), step=5.0)

            add_btn = st.form_submit_button("Add to Today's Diary")
            if add_btn and food_name:
                meal = MealEntry(food=food_name, protein_g=protein_g, quantity=quantity)
                current_log.meals.append(meal.to_dict())
                current_log.protein_intake_g = calculate_daily_protein(current_log.meals)
                current_log.protein_target_g = user_target
                log_svc.save_log(current_log)
                st.success(f"Added {food_name} (+{protein_g}g protein)!")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with col_diary:
        st.markdown(
            """
            <div class="k-card">
                <div class="k-card-header">
                    <h3 class="k-card-title">&#128221; Today's Food Diary</h3>
                    <span class="k-pill k-pill-orange">Auto-Summed</span>
                </div>
            """,
            unsafe_allow_html=True,
        )

        if current_log.meals:
            meal_df = pd.DataFrame(current_log.meals)
            st.dataframe(meal_df, use_container_width=True)

            if st.button("🗑️ Clear Today's Meals", key="clear_meals_btn"):
                current_log.meals = []
                current_log.protein_intake_g = 0.0
                log_svc.save_log(current_log)
                st.rerun()
        else:
            st.info("No meals logged for today yet. Use the form on the left or quick-add buttons.")

        st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================================
# PAGE 5: PHYSICAL ACTIVITY
# ===========================================================================

def page_activity_tracking(patient_id: str):
    """Track walking volume, step counts, duration, and personal activity baseline adherence."""
    render_topbar("Physical Activity & Mobility Volume", "Gait & Exercise Adherence", patient_id)

    log_svc = DailyLogService()
    today_str = datetime.now().strftime("%Y-%m-%d")
    current_log = log_svc.get_log(patient_id, today_str) or DailyLog(patient_id=patient_id, date=today_str)

    profile_svc = PatientProfileService()
    profile = profile_svc.get_profile(patient_id)
    step_target = profile.get("daily_step_goal", 5000) or 5000

    col_log, col_history = st.columns([1, 1.4])

    with col_log:
        st.markdown(
            """
            <div class="k-card">
                <div class="k-card-header">
                    <h3 class="k-card-title">&#128694; Log Daily Physical Activity</h3>
                </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("activity_entry_form"):
            steps = st.number_input("Step Count Today", min_value=0, max_value=50000, value=current_log.steps_count, step=500)
            source = st.selectbox("Measurement Source", ["Manual Entry", "Smartphone / HealthKit / Google Fit", "Insole Sensor Estimated"])
            duration_min = st.number_input("Total Walking Duration (Minutes)", min_value=0.0, max_value=360.0, value=float(current_log.walking_duration_minutes), step=5.0)

            save_btn = st.form_submit_button("Save Activity Log")
            if save_btn:
                current_log.steps_count = int(steps)
                current_log.steps_source = source
                current_log.walking_duration_minutes = float(duration_min)
                log_svc.save_log(current_log)
                st.success("Activity logged successfully!")
                st.rerun()

        st.markdown(
            """
            <div style="margin-top:14px; font-size:0.82rem; color:#888; line-height:1.45;">
                💡 <strong>Clinical Context:</strong> Maintaining a regular walking cadence without overloading the affected knee joint promotes cartilage nutrition and reduces morning stiffness.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_history:
        st.markdown(
            """
            <div class="k-card">
                <div class="k-card-header">
                    <h3 class="k-card-title">&#128202; 30-Day Step Count Progression</h3>
                    <span class="k-pill k-pill-orange">Baseline Adherence</span>
                </div>
            """,
            unsafe_allow_html=True,
        )

        history_df = log_svc.get_history_dataframe(patient_id, limit_days=30)
        if not history_df.empty and "steps_count" in history_df.columns:
            plot_df = history_df[["date", "steps_count"]].copy()
            plot_df["date"] = pd.to_datetime(plot_df["date"]).dt.strftime("%b %d")
            plot_df = plot_df.set_index("date")

            if _MATPLOTLIB_AVAILABLE:
                fig, ax = create_dark_figure(figsize=(8, 3.8))
                ax.bar(plot_df.index, plot_df["steps_count"], color="#FF6B00", width=0.6, edgecolor="#FF8A24")
                ax.axhline(step_target, color="#4ADE80", linestyle="--", linewidth=1.2, label=f"Goal ({step_target:,})")
                ax.set_ylabel("Steps", color="#9E9E9E", fontsize=9)
                ax.legend(facecolor="#202020", edgecolor="#333333", labelcolor="#FFFFFF", fontsize=8)
                plt.xticks(rotation=35, ha="right")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.bar_chart(plot_df)
        else:
            st.info("No 30-day activity logs found. Start logging your steps above.")

        st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================================
# PAGE 6: GAIT MONITORING (INSOLE SENSORS)
# ===========================================================================

def page_gait_monitoring(patient_id: str):
    """Wearable insole foot pressure and kinematic analysis with live meters and past trial playback."""
    render_topbar("Wearable Insole Gait Monitoring", "Sensor Biomechanics & Acquisition", patient_id)

    tab_record, tab_inspect = st.tabs(["⚡ Live Sensor Trial & Acquisition", "📈 Analyze Recorded Gait Sessions"])

    # -----------------------------------------------------------------------
    # TAB 1: Live Record / Simulation
    # -----------------------------------------------------------------------
    with tab_record:
        col_ctrl, col_meters = st.columns([1, 1.4])

        with col_ctrl:
            st.markdown(
                """
                <div class="k-card">
                    <div class="k-card-header">
                        <h3 class="k-card-title">&#9881; Acquisition Controls</h3>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            available_ports = list_serial_ports()
            mode = st.radio(
                "Acquisition Mode",
                ["Demonstration (Simulated Insole)", "Physical Hardware (ESP32 DevKit)"],
            )
            is_demo = mode.startswith("Demonstration")

            selected_port = None
            if not is_demo:
                if available_ports:
                    selected_port = st.selectbox("USB Serial Port", available_ports)
                else:
                    st.warning("No USB serial devices detected. Connect ESP32 or use Demo mode.")

            duration = st.slider("Walking Trial Duration (seconds)", min_value=3, max_value=60, value=10)
            sid = generate_session_id(patient_id)

            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            record_btn = st.button("🔴 Start Walking Session", use_container_width=True)

            st.markdown(
                """
                <div style="font-size:0.8rem; color:#888; margin-top:12px; line-height:1.4;">
                    <strong>Trial Protocol:</strong><br>
                    1. FSR sensors placed at Heel, Midfoot, and Forefoot.<br>
                    2. MPU6050 attached to shoe collar.<br>
                    3. Walk at normal cadence on a flat surface.<br>
                    4. Software records at ~50 Hz.
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_meters:
            st.markdown(
                """
                <div class="k-card">
                    <div class="k-card-header">
                        <h3 class="k-card-title">&#128225; Real-Time Sensor Telemetry (ESP32)</h3>
                        <span class="k-pill k-pill-orange">50 Hz Target</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            if record_btn:
                st.info(f"Acquiring gait data for **{patient_id}** ({duration}s)...")
                prog = st.progress(0.0)
                sm = SessionManager(
                    participant_id=patient_id,
                    session_id=sid,
                    port=selected_port,
                    demo=is_demo,
                    duration_seconds=duration,
                    display_readings=False,
                )
                res = sm.run(duration_seconds=duration)
                prog.progress(1.0)

                if res.is_successful:
                    st.success(f"Walking trial recorded: **{res.rows_written}** samples saved!")
                    proc = DataProcessor()
                    clean_path, rep = proc.process_and_save(res.csv_path)
                    ext = FeatureExtractor()
                    feat_path, feats = ext.process_and_save(clean_path)
                    st.success(f"Cleaned & extracted gait features: `{clean_path.name}` ({rep.estimated_sampling_hz:.1f} Hz).")
                else:
                    st.error(f"Recording failed: {res.error_message}")
            else:
                # Visual live sensor meters
                st.markdown(
                    """
                    <div class="k-sensor-box">
                        <div class="k-sensor-header"><span>HEEL PRESSURE (FSR 1 • GPIO 34)</span><span class="k-sensor-val">2,140 / 4,095</span></div>
                        <div class="k-sensor-track"><div class="k-sensor-fill" style="width: 52%;"></div></div>
                    </div>
                    <div class="k-sensor-box">
                        <div class="k-sensor-header"><span>MIDFOOT PRESSURE (FSR 2 • GPIO 35)</span><span class="k-sensor-val">1,480 / 4,095</span></div>
                        <div class="k-sensor-track"><div class="k-sensor-fill" style="width: 36%;"></div></div>
                    </div>
                    <div class="k-sensor-box">
                        <div class="k-sensor-header"><span>FOREFOOT PRESSURE (FSR 3 • GPIO 32)</span><span class="k-sensor-val">3,250 / 4,095</span></div>
                        <div class="k-sensor-track"><div class="k-sensor-fill" style="width: 79%;"></div></div>
                    </div>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:12px;">
                        <div style="background:#141414; padding:10px; border-radius:8px; border:1px solid #282828;">
                            <div style="color:#888; font-size:0.75rem;">ACCELEROMETER (MPU6050)</div>
                            <strong style="color:#FFF; font-size:0.9rem;">X: 0.12g | Y: 0.25g | Z: 0.98g</strong>
                        </div>
                        <div style="background:#141414; padding:10px; border-radius:8px; border:1px solid #282828;">
                            <div style="color:#888; font-size:0.75rem;">GYROSCOPE (MPU6050)</div>
                            <strong style="color:#FFF; font-size:0.9rem;">Roll: 0.3°/s | Pitch: 0.1°/s | Yaw: 0.4°/s</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # TAB 2: Inspect Past Walking Trials
    # -----------------------------------------------------------------------
    with tab_inspect:
        patient_files = sorted(
            list(config.DATA_RAW_DIR.glob(f"{patient_id}_*.csv"))
            + list(config.DATA_DEMO_DIR.glob(f"{patient_id}_*.csv")),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        if not patient_files:
            st.info(f"No walking trials saved for {patient_id} yet.")
            return

        sel_file = st.selectbox("Select Walking Session File", patient_files, format_func=lambda x: x.name)
        if not sel_file:
            return

        df = pd.read_csv(sel_file)
        time_s = (
            (df["timestamp_ms"] - df["timestamp_ms"].iloc[0]) / 1000.0
            if "timestamp_ms" in df.columns
            else np.arange(len(df))
        )

        st.markdown(
            """
            <div class="k-card">
                <div class="k-card-header">
                    <h3 class="k-card-title">&#128099; Stance Foot Pressure Loading Profiles (ADC Counts)</h3>
                    <span class="k-pill k-pill-orange">Heel -> Midfoot -> Forefoot</span>
                </div>
            """,
            unsafe_allow_html=True,
        )

        if all(c in df.columns for c in ("heel", "midfoot", "forefoot")):
            if _MATPLOTLIB_AVAILABLE:
                fig, ax = create_dark_figure(figsize=(9, 3.8))
                ax.plot(time_s[:400], df["heel"][:400], color="#FF6B00", label="Heel Sensor", linewidth=1.8)
                ax.plot(time_s[:400], df["midfoot"][:400], color="#FF8A24", label="Midfoot Sensor", linewidth=1.5, linestyle="--")
                ax.plot(time_s[:400], df["forefoot"][:400], color="#FFFFFF", label="Forefoot Push-Off", linewidth=1.6)
                ax.set_xlabel("Time (seconds)", color="#9E9E9E", fontsize=9)
                ax.set_ylabel("Sensor Counts (ADC)", color="#9E9E9E", fontsize=9)
                ax.legend(facecolor="#202020", edgecolor="#333333", labelcolor="#FFFFFF", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.line_chart(df[["heel", "midfoot", "forefoot"]].iloc[:400])

        st.markdown("</div>", unsafe_allow_html=True)

        # Feature Summary Table & CSV Export
        c_feat, c_raw = st.columns([1.2, 1])
        with c_feat:
            st.markdown(
                """
                <div class="k-card">
                    <div class="k-card-header">
                        <h3 class="k-card-title">&#128203; Extracted Biomechanical Features</h3>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            clean_name = sel_file.stem + "_clean_features.json"
            feat_p = config.DATA_PROCESSED_DIR / clean_name
            if feat_p.exists():
                with open(feat_p, "r", encoding="utf-8") as f:
                    feats = json.load(f)
            else:
                ext = FeatureExtractor()
                feats = ext.extract_features(df)

            f_items = [{"Metric": k, "Value": v} for k, v in feats.items() if isinstance(v, (int, float, str))]
            st.dataframe(pd.DataFrame(f_items), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c_raw:
            st.markdown(
                """
                <div class="k-card">
                    <div class="k-card-header">
                        <h3 class="k-card-title">&#128196; Export Trial Data</h3>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Raw Session CSV",
                data=csv_bytes,
                file_name=sel_file.name,
                mime="text/csv",
                use_container_width=True,
            )
            st.dataframe(df.head(25), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================================
# PAGE 7: SYMPTOM JOURNAL
# ===========================================================================

def page_symptom_journal(patient_id: str):
    """Log knee pain, morning stiffness, joint swelling, and therapeutic exercise adherence."""
    render_topbar("Daily Knee Symptom Journal", "Self-Reported Outcomes", patient_id)

    log_svc = DailyLogService()
    today_str = datetime.now().strftime("%Y-%m-%d")
    current_log = log_svc.get_log(patient_id, today_str) or DailyLog(patient_id=patient_id, date=today_str)

    col_entry, col_history = st.columns([1, 1.3])

    with col_entry:
        st.markdown(
            """
            <div class="k-card">
                <div class="k-card-header">
                    <h3 class="k-card-title">&#128221; Today's Symptoms & Joint Feedback</h3>
                </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("symptom_entry_form"):
            pain = st.slider("Knee Pain Score (0 = No Pain, 10 = Severe Pain)", min_value=0, max_value=10, value=current_log.pain_score)
            stiffness = st.number_input("Morning Stiffness Duration (Minutes)", min_value=0, max_value=240, value=current_log.stiffness_minutes, step=5)
            fatigue = st.slider("Fatigue / Energy Level (0 = Energetic, 10 = Exhausted)", min_value=0, max_value=10, value=current_log.fatigue_score)
            swelling = st.checkbox("Joint Swelling Observed Today", value=current_log.swelling_reported)
            exercise = st.checkbox("Completed Prescribed Knee Exercises", value=current_log.exercise_completed)
            notes = st.text_area("Personal Notes / Trigger Events", value=current_log.notes, placeholder="e.g. Mild pain after climbing stairs, stiffness eased after gentle walk")

            save_sym_btn = st.form_submit_button("Save Daily Symptoms")
            if save_sym_btn:
                current_log.pain_score = int(pain)
                current_log.stiffness_minutes = int(stiffness)
                current_log.fatigue_score = int(fatigue)
                current_log.swelling_reported = bool(swelling)
                current_log.exercise_completed = bool(exercise)
                current_log.notes = notes
                log_svc.save_log(current_log)
                st.success("Symptom log saved successfully!")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with col_history:
        st.markdown(
            """
            <div class="k-card">
                <div class="k-card-header">
                    <h3 class="k-card-title">&#128200; Pain & Stiffness Progression Over Time</h3>
                    <span class="k-pill k-pill-orange">30-Day Trend</span>
                </div>
            """,
            unsafe_allow_html=True,
        )

        history_df = log_svc.get_history_dataframe(patient_id, limit_days=30)
        if not history_df.empty and "pain_score" in history_df.columns:
            plot_df = history_df[["date", "pain_score", "stiffness_minutes"]].copy()
            plot_df["date"] = pd.to_datetime(plot_df["date"]).dt.strftime("%b %d")
            plot_df = plot_df.set_index("date")

            if _MATPLOTLIB_AVAILABLE:
                fig, ax1 = create_dark_figure(figsize=(8, 3.8))
                ax2 = ax1.twinx()

                ax1.plot(plot_df.index, plot_df["pain_score"], color="#FF6B00", marker="o", linewidth=2.0, label="Pain Score (0-10)")
                ax2.plot(plot_df.index, plot_df["stiffness_minutes"], color="#4ADE80", marker="s", linewidth=1.5, linestyle="--", label="Stiffness (min)")

                ax1.set_ylabel("Pain (0-10)", color="#FF6B00", fontsize=9)
                ax2.set_ylabel("Stiffness (Minutes)", color="#4ADE80", fontsize=9)
                ax1.tick_params(colors="#9E9E9E", labelsize=8)
                ax2.tick_params(colors="#9E9E9E", labelsize=8)
                plt.xticks(rotation=35, ha="right")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.line_chart(plot_df)
        else:
            st.info("Log your daily symptoms to generate a recovery trajectory.")

        st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================================
# PAGE 8: AI INSIGHTS & ANOMALY REPORTING
# ===========================================================================

def page_ai_insights(patient_id: str):
    """Personalized baseline anomaly evaluation, Gemini 2.5 Flash summaries, and physician talking points."""
    render_topbar("AI Patient Insights & Clinical Summaries", "Intelligent Assistance Layer", patient_id)

    st.markdown(
        f"""
        <div class="k-disclaimer">
            <strong>&#9888; MANDATORY MEDICAL NOTICE:</strong><br>
            {MEDICAL_DISCLAIMER_TEXT}
        </div>
        """,
        unsafe_allow_html=True,
    )

    coord = PatientMonitoringCoordinator()
    today_str = datetime.now().strftime("%Y-%m-%d")

    with st.spinner("Analyzing personal baseline and recent observations..."):
        eval_res = coord.run_daily_monitoring(patient_id, today_date_str=today_str, upload_to_firebase=True)

    insight = eval_res.get("ai_insight", {})
    anomaly = eval_res.get("anomaly_report", {})
    status_category = anomaly.get("overall_category", "Within personal baseline")

    if status_category == "Change to monitor":
        st.markdown(
            f"""
            <div class="k-alert-monitor">
                <h4 style="margin:0 0 6px 0; color:#F87171;">&#9888; Monitoring Status: Change to Monitor</h4>
                <p style="margin:0; font-size:0.9rem; color:#DDD;">{anomaly.get('summary_message', '')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="k-alert-baseline">
                <h4 style="margin:0 0 6px 0; color:#4ADE80;">&#10003; Monitoring Status: Within Personal Baseline</h4>
                <p style="margin:0; font-size:0.9rem; color:#DDD;">{anomaly.get('summary_message', 'Today’s measurements align with your established baseline.')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_gemini, col_deviations = st.columns([1.2, 1])

    with col_gemini:
        st.markdown(
            f"""
            <div class="k-card">
                <div class="k-card-header">
                    <h3 class="k-card-title">&#129302; AI Clinical Assistant Explanation</h3>
                    <span class="k-pill k-pill-orange">{insight.get('generator_source', 'Gemini AI')}</span>
                </div>
                <div style="font-size:0.92rem; color:#EEE; line-height:1.6; margin-bottom:14px;">
                    {insight.get('daily_summary', 'All daily gait and lifestyle observations have been compiled successfully.')}
                </div>
                <div style="font-size:0.85rem; color:#CCC; line-height:1.55;">
                    <div>• <strong>Physical Activity:</strong> {insight.get('activity_analysis', 'Stable volume.')}</div>
                    <div>• <strong>Nutrition & Protein:</strong> {insight.get('nutrition_analysis', 'Adequate intake.')}</div>
                    <div>• <strong>Gait Stability:</strong> {insight.get('gait_analysis', 'Smooth stance progression.')}</div>
                    <div>• <strong>Symptom Dynamics:</strong> {insight.get('symptom_analysis', 'No acute flares reported.')}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="k-card">
                <div class="k-card-header">
                    <h3 class="k-card-title">&#128172; Suggested Doctor Discussion Topics</h3>
                </div>
                <div style="font-size:0.88rem; color:#FF8A24; line-height:1.5;">
                    💡 {insight.get('suggested_follow_up', 'Review stair descent comfort and quad strength with your physiotherapist.')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_deviations:
        st.markdown(
            """
            <div class="k-card">
                <div class="k-card-header">
                    <h3 class="k-card-title">&#128202; Baseline Deviation Signals</h3>
                </div>
            """,
            unsafe_allow_html=True,
        )

        deviations = anomaly.get("deviations_detected", [])
        if deviations:
            dev_records = [
                {
                    "Metric": d.get("metric", "").replace("_", " ").title(),
                    "Today": d.get("current_value"),
                    "Median": d.get("baseline_median"),
                    "Deviation": f"{d.get('deviation_percent', 0):+.1f}%",
                    "Status": "Flagged" if d.get("status") == "change_to_monitor" else "Normal",
                }
                for d in deviations
            ]
            st.dataframe(pd.DataFrame(dev_records), use_container_width=True)
        else:
            st.info("No statistically significant deviations detected today.")

        st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================================
# PAGE 9: SYSTEM SETTINGS
# ===========================================================================

def page_settings(patient_id: str):
    """System diagnostics, hardware configuration, and Firebase / Gemini API management."""
    render_topbar("System Settings & Integration", "Configuration & Health", patient_id)

    c_hw, c_cloud = st.columns(2)

    with c_hw:
        st.markdown(
            """
            <div class="k-card">
                <div class="k-card-header">
                    <h3 class="k-card-title">&#9881; Hardware & Serial Setup</h3>
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(f"- **Configured Port:** `{config.SERIAL_PORT or 'Auto-Detect'}`")
        st.write(f"- **Baud Rate:** `{config.SERIAL_BAUD_RATE}`")
        st.write(f"- **Demo Mode Default:** `{config.DEMO_MODE}`")
        st.write(f"- **Target Sampling Rate:** `~50 Hz`")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_cloud:
        st.markdown(
            """
            <div class="k-card">
                <div class="k-card-header">
                    <h3 class="k-card-title">&#9729; Cloud & AI Integration</h3>
                </div>
            """,
            unsafe_allow_html=True,
        )
        fb = get_firebase_service()
        st.write(f"- **Firestore Status:** `{'Connected' if fb.is_connected else 'Offline / Local Storage Mode'}`")
        st.write(f"- **Credentials Path:** `{config.FIREBASE_CREDENTIALS_PATH}`")
        st.write(f"- **Gemini API Key:** `{'Configured (Active)' if config.GEMINI_API_KEY else 'Not Set'}`")
        st.write(f"- **Gemini Model:** `{config.GEMINI_MODEL_NAME}`")
        st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================================
# MAIN DISPATCHER & ALIASES
# ===========================================================================

def main():
    """Main Streamlit application entrypoint."""
    setup_page()
    if os.getenv("PUBLIC_DEPLOYMENT", "false").lower() in ("true", "1", "yes"):
        access_password = os.getenv("PUBLIC_APP_PASSWORD", "")
        if not access_password:
            st.error("This deployment is locked until its owner configures PUBLIC_APP_PASSWORD in hosting secrets.")
            st.stop()
        if not st.session_state.get("_public_app_authenticated", False):
            st.title("KINETIX OA — Private Access")
            with st.form("public_app_access"):
                entered_password = st.text_input("Access password", type="password")
                submitted = st.form_submit_button("Continue")
            if submitted:
                if hmac.compare_digest(entered_password, access_password):
                    st.session_state["_public_app_authenticated"] = True
                    st.rerun()
                st.error("That password did not match.")
            st.stop()

    page, active_patient = render_sidebar()

    if page == "1. Dashboard Overview":
        page_patient_overview(active_patient)
    elif page == "2. Patient Profile":
        page_patient_profile(active_patient)
    elif page == "3. Medical Reports":
        page_medical_reports(active_patient)
    elif page == "4. Nutrition Tracking":
        page_nutrition_tracking(active_patient)
    elif page == "5. Physical Activity":
        page_activity_tracking(active_patient)
    elif page == "6. Gait Monitoring":
        page_gait_monitoring(active_patient)
    elif page == "7. Symptom Journal":
        page_symptom_journal(active_patient)
    elif page == "8. AI Insights":
        page_ai_insights(active_patient)
    elif page == "9. Settings":
        page_settings(active_patient)


# ---------------------------------------------------------------------------
# Aliases preserved for strict backward compatibility with existing tests
# ---------------------------------------------------------------------------
page_overview = page_patient_overview
page_participants = page_patient_profile
page_data_acquisition = page_gait_monitoring
page_walking_analysis = page_gait_monitoring
page_assessment_report = page_ai_insights


if __name__ == "__main__":
    main()
