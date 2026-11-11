"""
ui_theme.py — Orange & Black Fitness-Tech Design System for Streamlit.

Visual Identity:
- Deep black background:   #101010
- Secondary black / bars:  #161616
- Card surfaces:           #202020
- High-contrast surface:   #262626
- Subtle borders:          #303030
- Primary vibrant orange:  #FF6B00
- Secondary warm orange:   #FF8A24
- Text primary on dark:    #FFFFFF
- Text secondary / muted:  #9E9E9E
- Positive / success:      #22C55E
- Alert / warning:         #EF4444
"""

from typing import Optional, Tuple
import matplotlib.pyplot as plt

THEME_CSS = """
<style>
/* ==========================================================================
   ORANGE & BLACK DESIGN SYSTEM — GLOBAL OVERRIDES
   ========================================================================== */

/* Main canvas background */
.stApp {
    background-color: #101010 !important;
    color: #FFFFFF !important;
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #141414 !important;
    border-right: 1px solid #262626 !important;
}

[data-testid="stSidebar"] hr {
    border-color: #262626 !important;
    margin: 12px 0 !important;
}

/* Header container */
header[data-testid="stHeader"] {
    background-color: rgba(16, 16, 16, 0.85) !important;
    backdrop-filter: blur(8px);
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

p, span, label {
    color: #E0E0E0;
}

/* Streamlit Buttons — Primary Orange */
.stButton > button {
    background: linear-gradient(135deg, #FF6B00 0%, #FF8A24 100%) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.25rem !important;
    box-shadow: 0 2px 8px rgba(255, 107, 0, 0.25) !important;
    transition: all 0.2s ease-in-out !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #FF8A24 0%, #FFA04D 100%) !important;
    box-shadow: 0 4px 14px rgba(255, 107, 0, 0.4) !important;
    transform: translateY(-1px);
}

.stButton > button:active {
    transform: translateY(0);
}

/* Secondary Button variant */
.stButton > button[kind="secondary"] {
    background: #242424 !important;
    color: #E0E0E0 !important;
    border: 1px solid #363636 !important;
    box-shadow: none !important;
}

.stButton > button[kind="secondary"]:hover {
    background: #2A2A2A !important;
    border-color: #FF6B00 !important;
    color: #FFFFFF !important;
}

/* Form Inputs, Selectboxes, Text Areas */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div {
    background-color: #1A1A1A !important;
    color: #FFFFFF !important;
    border: 1px solid #303030 !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #FF6B00 !important;
    box-shadow: 0 0 0 2px rgba(255, 107, 0, 0.2) !important;
}

/* Sliders */
.stSlider > div > div > div > div {
    background-color: #FF6B00 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #141414;
    padding: 6px;
    border-radius: 10px;
    border: 1px solid #262626;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    color: #9E9E9E !important;
    font-weight: 500;
    padding: 8px 16px;
    background-color: transparent;
    border: none !important;
}

.stTabs [aria-selected="true"] {
    background-color: #242424 !important;
    color: #FF6B00 !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
}

/* Radio button pills in sidebar */
[data-testid="stSidebar"] .stRadio > div {
    gap: 4px;
}

[data-testid="stSidebar"] .stRadio label {
    padding: 8px 12px;
    border-radius: 8px;
    transition: all 0.15s ease;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background-color: rgba(255, 107, 0, 0.08);
}

/* Dataframe & Tables */
[data-testid="stDataFrame"] {
    background-color: #1A1A1A;
    border: 1px solid #2D2D2D;
    border-radius: 8px;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}

[data-testid="stMetricLabel"] {
    color: #9E9E9E !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ==========================================================================
   CUSTOM CARD & VISUAL COMPONENT CLASSES
   ========================================================================== */

/* Brand Header */
.k-brand-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 4px 0 16px 0;
}

.k-brand-icon {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: linear-gradient(135deg, #FF6B00 0%, #FF8A24 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #FFFFFF;
    font-weight: 900;
    font-size: 1.1rem;
    box-shadow: 0 2px 8px rgba(255, 107, 0, 0.35);
}

.k-brand-title {
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #FFFFFF;
    margin: 0;
    line-height: 1.1;
}

.k-brand-sub {
    font-size: 0.72rem;
    color: #9E9E9E;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 2px 0 0 0;
}

/* Top App Bar */
.k-topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #171717;
    border: 1px solid #262626;
    border-radius: 12px;
    padding: 12px 18px;
    margin-bottom: 22px;
}

.k-topbar-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.k-topbar-patient {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #202020;
    border: 1px solid #333333;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
}

.k-patient-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #FF6B00;
    box-shadow: 0 0 6px #FF6B00;
}

/* Standard Content Card */
.k-card {
    background-color: #1A1A1A;
    border: 1px solid #282828;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 18px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    transition: border-color 0.2s ease;
}

.k-card:hover {
    border-color: #383838;
}

.k-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #262626;
}

.k-card-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.k-card-subtitle {
    font-size: 0.8rem;
    color: #8C8C8C;
    margin: 0;
}

/* Metric KPI Card (Orange Accents) */
.k-kpi-card {
    background: linear-gradient(180deg, #1E1E1E 0%, #161616 100%);
    border: 1px solid #2E2E2E;
    border-radius: 14px;
    padding: 18px;
    height: 100%;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.25);
    position: relative;
    overflow: hidden;
}

.k-kpi-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #FF6B00 0%, #FF8A24 100%);
}

.k-kpi-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #9E9E9E;
    font-weight: 600;
    margin-bottom: 6px;
}

.k-kpi-value {
    font-size: 2.1rem;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.1;
    margin-bottom: 6px;
    letter-spacing: -0.03em;
}

.k-kpi-unit {
    font-size: 1.0rem;
    font-weight: 500;
    color: #FF8A24;
    margin-left: 2px;
}

.k-kpi-sub {
    font-size: 0.8rem;
    color: #7A7A7A;
    display: flex;
    align-items: center;
    gap: 6px;
}

.k-kpi-bar {
    width: 100%;
    height: 6px;
    background-color: #282828;
    border-radius: 4px;
    margin-top: 10px;
    overflow: hidden;
}

.k-kpi-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #FF6B00 0%, #FF8A24 100%);
    border-radius: 4px;
    transition: width 0.4s ease;
}

/* Status Pill Badges */
.k-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.k-pill-orange {
    background: rgba(255, 107, 0, 0.15);
    color: #FF8A24;
    border: 1px solid rgba(255, 107, 0, 0.35);
}

.k-pill-green {
    background: rgba(34, 197, 94, 0.14);
    color: #4ADE80;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.k-pill-red {
    background: rgba(239, 68, 68, 0.15);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.35);
}

.k-pill-neutral {
    background: #242424;
    color: #A3A3A3;
    border: 1px solid #333333;
}

/* Clinical Medical Disclaimer Banner */
.k-disclaimer {
    background: #1A1612;
    border-left: 4px solid #FF6B00;
    border-top: 1px solid #33261C;
    border-right: 1px solid #33261C;
    border-bottom: 1px solid #33261C;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 16px;
    font-size: 0.82rem;
    color: #D1C5B8;
    line-height: 1.45;
}

.k-disclaimer strong {
    color: #FF8A24;
}

/* Anomaly / Alert Box */
.k-alert-monitor {
    background: #221414;
    border: 1px solid #4D1F1F;
    border-left: 4px solid #EF4444;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
}

.k-alert-baseline {
    background: #131E14;
    border: 1px solid #1C3820;
    border-left: 4px solid #22C55E;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
}

/* Sensor Visual Meter */
.k-sensor-box {
    background: #181818;
    border: 1px solid #292929;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 10px;
}

.k-sensor-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    font-weight: 600;
    color: #E0E0E0;
    margin-bottom: 6px;
}

.k-sensor-val {
    color: #FF8A24;
    font-weight: 700;
}

.k-sensor-track {
    width: 100%;
    height: 8px;
    background: #262626;
    border-radius: 4px;
    overflow: hidden;
}

.k-sensor-fill {
    height: 100%;
    background: linear-gradient(90deg, #FF6B00 0%, #FF8A24 100%);
    border-radius: 4px;
}

/* Light, spacious dashboard inspired by the supplied workspace references.
   :has keeps the other clinical modules on their existing theme. */
.stApp:has(.k-dashboard-shell) {
    background: #F3F2F9 !important;
    color: #202334 !important;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.stApp:has(.k-dashboard-shell) .k-dashboard-shell { display: none; }
.stApp:has(.k-dashboard-shell) [data-testid="stAppViewContainer"] { background: #F3F2F9 !important; }
.stApp:has(.k-dashboard-shell) [data-testid="stMain"] { background: transparent !important; }
.stApp:has(.k-dashboard-shell) [data-testid="stMainBlockContainer"] {
    max-width: 1450px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}
.stApp:has(.k-dashboard-shell) [data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E7E8EF !important;
    box-shadow: 4px 0 22px rgba(31, 35, 54, .035);
}
.stApp:has(.k-dashboard-shell) [data-testid="stHeader"] {
    background: rgba(244, 245, 250, .92) !important;
}
.stApp:has(.k-dashboard-shell) h1,
.stApp:has(.k-dashboard-shell) h2,
.stApp:has(.k-dashboard-shell) h3,
.stApp:has(.k-dashboard-shell) h4,
.stApp:has(.k-dashboard-shell) p,
.stApp:has(.k-dashboard-shell) label,
.stApp:has(.k-dashboard-shell) [data-testid="stMarkdownContainer"] {
    color: #25283A;
}
.stApp:has(.k-dashboard-shell) [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: #5F6475; }
.stApp:has(.k-dashboard-shell) .k-brand-icon {
    background: #17152B;
    box-shadow: none;
}
.stApp:has(.k-dashboard-shell) .k-brand-title { color: #202238 !important; }
.stApp:has(.k-dashboard-shell) .k-brand-sub { color: #8B8FA0; }
.stApp:has(.k-dashboard-shell) .k-disclaimer {
    background: #F7F6FF;
    border: 1px solid #E9E5FF;
    border-left: 3px solid #7969DB;
    color: #66627B;
}
.stApp:has(.k-dashboard-shell) .k-disclaimer strong { color: #6254C4; }
.stApp:has(.k-dashboard-shell) [data-testid="stSidebar"] hr { border-color: #ECECF2 !important; }
.stApp:has(.k-dashboard-shell) [data-testid="stSidebar"] .stRadio label { color: #555A6C !important; }
.stApp:has(.k-dashboard-shell) [data-testid="stSidebar"] .stRadio label:hover { background: #F3F1FF; }
.stApp:has(.k-dashboard-shell) [data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: #F0EEFF;
    color: #5144B7 !important;
}
.stApp:has(.k-dashboard-shell) [data-testid="stSidebar"] input,
.stApp:has(.k-dashboard-shell) [data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #FFFFFF !important;
    color: #34384A !important;
    border-color: #E5E7EF !important;
}
.stApp:has(.k-dashboard-shell) .k-topbar {
    background: transparent;
    border: 0;
    border-radius: 0;
    border-bottom: 1px solid #E5E7EF;
    padding: 4px 2px 18px;
    margin-bottom: 20px;
}
.stApp:has(.k-dashboard-shell) .k-topbar-title { color: #202238 !important; font-size: 1.65rem; }
.stApp:has(.k-dashboard-shell) .k-topbar > div:first-child > div { color: #7162D2 !important; }
.stApp:has(.k-dashboard-shell) .k-topbar-patient strong { color: #292C3E !important; }
.stApp:has(.k-dashboard-shell) .k-topbar-patient span { color: #73788A !important; }
.stApp:has(.k-dashboard-shell) .k-topbar-patient {
    background: #FFFFFF;
    border-color: #E8E9F0;
    color: #5C6070;
    box-shadow: 0 2px 8px rgba(31, 35, 54, .035);
}
.stApp:has(.k-dashboard-shell) .k-patient-dot { background: #35A980; box-shadow: 0 0 0 3px #E1F5EC; }
.stApp:has(.k-dashboard-shell) .k-kpi-card,
.stApp:has(.k-dashboard-shell) .k-card {
    background: #FFFFFF;
    border: 1px solid #E8E9F0;
    border-radius: 13px;
    box-shadow: 0 3px 14px rgba(31, 35, 54, .035);
}
.stApp:has(.k-dashboard-shell) .k-kpi-card { padding: 18px 20px 16px; min-height: 145px; }
.stApp:has(.k-dashboard-shell) .k-kpi-card::before { height: 3px; background: var(--k-accent, #7969DB); }
.stApp:has(.k-dashboard-shell) .k-kpi-heading { display:flex; justify-content:space-between; align-items:center; }
.stApp:has(.k-dashboard-shell) .k-kpi-mark { width:9px; height:9px; background:var(--k-accent, #7969DB); border-radius:50%; box-shadow:0 0 0 4px color-mix(in srgb, var(--k-accent, #7969DB) 13%, white); }
.stApp:has(.k-dashboard-shell) .k-kpi-label { color: #777C8D; letter-spacing: .035em; }
.stApp:has(.k-dashboard-shell) .k-kpi-value { color: #24273A; font-size: 1.9rem; }
.stApp:has(.k-dashboard-shell) .k-kpi-unit { color: #7770CA; }
.stApp:has(.k-dashboard-shell) .k-kpi-sub { color: #85899A; min-height:1.1rem; }
.stApp:has(.k-dashboard-shell) .k-kpi-bar { background: #F0F0F5; height:5px; margin-top:13px; }
.stApp:has(.k-dashboard-shell) .k-kpi-bar-fill { background: var(--k-accent, #7969DB); }
.stApp:has(.k-dashboard-shell) .k-page-intro { margin: -8px 0 18px; }
.stApp:has(.k-dashboard-shell) .k-breadcrumb { color:#777C8D; font-size:.8rem; font-weight:600; letter-spacing:.02em; }
.stApp:has(.k-dashboard-shell) .k-breadcrumb span { color:#B0B3BF; padding:0 7px; }
.stApp:has(.k-dashboard-shell) .k-page-intro p { color:#85899A; font-size:.91rem; margin:.28rem 0 0; }
.stApp:has(.k-dashboard-shell) .k-dashboard-alert { display:flex; gap:12px; align-items:center; padding:12px 15px; margin:0 0 17px; border:1px solid #DDEFE5; border-left:3px solid #39A77E; border-radius:11px; background:#F2FAF6; }
.stApp:has(.k-dashboard-shell) .k-dashboard-alert strong { display:block; color:#2C795F !important; font-size:.88rem; margin-bottom:2px; }
.stApp:has(.k-dashboard-shell) .k-dashboard-alert p { color:#667D73 !important; font-size:.82rem; margin:0; line-height:1.4; }
.stApp:has(.k-dashboard-shell) .k-dashboard-alert > .k-pill { margin-left:auto; flex-shrink:0; }
.stApp:has(.k-dashboard-shell) .k-alert-icon { width:23px; height:23px; display:grid; place-items:center; border-radius:50%; background:#DFF2E8; color:#27825F; font-weight:800; flex:none; }
.stApp:has(.k-dashboard-shell) .k-alert-monitor.k-dashboard-alert { background:#FFF8F6; border-color:#F3DDD7; border-left-color:#D97862; }
.stApp:has(.k-dashboard-shell) .k-alert-monitor.k-dashboard-alert strong { color:#A65342 !important; }
.stApp:has(.k-dashboard-shell) .k-alert-progress.k-dashboard-alert { background:#F7F6FF; border-color:#E9E5FF; border-left-color:#7969DB; }
.stApp:has(.k-dashboard-shell) .k-alert-progress.k-dashboard-alert strong { color:#5D50BD !important; }
.stApp:has(.k-dashboard-shell) .k-section-heading { display:flex; align-items:center; justify-content:space-between; min-height:45px; margin:13px 0 8px; }
.stApp:has(.k-dashboard-shell) .k-section-heading h3 { color:#272A3D !important; font-size:1rem !important; margin:0; letter-spacing:-.01em; }
.stApp:has(.k-dashboard-shell) .k-section-heading p { color:#85899A !important; font-size:.78rem; margin:3px 0 0; }
.stApp:has(.k-dashboard-shell) .k-range-chip { display:inline-flex; align-items:center; padding:6px 10px; border:1px solid #E7E8EF; border-radius:7px; background:#fff; color:#666B7C; font-size:.74rem; white-space:nowrap; }
.stApp:has(.k-dashboard-shell) .k-chart-card { background:#fff; border:1px solid #E8E9F0; border-radius:13px; padding:12px 13px 7px; box-shadow:0 3px 14px rgba(31,35,54,.035); }
.stApp:has(.k-dashboard-shell) [data-testid="stVerticalBlockBorderWrapper"] { border-color:#E8E9F0 !important; border-radius:13px !important; background:#FFFFFF !important; box-shadow:0 3px 14px rgba(31,35,54,.035); padding:8px 12px !important; }
.stApp:has(.k-dashboard-shell) .st-key-steps_trend_card,
.stApp:has(.k-dashboard-shell) .st-key-nutrition_trend_card { border:0 !important; background:transparent !important; padding:0 !important; }
.stApp:has(.k-dashboard-shell) .k-chart-card [data-testid="stImage"] img { border-radius:7px; }
.stApp:has(.k-dashboard-shell) .k-profile-card,
.stApp:has(.k-dashboard-shell) .k-care-card { background:#fff; border:1px solid #E8E9F0; border-radius:13px; padding:17px; box-shadow:0 3px 14px rgba(31,35,54,.035); }
.stApp:has(.k-dashboard-shell) .k-profile-card-top { display:flex; align-items:center; gap:10px; padding-bottom:14px; border-bottom:1px solid #EFF0F4; }
.stApp:has(.k-dashboard-shell) .k-profile-card-top > div:nth-child(2) { min-width:0; flex:1; }
.stApp:has(.k-dashboard-shell) .k-profile-card-top strong { display:block; color:#282B3E; font-size:.89rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.stApp:has(.k-dashboard-shell) .k-profile-card-top span:not(.k-pill) { display:block; color:#888D9D; font-size:.75rem; margin-top:2px; }
.stApp:has(.k-dashboard-shell) .k-avatar { width:35px; height:35px; display:grid; place-items:center; border-radius:50%; background:#F0EEFF; color:#5D50BD; font-size:.83rem; font-weight:800; flex:none; }
.stApp:has(.k-dashboard-shell) .k-profile-grid { display:grid; grid-template-columns:1fr 1fr; gap:14px 10px; padding:15px 0 12px; }
.stApp:has(.k-dashboard-shell) .k-profile-grid span { display:block; color:#85899A; font-size:.72rem; margin-bottom:4px; }
.stApp:has(.k-dashboard-shell) .k-profile-grid strong { color:#303347; font-size:.84rem; }
.stApp:has(.k-dashboard-shell) .k-profile-note { color:#73788A; background:#F7F7FA; border-radius:8px; padding:9px 10px; font-size:.76rem; line-height:1.4; }
.stApp:has(.k-dashboard-shell) .k-gait-status { display:flex; align-items:center; gap:10px; padding:13px 2px 4px; }
.stApp:has(.k-dashboard-shell) .k-gait-status-dot { width:8px; height:8px; background:#BFC3CF; border-radius:50%; }
.stApp:has(.k-dashboard-shell) .k-gait-status > div { flex:1; }
.stApp:has(.k-dashboard-shell) .k-gait-status strong,
.stApp:has(.k-dashboard-shell) .k-gait-status span:not(.k-gait-status-dot):not(.k-gait-frequency) { display:block; color:#52576A; font-size:.77rem; }
.stApp:has(.k-dashboard-shell) .k-gait-status > div span { color:#8A8F9F; font-size:.72rem; margin-top:2px; }
.stApp:has(.k-dashboard-shell) .k-gait-frequency { color:#7870C4; background:#F0EEFF; border-radius:6px; padding:4px 7px; font-size:.7rem; }
.stApp:has(.k-dashboard-shell) .k-care-line { display:flex; align-items:center; justify-content:space-between; min-height:39px; border-bottom:1px solid #F0F1F5; }
.stApp:has(.k-dashboard-shell) .k-care-line:last-child { border:0; }
.stApp:has(.k-dashboard-shell) .k-care-line span { color:#85899A; font-size:.79rem; }
.stApp:has(.k-dashboard-shell) .k-care-line strong { color:#3A3E50; font-size:.8rem; }
.stApp:has(.k-dashboard-shell) .k-empty-state { padding:28px 14px; color:#85899A; text-align:center; font-size:.83rem; }
.stApp:has(.k-dashboard-shell) .k-recent-heading { margin-top:20px; }
.stApp:has(.k-dashboard-shell) [data-testid="stDataFrame"] { overflow:hidden; border-radius:12px; box-shadow:0 3px 14px rgba(31,35,54,.035); }
.stApp:has(.k-dashboard-shell) [data-testid="stCaptionContainer"] { color: #868B9B !important; font-size:.75rem; }
.stApp:has(.k-dashboard-shell) [data-testid="stAlert"] { background:#F7F6FF !important; border-color:#E9E5FF !important; }
.stApp:has(.k-dashboard-shell) .k-card-title { color: #272A3D !important; }
.stApp:has(.k-dashboard-shell) .k-card-subtitle { color: #85899A; }
.stApp:has(.k-dashboard-shell) .k-card-header { border-color: #EFF0F4; }
.stApp:has(.k-dashboard-shell) .k-pill-orange { background: #F0EEFF; color: #5D50BD; border: 1px solid #E5E1FF; }
.stApp:has(.k-dashboard-shell) .k-pill-green { background: #EAF7F0; color: #27825F; border: 1px solid #D9F0E3; }
.stApp:has(.k-dashboard-shell) .k-pill-red { background: #FFF0EC; color: #B85B46; border: 1px solid #F7DED7; }
.stApp:has(.k-dashboard-shell) .k-pill-neutral { background: #F3F4F8; color: #707587; border: 1px solid #E7E9F0; }
.stApp:has(.k-dashboard-shell) .k-alert-monitor {
    background: #FFF8F6;
    border-color: #F3DDD7;
    border-left-color: #D97862;
}
.stApp:has(.k-dashboard-shell) .k-alert-baseline {
    background: #F2FAF6;
    border-color: #DDEFE5;
    border-left-color: #39A77E;
}
.stApp:has(.k-dashboard-shell) [data-testid="stDataFrame"] {
    background: #FFFFFF;
    border-color: #E8E9F0;
}
.stApp:has(.k-dashboard-shell) .stButton > button {
    background: #7162D2 !important;
    box-shadow: none !important;
    border-radius: 8px !important;
}
.stApp:has(.k-dashboard-shell) .stButton > button:hover { background: #5E50BD !important; }
.stApp:has(.k-dashboard-shell) [data-testid="stCaptionContainer"] { color: #868B9B !important; }
@media (max-width: 900px) {
    .stApp:has(.k-dashboard-shell) [data-testid="stMainBlockContainer"] { padding:1rem 1rem 2rem; }
    .stApp:has(.k-dashboard-shell) .k-kpi-value { font-size:1.55rem; }
    .stApp:has(.k-dashboard-shell) .k-dashboard-alert { align-items:flex-start; }
}

</style>
"""


def apply_theme():
    """Inject the orange & black theme stylesheet into the active Streamlit app."""
    import streamlit as st
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def create_dark_figure(figsize: Tuple[float, float] = (8, 3.5)):
    """
    Create a Matplotlib figure and axes pre-styled with the dark & orange theme.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="#161616")
    ax.set_facecolor("#1A1A1A")
    ax.tick_params(colors="#9E9E9E", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#2E2E2E")
    ax.grid(True, color="#262626", linestyle="--", linewidth=0.7, alpha=0.8)
    return fig, ax


def create_light_figure(figsize: Tuple[float, float] = (8, 3.5)):
    """Create a chart styled for the light dashboard surface."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.tick_params(colors="#73788A", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#E7E9F0")
    ax.grid(True, color="#EEF0F5", linestyle="-", linewidth=0.7, alpha=0.9)
    return fig, ax

