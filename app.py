"""Delivery Time Prediction - Streamlit frontend.

Run from the project root with:  streamlit run app.py
Only loads saved artifacts and calls transform()/predict(); nothing is fitted here.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

ARTIFACT_FILES = {
    "model": "best_xgb_model.pkl",
    "encoder": "onehot_encoder.pkl",
    "feature_columns": "feature_columns.pkl",
    "traffic_mapping": "traffic_mapping.pkl",
    "numeric_median": "numeric_median.pkl",
    "categorical_modes": "categorical_modes.pkl",
}

CATEGORICAL_COLUMNS = ["Weather", "Time_of_Day", "Vehicle_Type"]
NUMERIC_EXPERIENCE_COL = "Courier_Experience_yrs"

st.set_page_config(page_title="Delivery Time Predictor", page_icon="🚚", layout="centered")


# ----------------------------------------------------------------------------
# Styling
# Palette: Graphite #3A3E3B, Rich Black #090C08, Mist #F0EFF4
# Font: Geist (with Geist Mono for the section numbers)
# ----------------------------------------------------------------------------
APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;500&display=swap');

:root {
    --bg: #090C08;            /* Black */
    --surface: #3A3E3B;       /* Graphite */
    --field: #2E3230;
    --border: #525754;
    --border-strong: #525754;
    --text: #F0EFF4;          /* Mist */
    --muted: #B7B8BC;
    --dim: #9A9E9B;
    --accent: #F0EFF4;
    --accent-hover: #DAD9DF;
    --accent-light: #F0EFF4;
    --on-accent: #090C08;
    --radius: 14px;
    --font: 'Geist', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    --mono: 'Geist Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

/* Page */
.stApp { background: var(--bg); color: var(--text); }
.stApp, .stApp p, .stApp label, .stApp input, .stApp button, .stApp h1, .stApp h2, .stApp h3 {
    font-family: var(--font);
}
.block-container { max-width: 860px; padding-top: 2.75rem; padding-bottom: 3rem; }
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }

/* Header */
.app-badge {
    display: inline-block; padding: 0.25rem 0.7rem; margin-bottom: 1.1rem;
    font-family: var(--mono); font-size: 0.72rem; font-weight: 500;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--accent-light); background: rgba(58, 62, 59, 0.6);
    border: 1px solid var(--border-strong); border-radius: 999px;
}
.app-title {
    font-size: 2.4rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.1;
    text-transform: uppercase; color: var(--text); margin: 0 0 0.9rem 0;
}
.app-title .dim { color: var(--dim); }
.app-subtitle { font-size: 1rem; line-height: 1.6; color: var(--muted); margin: 0 0 2rem 0; max-width: 36rem; }

/* Cards (bordered containers) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 0.4rem 0.35rem;
}
.section-head { display: flex; align-items: baseline; gap: 0.9rem; margin: 0 0 0.35rem 0; }
.section-num { font-family: var(--mono); font-size: 0.95rem; color: var(--accent); opacity: 0.6; }
.section-title {
    font-size: 1.55rem; font-weight: 400; letter-spacing: 0.01em; text-transform: uppercase;
    color: var(--text); line-height: 1.2;
}
.section-desc { font-size: 0.875rem; color: var(--muted); margin: 0 0 1rem 0; }

/* Field labels and helper text */
[data-testid="stWidgetLabel"] p { font-size: 0.82rem; font-weight: 500; color: var(--muted); }
[data-testid="stCaptionContainer"], .stCaption { color: var(--muted); font-size: 0.85rem; }

/* Inputs */
div[data-baseweb="input"], div[data-baseweb="select"] > div {
    background: var(--field); border: 1px solid var(--border); border-radius: 10px;
}
div[data-baseweb="input"] input, div[data-baseweb="select"] div { color: var(--text); }
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within {
    border-color: var(--accent); box-shadow: 0 0 0 3px rgba(240, 239, 244, 0.2);
}
[data-testid="stNumberInput"] button { background: transparent; color: var(--muted); border: none; }
[data-testid="stNumberInput"] button:hover { color: var(--accent); }
[data-testid="stRadio"] label p { font-size: 0.92rem; color: var(--text); }
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] [role="listbox"] {
    background: var(--surface); color: var(--text);
}
div[data-baseweb="popover"] li { color: var(--text); }
div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] li[aria-selected="true"] {
    background: rgba(240, 239, 244, 0.18);
}

/* Primary button */
.stButton > button {
    width: 100%; height: 3.1rem; border-radius: 12px; border: 1px solid var(--accent);
    background: var(--accent); color: var(--on-accent); font-size: 1rem; font-weight: 600;
    letter-spacing: 0.02em; transition: background 0.15s ease;
}
.stButton > button p { color: var(--on-accent); font-weight: 600; }
.stButton > button:hover, .stButton > button:focus:not(:active) {
    background: var(--accent-hover); border-color: var(--accent-hover); color: var(--on-accent);
}

/* Model menu: trigger button (top right) */
[data-testid="stPopover"] { margin-top: 0.2rem; }
[data-testid="stPopover"] button {
    width: 100%; height: 2.6rem; border-radius: 999px; background: var(--surface);
    border: 1px solid var(--border); color: var(--text); font-size: 0.88rem; font-weight: 500;
    transition: border-color 0.15s ease, background 0.15s ease;
}
[data-testid="stPopover"] button p { color: var(--text); font-weight: 500; }
[data-testid="stPopover"] button:hover, [data-testid="stPopover"] button:focus:not(:active) {
    background: var(--field); border-color: var(--accent); color: var(--text);
}

/* Model menu: dropdown panel */
[data-testid="stPopoverBody"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 1.1rem 1.25rem; min-width: min(560px, 92vw);
}
.menu-title {
    font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--dim); margin-bottom: 0.35rem;
}
.menu-row {
    display: flex; align-items: center; gap: 0.8rem; padding: 0.7rem 0;
    border-bottom: 1px solid var(--border);
}
.menu-icon {
    flex: 0 0 auto; display: flex; align-items: center; justify-content: center;
    width: 2rem; height: 2rem; border-radius: 9px; color: var(--accent);
    background: rgba(240, 239, 244, 0.08);
}
.menu-label { flex: 1 1 auto; font-size: 0.88rem; color: var(--muted); }
.menu-value { font-size: 0.92rem; font-weight: 500; color: var(--text); text-align: right; }
.chip {
    display: inline-block; padding: 0.05rem 0.5rem; margin: 0.1rem 0 0.1rem 0.25rem;
    font-family: var(--mono); font-size: 0.78rem; color: var(--accent-light);
    border: 1px solid var(--border); border-radius: 999px;
}
.menu-title-spaced { margin-top: 1.4rem; margin-bottom: 0.7rem; }
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: 0.65rem; }
.metric-card {
    padding: 0.95rem 1rem; background: var(--field);
    border: 1px solid var(--border); border-radius: 12px;
}
.metric-name {
    font-family: var(--mono); font-size: 0.7rem; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--dim); margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 1.85rem; font-weight: 300; letter-spacing: -0.02em; line-height: 1.1; color: var(--text);
}
.metric-unit { font-size: 0.85rem; font-weight: 400; letter-spacing: 0; margin-left: 0.35rem; color: var(--muted); }
.metric-desc { font-size: 0.78rem; line-height: 1.45; color: var(--muted); margin-top: 0.55rem; }
.metric-extra { font-family: var(--mono); font-size: 0.75rem; color: var(--dim); margin-top: 0.7rem; }
.metric-extra strong { color: var(--muted); font-weight: 500; }
.menu-note {
    display: flex; gap: 0.7rem; align-items: flex-start; margin-top: 0.9rem;
    font-size: 0.82rem; line-height: 1.5; color: var(--muted);
}
.menu-note svg { flex: 0 0 auto; margin-top: 0.15rem; color: var(--dim); }

/* Result card */
.result-card {
    margin-top: 1.25rem; padding: 1.6rem 1.9rem; border-radius: var(--radius);
    background: var(--accent); border: 1px solid var(--accent);
}
.result-head { display: flex; align-items: baseline; gap: 0.9rem; margin-bottom: 0.6rem; }
.result-num { font-family: var(--mono); font-size: 0.95rem; color: var(--surface); opacity: 0.8; }
.result-label {
    font-size: 0.8rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--surface);
}
.result-value { font-size: 3.4rem; font-weight: 300; letter-spacing: -0.03em; line-height: 1.05; color: var(--on-accent); }
.result-unit { font-size: 1.1rem; font-weight: 400; margin-left: 0.5rem; letter-spacing: 0; color: var(--surface); }
.result-note { font-size: 0.85rem; color: rgba(9, 12, 8, 0.65); margin-top: 0.75rem; }

.app-footer {
    font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--dim); text-align: center; margin-top: 2.25rem;
}

@media (max-width: 640px) {
    .block-container { padding-top: 1.5rem; }
    .app-title { font-size: 1.9rem; }
    .section-title { font-size: 1.3rem; }
    .result-value { font-size: 2.6rem; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        """
        <div class="app-badge">XGBoost regression</div>
        <div class="app-title">Delivery Time<br><span class="dim">Predictor</span></div>
        <div class="app-subtitle">
            Enter the details of an order to estimate how long the delivery will take.
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(number: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="section-head"><span class="section-num">{number}</span>'
        f'<span class="section-title">{title}</span></div>'
        f'<div class="section-desc">{description}</div>',
        unsafe_allow_html=True,
    )


def render_result(minutes: float) -> None:
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-head">
                <span class="result-num">04</span>
                <span class="result-label">Estimated delivery time</span>
            </div>
            <div class="result-value">{minutes:.2f}<span class="result-unit">minutes</span></div>
            <div class="result-note">Estimate from the trained XGBoost model using the details above.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Model details menu (static information; metric values are intentionally omitted)
# ----------------------------------------------------------------------------
ICON_PATHS = {
    "model": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>'
             '<path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>',
    "type": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "features": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/>'
                '<polyline points="2 12 12 17 22 12"/>',
    "train": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>'
             '<path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
    "test": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "metrics": '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/>'
               '<line x1="6" y1="20" x2="6" y2="14"/>',
    "info": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/>'
            '<line x1="12" y1="8" x2="12.01" y2="8"/>',
}

MODEL_INFO = [
    ("model", "Model", "XGBoost Regressor"),
    ("type", "Prediction type", "Regression"),
    ("target", "Prediction target", "Delivery Time (minutes)"),
    ("features", "Input features", "7"),
    ("train", "Training data", "80%"),
    ("test", "Testing data", "20%"),
]

# Test-set evaluation results (fixed values supplied from the trained model's evaluation).
METRICS = [
    ("MAE", "6.58", "min", "Average size of the error between predicted and actual delivery times."),
    ("RMSE", "9.44", "min", "Like MAE, but larger errors count more, so it flags big misses."),
    ("R² Score", "0.8011", "80.11%", "Share of the variation in delivery times the model explains."),
]
MSE_VALUE = "89.15"

DISCLAIMER = (
    "Predictions are estimates based on historical delivery data. "
    "Actual delivery times may vary depending on real-world conditions."
)


def icon_svg(key: str, size: int = 17) -> str:
    return (
        f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="currentColor" '
        f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{ICON_PATHS[key]}</svg>'
    )


def model_menu_html() -> str:
    rows = "".join(
        f'<div class="menu-row"><div class="menu-icon">{icon_svg(key)}</div>'
        f'<div class="menu-label">{label}</div><div class="menu-value">{value}</div></div>'
        for key, label, value in MODEL_INFO
    )
    cards = "".join(
        f'<div class="metric-card"><div class="metric-name">{name}</div>'
        f'<div class="metric-value">{value}<span class="metric-unit">{unit}</span></div>'
        f'<div class="metric-desc">{desc}</div></div>'
        for name, value, unit, desc in METRICS
    )
    metrics = (
        '<div class="menu-title menu-title-spaced">Test-set performance</div>'
        f'<div class="metric-grid">{cards}</div>'
        f'<div class="metric-extra">MSE <strong>{MSE_VALUE}</strong></div>'
    )
    note = f'<div class="menu-note">{icon_svg("info", 16)}<span>{DISCLAIMER}</span></div>'
    return f'<div class="menu-title">About the model</div>{rows}{metrics}{note}'


def render_model_menu() -> None:
    """Dropdown menu in the top-right corner (falls back to an expander on old Streamlit)."""
    if hasattr(st, "popover"):
        container = st.popover("Model details", use_container_width=True)
    else:
        container = st.expander("Model details")
    with container:
        st.markdown(model_menu_html(), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Artifact loading
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model and preprocessing artifacts...")
def load_artifacts() -> dict:
    """Load all six artifacts. Raises FileNotFoundError listing any missing files."""
    missing = [name for name in ARTIFACT_FILES.values() if not (MODEL_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing file(s) in '{MODEL_DIR}': " + ", ".join(missing)
        )
    return {key: joblib.load(MODEL_DIR / fname) for key, fname in ARTIFACT_FILES.items()}


# ----------------------------------------------------------------------------
# Helpers to read saved imputation values (scalar, dict, or pandas object)
# ----------------------------------------------------------------------------
def get_median_value(numeric_median):
    if isinstance(numeric_median, (pd.Series, dict)):
        return numeric_median[NUMERIC_EXPERIENCE_COL]
    if isinstance(numeric_median, pd.DataFrame):
        return numeric_median[NUMERIC_EXPERIENCE_COL].iloc[0]
    return numeric_median


def get_mode_value(categorical_modes, column):
    if isinstance(categorical_modes, pd.DataFrame):
        return categorical_modes[column].iloc[0]
    return categorical_modes[column]  # dict or Series


def get_categories(encoder) -> dict:
    """Map each categorical column to its categories from the fitted encoder."""
    names = getattr(encoder, "feature_names_in_", None)
    cols = list(names) if names is not None else CATEGORICAL_COLUMNS
    if sorted(cols) != sorted(CATEGORICAL_COLUMNS):
        raise ValueError(
            f"Encoder was fitted on {cols}, expected {CATEGORICAL_COLUMNS}."
        )
    return {col: list(cats) for col, cats in zip(cols, encoder.categories_)}


# ----------------------------------------------------------------------------
# Preprocessing (mirrors training; no fitting)
# ----------------------------------------------------------------------------
def preprocess(raw_input: dict, artifacts: dict) -> pd.DataFrame:
    encoder = artifacts["encoder"]
    feature_columns = list(artifacts["feature_columns"])
    traffic_mapping = artifacts["traffic_mapping"]

    # 1-2. DataFrame with the seven features
    df = pd.DataFrame([raw_input])

    # 3. Ordinal-encode Traffic_Level with the saved mapping
    df["Traffic_Level"] = df["Traffic_Level"].map(traffic_mapping)
    if df["Traffic_Level"].isna().any():
        raise ValueError(
            f"Traffic_Level must be one of {list(traffic_mapping.keys())}."
        )

    # 4. Impute missing experience with the saved training median
    if df[NUMERIC_EXPERIENCE_COL].isna().any():
        df[NUMERIC_EXPERIENCE_COL] = df[NUMERIC_EXPERIENCE_COL].fillna(
            get_median_value(artifacts["numeric_median"])
        )

    # 5. Impute missing categoricals with the saved training modes
    for col in CATEGORICAL_COLUMNS:
        if df[col].isna().any():
            df[col] = df[col].fillna(get_mode_value(artifacts["categorical_modes"], col))

    # 6. One-hot encode with the fitted encoder (transform only), handling sparse output
    encoded = encoder.transform(df[CATEGORICAL_COLUMNS])
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    encoded_df = pd.DataFrame(
        np.asarray(encoded),
        columns=encoder.get_feature_names_out(CATEGORICAL_COLUMNS),
        index=df.index,
    )

    # 7. Combine numeric features with encoded categoricals
    numeric_df = df.drop(columns=CATEGORICAL_COLUMNS)
    combined = pd.concat([numeric_df, encoded_df], axis=1)

    # 8. Enforce exact training columns and order (fail loudly if anything is missing)
    missing = [c for c in feature_columns if c not in combined.columns]
    if missing:
        raise ValueError(f"Processed input is missing training feature(s): {missing}")
    return combined[feature_columns]


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
def predict(raw_input: dict, artifacts: dict) -> float:
    features = preprocess(raw_input, artifacts)
    return float(artifacts["model"].predict(features)[0])


# ----------------------------------------------------------------------------
# UI state helpers
# ----------------------------------------------------------------------------
def input_signature(raw_input: dict) -> tuple:
    """Hashable snapshot of the inputs (NaN -> None) used to detect stale results."""
    return tuple(
        (key, None if isinstance(value, float) and np.isnan(value) else value)
        for key, value in raw_input.items()
    )


# ----------------------------------------------------------------------------
# UI sections
# ----------------------------------------------------------------------------
def render_order_section() -> tuple[float, float]:
    with st.container(border=True):
        section_header("01", "Order", "Distance to travel and how long the kitchen needs.")
        col1, col2 = st.columns(2)
        with col1:
            distance = st.number_input("Distance (km)", min_value=0.0, value=10.0, step=0.5)
        with col2:
            prep_time = st.number_input(
                "Preparation time (minutes)", min_value=0.0, value=15.0, step=1.0
            )
    return distance, prep_time


def render_conditions_section(categories: dict, traffic_options: list) -> tuple:
    with st.container(border=True):
        section_header("02", "Conditions", "Weather, traffic and time of day during the delivery.")
        col1, col2, col3 = st.columns(3)
        with col1:
            weather = st.selectbox("Weather", categories["Weather"])
        with col2:
            traffic = st.selectbox("Traffic level", traffic_options)
        with col3:
            time_of_day = st.selectbox("Time of day", categories["Time_of_Day"])
    return weather, traffic, time_of_day


def render_courier_section(categories: dict, median_exp: float) -> tuple:
    with st.container(border=True):
        section_header("03", "Courier", "Vehicle used and the courier's experience.")
        col1, col2 = st.columns(2)
        with col1:
            vehicle = st.selectbox("Vehicle type", categories["Vehicle_Type"])
        with col2:
            experience_known = st.radio(
                "Is courier experience known?", ["Yes", "No"], index=0, horizontal=True
            )

        if experience_known == "Yes":
            exp_col, _ = st.columns(2)
            with exp_col:
                experience = st.number_input(
                    "Courier Experience (years)", min_value=0.0, value=3.0, step=0.5
                )
        else:
            # Left missing on purpose: preprocess() fills it with the saved training median.
            experience = np.nan
            st.caption(
                f"Using the training median of {median_exp:g} years (loaded from numeric_median.pkl)."
            )
    return vehicle, experience


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
def main() -> None:
    inject_css()

    head_col, menu_col = st.columns([3, 1.3])
    with head_col:
        render_header()
    with menu_col:
        render_model_menu()

    try:
        artifacts = load_artifacts()
        categories = get_categories(artifacts["encoder"])
        traffic_options = list(artifacts["traffic_mapping"].keys())
        median_exp = float(get_median_value(artifacts["numeric_median"]))
    except FileNotFoundError as e:
        st.error(f"Could not load required files. {e}")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load artifacts: {type(e).__name__}: {e}")
        st.stop()

    distance, prep_time = render_order_section()
    weather, traffic, time_of_day = render_conditions_section(categories, traffic_options)
    vehicle, experience = render_courier_section(categories, median_exp)

    raw_input = {
        "Distance_km": distance,
        "Weather": weather,
        "Traffic_Level": traffic,
        "Time_of_Day": time_of_day,
        "Vehicle_Type": vehicle,
        "Preparation_Time_min": prep_time,
        "Courier_Experience_yrs": experience,
    }
    signature = input_signature(raw_input)

    st.write("")
    if st.button("Predict Delivery Time", type="primary", use_container_width=True):
        try:
            minutes = predict(raw_input, artifacts)
            st.session_state["last_prediction"] = (signature, minutes)
        except Exception as e:
            st.session_state.pop("last_prediction", None)
            st.error(f"Prediction failed: {type(e).__name__}: {e}")

    # Show the result only while the inputs still match the ones it was generated for.
    last = st.session_state.get("last_prediction")
    if last is not None and last[0] == signature:
        render_result(last[1])

    st.markdown(
        '<div class="app-footer">Estimates from a trained XGBoost regression model</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()