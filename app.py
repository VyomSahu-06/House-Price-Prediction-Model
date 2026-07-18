import pickle
import numpy as np
import pandas as pd
import os
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
CURRENT_YEAR = 2026

FEATURE_ORDER = [
    "OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath",
    "BedroomAbvGr", "LotArea", "House Age", "Total Size", "AreaperBed", "BathperBed",
]

# Fill in test-set scores for each model from your notebook (or leave None to
# compute them live from an uploaded test CSV further down).
PRECOMPUTED_METRICS = {
    "Linear Regression": {"MAE": None, "MSE": None, "R2": None},
    "Decision Tree":      {"MAE": None, "MSE": None, "R2": None},
    "Random Forest":      {"MAE": None, "MSE": None, "R2": None},
}

# Short blurb shown in the sidebar for whichever model is selected
MODEL_DESCRIPTIONS = {
    "Linear Regression": (
        "Fits a straight-line relationship between the house features and its "
        "sale price. Fast and easy to interpret — each feature gets a fixed "
        "weight — but it assumes prices change at a constant rate as features "
        "increase, which can miss more complex, non-linear patterns."
    ),
    "Decision Tree": (
        "Splits the data into a series of yes/no questions (e.g. \"Is GrLivArea "
        "> 2000?\") to arrive at a price. Captures non-linear patterns and "
        "feature interactions well, but a single tree can overfit and be "
        "sensitive to small changes in the data."
    ),
    "Random Forest": (
        "Trains many decision trees on random subsets of the data and averages "
        "their predictions. Usually more accurate and stable than a single "
        "tree since averaging smooths out individual trees' overfitting, at "
        "the cost of being less directly interpretable."
    ),
}

st.set_page_config(page_title="House Price Prediction", page_icon="Assets/logo.png", layout="centered")

# ----------------------------------------------------------------------------
# Design system — "Blueprint Ledger"
# Ink navy backdrop, brass accent, serif display + mono numerals.
# ----------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
 
    :root {
        --ink: #1B2430;
        --panel: #212B38;
        --panel-line: #384457;
        --brass: #C9A24B;
        --brass-soft: #E4C77A;
        --paper: #EDEAE1;
        --muted: #93A0AF;
    }
 
    /* App background */
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: var(--ink);
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 760px;
    }
 
    /* Base typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--paper);
    }
    p, label, span, div {
        color: var(--paper);
    }
 
    /* Hero block */
    .hero {
        position: relative;
        padding: 2.2rem 1.5rem 1.6rem 1.5rem;
        margin-bottom: 1.8rem;
        border: 1px solid var(--panel-line);
        border-radius: 4px;
        background-image:
            linear-gradient(var(--panel), var(--panel)),
            repeating-linear-gradient(0deg, rgba(201,162,75,0.06) 0 1px, transparent 1px 28px),
            repeating-linear-gradient(90deg, rgba(201,162,75,0.06) 0 1px, transparent 1px 28px);
        background-blend-mode: normal;
        text-align: center;
    }
    .hero-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.18em;
        font-size: 0.72rem;
        color: var(--brass-soft);
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .hero-title {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 2.4rem;
        line-height: 1.15;
        color: var(--paper);
        margin: 0;
    }
    .hero-rule {
        width: 120px;
        height: 2px;
        margin: 1rem auto 0.9rem auto;
        background: var(--brass);
        position: relative;
    }
    .hero-rule::before, .hero-rule::after {
        content: '';
        position: absolute;
        top: -3px;
        width: 1px;
        height: 8px;
        background: var(--brass);
    }
    .hero-rule::before { left: 0; }
    .hero-rule::after { right: 0; }
    .hero-sub {
        font-size: 0.92rem;
        color: var(--muted);
        max-width: 480px;
        margin: 0 auto;
    }
 
    /* Section labels (used instead of st.subheader) */
    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--brass-soft);
        border-bottom: 1px solid var(--panel-line);
        padding-bottom: 0.5rem;
        margin: 1.8rem 0 1rem 0;
    }
 
    /* Card wrapper */
    .card {
        background: var(--panel);
        border: 1px solid var(--panel-line);
        border-radius: 4px;
        padding: 1.4rem 1.4rem 0.6rem 1.4rem;
        margin-bottom: 1rem;
    }
 
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--panel);
        border-right: 1px solid var(--panel-line);
    }
    [data-testid="stSidebar"] * {
        color: var(--paper);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* Sidebar brand block */
    .sb-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.16em;
        font-size: 0.65rem;
        color: var(--brass-soft);
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .sb-title {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 1.6rem;
        line-height: 1.2;
        color: var(--paper);
        margin: 0 0 0.6rem 0;
    }
    .sb-rule {
        width: 56px;
        height: 2px;
        background: var(--brass);
        margin: 0 0 1.2rem 0;
    }
    .sb-section {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--brass-soft);
        margin: 1.4rem 0 0.6rem 0;
    }
    .sb-note {
        font-size: 0.8rem;
        color: var(--muted);
        line-height: 1.5;
    }
    .sb-footer {
        margin-top: 2rem;
        padding-top: 0.9rem;
        border-top: 1px solid var(--panel-line);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.06em;
        color: var(--muted);
    }

    /* Sidebar radio styled like a nav list */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: var(--ink);
        border: 1px solid var(--panel-line);
        border-radius: 3px;
        padding: 0.5rem 0.7rem;
        margin-bottom: 0.4rem;
        width: 100%;
        transition: border-color 0.15s ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        border-color: var(--brass);
    }

    /* Sidebar model description card */
    .sb-model-desc {
        background: var(--ink);
        border: 1px solid var(--panel-line);
        border-left: 2px solid var(--brass);
        border-radius: 3px;
        padding: 0.7rem 0.8rem;
        margin: 0.4rem 0 0.2rem 0;
        font-size: 0.78rem;
        line-height: 1.5;
        color: var(--muted);
    }
    .sb-model-desc b {
        color: var(--brass-soft);
    }

    /* Inputs */
    [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {
        background-color: var(--ink) !important;
        color: var(--paper) !important;
        border: 1px solid var(--panel-line) !important;
        border-radius: 3px !important;
    }
    [data-testid="stSlider"] [role="slider"] {
        background-color: var(--brass) !important;
    }
    .stSlider [data-baseweb="slider"] div div div {
        background: var(--brass) !important;
    }
 
    /* Buttons */
    .stButton > button {
        width: 100%;
        background-color: var(--brass);
        color: var(--ink);
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 500;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-size: 0.85rem;
        border: none;
        border-radius: 3px;
        padding: 0.7rem 0;
        transition: background-color 0.15s ease;
    }
    .stButton > button:hover {
        background-color: var(--brass-soft);
        color: var(--ink);
    }
 
    /* Metric / result */
    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--brass);
        border-radius: 4px;
        padding: 1.2rem 1rem;
        text-align: center;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--brass-soft) !important;
        justify-content: center;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Fraunces', serif;
        font-size: 2.4rem !important;
        color: var(--paper) !important;
    }
 
    /* Expander */
    [data-testid="stExpander"] {
        background: var(--panel);
        border: 1px solid var(--panel-line);
        border-radius: 4px;
    }
 
    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--panel-line);
        border-radius: 4px;
    }
 
    /* Stat row (MAE / MSE / R2) */
    .stat-card {
        background: var(--panel);
        border: 1px solid var(--panel-line);
        border-radius: 4px;
        padding: 0.9rem 0.5rem;
        text-align: center;
    }
    .stat-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--brass-soft);
        margin-bottom: 0.35rem;
    }
    .stat-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.3rem;
        font-weight: 500;
        color: var(--paper);
    }

    /* Footer strip */
    .ledger-footer {
        margin-top: 2rem;
        padding-top: 0.8rem;
        border-top: 1px solid var(--panel-line);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.08em;
        color: var(--muted);
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
 
# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">Automated Appraisal · Est. 2026</div>
        <div class="hero-title">House Price Appraisal</div>
        <div class="hero-rule"></div>
        <div class="hero-sub">
            Enter a property's details below and get an instant, model-backed
            estimate of its sale price.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Project Overview</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="card">
    This project predicts a house's sale price from 11 property features —
    size, quality, layout, age, and a few engineered ratios such as
    area-per-bedroom and bath-per-bedroom. It's trained on a housing dataset
    and compares three regression approaches side by side: <b>Linear
    Regression</b>, <b>Decision Tree</b>, and <b>Random Forest</b>. Pick a
    model from the sidebar, enter a property's details below, and get an
    instant estimate along with that model's MAE, MSE, and R² on a held-out
    test set.
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Data visualizations (exported from the notebook's EDA)
# ----------------------------------------------------------------------------
st.markdown('<div class="section-label">Data Visualizations</div>', unsafe_allow_html=True)
 
ASSETS_DIR = "Assets"
 
 
def show_viz(filename, caption):
    path = f"{ASSETS_DIR}/{filename}"
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.caption(f"⚠️ Missing image: `{path}` — add it to the `assets/` folder next to app.py.")
 
 
viz_tabs = st.tabs(["Distributions", "Relationships", "Correlation"])
 
with viz_tabs[0]:
    v1, v2 = st.columns(2)
    with v1:
        show_viz("sale_price_distribution.png", "Sale Price Distribution")
    with v2:
        show_viz("avg_price_by_quality.png", "Average Price by Overall Quality")
    show_viz("sale_price_outliers.png", "Sale Price Outliers")
 
with viz_tabs[1]:
    v1, v2 = st.columns(2)
    with v1:
        show_viz("area_vs_price.png", "Living Area vs Sale Price")
        show_viz("basement_area_vs_price.png", "Basement Area vs Sale Price")
    with v2:
        show_viz("garage_cars_vs_price.png", "Garage Cars vs Sale Price")
        show_viz("price_by_year_built.png", "Sale Price by Year Built")
 
with viz_tabs[2]:
    show_viz("correlation_heatmap.png", "Correlation Heatmap")
 
 

# ----------------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


st.title("🏠 House Price Prediction")

with st.sidebar:
    st.markdown(
        """
        <div class="sb-eyebrow">Blueprint Ledger</div>
        <div class="sb-title">House Price<br>Appraisal</div>
        <div class="sb-rule"></div>
        <div class="sb-note">
            An instant, model-backed estimate of a property's sale price —
            pick a model and fill in the details on the right.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-section">Model</div>', unsafe_allow_html=True)
    select_model = st.radio(
        "Select a model",
        ["Linear Regression", "Decision Tree", "Random Forest"],
        label_visibility="collapsed",
    )
    st.markdown(
        f'<div class="sb-model-desc"><b>{select_model}</b><br>'
        f'{MODEL_DESCRIPTIONS[select_model]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-section">Features Used</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sb-note">
        Overall Quality · Living Area · Garage Cars · Basement Area ·
        Full Baths · Bedrooms · Lot Area · House Age · Total Size ·
        Area/Bed · Bath/Bed
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sb-footer">Model-generated estimate · Not a substitute '
        'for a licensed appraisal</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        '<div class="sb-section">About the Developer</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sb-note">Hello I am Vyom Sahu and I make this Project of House Price Prediction where I'
        ' have used 3 regression models to predict the price of the houses using the features mentioned above.</div>',
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
st.subheader("Enter house details")

c1, c2, c3 = st.columns(3)
with c1:
    gr_liv_area = st.number_input("Ground Living Area (sq ft)", min_value=0, value=1500)
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    garage_cars = st.slider("Garage Cars", 0, 4, 1)
with c2:
    total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, value=800)
    full_bath = st.slider("Full Bathrooms", 0, 4, 2)
    bedroom_abv_gr = st.slider("Bedrooms Above Grade", 1, 8, 3)
with c3:
    year_built = st.number_input("Year Built", min_value=1870, max_value=CURRENT_YEAR, value=2000)
    lot_area = st.number_input("Enter Lot Area (sq ft)", min_value=0, value=2000)

if select_model=="Linear Regression":
    try:
        model_path="Models/model_lr.pkl"
        model = load_model(model_path)
    except FileNotFoundError:
        st.error(f"Couldn't find **{model_path}**. Place your pickled model next to app.py.")
        st.stop()
    except Exception as e:
        st.error(f"Couldn't load the model: {e}")
        st.stop()

if select_model=="Decision Tree":
    try:
        model_path="Models/model_dt.pkl"
        model = load_model(model_path)
    except FileNotFoundError:
        st.error(f"Couldn't find **{model_path}**. Place your pickled model next to app.py.")
        st.stop()
    except Exception as e:
        st.error(f"Couldn't load the model: {e}")
        st.stop()

if select_model=="Random Forest":
    try:
        model_path="Models/model_rf.pkl"
        model = load_model(model_path)
    except FileNotFoundError:
        st.error(f"Couldn't find **{model_path}**. Place your pickled model next to app.py.")
        st.stop()
    except Exception as e:
        st.error(f"Couldn't load the model: {e}")
        st.stop()


# ----------------------------------------------------------------------------
# Model performance (MAE / MSE / R²) for the currently selected model
# ----------------------------------------------------------------------------
st.markdown('<div class="section-label">Model Performance</div>', unsafe_allow_html=True)

metrics = PRECOMPUTED_METRICS[select_model]
mae, mse, r2 = metrics["MAE"], metrics["MSE"], metrics["R2"]

if mae is None or mse is None or r2 is None:
    with st.expander(f"Upload a test set to score {select_model}"):
        st.caption(
            "CSV must contain the same raw columns your model was engineered from, "
            "plus the true `SalePrice` column."
        )
        test_file = st.file_uploader("Test set CSV", type=["csv"], key=f"test_set_{select_model}")
        if test_file is not None:
            try:
                test_df = pd.read_csv(test_file)
                if "SalePrice" not in test_df.columns:
                    st.error("The uploaded file needs a `SalePrice` column with the true values.")
                else:
                    y_true = test_df["SalePrice"]
                    x_test = test_df[FEATURE_ORDER]
                    y_pred = model.predict(x_test)
                    mae = mean_absolute_error(y_true, y_pred)
                    mse = mean_squared_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)
            except KeyError as e:
                st.error(f"Missing expected column in the test set: {e}")
            except Exception as e:
                st.error(f"Couldn't score the test set: {e}")

if mae is not None and mse is not None and r2 is not None:
    m1, m2, m3 = st.columns(3)
    m1.markdown(
        f'<div class="stat-card"><div class="stat-label">MAE</div>'
        f'<div class="stat-value">₹{mae:,.0f}</div></div>',
        unsafe_allow_html=True,
    )
    m2.markdown(
        f'<div class="stat-card"><div class="stat-label">MSE</div>'
        f'<div class="stat-value">{mse:,.0f}</div></div>',
        unsafe_allow_html=True,
    )
    m3.markdown(
        f'<div class="stat-card"><div class="stat-label">R² Score</div>'
        f'<div class="stat-value">{r2:.3f}</div></div>',
        unsafe_allow_html=True,
    )
else:
    st.caption(
        f"No metrics yet for {select_model} — fill in `PRECOMPUTED_METRICS` at the "
        "top of the file, or upload a test set above."
    )

# Engineered features (must match how the model was trained)
house_age = CURRENT_YEAR - year_built
total_size = gr_liv_area + total_bsmt_sf
area_per_bed = gr_liv_area / bedroom_abv_gr
bath_per_bed = full_bath / bedroom_abv_gr

input_row = {
    "OverallQual": overall_qual,
    "GrLivArea": gr_liv_area,
    "GarageCars": garage_cars,
    "TotalBsmtSF": total_bsmt_sf,
    "FullBath": full_bath,
    "BedroomAbvGr": bedroom_abv_gr,
    "LotArea": lot_area,
    "House Age": house_age,
    "Total Size": total_size,
    "AreaperBed": area_per_bed,
    "BathperBed": bath_per_bed,
}
# Built directly from FEATURE_ORDER so the model always gets columns in the
# exact order it was trained on — no separate hand-typed list to drift out of sync.
input_df = pd.DataFrame([input_row])[FEATURE_ORDER]
display_df = pd.concat([pd.DataFrame({"Model Selected": [select_model]}), input_df], axis=1)

st.markdown('<div class="section-label">Appraisal</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------------
if st.button("Predict Sale Price", type="primary"):
    prediction = model.predict(input_df)
    st.metric("Predicted Sale Price", f"₹{prediction[0]:,.0f}")
    with st.expander("See the feature row sent to the model"):
        st.dataframe(display_df, use_container_width=True)

st.markdown('<div class="section-label">Appraisal</div>', unsafe_allow_html=True)
