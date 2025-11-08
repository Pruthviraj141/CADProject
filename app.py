import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

# --- Configuration ---
FEATURE_COLS = ['users_rated', 'minplayers', 'maxplayers', 'playingtime', 'yearpublished', 'minage']

# --- Simulated Model Functions (Heuristic) ---

def predict_rf_dummy(users_rated, minplayers, maxplayers, playingtime, yearpublished, minage):
    users_influence = np.log1p(users_rated) / 6.0
    age_influence = min(1, max(0, (minage - 8) / 10.0)) * 0.5
    time_deviation = abs(playingtime - 90) / 100.0
    time_influence = max(0, 0.5 - time_deviation)
    player_influence = max(0, 0.5 - abs(maxplayers - 4) / 4.0)
    base_rating = 6.0

    raw_pred = (
        base_rating +
        users_influence +
        age_influence +
        time_influence * 0.8 +
        player_influence * 0.7 +
        np.random.uniform(-0.3, 0.3)
    )

    return max(1.0, min(10.0, raw_pred))

def predict_lr_dummy(users_rated, minplayers, maxplayers, playingtime, yearpublished, minage):
    raw_pred = (
        5.0 +
        (np.log1p(users_rated) * 0.2) +
        (minplayers * 0.05) -
        (abs(maxplayers - 5) * 0.05) +
        (playingtime * 0.005) +
        ((yearpublished - 2000) * 0.01) +
        np.random.uniform(-0.2, 0.2)
    )

    return max(1.0, min(10.0, raw_pred))


# --- Streamlit UI Setup ---
st.set_page_config(page_title="🎯 Board Game Rating Predictor", layout="centered")

# --- Custom CSS for better visuals ---
st.markdown("""
    <style>
        .main {
            padding: 2rem;
            background-color: #f9fafb;
        }
        h1, h2, h3 {
            text-align: center;
        }
        div[data-testid="stMetricValue"] {
            font-size: 32px !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 16px !important;
            color: #555 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🎲 Board Game Rating Predictor")
st.caption("Estimate how your board game might be rated by players based on its specifications.")
st.markdown("---")

# --- Sidebar ---
st.sidebar.header("Model Settings")
model_choice = st.sidebar.radio(
    "Select Prediction Model",
    ["Random Forest", "Linear Regression", "Both"],
    help="Random Forest is more dynamic, while Linear Regression gives more stable results."
)
st.sidebar.info("Adjust parameters on the right, then click **Predict Rating** to see results.")

# --- User Inputs ---
st.subheader("Enter Game Details")

col1, col2 = st.columns(2)
with col1:
    users_rated = st.number_input(
        "1. Users Rated", min_value=0, max_value=100000, value=1000, step=100,
        help="Number of players who rated the game."
    )
    minplayers = st.number_input(
        "2. Minimum Players", min_value=1, max_value=20, value=2, step=1
    )
    maxplayers = st.number_input(
        "3. Maximum Players", min_value=1, max_value=20, value=4, step=1
    )

with col2:
    playingtime = st.number_input(
        "4. Playing Time (minutes)", min_value=1, max_value=1000, value=60, step=15
    )
    yearpublished = st.number_input(
        "5. Year Published", min_value=1900, max_value=2025, value=2015, step=1
    )
    minage = st.number_input(
        "6. Minimum Age", min_value=0, max_value=100, value=12, step=1
    )

st.markdown("---")

# --- Predict Button ---
if st.button("🚀 Predict Rating", type="primary"):
    with st.spinner("Fetching model predictions..."):
        time.sleep(1)  # simulate loading delay

    predictions = {}

    if model_choice in ["Linear Regression", "Both"]:
        lr_pred = predict_lr_dummy(users_rated, minplayers, maxplayers, playingtime, yearpublished, minage)
        predictions["Linear Regression"] = round(lr_pred, 2)

    if model_choice in ["Random Forest", "Both"]:
        rf_pred = predict_rf_dummy(users_rated, minplayers, maxplayers, playingtime, yearpublished, minage)
        predictions["Random Forest"] = round(rf_pred, 2)

    # --- Results Display ---
    st.success("✅ Prediction Complete!")
    st.subheader("Predicted Ratings")

    result_cols = st.columns(len(predictions))
    for i, (model, pred) in enumerate(predictions.items()):
        with result_cols[i]:
            st.metric(label=f"{model}", value=f"{pred:.2f}", delta="Rating /10")

    # --- Chart Visualization ---
    if predictions:
        st.markdown("### 📊 Model Comparison")

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#1E88E5" if k == "Linear Regression" else "#F4511E" for k in predictions.keys()]
        bars = ax.bar(predictions.keys(), predictions.values(), color=colors)

        ax.set_ylim(0, 10)
        ax.set_ylabel("Predicted Rating (1–10)")
        ax.set_title("Model Comparison", pad=15)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.2f}", ha='center', va='bottom', fontsize=10)

        st.pyplot(fig)

    st.markdown("---")
    st.info("These results are generated using a simulation of real machine learning models.")

else:
    st.info("Enter your board game details and click **Predict Rating** to start.")


