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
st.set_page_config(page_title="🎯 Board Game Rating Predictor Using ML", layout="centered")

# --- Header ---
st.title("🎲 Board Game Rating Prediction..")
st.caption("Estimate how your board game might be rated by players based on its specifications.")
st.markdown("---")

# --- Sidebar ---
st.sidebar.header("Model Settings")
model_choice = st.sidebar.radio(
    "Select Prediction Model",
    ["Random Forest", "Linear Regression", "Both"]
)

# --- User Inputs ---
st.subheader("Enter Game Details")

col1, col2 = st.columns(2)
with col1:
    users_rated = st.number_input("1. Users Rated", min_value=0, max_value=100000, value=1000, step=100)
    minplayers = st.number_input("2. Minimum Players", min_value=1, max_value=20, value=2, step=1)
    maxplayers = st.number_input("3. Maximum Players", min_value=1, max_value=20, value=4, step=1)

with col2:
    playingtime = st.number_input("4. Playing Time (minutes)", min_value=1, max_value=1000, value=60, step=15)
    yearpublished = st.number_input("5. Year Published", min_value=1900, max_value=2025, value=2015, step=1)
    minage = st.number_input("6. Minimum Age", min_value=0, max_value=100, value=12, step=1)

st.markdown("---")

# --- Predict Button ---
if st.button("🚀 Predict Rating"):
    with st.spinner("Calculating..."):
        time.sleep(1)

    predictions = {}

    if model_choice in ["Linear Regression", "Both"]:
        predictions["Linear Regression"] = round(predict_lr_dummy(users_rated, minplayers, maxplayers, playingtime, yearpublished, minage), 2)

    if model_choice in ["Random Forest", "Both"]:
        predictions["Random Forest"] = round(predict_rf_dummy(users_rated, minplayers, maxplayers, playingtime, yearpublished, minage), 2)

    st.success("Prediction Complete!")
    st.subheader("Predicted Ratings")

    for model, pred in predictions.items():
        st.metric(label=model, value=f"{pred:.2f}")

    # --- Bar Chart Comparison ---
    fig, ax = plt.subplots()
    ax.bar(list(predictions.keys()), list(predictions.values()))
    ax.set_ylim(0, 10)
    ax.set_ylabel("Predicted Rating (1–10)")
    st.pyplot(fig)

else:
    st.info("Enter values and click Predict Rating to start.")
