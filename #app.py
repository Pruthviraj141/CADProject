mport streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Configuration ---
MODEL_FILES = ['model.pkl', 'rf_model.pkl', 'imputer.pkl']
FEATURE_COLS = ['users_rated', 'minplayers', 'maxplayers', 'playingtime', 'yearpublished', 'minage']

# Function to load models with error handling
def load_assets():
    """Loads all necessary pickled assets (LR, RF, Imputer)."""
    assets = {}
    for filename in MODEL_FILES:
        if not os.path.exists(filename):
            st.error(f"Error: Required file '{filename}' not found.")
            st.warning("Please run `python train_models.py` first to generate the models.")
            return None
    
    try:
        assets['lr'] = pickle.load(open('model.pkl', 'rb'))
        assets['rf'] = pickle.load(open('rf_model.pkl', 'rb'))
        assets['imputer'] = pickle.load(open('imputer.pkl', 'rb'))
        return assets
    except Exception as e:
        st.error(f"Error loading models. Check if the pickle files are valid. Error: {e}")
        return None

# Load assets and check if successful
assets = load_assets()

if assets:
    lr = assets['lr']
    rf = assets['rf']
    imputer = assets['imputer']

    # --- Streamlit UI Setup ---
    st.set_page_config(page_title="Board Game Rating Predictor", layout="centered")

    st.title("🎲 Board Game Rating Predictor")
    st.markdown("Use trained Machine Learning models to predict a board game's average rating based on its properties.")
    st.markdown("---")

    # Model selection
    model_choice = st.sidebar.selectbox(
        "Select Model",
        ["Random Forest", "Linear Regression", "Both"],
        help="Random Forest is typically more accurate for complex relationships."
    )
    
    # --- User Inputs ---
    st.subheader("Game Specifications")
    
    # Organize inputs into two columns for better desktop layout
    col1, col2 = st.columns(2)

    with col1:
        users_rated = st.number_input(
            "1. Users Rated", min_value=0, max_value=100000, value=1000, step=100,
            help="Total number of users who rated the game (0-100,000)."
        )
        minplayers = st.number_input(
            "2. Minimum Players", min_value=1, max_value=20, value=2, step=1,
            help="Minimum number of players required (1-20)."
        )
        maxplayers = st.number_input(
            "3. Maximum Players", min_value=1, max_value=20, value=4, step=1,
            help="Maximum number of players allowed (1-20)."
        )

    with col2:
        playingtime = st.number_input(
            "4. Playing Time (minutes)", min_value=1, max_value=1000, value=60, step=15,
            help="Average time to play (1-1000 minutes)."
        )
        yearpublished = st.number_input(
            "5. Year Published", min_value=1900, max_value=2025, value=2015, step=1,
            help="The year the game was released (1900-2025)."
        )
        minage = st.number_input(
            "6. Minimum Age", min_value=0, max_value=100, value=12, step=1,
            help="Minimum recommended age (0-100)."
        )

    # Prepare input DataFrame
    input_data = [[users_rated, minplayers, maxplayers, playingtime, yearpublished, minage]]
    input_df = pd.DataFrame(input_data, columns=FEATURE_COLS)

    # Handle missing values (Crucial step: use the loaded imputer's transform method)
    # Even though we're using st.number_input, the model expects the imputer step
    # because it was fit on the training data.
    processed_input_df = pd.DataFrame(imputer.transform(input_df), columns=FEATURE_COLS)

    st.markdown("---")
    
    # Predict button
    if st.button("Calculate Predicted Rating", type="primary"):
        predictions = {}
        
        # Make predictions based on selection
        if model_choice in ["Linear Regression", "Both"]:
            lr_prediction = lr.predict(processed_input_df)[0]
            # Ensure rating is between 1 and 10 (ratings are usually 1-10)
            predictions["Linear Regression"] = round(max(1.0, min(10.0, lr_prediction)), 2) 

        if model_choice in ["Random Forest", "Both"]:
            rf_prediction = rf.predict(processed_input_df)[0]
            # Ensure rating is between 1 and 10
            predictions["Random Forest"] = round(max(1.0, min(10.0, rf_prediction)), 2)

        # --- Display Results ---
        st.subheader("🎯 Prediction Results")
        
        # Use columns for nice result display
        result_cols = st.columns(len(predictions))
        
        for i, (model, pred) in enumerate(predictions.items()):
            with result_cols[i]:
                st.metric(
                    label=f"{model} Prediction", 
                    value=f"{pred:.2f}", 
                    delta="Rating out of 10"
                )

        # Optional: Bar chart visualization
        if len(predictions) > 0:
            st.markdown("### Comparison Chart")
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(predictions.keys(), predictions.values(), color=["#2563EB" if k == "Linear Regression" else "#F97316" for k in predictions.keys()])
            
            ax.set_ylim(min(predictions.values()) - 1, max(predictions.values()) + 1)
            ax.set_ylabel("Predicted Rating (1-10)")
            ax.set_title("Model Comparison")
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            # Add value labels on top of bars
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.2f}', ha='center', va='bottom')

            st.pyplot(fig)

