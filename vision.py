import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model
from sklearn.preprocessing import StandardScaler


@st.cache_resource
def load_models():
    ann = load_model("ANN.pkl")
    with open("RF.pkl", "rb") as f:
        rf = pickle.load(f)
    with open("XGB.pkl", "rb") as f:
        xgb = pickle.load(f)
    return ann, rf, xgb


def get_scaler_stats():
    means = np.array([250, 100, 50, 150, 10, 800, 600, 28])
    stds = np.array([100, 80, 40, 30, 7, 150, 120, 50])
    return means, stds


FEATURES = [
    'cement', 'blast_furnace_slag', 'fly_ash', 'water',
    'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age'
]


st.set_page_config(page_title="Concrete Strength Predictor", layout="centered")
st.title("Concrete Strength Predictor")
st.markdown("Enter the concrete mix properties below:")

user_input = []
defaults = [250, 100, 50, 150, 10, 800, 600, 28]
mins = [0, 0, 0, 50, 0, 600, 400, 1]
maxs = [600, 300, 200, 250, 35, 1200, 1000, 365]

for i, feature in enumerate(FEATURES):
    val = st.number_input(f"{feature.replace('_', ' ').title()}", min_value=float(mins[i]), max_value=float(maxs[i]), value=float(defaults[i]))
    user_input.append(val)

if st.button("🔮 Predict Strength"):
    ann_model, rf_model, xgb_model = load_models()
    X_input = np.array(user_input).reshape(1, -1)

   
    means, stds = get_scaler_stats()
    X_scaled = (X_input - means) / stds

    ann_pred = ann_model.predict(X_scaled)[0][0]
    rf_pred = rf_model.predict(X_input)[0]
    xgb_pred = xgb_model.predict(X_input)[0]

    st.subheader(" Predicted Compressive Strength (MPa)")
    st.write(f"**Artificial Neural Network:** {ann_pred:.2f}")
    st.write(f"**Random Forest:** {rf_pred:.2f}")
    st.write(f"**XGBoost:** {xgb_pred:.2f}")

    
    st.subheader(" Model Comparison")
    fig, ax = plt.subplots()
    models = ['ANN', 'Random Forest', 'XGBoost']
    predictions = [ann_pred, rf_pred, xgb_pred]
    ax.bar(models, predictions, color=['blue', 'green', 'orange'])
    ax.set_ylabel("Strength (MPa)")
    ax.set_ylim([0, max(predictions)*1.2])
    st.pyplot(fig)

st.markdown("---")
st.caption("Developed for major project — Concrete Compressive Strength Predictor")
