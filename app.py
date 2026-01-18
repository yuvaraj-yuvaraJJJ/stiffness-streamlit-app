# ============================================================
# AI-Based Stamped Panel Stiffness Predictor
# Physics-based LF Model + HF-Corrected ML (XGBoost)
# FINAL CLEAN DEPLOYMENT VERSION
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI-Based Stamped Panel Stiffness Predictor",
    layout="centered"
)

# ------------------------------------------------------------
# TITLE & CONTEXT (SHORT, NON-CLUMPY)
# ------------------------------------------------------------
st.title("AI-Based Stamped Panel Stiffness Predictor")

st.markdown(
    "Fast early-stage stiffness and deformation estimation for stamped "
    "sheet-metal panels using **physics-guided AI**."
)

st.divider()

# ------------------------------------------------------------
# LOAD MODEL & SCALER
# ------------------------------------------------------------
@st.cache_resource
def load_model_assets():
    model = joblib.load("stiffness_model_FINAL.joblib")
    scaler = joblib.load("scaler_FINAL.joblib")
    features = scaler.feature_names_in_.tolist()
    return model, scaler, features

model, scaler, trained_features = load_model_assets()

# ------------------------------------------------------------
# SIDEBAR — INPUTS (CLEAN + ENGINEERING-FOCUSED)
# ------------------------------------------------------------
st.sidebar.header("Panel Geometry & Load")

thk_mm = st.sidebar.number_input(
    "Thickness (mm)", 0.5, 10.0, 2.0
)

L_mm = st.sidebar.number_input(
    "Plate Length (mm)", 100.0, 1000.0, 300.0
)

b_mm = st.sidebar.number_input(
    "Plate Width (mm)", 50.0, 500.0, 200.0
)

bead_mm = st.sidebar.number_input(
    "Bead Depth (mm)", 0.0, 20.0, 2.0
)

load_N = st.sidebar.number_input(
    "Applied Load (N)", 100.0, 5000.0, 1000.0
)

material = st.sidebar.selectbox(
    "Material",
    [
        "Mild Steel",
        "AHSS (Advanced High Strength Steel)",
        "Aluminium",
        "Custom Material (User-defined)"
    ]
)

# ------------------------------------------------------------
# MATERIAL PROPERTIES (GPa)
# ------------------------------------------------------------
if material == "Mild Steel":
    E = 210
elif material == "AHSS (Advanced High Strength Steel)":
    E = 210
elif material == "Aluminium":
    E = 70
else:
    E = st.sidebar.number_input(
        "Young’s Modulus (GPa)", 10.0, 300.0, 200.0
    )

# ------------------------------------------------------------
# LOW-FIDELITY PHYSICS MODEL (UNCHANGED)
# ------------------------------------------------------------
thk = thk_mm / 1000.0
L = L_mm / 1000.0
b = b_mm / 1000.0
load = load_N

I = (b * thk**3) / 12.0
delta_LF = (load * L**3) / (48.0 * (E * 1e9) * I)
LF_k = load / delta_LF

# ------------------------------------------------------------
# BUILD ML INPUT
# ------------------------------------------------------------
user_input = {
    "thk": thk_mm,
    "L": L_mm,
    "b": b_mm,
    "bead": bead_mm,
    "load": load_N,
    "E": E,
    "LF_k": LF_k
}

X_user = pd.DataFrame([user_input])
X_user = X_user[trained_features]
X_user_scaled = scaler.transform(X_user)

# ------------------------------------------------------------
# AI CORRECTION
# ------------------------------------------------------------
duser = xgb.DMatrix(X_user_scaled)
delta_k = model.predict(duser)[0]

final_k = LF_k + delta_k
deformation_mm = (load / final_k) * 1000.0

# ------------------------------------------------------------
# RESULTS (CLEAN, ENGINEERING LANGUAGE)
# ------------------------------------------------------------
st.header("Prediction Results")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Low-Fidelity Stiffness (N/m)",
        f"{LF_k:,.2e}"
    )

with col2:
    st.metric(
        "AI-Corrected Stiffness (N/m)",
        f"{final_k:,.2e}"
    )

st.metric(
    "Predicted Deformation (mm)",
    f"{deformation_mm:.3f}"
)

# ------------------------------------------------------------
# FEASIBILITY FEEDBACK (SIMPLE)
# ------------------------------------------------------------
st.subheader("Design Feasibility")

if deformation_mm < 1.0:
    st.success("Design is feasible with low deformation.")
elif deformation_mm < 2.5:
    st.warning("Moderate deformation detected. Optimization recommended.")
else:
    st.error("High deformation detected. Design revision required.")

# ------------------------------------------------------------
# VISUAL COMPARISON (MINIMAL)
# ------------------------------------------------------------
st.subheader("Stiffness Comparison")

fig, ax = plt.subplots()
ax.bar(["LF Prediction", "AI-Corrected"], [LF_k, final_k])
ax.set_ylabel("Stiffness (N/m)")
ax.grid(True)

st.pyplot(fig)

# ------------------------------------------------------------
# COLLAPSIBLE ENGINEERING ASSUMPTIONS (NO CLUTTER)
# ------------------------------------------------------------
with st.expander("Engineering Assumptions"):
    st.markdown("""
- Elastic behavior assumed  
- Identical boundary conditions across all predictions  
- Load applied on base panel surface  
- Geometry-driven stiffness comparison only  
    """)

with st.expander("How Material Is Handled"):
    st.markdown("""
- AI learns **geometry-driven stiffness behavior**
- Material influence is applied analytically using Young’s modulus
- This allows prediction across **any sheet-metal material**
    """)

# ------------------------------------------------------------
# MODEL INFORMATION (SHORT & SAFE)
# ------------------------------------------------------------
st.divider()

st.markdown("""
**Model Summary**

• Physics-based low-fidelity stiffness model  
• AI learns only nonlinear geometric correction  
• Limited CAE anchors used during training  
• Fast, stable, and scalable for early design
""")

st.success(
    "This tool accelerates early design screening before detailed CAE is performed."
)

st.caption(
    "Disclaimer: Intended for early-stage design decision support only. "
    "Not a replacement for detailed CAE or physical validation."
)
