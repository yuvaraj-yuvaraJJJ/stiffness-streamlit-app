# AI-Based Stamped Panel Stiffness Predictor

## Overview
Early-stage stiffness evaluation of stamped sheet-metal panels using full CAE
simulations is time-consuming and slows design iteration.

This project presents a **physics-guided, multi-fidelity AI approach** that
provides **instant stiffness and deformation estimates** during early design.

---

## Key Capabilities
- Low-fidelity physics-based stiffness estimation
- High-fidelity AI correction trained on CAE anchors
- Supports multiple materials via Young’s modulus
- Real-time prediction using Streamlit
- Physically consistent and unit-safe pipeline

---

## Methodology

1. **User Inputs**
   - Thickness, length, width
   - Applied load
   - Material (or custom Young’s modulus)

2. **Low-Fidelity Physics**
   - Elastic bending formulation
   - Captures smooth global trends

3. **AI Correction**
   - XGBoost model predicts LF–HF stiffness correction
   - Learns nonlinear geometric effects implicitly

4. **Final Output**
   - AI-corrected stiffness (N/mm)
   - Predicted deformation (mm)
   - Design feasibility feedback

---

## Why Multi-Fidelity AI?
- Physics alone misses nonlinear effects
- Pure AI requires large datasets
- Multi-fidelity combines both efficiently
- Accurate, fast, and industry-aligned

---

## Technologies Used
- Python
- Streamlit
- XGBoost
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

---

## Disclaimer
This tool is intended for **early-stage design screening only** and does not
replace detailed CAE or physical testing.
