# AI-Based Stamped Panel Stiffness & Deformation Predictor

## Overview
Early-stage stiffness evaluation of stamped sheet metal panels using full CAE
simulations is time-consuming and slows down design iterations.
This project presents a **physics-guided, multi-fidelity AI approach**
to instantly predict stiffness and deformation during early design stages.

The system combines:
- A low-fidelity physics-based analytical model
- A high-fidelity AI correction model (XGBoost)

This ensures fast, reliable, and physically consistent predictions.

---

## Key Features
- Physics-based low-fidelity stiffness calculation
- High-fidelity AI correction using machine learning
- Multi-fidelity learning strategy
- Real-time prediction through a Streamlit web application
- Industry-grade preprocessing using feature scaling
- Deployment-ready ML pipeline

---

## Methodology (How It Works)

1. **User Inputs**
   - Plate geometry (thickness, length, width, bead depth)
   - Applied load
   - Material selection

2. **Low-Fidelity Physics Model**
   - Analytical stiffness computed using beam/plate theory
   - Captures dominant structural behavior

3. **AI-Based Correction**
   - XGBoost model predicts stiffness correction (Δk)
   - Trained on LF–HF residuals
   - Improves accuracy without full CAE simulation

4. **Final Output**
   - AI-corrected stiffness
   - Predicted deformation
   - Design feasibility feedback

---

## Why Multi-Fidelity AI?
- Pure AI models require large datasets
- Pure physics models lack nonlinear accuracy
- Multi-fidelity approach combines the strengths of both
- Faster, more reliable, and data-efficient

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

## File Structure

