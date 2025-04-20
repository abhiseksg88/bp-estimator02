# Blood Pressure Estimation App

A real-time web app that uses webcam-based rPPG or PPG input to estimate:
- Systolic and Diastolic Blood Pressure
- Heart Rate Variability (HRV)
- SpO₂ (approx)
- Shows waveform, filters, and allows calibration

## Features
- Real-time camera input for rPPG
- Optional external PPG input
- Butterworth / Moving Average filters
- Baseline calibration
- Visualization of waveform

## How to Run
1. Upload this repo to [https://share.streamlit.io](https://share.streamlit.io)
2. Select `bp_app.py` as the main file
3. Allow camera permissions if using rPPG mode
