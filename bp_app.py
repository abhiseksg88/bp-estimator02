import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import av
import cv2
from scipy.signal import butter, filtfilt
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load trained LightGBM models
sbp_model = joblib.load("models/real_sbp_model.pkl")
dbp_model = joblib.load("models/real_dbp_model.pkl")

calibration_ppg = []

# Signal processing functions
def butterworth_filter(data, lowcut=0.5, highcut=3.5, fs=30.0, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def extract_hrv(ppg_waveform):
    ibi = np.diff(np.array(ppg_waveform))
    if len(ibi) < 2:
        return 0.0
    return np.sqrt(np.mean(np.square(np.diff(ibi))))

def estimate_spo2(ppg):
    ac = np.std(ppg)
    dc = np.mean(ppg)
    if dc == 0:
        return 0.0
    ratio = ac / dc
    spo2 = 110 - 25 * ratio
    return round(np.clip(spo2, 85, 100), 1)

def predict_bp(hr, age, bmi, rr, ppg, baseline_ppg=None):
    hrv = extract_hrv(ppg)
    features = np.array([[hr, rr, age, bmi, hrv]])
    if baseline_ppg:
        baseline_hrv = extract_hrv(baseline_ppg)
        hrv_diff = hrv - baseline_hrv
        features[0, -1] += hrv_diff
    sbp = sbp_model.predict(features)[0]
    dbp = dbp_model.predict(features)[0]
    return round(sbp, 1), round(dbp, 1), 0.95

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.signal = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        roi = img[100:200, 100:200]  # Simple center face ROI
        green_avg = np.mean(roi[:, :, 1])
        self.signal.append(green_avg)
        self.frame_count += 1
        return img

# --- Streamlit App UI ---
st.set_page_config(page_title="BP Estimator", layout="centered")
st.title("🩺 Blood Pressure Estimation App")

st.markdown("Choose your input method:")
mode = st.radio("Select Input Type", ["Use Camera (rPPG)", "Use External Sensor (PPG)"])

hr = st.number_input("Heart Rate (BPM)", 30, 200, 75)
age = st.number_input("Age", 1, 120, 35)
bmi = st.number_input("BMI", 10.0, 50.0, 22.5)
rr = st.number_input("Respiratory Rate (breaths/min)", 5, 40, 16)

filter_option = st.selectbox("Signal Filter", ["None", "Butterworth", "Moving Average"])

ppg_waveform = []
calibration_data = st.session_state.get("calibration_data", [])

if mode == "Use Camera (rPPG)":
    st.markdown("**Activating webcam for rPPG signal collection...**")
    ctx = webrtc_streamer(key="rppg", video_transformer_factory=VideoTransformer)

    if ctx.video_transformer:
        if st.button("🔧 Calibrate Me"):
            calibration_ppg = ctx.video_transformer.signal.copy()
            st.session_state.calibration_data = calibration_ppg
            st.success("Baseline signal captured successfully.")

        if st.button("🔍 Analyze rPPG Signal"):
            ppg_waveform = ctx.video_transformer.signal

elif mode == "Use External Sensor (PPG)":
    ppg_input = st.text_area("Paste PPG waveform (comma-separated)", "0.01, 0.03, 0.04, 0.06, 0.05")
    try:
        ppg_waveform = [float(val.strip()) for val in ppg_input.split(",") if val.strip()]
    except Exception:
        st.error("❌ Invalid PPG input.")

if ppg_waveform:
    if filter_option == "Butterworth":
        ppg_waveform = butterworth_filter(ppg_waveform)
    elif filter_option == "Moving Average":
        ppg_waveform = moving_average(ppg_waveform)

    if st.button("🎯 Predict Blood Pressure"):
        try:
            sbp, dbp, conf = predict_bp(hr, age, bmi, rr, ppg_waveform, baseline_ppg=calibration_data if calibration_data else None)
            spo2 = estimate_spo2(ppg_waveform)

            st.success(f"🧠 Systolic BP: {sbp} mmHg")
            st.success(f"🧠 Diastolic BP: {dbp} mmHg")
            st.success(f"🫁 Estimated SpO₂: {spo2}%")
            st.info(f"📊 Model Confidence: {conf * 100:.1f}%")

            st.markdown("### 📈 Filtered PPG Signal")
            fig, ax = plt.subplots()
            ax.plot(ppg_waveform, color='blue')
            ax.set_title("PPG / rPPG Waveform")
            st.pyplot(fig)

            if calibration_data:
                st.markdown("### 🧪 Calibration Baseline")
                fig2, ax2 = plt.subplots()
                ax2.plot(calibration_data, color='green')
                ax2.set_title("Baseline PPG")
                st.pyplot(fig2)

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

