import streamlit as st
import os
import joblib
import pandas as pd
from src.vision_engine import extract_universal_screentime

# --- 1. THEME & STYLING ---
st.set_page_config(page_title="Detox Detach AI", page_icon="🌱", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f0f4f1; color: #1e3d33; } /* Soft Green Detox Theme */
    .stButton>button {
        border-radius: 20px;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
    }
    .stButton>button:hover { background-color: #45a049; }
    .status-card {
        padding: 25px;
        border-radius: 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_exist_ok=True)

# --- 2. ASSETS & LOGIC ---
@st.cache_resource
def load_assets():
    return joblib.load("models/addiction_model.pkl")

model = load_assets()

def get_detox_status(risk_level):
    # PLANT STATUS & ADVICE
    if risk_level == 0: # Low Risk
        return "Flourishing Tree", "🌲", "Your digital forest is healthy! Keep nurturing your focus.", "green"
    elif risk_level == 1: # Medium Risk
        return "Thirsty Sprout", "🌱", "Your focus is starting to wilt. Give it some screen-free time.", "orange"
    else: # High Risk
        return "Wilted Leaf", "🍂", "Your digital plant is dying. Immediate detox required!", "red"

# --- 3. DASHBOARD ---
st.title("🌿 Detox Detach: Focus Growth Engine")
st.write("Transform your digital habits into a thriving ecosystem.")

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("📸 Submit Your Digital Signature")
    st.caption("Upload screenshots of your Screen Time or Digital Wellbeing pages.")
    uploaded_files = st.file_uploader("", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    
    if st.button("START DETOX ANALYSIS"):
        if uploaded_files:
            with st.spinner("Gemini Vision AI is analyzing your habits..."):
                temp_paths = []
                if not os.path.exists("uploads"): os.makedirs("uploads")
                for file in uploaded_files:
                    path = os.path.join("uploads", file.name)
                    with open(path, "wb") as f: f.write(file.getbuffer())
                    temp_paths.append(path)

                # Day 3 Logic
                extracted = extract_universal_screentime(temp_paths)

                if "error" in extracted:
                    st.error(extracted['error'])
                else:
                    # Map to model features (Day 2)
                    # Use extracted data for primary features, 1.0 for binary placeholders
                    feature_vector = [[
                        extracted['Daily_Usage_Hours'], extracted['Phone_Checks_Per_Day'], 
                        extracted['Time_on_Social_Media'], 20, 1, 7, 3, 5, 2, 1, 8, 4, 3, 15, 12, 1, 1, 1, 1, 1, 1
                    ]]
                    st.session_state['pred'] = model.predict(feature_vector)[0]
                    st.session_state['data'] = extracted
        else:
            st.warning("Please upload at least one image first.")

with col2:
    if 'pred' in st.session_state:
        plant_name, icon, msg, color = get_detox_status(st.session_state['pred'])
        
        st.markdown(f"""
            <div class="status-card">
                <h1 style='font-size: 100px; margin: 0;'>{icon}</h1>
                <h2 style='color: {color};'>{plant_name}</h2>
                <p style='font-style: italic;'>"{msg}"</p>
            </div>
        """, unsafe_allow_exist_ok=True)
        
        # Display Metrics
        d = st.session_state['data']
        m1, m2, m3 = st.columns(3)
        m1.metric("Screen Hours", f"{d['Daily_Usage_Hours']}h")
        m2.metric("Total Pickups", d['Phone_Checks_Per_Day'])
        m3.metric("Social Media", f"{d['Time_on_Social_Media']}h")
        
        # Advisory Engine
        st.write("---")
        st.subheader("📝 Personalized Growth Plan")
        if st.session_state['pred'] == 2:
            st.error("• Enable Screen Time Downtime at 10 PM\n\n• Delete Social Media apps for 48 hours\n\n• Leave phone in another room during study.")
        elif st.session_state['pred'] == 1:
            st.warning("• Use a 'Zen Mode' app for 30 minutes daily\n\n• Turn off non-human notifications\n\n• Limit social media to 45 mins.")
        else:
            st.success("• Keep doing what you're doing!\n\n• Share your healthy habits with others.")
    else:
        st.markdown("""
            <div style='text-align: center; padding: 50px; border: 2px dashed #ccc; border-radius: 20px;'>
                <p>Upload your data to see your plant's status</p>
            </div>
        """, unsafe_allow_exist_ok=True)

# --- 4. FOOTER ---
st.divider()
st.caption("Detox Detach System | AI Powered Behavioral Health Prototype")