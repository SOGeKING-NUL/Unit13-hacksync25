import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_extras.metric_cards import style_metric_cards

# Configuration
BACKEND_URL = "http://localhost:8000"
st.set_page_config(page_title="HydroSync", page_icon="💧", layout="wide")

# Custom CSS for dark mode theme
st.markdown("""
<style>
    /* Dark theme styling */
    .stApp { background-color: #121212; color: #e0e0e0; }
    
    /* Title styling */
    h1 { color: #bb86fc !important; border-bottom: 2px solid #6200ea; padding-bottom: 0.5rem; text-align: center; font-family: 'Arial', sans-serif; }
    
    /* Card styling */
    .metric-card { 
        background: #1e1e1e !important;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        text-align: center;
    }
    
    /* Button styling */
    .stButton>button {
        background: #3700b3;
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        font-family: 'Arial', sans-serif;
    }
    
    .stButton>button:hover {
        background: #bb86fc;
        transform: scale(1.05);
    }
    
    /* Map container */
    .folium-map { border-radius: 10px; overflow: hidden; }
    
    /* Sidebar styling */
    .stSidebar { background-color: #1e1e1e !important; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("HydroSync - Intelligent Flood Management System")
    
    # ========== Sidebar Controls ==========
    with st.sidebar:
        st.header("Risk Parameters")
        with st.form("prediction_form"):
            input_features = get_feature_inputs()
            submitted = st.form_submit_button("Analyze Flood Risk")
        
        st.header("Additional Settings")
        with st.expander("Advanced Options"):
            st.checkbox("Enable Real-time Updates", value=True)
            st.checkbox("Show Historical Data", value=False)
        
        with st.expander("Data Sources"):
            st.checkbox("Use Satellite Data", value=True)
            st.checkbox("Use IoT Sensor Data", value=True)
    
    # ========== Main Content ==========
    if submitted:
        handle_prediction(input_features)
    else:
        show_landing_page()

def get_feature_inputs():
    features = []
    
    # Environmental Factors
    with st.expander("Environmental Factors", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            features.append(st.slider("Monsoon Intensity", 1, 5, 3))
            features.append(st.slider("River Level", 1, 5, 2))
            features.append(st.slider("Soil Moisture", 1, 5, 3))
        with col2:
            features.append(st.slider("Deforestation Rate", 1, 5, 4))
            features.append(st.slider("Snowmelt Rate", 1, 5, 2))
            features.append(st.slider("Coastal Vulnerability", 1, 5, 3))

    return features

def handle_prediction(features):
    try:
        response = requests.post(
            f"{BACKEND_URL}/predict",
            json={"features": features}
        )
        
        if response.status_code == 200:
            result = response.json()
            display_results(result)
        else:
            st.error("Prediction service unavailable. Try again later.")
            
    except requests.ConnectionError:
        st.error("Connection to prediction server failed")

def display_results(result):
    alert_level = result.get('alert', 'Low Risk')
    alert_config = {
        "High Risk": ("EVACUATION ALERT", "#cf6679"),
        "Moderate Risk": ("FLOOD WARNING", "#ffb74d"),
        "Low Risk": ("ALL CLEAR", "#03dac6")
    }
    title, color = alert_config[alert_level]
    st.markdown(f"""
    <div style="background: {color}; padding: 1rem; border-radius: 10px; color: white; margin-bottom: 2rem; text-align: center;">
        <h2 style="color: white; margin: 0;">{title}</h2>
    </div>
    """, unsafe_allow_html=True)

def show_landing_page():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #3700b3, #6200ea);
                padding: 3rem;
                border-radius: 20px;
                margin: 2rem 0;
                text-align: center;">
        <h1 style="color: white; font-size: 2.5rem;">Next-Generation Flood Management System</h1>
        <p style="color: #bb86fc; font-size: 1.2rem;">
            AI-powered flood prediction combined with real-time emergency coordination
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    cards = [
        ("Satellite Monitoring", "Real-time satellite data integration"),
        ("AI Predictions", "Deep learning flood forecasting models"),
        ("Smart Alerts", "Multi-channel emergency notifications")
    ]
    
    for col, (title, desc) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div style="background: #1e1e1e; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; text-align: center;">
                <h3 style="color: #bb86fc;">{title}</h3>
                <p style="color: #e0e0e0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()
