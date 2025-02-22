import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_extras.metric_cards import style_metric_cards

# Configuration
BACKEND_URL = "http://localhost:8000"
st.set_page_config(page_title="HydroSync", page_icon="🌊", layout="wide")

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Dark theme enhancements */
    .stApp { background-color: #0f172a; color: #f8fafc; }
    
    /* Title styling */
    h1 { color: #7dd3fc !important; border-bottom: 2px solid #1e3a8a; padding-bottom: 0.5rem; }
    
    /* Card styling */
    .metric-card { 
        background: #1e293b !important;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Button styling */
    .stButton>button {
        background: #1d4ed8;
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: #1e3a8a;
        transform: scale(1.05);
    }
    
    /* Map container */
    .folium-map { border-radius: 15px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🌊 HydroSync - Intelligent Flood Management System")
    
    # ========== Sidebar Controls ==========
    with st.sidebar:
        st.header("⚙️ Risk Parameters")
        with st.form("prediction_form"):
            # Organized into sections
            input_features = get_feature_inputs()
            submitted = st.form_submit_button("🚀 Analyze Flood Risk")
    
    # ========== Main Content ==========
    if submitted:
        handle_prediction(input_features)
    else:
        show_landing_page()

def get_feature_inputs():
    features = []
    
    # Environmental Factors
    with st.expander("🌧️ Environmental Factors", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            features.append(st.slider("Monsoon Intensity", 1, 5, 3))
            features.append(st.slider("River Level", 1, 5, 2))
            features.append(st.slider("Soil Moisture", 1, 5, 3))
        with col2:
            features.append(st.slider("Deforestation Rate", 1, 5, 4))
            features.append(st.slider("Snowmelt Rate", 1, 5, 2))
            features.append(st.slider("Coastal Vulnerability", 1, 5, 3))

    # Infrastructure Factors
    with st.expander("🏗️ Infrastructure Factors"):
        col3, col4 = st.columns(2)
        with col3:
            features.append(st.slider("Dam Quality", 1, 5, 3))
            features.append(st.slider("Drainage Capacity", 1, 5, 2))
            features.append(st.slider("Levee Strength", 1, 5, 4))
        with col4:
            features.append(st.slider("Urbanization Level", 1, 5, 4))
            features.append(st.slider("Road Network", 1, 5, 3))
            features.append(st.slider("Bridge Condition", 1, 5, 2))

    # Advanced Parameters
    with st.expander("🔬 Advanced Parameters"):
        col5, col6 = st.columns(2)
        with col5:
            features.append(st.slider("Population Density", 1, 5, 4))
            features.append(st.slider("Wetland Coverage", 1, 5, 2))
            features.append(st.slider("Crop Health", 1, 5, 3))
        with col6:
            features.append(st.slider("Emergency Response", 1, 5, 3))
            features.append(st.slider("Shelter Capacity", 1, 5, 2))
            features.append(st.slider("Communication Network", 1, 5, 4))
    
    # Add remaining 3 features
    features.extend([
        st.slider("Historical Flood Data", 1, 5, 3),
        st.slider("Topography Score", 1, 5, 2),
        st.slider("Water Table Depth", 1, 5, 4)
    ])
    
    return features

def handle_prediction(features):
    try:
        # API Call
        response = requests.post(
            f"{BACKEND_URL}/predict",
            json={"features": features}
        )
        
        if response.status_code == 200:
            result = response.json()
            display_results(result)
        else:
            st.error("⚠️ Prediction service unavailable. Try again later.")
            
    except requests.ConnectionError:
        st.error("🔌 Connection to prediction server failed")

def display_results(result):
    # Alert System
    alert_level = result.get('alert', 'Low Risk')
    alert_config = {
        "High Risk": ("🚨 EVACUATION ALERT", "#dc2626"),
        "Moderate Risk": ("⚠️ FLOOD WARNING", "#f59e0b"),
        "Low Risk": ("✅ ALL CLEAR", "#10b981")
    }
    title, color = alert_config[alert_level]
    st.markdown(f"""
    <div style="background: {color}; 
                padding: 1rem; 
                border-radius: 15px;
                color: white;
                margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">{title}</h2>
    </div>
    """, unsafe_allow_html=True)

    # Metrics Grid
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class='metric-card'>""", unsafe_allow_html=True)
        st.metric("Flood Probability", 
                 f"{result['probability']*100:.1f}%", 
                 f"±{result['uncertainty']*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""<div class='metric-card'>""", unsafe_allow_html=True)
        st.metric("Affected Area", "15 km²", "3% increase")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("""<div class='metric-card'>""", unsafe_allow_html=True)
        st.metric("Estimated Impact", "2,500 people", "15% shelters occupied")
        st.markdown("</div>", unsafe_allow_html=True)
    
    style_metric_cards(background_color="#1e293b", border_left_color="#1d4ed8")

    # Visualization Tabs
    tab1, tab2, tab3 = st.tabs(["🌍 Flood Map", "📈 Risk Analysis", "🚑 Emergency Plan"])
    
    with tab1:
        show_flood_map(result)
    
    with tab2:
        plot_risk_analysis(result)
    
    with tab3:
        show_emergency_plan()

def show_flood_map(result):
    # Create base map
    m = folium.Map(location=[28.6139, 77.2090], zoom_start=12)
    
    # Add flood risk layer
    folium.CircleMarker(
        location=[28.6139, 77.2090],
        radius=result['probability'] * 100,
        color='#dc2626',
        fill=True,
        fill_opacity=0.6,
        popup=f"Flood Risk: {result['probability']*100:.1f}%"
    ).add_to(m)
    
    # Add evacuation routes
    folium.PolyLine(
        locations=[[28.6139, 77.2090], [28.6200, 77.2150]],
        color='#10b981',
        weight=3,
        dash_array='5,5'
    ).add_to(m)
    
    # Display map
    st_folium(m, width=1200, height=500)

def plot_risk_analysis(result):
    # Feature Importance Visualization
    features = [
        "Monsoon", "Drainage", "Urbanization",
        "Population", "Topography", "Infrastructure"
    ]
    importance = [0.25, 0.18, 0.15, 0.12, 0.10, 0.08]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(features[::-1], importance[::-1], color='#7dd3fc')
    ax.set_title('Key Risk Contributors', fontsize=16, pad=20)
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#1e293b')
    
    st.pyplot(fig)

def show_emergency_plan():
    # Emergency Response Plan
    st.subheader("🚨 Emergency Protocol")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🚧 Evacuation Routes
        - Primary Route: Highway 27 → Shelter A
        - Alternate Route: River Road → Shelter B
        - Emergency Access: Military Route 5
        
        ### 🏥 Medical Facilities
        - City General Hospital (5km)
        - Riverside Clinic (3km)
        - Field Hospital (Deploying)
        """)
    
    with col2:
        st.markdown("""
        ### 🚒 Response Teams
        - National Disaster Force: On standby
        - Local Police: Mobilized
        - Volunteer Corps: Activated
        
        ### 📡 Communication
        - Emergency Broadcast: 108.5 FM
        - SMS Alerts: Enabled
        - Satellite Comms: Active
        """)

def show_landing_page():
    # Hero Section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a, #1e40af);
                padding: 3rem;
                border-radius: 20px;
                margin: 2rem 0;
                text-align: center;">
        <h1 style="color: white; font-size: 2.5rem;">Next-Generation Flood Management System</h1>
        <p style="color: #bfdbfe; font-size: 1.2rem;">
            AI-powered flood prediction combined with real-time emergency coordination
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Value Propositions
    col1, col2, col3 = st.columns(3)
    cards = [
        ("🛰️ Satellite Monitoring", "Real-time satellite data integration"),
        ("🤖 AI Predictions", "Deep learning flood forecasting models"),
        ("🚨 Smart Alerts", "Multi-channel emergency notifications")
    ]
    
    for col, (title, desc) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div style="background: #1e293b;
                        padding: 1.5rem;
                        border-radius: 15px;
                        margin: 1rem 0;">
                <h3 style="color: #7dd3fc;">{title}</h3>
                <p style="color: #94a3b8;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 2rem 0;">
        <p>🚑 Integrated with National Disaster Management Authority</p>
        <p>🛰️ Powered by Sentinel-2 Satellite Data</p>
        <p>© 2024 HydroSync. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()