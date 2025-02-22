import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_extras.metric_cards import style_metric_cards

# Page config MUST BE FIRST
st.set_page_config(
    page_title="HydroSync",
    page_icon="🌊",
    layout="wide"
)

# Dark mode CSS
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Titles */
    h1, h2, h3, h4, h5, h6 {
        color: #76B7F9 !important;
    }
    
    /* Input sections */
    .stForm {
        background-color: #1E1E1E !important;
        border-radius: 10px;
        border: 1px solid #2D2D2D;
        padding: 1rem;
    }
    
    /* Sliders */
    .stSlider [data-baseweb="slider"] {
        background: #2D2D2D;
    }
    
    /* Buttons */
    .stButton>button {
        background: #1F77B4;
        color: white;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: #1668A1;
        transform: scale(1.05);
    }
    
    /* Tabs */
    [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    [data-baseweb="tab"] {
        background: #1E1E1E !important;
        color: #FAFAFA !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
    }
    
    [data-baseweb="tab"][aria-selected="true"] {
        background: #1F77B4 !important;
    }
</style>
""", unsafe_allow_html=True)

# ========== Page Content ==========
st.markdown("# 🌊 FloodAI: Intelligent Flood Prediction System")

# ========== Sidebar ==========
with st.sidebar:
    st.header("⚙️ Risk Parameters")
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Environmental Factors")
            monsoon = st.slider("Monsoon Intensity", 1, 5, 3, 
                              help="Intensity of monsoon rains")
            drainage = st.slider("Drainage Quality", 1, 5, 3,
                               help="Effectiveness of drainage systems")
            urbanization = st.slider("Urbanization Level", 1, 5, 3,
                                   help="Degree of urban development")
            
        with col2:
            st.subheader("Infrastructure Factors")
            dams = st.slider("Dam Quality", 1, 5, 3,
                           help="Condition of dam infrastructure")
            siltation = st.slider("Siltation Level", 1, 5, 3,
                                help="Sediment accumulation in rivers")
            wetlands = st.slider("Wetland Preservation", 1, 5, 3,
                               help="Percentage of wetlands remaining")

        # Advanced parameters
        with st.expander("🔧 Advanced Parameters"):
            col3, col4 = st.columns(2)
            with col3:
                deforestation = st.slider("Deforestation Rate", 1, 5, 3)
                climate_change = st.slider("Climate Change Impact", 1, 5, 3)
                
            with col4:
                population = st.slider("Population Density", 1, 5, 3)
                preparedness = st.slider("Disaster Preparedness", 1, 5, 3)

        submitted = st.form_submit_button("🔍 Predict Flood Risk")

# ========== Main Content ==========
if submitted:
    # Mock response for demo
    result = {
        "probability": 0.82,
        "uncertainty": 0.05,
        "alert": "High Risk"
    }

    # Alert System
    if result['alert'] == "High Risk":
        st.error("🚨 EMERGENCY ALERT: Immediate evacuation recommended!")
    elif result['alert'] == "Moderate Risk":
        st.warning("⚠️ WARNING: Flood conditions developing")
    else:
        st.success("✅ No immediate flood threat detected")

    # Metrics Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Flood Probability", 
                f"{result['probability']*100:.1f}%", 
                delta=f"±{result['uncertainty']*100:.1f}%")
    with col2:
        st.metric("Affected Area", "25 km²", "-12% from baseline")
    with col3:
        st.metric("Response Time", "45 mins", "8 mins faster than average")
    style_metric_cards(background_color="#1E1E1E", border_left_color="#1F77B4")

    # Visualization Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Risk Analysis", "🗺️ Live Map", "📜 Action Plan"])
    
    with tab1:
        # Feature Importance Chart
        features = ["Monsoon", "Drainage", "Urban", "Dams", "Siltation"]
        importance = [0.32, 0.25, 0.18, 0.15, 0.10]
        
        fig, ax = plt.subplots()
        ax.barh(features[::-1], importance[::-1], color="#1F77B4")
        ax.set_facecolor('#0E1117')
        fig.patch.set_facecolor('#0E1117')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        st.pyplot(fig)
    
    with tab2:
        # Interactive Map with Dark Theme
        df = pd.DataFrame({
            "lat": [28.6139, 28.6200, 28.6000],
            "lon": [77.2090, 77.2150, 77.2000],
            "risk": [result['probability'], 0.65, 0.45]
        })
        st.map(df, color="#1F77B4", size="risk")
    
    with tab3:
        st.subheader("Recommended Actions")
        st.markdown("""
        <div style="background-color: #1E1E1E; padding: 1rem; border-radius: 10px;">
            <p>🚑 Activate emergency response teams</p>
            <p>🏠 Open shelters in safe zones</p>
            <p>🛡️ Deploy flood defenses</p>
            <p>🚧 Divert traffic from high-risk areas</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Hero Section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1E1E1E, #2D2D2D); 
                padding: 2rem; 
                border-radius: 15px;
                margin: 2rem 0;">
        <h2 style="color: #76B7F9;">Next-Gen Flood Prediction</h2>
        <p style="color: #CCCCCC;">AI-powered risk assessment combining 20+ environmental and infrastructure factors</p>
    </div>
    """, unsafe_allow_html=True)

    # Value Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background-color: #1E1E1E; 
                    padding: 1.5rem; 
                    border-radius: 10px;
                    border: 1px solid #2D2D2D;
                    margin: 1rem 0;">
            <h3 style="color: #76B7F9;">🕒 Real-Time Analysis</h3>
            <p style="color: #CCCCCC;">Continuous monitoring of risk factors</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #1E1E1E; 
                    padding: 1.5rem; 
                    border-radius: 10px;
                    border: 1px solid #2D2D2D;
                    margin: 1rem 0;">
            <h3 style="color: #76B7F9;">🧠 AI Forecasting</h3>
            <p style="color: #CCCCCC;">Deep learning with uncertainty estimates</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color: #1E1E1E; 
                    padding: 1.5rem; 
                    border-radius: 10px;
                    border: 1px solid #2D2D2D;
                    margin: 1rem 0;">
            <h3 style="color: #76B7F9;">📡 Instant Alerts</h3>
            <p style="color: #CCCCCC;">Multi-channel emergency notifications</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666666;">
    <p>🚑 Integrated Emergency Response | 🛰️ Satellite Data Integration | 📊 Real-time Analytics</p>
    <p>© 2024 FloodAI. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)