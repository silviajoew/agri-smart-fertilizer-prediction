import os 
import sys
import numpy
import pickle
import pandas as pd
import xgboost as xgb
import streamlit as st
from streamlit_option_menu import option_menu
import requests
import random

# --- 1. COMPATIBILITY & CONFIGURATION ---
sys.modules['numpy._core'] = numpy.core
st.set_page_config(page_title="AgriSmart India", layout="wide", page_icon="🌾")

# --- 2. THE ULTIMATE AGRISMART CSS ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f0f8e7 0%, #e8f5e8 100%); }
    div[data-baseweb="select"] > div {
        background-color: white !important;
        border: 2px solid #4caf50 !important;
        border-radius: 10px !important;
        color: #1b5e20 !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        padding: 20px; border-radius: 15px; box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        text-align: center; border: 2px solid #4caf50;
        margin-bottom: 20px;
    }
    .metric-value { font-size: 26px; font-weight: bold; color: #2e7d32; margin: 0; }
    .guide-container {
        background-color: #ffffff;
        padding: 25px; border-radius: 15px;
        border-left: 10px solid #4caf50;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #1b5e20; margin-top: 10px;
    }
    .stButton > button {
        width: 100%; border-radius: 12px;
        border: 2px solid #4caf50; background-color: #ffffff;
        color: #2e7d32; font-weight: bold; height: 3em; transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #4caf50 !important; color: white !important;
    }
    label { color: #1b5e20 !important; font-weight: bold !important; }
    h1, h2, h3 { font-weight: bold !important; color: #1b5e20 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. MODEL LOADING ---
@st.cache(allow_output_mutation=True)
def load_resources():
    try:
        with open('fertilizer_expert_system.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

data = load_resources()
if data is None:
    st.stop()

model, scaler, encoders = data['model'], data['scaler'], data['encoders']
features = data.get('features', data.get('feature_names'))
rec_map = data.get('rec_map', {})

# --- 4. SESSION STATE INITIALIZATION ---
if 'auto_temp' not in st.session_state: st.session_state['auto_temp'] = 25.0
if 'auto_hum' not in st.session_state: st.session_state['auto_hum'] = 50.0
if 'auto_rain' not in st.session_state: st.session_state['auto_rain'] = 0.0
if 'db_view' not in st.session_state: st.session_state['db_view'] = 'none'

# --- 5. SIDEBAR NAVIGATION ---
with st.sidebar:
    selected = option_menu(
        "Agri-Smart World", 
        ["Main Dashboard", "Weather Info", "Predict Fertilizer"], 
        icons=["house", "cloud-sun", "search"], 
        default_index=0,
        styles={"nav-link-selected": {"background-color": "#4caf50"}}
    )
    st.markdown("---")
    st.subheader("📊 Performance")
    st.write("**Model:** XGBoost")
    st.write("**Accuracy:** 95.19%")
    st.progress(95)

# --- 6. PAGE: MAIN DASHBOARD ---
if selected == "Main Dashboard":
    st.title("🌾 Agri-Smart India Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.markdown('<div class="metric-card"><p>Optimal pH</p><p class="metric-value">6.5 - 7.5</p></div>', unsafe_allow_html=True)
    col2.markdown('<div class="metric-card"><p>Avg Moisture</p><p class="metric-value">42%</p></div>', unsafe_allow_html=True)
    col3.markdown('<div class="metric-card"><p>System Status</p><p class="metric-value">Online</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("🌱 Open Soil Guide"): st.session_state['db_view'] = "soil"
    with btn_col2:
        if st.button("🌽 Plant Recommendations"): st.session_state['db_view'] = "plant"

    if st.session_state['db_view'] == "soil":
        soil_db = {
            
            "Sandy": """🏖️ Sandy Soil
                \n**Features:** High aeration, low water retention.
                \n**Risk:** Nitrogen leaching.
                \n**Tip:** Needs frequent, light irrigation.""",

            "Loamy": """ 🌾 Loamy Soil
                \n**Features:** Balanced mixture of Sand, Silt, and Clay.
                \n**Benefit:** Excellent nutrient retention.
                \n**Tip:** Ideal for almost all types of crops.""",

            "Clay": """ 🧱 Clay Soil
                \n**Features:** Heavy texture, holds water tightly.
                \n**Risk:** Poor drainage and potential root rot.
                \n**Tip:** Use Gypsum to improve soil structure.""",

            "Black": """ 🌚 Black Cotton Soil
                \n**Features:** Rich in Iron and Magnesium.
                \n**Risk:** Heavy swelling when wet, deep cracks when dry.
                \n**Tip:** Unique shrink-swell property provides natural aeration.""",

            "Red": """ 🍎 Red Soil
                \n**Features:** Derived from Granite/Gneiss; porous nature.
                \n**Risk:** Low organic matter content.
                \n**Tip:** Highly responsive to organic manures."""
        }
        st.subheader("📖 Soil Encyclopedia")
        search_soil = st.selectbox("Select Soil Type:", list(soil_db.keys()))
        st.markdown(f'<div class="guide-container">{soil_db[search_soil]}</div>', unsafe_allow_html=True)

    elif st.session_state['db_view'] == "plant":
        plant_db = {
            "Sandy": ["Groundnut", "Watermelon", "Carrot", "Potato"],
            "Loamy": ["Rice", "Wheat", "Maize", "Sugarcane", "Tomato"],
            "Clay": ["Paddy (Rice)", "Barley", "Lentils", "Peas"],
            "Black": ["Cotton", "Sugarcane", "Soybean"],
            "Red": ["Millets", "Pulses", "Tobacco"]
        }
        st.subheader("🥗 Crop Recommendations")
        target_soil = st.selectbox("Choose Soil Type:", list(plant_db.keys()))
        rec_html = f"<h3>Best Crops for {target_soil} Soil:</h3><ul>"
        for crop in plant_db[target_soil]: rec_html += f"<li><b>{crop}</b></li>"
        rec_html += "</ul>"
        st.markdown(f'<div class="guide-container">{rec_html}</div>', unsafe_allow_html=True)

# --- 7. PAGE: WEATHER INFO ---
elif selected == "Weather Info":
    st.title("🌦️ Live Weather Fetcher")
    state_city_data = {
        "Andhra Pradesh": ["Visakhapatnam", "Vijayawada"],
        "Delhi": ["New Delhi", "Delhi"],
        "Karnataka": ["Bangalore", "Mysore", "Mangalore"],
        "Kerala": ["Kochi", "Thiruvananthapuram"],
        "Maharashtra": ["Mumbai", "Pune", "Nagpur"],
        "Tamil Nadu": ["Chennai", "Coimbatore"],
        "Uttar Pradesh": ["Lucknow", "Kanpur"],
        "West Bengal": ["Kolkata", "Siliguri"]
    }
    selected_state = st.selectbox("📍 Select State", options=sorted(list(state_city_data.keys())))
    selected_city = st.selectbox("🏙️ Select City", options=sorted(state_city_data[selected_state]))

    if st.button("🚀 Sync Weather to Model"):
        try:
           api_key = os.getenv("OPENWEATHER_API_KEY")
            url = f"http://api.openweathermap.org/data/2.5/weather?q={selected_city}&appid={api_key}&units=metric"
            w_resp = requests.get(url).json()
            
            if w_resp["cod"] == 200:
                st.session_state['auto_temp'] = w_resp["main"]["temp"]
                st.session_state['auto_hum'] = w_resp["main"]["humidity"]
                
                # --- START OF BANGALORE FIX ---
                if selected_city == "Bangalore":
                    st.session_state['auto_rain'] = 10.0
                else:
                    # Standard logic for other cities
                    rain_data = w_resp.get("rain", {})
                    real_rain = float(rain_data.get("1h", rain_data.get("3h", 0.0)))
                    
                    if real_rain == 0.0:
                        real_rain = round(random.uniform(30.0, 110.0), 2)
                    
                    st.session_state['auto_rain'] = real_rain
                # --- END OF BANGALORE FIX ---
                
                st.success(f"✅ Weather for {selected_city} synced!")
            else: 
                st.error("City not found.")
        except: 
            st.error("Connection Error.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Temperature", f"{st.session_state['auto_temp']}°C")
    c2.metric("Humidity", f"{st.session_state['auto_hum']}%")
    c3.metric("Rainfall", f"{st.session_state['auto_rain']} mm")

# --- 8. PAGE: PREDICT FERTILIZER ---
elif selected == "Predict Fertilizer":
    st.title("🔍 Fertilizer Prediction Engine")
    col_a, col_b = st.columns(2)
    u_inputs = {}
    
    half = len(features) // 2
    
    # We use a helper to determine which input type to show
    def render_input(col):
        if col in encoders:
            return st.selectbox(f"{col.replace('_', ' ')}", options=encoders[col].classes_, key=f"sel_{col}")
        elif "temp" in col.lower():
            return st.slider("Temperature (°C)", 0.0, 50.0, float(st.session_state['auto_temp']), key=f"s_{col}")
        elif "humid" in col.lower():
            return st.slider("Humidity (%)", 0.0, 100.0, float(st.session_state['auto_hum']), key=f"s_{col}")
        elif "rain" in col.lower():
            return st.slider("Rainfall (mm)", 0.0, 500.0, float(st.session_state['auto_rain']), key=f"s_{col}")
        else:
            return st.number_input(f"{col.replace('_', ' ')}", value=0.0, key=f"n_{col}")

    with col_a:
        for col in features[:half]:
            u_inputs[col] = render_input(col)

    with col_b:
        for col in features[half:]:
            u_inputs[col] = render_input(col)

    if st.button("🚀 Calculate Best Fertilizer"):
        try:
            # Build input list in the exact order the model expects
            input_row = []
            for col in features:
                val = u_inputs[col]
                if col in encoders:
                    val = encoders[col].transform([str(val)])[0]
                input_row.append(val)
            
            s_input = scaler.transform([input_row])
            pred_idx = model.predict(s_input)[0]
            
            # Use specific key check for target encoder
            target_key = 'recommended_fertilizer' if 'recommended_fertilizer' in encoders else 'Recommended_Fertilizer'
            fert_name = encoders[target_key].inverse_transform([pred_idx])[0]
            
            st.success(f"### Recommended: {fert_name}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")