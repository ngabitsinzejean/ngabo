import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import io

# ----------------------------------------------------
# 0. STREAMLIT CONFIGURATION
# ----------------------------------------------------
st.set_page_config(page_title="Umuhinzi AI Platform", layout="wide", page_icon="🇷🇼")

# ----------------------------------------------------
# 1. GENERATE ADVANCED SYNTHETIC DATASET (NISR Aligned)
# ----------------------------------------------------
@st.cache_data
def load_advanced_nisr_data():
    np.random.seed(42)
    districts = ['Musanze', 'Nyagatare', 'Bugesera', 'Huye', 'Rubavu', 'Gatsibo']
    crops = ['Irish Potato', 'Maize', 'Beans', 'Rice', 'Sorghum']
    
    # Coordinates mapping for Rwanda Districts to simulate map view
    coords = {
        'Musanze': [-1.5003, 29.6343],
        'Nyagatare': [-1.4286, 30.3251],
        'Bugesera': [-2.2227, 30.1344],
        'Huye': [-2.5161, 29.7401],
        'Rubavu': [-1.6931, 29.4148],
        'Gatsibo': [-1.5972, 30.4578]
    }
    
    data = []
    for _ in range(1000):
        district = np.random.choice(districts)
        crop = np.random.choice(crops)
        lat, lon = coords[district]
        
        if district == 'Musanze' and crop == 'Irish Potato':
            rainfall = np.random.uniform(1200, 1600)
            yield_tonnes = np.random.uniform(18, 25)
            price_per_kg = np.random.uniform(300, 450)
        elif district == 'Nyagatare' and crop == 'Maize':
            rainfall = np.random.uniform(800, 1100)
            yield_tonnes = np.random.uniform(4, 7)
            price_per_kg = np.random.uniform(350, 500)
        else:
            rainfall = np.random.uniform(700, 1400)
            yield_tonnes = np.random.uniform(2, 12)
            price_per_kg = np.random.uniform(400, 800)
            
        data.append([district, crop, round(rainfall, 1), round(yield_tonnes, 2), int(price_per_kg), lat, lon])
        
    return pd.DataFrame(data, columns=['District', 'Crop_Type', 'Avg_Rainfall_mm', 'Yield_Tonnes_Per_HA', 'Market_Price_FRW', 'lat', 'lon'])

df = load_advanced_nisr_data()

# ----------------------------------------------------
# 2. AI MODEL TRAINING ENGINE
# ----------------------------------------------------
@st.cache_resource
def train_ai_model(data):
    le_dist = LabelEncoder()
    le_crop = LabelEncoder()
    
    data_encoded = data.copy()
    data_encoded['District'] = le_dist.fit_transform(data['District'])
    data_encoded['Crop_Type'] = le_crop.fit_transform(data['Crop_Type'])
    
    X = data_encoded[['District', 'Crop_Type', 'Avg_Rainfall_mm']]
    y_yield = data_encoded['Yield_Tonnes_Per_HA']
    y_price = data_encoded['Market_Price_FRW']
    
    model_yield = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y_yield)
    model_price = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y_price)
    
    return model_yield, model_price, le_dist, le_crop

model_yield, model_price, le_dist, le_crop = train_ai_model(df)

# ----------------------------------------------------
# 3. INTERACTIVE DASHBOARD PRESENTATION
# ----------------------------------------------------
st.title("🇷🇼 Umuhinzi AI: Advanced Agritech Forecasting Platform")
st.markdown("### NISR 2026 Big Data Project Framework — *Aligned with NST2 Pillars*")

st.divider()

# Sidebar Setup
st.sidebar.header("⚙️ Configuration Panel")
input_district = st.sidebar.selectbox("Hitamo Akarere (District):", df['District'].unique())
input_crop = st.sidebar.selectbox("Hitamo Igihingwa (Crop Type):", df['Crop_Type'].unique())
input_rainfall = st.sidebar.slider("Ikigereranyo cy'Imvura (Rainfall in mm):", 500, 1800, 1000)

st.sidebar.divider()
st.sidebar.header("💲 Cost Parameters (Ikiguzi)")
cost_per_ha = st.sidebar.number_input("Ikiguzi cy'Imbuto n'Ifumbire kuri Hectare (RWF):", min_value=50000, max_value=1000000, value=250000, step=5000)

# Run Inference Automatically or On-Demand
dist_enc = le_dist.transform([input_district])[0]
crop_enc = le_crop.transform([input_crop])[0]

pred_input = pd.DataFrame([[dist_enc, crop_enc, input_rainfall]], columns=['District', 'Crop_Type', 'Avg_Rainfall_mm'])
pred_yield = model_yield.predict(pred_input)[0]
pred_price = model_price.predict(pred_input)[0]

# --- SECTION 1: METRICS AND ROI CALCULATOR ---
st.header("🎯 Live AI Projections & Financial Estimates (ROI)")
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

total_revenue_per_ha = (pred_yield * 1000) * pred_price
net_profit_per_ha = total_revenue_per_ha - cost_per_ha

with m_col1:
    st.metric(label="📊 Umusaruro Uteganyijwe", value=f"{pred_yield:.2f} Tons/HA")
with m_col2:
    st.metric(label="💰 Igiciro ku Isoko (Estimated)", value=f"{int(pred_price):,} RWF/KG")
with m_col3:
    st.metric(label="📈 Isoko Ryose Tutaruye (Gross/HA)", value=f"{int(total_revenue_per_ha):,} RWF")
with m_col4:
    color_roi = "normal" if net_profit_per_ha > 0 else "inverse"
    st.metric(label="💵 Inyungu Isigaye (Net Profit/HA)", value=f"{int(net_profit_per_ha):,} RWF")

# REPORT DOWNLOAD INTERFACE
report_text = f"""Umuhinzi AI Report Summary
--------------------------
Akarere: {input_district}
Igihingwa: {input_crop}
Imvura: {input_rainfall} mm
Umusaruro Uhanurwa: {pred_yield:.2f} Tons/HA
Igiciro ku Isoko: {int(pred_price)} RWF/KG
Inyungu Nyayo Isigaye: {int(net_profit_per_ha)} RWF/HA
Generated via NISR Big Data Portal Model Stack 2026.
"""
st.download_button(label="📥 Gakura Advisory Report (Text Format)", data=report_text, file_name=f"Umuhinzi_AI_{input_district}.txt", mime="text/plain")

st.divider()

# --- SECTION 2: MAP & GRAPH VISUALIZATIONS ---
g_col1, g_col2 = st.columns([4, 6])

with g_col1:
    st.subheader("🗺️ Geographic Context Map")
    # Dynamic Map Filtering to target selected district marker
    district_geo = df[df['District'] == input_district].head(1)
    st.map(district_geo, latitude='lat', longitude='lon', zoom=9, use_container_width=True)
    st.caption(f"Ikarita yerekana amerekezo y'Akarere ka **{input_district}** muri gahunda y'ubuhinzi.")

with g_col2:
    st.subheader("📊 Interactive Analytical Diagrams")
    tab1, tab2 = st.tabs(["Market Prices (RWF/KG)", "Regional Crop Yield Profiles"])
    
    with tab1:
        crop_filtered = df[df['Crop_Type'] == input_crop]
        price_data = crop_filtered.groupby('District')['Market_Price_FRW'].mean().reset_index()
        fig_price = px.bar(price_data, x='District', y='Market_Price_FRW', 
                           title=f"Ibiciro bya {input_crop} mu Turere Twose",
                           labels={'Market_Price_FRW': 'Average Price (RWF / KG)'},
                           color='Market_Price_FRW', color_continuous_scale='YlOrRd')
        st.plotly_chart(fig_price, use_container_width=True)
        
    with tab2:
        yield_data = df.groupby('Crop_Type')['Yield_Tonnes_Per_HA'].mean().sort_values().reset_index()
        fig_yield = px.bar(yield_data, x='Yield_Tonnes_Per_HA', y='Crop_Type', orientation='h',
                           title="Ikigereranyo cy'Umusaruro ku Gihingwa (Tons/HA)",
                           labels={'Yield_Tonnes_Per_HA': 'Yield (Tonnes per HA)'},
                           color='Yield_Tonnes_Per_HA', color_continuous_scale='Greens')
        st.plotly_chart(fig_yield, use_container_width=True)

st.divider()
st.subheader("📋 Advanced Model Insights (Historical Preview)")
st.dataframe(df.head(10), use_container_width=True)
