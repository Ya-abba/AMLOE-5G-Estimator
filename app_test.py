import streamlit as st
import pandas as pd
import joblib
import math
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import numpy as np
import os
import gdown

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="AMLOE Web Platform", page_icon="📡")

# --- Constants & Helper Functions ---
BASE_LAT = 2.927778
BASE_LON = 101.785556


def haversine(lat1, lon1, lat2, lon2):
    """Distance between points in meters."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculates compass bearing (0-360) between two points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_lambda = math.radians(lon2 - lon1)
    y = math.sin(d_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def get_rsrp_color(rsrp):
    """Hex color mapping for RSRP levels."""
    if rsrp >= -80:
        return "#FF0000"  # Red (Excellent)
    elif rsrp >= -90:
        return "#FFA500"  # Orange (Good)
    elif rsrp >= -100:
        return "#FFFF00"  # Yellow (Fair)
    elif rsrp >= -110:
        return "#008000"  # Green (Poor)
    elif rsrp >= -120:
        return "#00FFFF"  # Cyan (Weak)
    else:
        return "#0000FF"  # Blue (No Signal)


# --- Load Data and Model (Decoupled Storage Logic) ---
@st.cache_resource
def load_resources():
    model_path = "rsrp_model.pkl"
    csv_path = "5G_BS_UKM.csv"

    file_id = '1U4hKKfedicyfJxvEXVV-uevY-B0SNesH/view?usp=sharing'
    url = f'https://drive.google.com/uc?id={file_id}'

    try:
        if not os.path.exists(model_path):
            with st.spinner("Downloading 4.5GB model from Google Drive... Please wait."):
                # quiet=False allows you to see download progress in the logs
                gdown.download(url, model_path, quiet=False)

        # 2. Load the CSV and the large Model
        df = pd.read_csv(csv_path)
        model = joblib.load(model_path)
        return df, model
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.info("Check if the model file on Google Drive is shared with 'Anyone with the link'.")
        return None, None


bs_df, model = load_resources()

# --- Application Layout ---
st.title("📡 AMLOE: Aerial Machine Learning Based Online Coverage Estimator")
st.markdown("Predicting **5G Downlink RSRP** using Random Forest Regression on Drone-collected datasets.")

col1, col2 = st.columns([1, 2.5])

with col1:
    st.header("⚙️ Settings")

    if bs_df is not None:
        st.subheader("1. Transmitter (Tx)")
        selected_station = st.selectbox("Select Base Station", bs_df['Location'].unique())
        row = bs_df[bs_df['Location'] == selected_station].iloc[0]

        lat0 = float(row['Latitude'])
        lon0 = float(row['Longtitude'])
        h_tx = float(row['Base station Height (m)'])
        tilt_tx = float(row['Tilt (°)'])
        az_tx = int(row['Azimuth (°)'])

        with st.expander("✅ BS Data Loaded", expanded=True):
            st.write(f"**Height:** {h_tx}m | **Tilt:** {tilt_tx}°")
            st.write(f"**Coordinates:** {lat0}, {lon0}")

        selected_azimuth = st.selectbox("Antenna Azimuth (°)", [az_tx])
        freq = st.radio("Frequency (MHz)", [700, 3500], index=1, horizontal=True)

    st.divider()

    st.subheader("2. Receiver (Rx)")
    lat1 = st.number_input("Drone Latitude", value=BASE_LAT, format="%.6f")
    lon1 = st.number_input("Drone Longitude", value=BASE_LON, format="%.6f")
    h_rx = st.number_input("Drone Altitude (m)", value=25.0)

    st.markdown("---")
    show_heatmap = st.checkbox("🔥 Generate Coverage Heatmap", value=False)

    execute = st.button("🚀 Execute Prediction", type="primary")

    st.markdown("### 📊 Legend")
    st.markdown("""
    <div style='line-height:1.8; font-family:sans-serif;'>
    <span style='color:red'>■</span> Excellent (≥ -80 dBm)<br>
    <span style='color:orange'>■</span> Good (-80 to -90)<br>
    <span style='color:yellow'>■</span> Fair (-90 to -100)<br>
    <span style='color:green'>■</span> Poor (-100 to -110)<br>
    <span style='color:cyan'>■</span> Weak (-110 to -120)<br>
    <span style='color:blue'>■</span> No Signal (≤ -120)
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.header("🗺️ Map Interface")

    # Initialize map
    m = folium.Map(location=[lat0, lon0] if bs_df is not None else [BASE_LAT, BASE_LON], zoom_start=17)

    # --- HEATMAP LOGIC ---
    if show_heatmap and model is not None:
        with st.spinner("Calculating area coverage grid..."):
            grid_res = 20  # Reduced grid density slightly for performance
            span = 0.004
            lat_grid = np.linspace(lat0 - span, lat0 + span, grid_res)
            lon_grid = np.linspace(lon0 - span, lon0 + span, grid_res)

            heat_data = []
            for g_lat in lat_grid:
                for g_lon in lon_grid:
                    dist = haversine(lat0, lon0, g_lat, g_lon)
                    if dist < 5: continue

                    bearing = calculate_bearing(lat0, lon0, g_lat, g_lon)
                    az_off = abs(selected_azimuth - bearing)
                    elev = math.degrees(math.atan2(h_tx - h_rx, dist))
                    tilt_off = tilt_tx + elev

                    X_grid = [[dist, az_off, elev, tilt_off, freq, 0]]
                    rsrp_pred = model.predict(X_grid)[0]

                    weight = np.clip((rsrp_pred + 120) / 50, 0, 1)
                    heat_data.append([g_lat, g_lon, weight])

            HeatMap(heat_data, radius=18, blur=12, min_opacity=0.4).add_to(m)

    # --- PREDICTION LOGIC ---
    if execute and model is not None:
        try:
            folium.Marker([lat0, lon0], tooltip="Base Station",
                          icon=folium.Icon(color='red', icon='broadcast-tower', prefix='fa')).add_to(m)
            folium.Marker([lat1, lon1], tooltip="Drone",
                          icon=folium.Icon(color='blue', icon='plane', prefix='fa')).add_to(m)

            total_dist = haversine(lat0, lon0, lat1, lon1)
            num_steps = max(2, int(total_dist / 15))

            lats = np.linspace(lat0, lat1, num_steps)
            lons = np.linspace(lon0, lon1, num_steps)

            for i in range(len(lats) - 1):
                s_lat, s_lon = lats[i], lons[i]
                e_lat, e_lon = lats[i + 1], lons[i + 1]

                d_seg = haversine(lat0, lon0, e_lat, e_lon)
                bearing = calculate_bearing(lat0, lon0, e_lat, e_lon)
                az_off = abs(selected_azimuth - bearing)
                elev = math.degrees(math.atan2(h_tx - h_rx, d_seg))
                tilt_off = tilt_tx + elev

                X = [[d_seg, az_off, elev, tilt_off, freq, 0]]
                rsrp_seg = model.predict(X)[0]

                folium.PolyLine(
                    locations=[(s_lat, s_lon), (e_lat, e_lon)],
                    color=get_rsrp_color(rsrp_seg),
                    weight=8,
                    opacity=0.85
                ).add_to(m)

            # Final Prediction Results
            f_bearing = calculate_bearing(lat0, lon0, lat1, lon1)
            f_az_off = abs(selected_azimuth - f_bearing)
            f_elev = math.degrees(math.atan2(h_tx - h_rx, total_dist))
            X_final = [[total_dist, f_az_off, f_elev, tilt_tx + f_elev, freq, 0]]
            final_rsrp = model.predict(X_final)[0]

            st.success(f"Path Prediction Complete!")
            st.metric("Predicted RSRP at Drone", f"{final_rsrp:.2f} dBm")
            st.info(f"Direct Link Distance: {total_dist:.2f} meters")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

    # Render Map
    st_folium(m, width="100%", height=650, returned_objects=[])