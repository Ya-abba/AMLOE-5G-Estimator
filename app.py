import streamlit as st
import pandas as pd
import joblib
import math
import folium
from streamlit_folium import st_folium

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="AMLOE Web Platform")

# --- Constants & Helper Functions (Ported from Original Code) ---
BASE_LAT = 2.927778
BASE_LON = 101.785556

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_rsrp_color(rsrp):
    # Logic from [cite: 25]
    if rsrp >= -80: return "red"
    elif rsrp >= -90: return "orange"
    elif rsrp >= -100: return "yellow"
    elif rsrp >= -110: return "green"
    elif rsrp >= -120: return "cyan"
    else: return "blue"

# --- Load Data and Model ---
@st.cache_resource
def load_resources():
    try:
        # Load dataset [cite: 11]
        df = pd.read_csv("5G_BS_UKM.csv")
        df = df[df['Location'].notna()]
        
        # Load Model [cite: 12]
        model = joblib.load("rsrp_model.pkl")
        return df, model
    except FileNotFoundError as e:
        st.error(f"File not found: {e}. Please ensure '5G_BS_UKM.csv' and 'rsrp_model.pkl' are in the directory.")
        return None, None

bs_df, model = load_resources()

# --- Application Layout ---
st.title("AMLOE: Aerial Machine Learning Based Online Coverage Estimator")
st.markdown("Predict 5G Downlink RSRP coverage using Random Forest Regression[cite: 7].")

col1, col2 = st.columns([1, 2])

# --- Left Column: Controls (Tx & Rx) ---
with col1:
    st.header("1. Transmitter (Tx) Settings")
    
    if bs_df is not None:
        # Station Selection [cite: 14]
        station_names = bs_df['Location'].tolist()
        selected_station = st.selectbox("Select Base Station", station_names)
        
        # Get data for selected station
        row = bs_df[bs_df['Location'] == selected_station].iloc[0]
        
        # Dynamic Azimuth Selection [cite: 15]
        azimuths = [int(row['Azimuth (°)'])]
        # Check optional azimuth columns if they exist in your CSV structure
        for col in ['Unnamed: 6', 'Unnamed: 7']:
            if col in bs_df.columns and pd.notna(row[col]):
                try:
                    azimuths.append(int(row[col]))
                except:
                    pass
        
        selected_azimuth = st.selectbox("Antenna Azimuth (°)", azimuths)
        
        # Frequency Selection [cite: 16]
        freq = st.radio("Frequency (MHz)", [700, 3500], horizontal=True)
        
        # Display Auto-populated Info [cite: 15]
        with st.expander("Base Station Details", expanded=True):
            st.write(f"**Lat:** {row['Latitude']}")
            st.write(f"**Lon:** {row['Longtitude']}")
            st.write(f"**Height:** {row['Base station Height (m)']} m")
            st.write(f"**Tilt:** {row['Tilt (°)']}°")
            
            # Store Tx vars for calculation
            lat0 = float(row['Latitude'])
            lon0 = float(row['Longtitude'])
            height_tx = float(row['Base station Height (m)'])
            # Clean tilt string if necessary
            try:
                tilt_tx = float(str(row['Tilt (°)']).strip())
            except:
                tilt_tx = 0.0

    st.divider()
    
    st.header("2. Receiver (Rx) Settings")
    # Manual Input for Rx [cite: 16]
    lat1 = st.number_input("Rx Latitude (DD)", value=BASE_LAT, format="%.6f")
    lon1 = st.number_input("Rx Longitude (DD)", value=BASE_LON, format="%.6f")
    height_rx = st.number_input("Rx Height ASL (m)", value=25.0)

    execute = st.button("Execute Prediction", type="primary")

    # Legend Display (Using Markdown to mimic the Canvas) [cite: 25]
    st.markdown("### Legend")
    st.markdown("""
    <div style='font-size:12px'>
    <span style='color:red'>█</span> ≥ -80 dBm<br>
    <span style='color:orange'>█</span> -80 to -90 dBm<br>
    <span style='color:gold'>█</span> -90 to -100 dBm<br>
    <span style='color:green'>█</span> -100 to -110 dBm<br>
    <span style='color:cyan'>█</span> -110 to -120 dBm<br>
    <span style='color:blue'>█</span> ≤ -120 dBm
    </div>
    """, unsafe_allow_html=True)

# --- Right Column: Map & Results ---
with col2:
    st.header("Map Interface")
    
    # Initialize Map
    m = folium.Map(location=[BASE_LAT, BASE_LON], zoom_start=15)

    # Calculation Logic triggers on button click
    if execute and model is not None:
        try:
            total_distance = haversine(lat0, lon0, lat1, lon1)
            steps = max(1, int(total_distance // 100)) # 100m step size
            
            # 1. Add Markers [cite: 24]
            folium.Marker(
                [lat0, lon0], popup="Tx (Base Station)", 
                icon=folium.Icon(color='red', icon='tower', prefix='fa')
            ).add_to(m)
            
            folium.Marker(
                [lat1, lon1], popup="Rx (Receiver)", 
                icon=folium.Icon(color='blue', icon='user', prefix='fa')
            ).add_to(m)

            path_coords = []
            
            # 2. Iterate path segments [cite: 24]
            for i in range(steps):
                f1 = i / steps
                f2 = (i + 1) / steps
                
                # Interpolate coordinates
                la1 = lat0 + (lat1 - lat0) * f1
                lo1 = lon0 + (lon1 - lon0) * f1
                la2 = lat0 + (lat1 - lat0) * f2
                lo2 = lon0 + (lon1 - lon0) * f2
                
                # Calculate features for model
                d = haversine(lat0, lon0, la2, lo2)
                az_off = abs(float(selected_azimuth) - math.degrees(math.atan2(lo2 - lon0, la2 - lat0)))
                elev = math.degrees(math.atan2(height_tx - height_rx, d))
                tilt_off = tilt_tx + elev
                
                # Predict
                X = [[d, az_off, elev, tilt_off, freq, 0]]
                rsrp = model.predict(X)[0]
                color = get_rsrp_color(rsrp)
                
                # Draw segment
                folium.PolyLine(
                    locations=[(la1, lo1), (la2, lo2)],
                    color=color,
                    weight=5,
                    opacity=0.8
                ).add_to(m)

            # 3. Final Prediction for Display
            final_d = haversine(lat0, lon0, lat1, lon1)
            final_elev = math.degrees(math.atan2(height_tx - height_rx, final_d))
            final_tilt_off = tilt_tx + final_elev
            # Re-calculate final azimuth offset 
            final_az_off = abs(float(selected_azimuth) - math.degrees(math.atan2(lon1 - lon0, lat1 - lat0)))
            
            Xf = [[final_d, final_az_off, final_elev, final_tilt_off, freq, 0]]
            rsrp_final = model.predict(Xf)[0]
            
            st.success(f"Prediction Complete! Distance: {final_d:.2f}m")
            st.metric(label="Predicted RSRP at Receiver", value=f"{rsrp_final:.2f} dBm") [cite: 18]

        except Exception as e:
            st.error(f"Error during calculation: {e}")

    # Render Map 
    st_folium(m, width="100%", height=600)