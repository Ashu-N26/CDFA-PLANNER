import streamlit as st
import pandas as pd
import numpy as np
import math

def compute_gp_angle(tod_altitude, mda_altitude, distance_nm):
    try:
        if distance_nm <= 0:
            return None
        altitude_diff = tod_altitude - mda_altitude
        distance_ft = distance_nm * 6076.12
        gp_angle_deg = math.degrees(math.atan(altitude_diff / distance_ft))
        return round(gp_angle_deg, 2)
    except Exception:
        return None

st.set_page_config(page_title="CDFA-PLANNER", layout="wide")
st.title("🛬 CDFA-PLANNER")

# Basic Inputs
col1, col2 = st.columns(2)
with col1:
    tod_altitude = st.number_input("Top of Descent (TOD) Altitude (FT)", min_value=0)
    mda = st.number_input("Minimum Descent Altitude (MDA) (FT)", min_value=0)
with col2:
    dme_thr = st.number_input("DME at Threshold (NM)", min_value=0.0, format="%.2f")
    dme_mapt = st.number_input("DME at MAPt (NM)", min_value=0.0, format="%.2f")

# Optional: FAF-MAPT (backup for GP angle)
faf_mapt_dist = st.number_input("FAF to MAPt Distance (NM)", min_value=0.0, format="%.2f")

# Optional SDF Inputs
st.subheader("Step-Down Fixes (Optional)")
sdf_count = st.number_input("Number of SDFs (max 6)", min_value=0, max_value=6, step=1)

sdfs = []
for i in range(sdf_count):
    col1, col2 = st.columns(2)
    with col1:
        dme = st.number_input(f"SDF {i+1} - DME (NM)", min_value=0.0, format="%.2f", key=f"sdf_dme_{i}")
    with col2:
        alt = st.number_input(f"SDF {i+1} - Altitude (FT)", min_value=0, key=f"sdf_alt_{i}")
    sdfs.append((dme, alt))

# Manual GP angle override (optional)
manual_gp = st.text_input("Glide Path Angle (optional, leave blank to auto-compute)", "")

# Compute GP angle
gp_angle = None
error_msg = ""

if manual_gp.strip():
    try:
        gp_angle = float(manual_gp)
    except ValueError:
        error_msg = "Invalid manual GP angle entered."
else:
    dist = None
    if dme_thr and dme_mapt and abs(dme_thr - dme_mapt) > 0.1:
        dist = abs(dme_thr - dme_mapt)
    elif faf_mapt_dist and faf_mapt_dist > 0.1:
        dist = faf_mapt_dist

    if dist:
        gp_angle = compute_gp_angle(tod_altitude, mda, dist)
    else:
        error_msg = "Invalid DME THR - MAPT or FAF-MAPT distance to compute GP angle."

# Button
if st.button("Generate CDFA Profile"):
    if error_msg:
        st.error(error_msg)
    elif gp_angle is None:
        st.error("Could not compute GP angle.")
    else:
        st.success(f"CDFA Slope: {gp_angle:.2f}°")

        # Generate DME Table (example logic)
        dme_table = []
        altitude_range = tod_altitude - mda
        dme_range = dist
        for i in range(8):
            dme_nm = round(dme_mapt + (dme_range / 7) * (7 - i), 1)
            altitude = round(mda + (altitude_range / 7) * (i), 0)
            dme_table.append((dme_nm, altitude))
        dme_df = pd.DataFrame(dme_table, columns=["DME (NM)", "Altitude (FT)"])

        st.subheader("📍 DME Table")
        st.dataframe(dme_df)

        # Generate ROD Table
        st.subheader("📉 ROD Table")
        rod_data = []
        for gs in [80, 100, 120, 140, 160]:
            time_min = (dist / gs) * 60
            rod = (tod_altitude - mda) / time_min
            rod_data.append((gs, int(round(rod)), f"{int(time_min)}:{int((time_min%1)*60):02d}"))
        rod_df = pd.DataFrame(rod_data, columns=["Ground Speed (kt)", "ROD (ft/min)", "Time (min:sec)"])
        st.dataframe(rod_df)


