import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from fpdf import FPDF
import base64
import os

st.set_page_config(page_title="DME/CDFA Descent Planner Tool", layout="wide")

st.title("🛬 DME/CDFA Descent Planner Tool")

# --- Sidebar Inputs ---
st.sidebar.header("Approach Inputs")

thr_elevation = st.sidebar.number_input("Runway Threshold Elevation (ft)", value=50)
mda = st.sidebar.number_input("Minimum Descent Altitude (MDA) (ft)", value=600)
gp_angle = st.sidebar.number_input("Glide Path Angle (degrees)", value=3.0)
dme_at_thr = st.sidebar.number_input("DME at Threshold / MAPt (NM)", value=0.5)
distance_faf_to_mapt = st.sidebar.number_input("FAF to MAPt Distance (NM)", value=5.0)

sdf_count = st.sidebar.slider("Number of Step-Down Fixes (SDFs)", 0, 6, 2)

sdf_distances = []
sdf_altitudes = []

for i in range(sdf_count):
    col1, col2 = st.sidebar.columns(2)
    with col1:
        d = st.number_input(f"SDF {i+1} DME (NM)", value=dme_at_thr + 1.5 + i, key=f"sdf_d_{i}")
        sdf_distances.append(d)
    with col2:
        a = st.number_input(f"SDF {i+1} Altitude (ft)", value=mda + 300 + (i * 200), key=f"sdf_a_{i}")
        sdf_altitudes.append(a)

# --- Calculations ---
def generate_dme_table(thr_elev, mda, gp_angle, dme_thr, faf_to_mapt_dist):
    descent_angle_rad = np.deg2rad(gp_angle)
    total_distance = faf_to_mapt_dist + dme_thr
    dme_points = np.linspace(dme_thr + faf_to_mapt_dist, dme_thr, 8)
    dme_table = []

    for dme in dme_points:
        slant_range = dme / np.cos(descent_angle_rad)
        height_above_thr = np.tan(descent_angle_rad) * slant_range * 6076.12 / 100  # feet
        altitude = max(mda, thr_elev + height_above_thr)
        dme_table.append((round(dme, 2), round(altitude)))

    return pd.DataFrame(dme_table, columns=["DME (NM)", "Altitude (ft)"])

def generate_rod_table(faf_to_mapt_dist, gp_angle):
    speeds = [70, 90, 120, 140, 160]
    gradient = np.tan(np.deg2rad(gp_angle)) * 100  # feet per 100ft
    rodes = []

    for spd in speeds:
        fpm = (spd * 101.27 * gradient)  # feet per min
        rodes.append((spd, round(fpm)))

    return pd.DataFrame(rodes, columns=["Ground Speed (kt)", "ROD (ft/min)"])

dme_df = generate_dme_table(thr_elevation, mda, gp_angle, dme_at_thr, distance_faf_to_mapt)
rod_df = generate_rod_table(distance_faf_to_mapt, gp_angle)

# --- Display ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("📋 DME Table")
    st.dataframe(dme_df)

with col2:
    st.subheader("📉 Rate of Descent Table")
    st.dataframe(rod_df)

# --- Descent Profile Plot ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dme_df["DME (NM)"], dme_df["Altitude (ft)"], marker='o', label="CDFA Path")
ax.axhline(y=mda, color='red', linestyle='--', label="MDA")
for i in range(sdf_count):
    ax.plot(sdf_distances[i], sdf_altitudes[i], 'kx')
    ax.text(sdf_distances[i], sdf_altitudes[i]+50, f"SDF{i+1}", ha='center', fontsize=8)

ax.set_title("CDFA Descent Profile")
ax.set_xlabel("DME (NM)")
ax.set_ylabel("Altitude (ft)")
ax.invert_xaxis()
ax.grid(True)
ax.legend()
st.pyplot(fig)

# --- PDF Export ---
def generate_pdf(dme_df, rod_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "DME/CDFA Descent Planner Report", ln=True, align="C")

    pdf.cell(200, 10, "DME Table", ln=True)
    for i, row in dme_df.iterrows():
        pdf.cell(200, 10, f"{row['DME (NM)']} NM — {row['Altitude (ft)']} ft", ln=True)

    pdf.cell(200, 10, "", ln=True)
    pdf.cell(200, 10, "Rate of Descent Table", ln=True)
    for i, row in rod_df.iterrows():
        pdf.cell(200, 10, f"{row['Ground Speed (kt)']} kt — {row['ROD (ft/min)']} ft/min", ln=True)

    return pdf.output(dest='S').encode('latin1')

st.subheader("📤 Export Reports")
pdf_bytes = generate_pdf(dme_df, rod_df)
b64 = base64.b64encode(pdf_bytes).decode()
st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="CDFA_Descent_Report.pdf">📄 Download PDF Report</a>', unsafe_allow_html=True)

csv_dme = dme_df.to_csv(index=False).encode()
st.download_button("⬇️ Download DME Table (CSV)", csv_dme, "dme_table.csv", "text/csv")

csv_rod = rod_df.to_csv(index=False).encode()
st.download_button("⬇️ Download ROD Table (CSV)", csv_rod, "rod_table.csv", "text/csv")




