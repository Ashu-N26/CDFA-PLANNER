import streamlit as st
import pandas as pd
import numpy as np
import math
import io
from fpdf import FPDF

st.set_page_config(layout="wide")
st.title("🛬 CDFA Descent Planner (Advanced Version)")

# Input Section
st.header("📥 Input Parameters")

col1, col2 = st.columns(2)
with col1:
    thr_lat = st.number_input("THR Latitude", format="%.6f", value=0.0)
    thr_lon = st.number_input("THR Longitude", format="%.6f", value=0.0)
    thr_elev = st.number_input("THR/TDZE Elevation (ft)", value=100)
    dme_lat = st.number_input("DME Latitude", format="%.6f", value=0.0)
    dme_lon = st.number_input("DME Longitude", format="%.6f", value=0.0)
    dme_at_thr = st.number_input("DME Distance at THR (NM)", format="%.2f", value=1.0)
    dme_at_mapt = st.number_input("DME Distance at MAPt (NM)", format="%.2f", value=1.0)
    mda = st.number_input("MDA (ft)", value=1000)
    tod_alt = st.number_input("TOD Altitude (ft)", value=3000)

with col2:
    num_sdfs = st.number_input("Number of Step Down Fixes (0–6)", min_value=0, max_value=6, value=3, step=1)
    sdf_distances = []
    sdf_altitudes = []
    for i in range(num_sdfs):
        sdf_distances.append(st.number_input(f"SDF {i+1} Distance (NM)", key=f"sdfd_{i}", value=float(7 - i)))
        sdf_altitudes.append(st.number_input(f"SDF {i+1} Altitude (ft)", key=f"sdfa_{i}", value=int(3000 - i*500)))

    faf_to_mapt = st.number_input("FAF → MAPt Distance (NM)", format="%.2f", value=5.0)
    gp_angle = st.number_input("Glide Path Angle (°)", value=3.00, format="%.2f", step=0.01)

# Computation
def generate_dme_table(tod_alt, mda, dme_at_thr, sdf_points):
    total_distance = dme_at_thr + 1.0  # Add extra to ensure 8 segments
    dme_points = np.linspace(total_distance, dme_at_thr, 8)
    dme_points = np.round(dme_points, 1)
    altitudes = []
    slope_rad = math.radians(gp_angle)
    for dme in dme_points:
        delta_ft = (dme - dme_at_thr) * 6076.12 * math.tan(slope_rad)
        alt = thr_elev + delta_ft
        altitudes.append(max(alt, mda))
    # Inject SDFs if close match
    fixes = []
    for d in dme_points:
        label = ""
        for i, sdf_d in enumerate(sdf_points.keys()):
            if abs(sdf_d - d) <= 0.2:
                label = f"SDF{i+1}"
        if d == dme_at_thr:
            label = "MAPt"
        fixes.append(label)
    return pd.DataFrame({
        "DME": dme_points,
        "Altitude": [int(round(a)) for a in altitudes],
        "Fix": fixes
    })

def generate_rod_table(gp_angle, distance_nm):
    slope_rad = math.radians(gp_angle)
    gradient_ft_per_nm = 6076.12 * math.tan(slope_rad)
    gs_list = [80, 100, 120, 140, 160]
    rod_list = [int(round(gs * gradient_ft_per_nm / 60)) for gs in gs_list]
    time_list = [round(distance_nm / gs * 60, 2) for gs in gs_list]
    return pd.DataFrame({
        "GS": gs_list,
        "ROD (fpm)": rod_list,
        "Time (min)": time_list
    })

# Output Section
if st.button("🧮 Generate Descent Plan"):
    sdf_dict = dict(zip(sdf_distances, sdf_altitudes))
    dme_table = generate_dme_table(tod_alt, mda, dme_at_thr, sdf_dict)
    rod_table = generate_rod_table(gp_angle, faf_to_mapt)

    st.subheader("📊 DME Descent Table")
    st.table(dme_table)

    st.subheader("📉 ROD Table (FAF → MAPt)")
    st.table(rod_table)

    st.subheader("🖨️ Download PDF Report")
    def export_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "CDFA Descent Report", ln=1)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "DME Table", ln=1)
        for i, row in dme_table.iterrows():
            pdf.cell(0, 8, f"{row['DME']} NM | {row['Altitude']} ft | {row['Fix']}", ln=1)

        pdf.cell(0, 10, "ROD Table", ln=1)
        for i, row in rod_table.iterrows():
            pdf.cell(0, 8, f"{row['GS']} kt | {row['ROD (fpm)']} fpm | {row['Time (min)']} min", ln=1)

        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        pdf_output.seek(0)
        return pdf_output

    st.download_button("📥 Download PDF", export_pdf(), file_name="CDFA_Descent_Report.pdf")



