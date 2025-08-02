import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from io import BytesIO

st.set_page_config(page_title="DME/CDFA Descent Planner", layout="wide")

# Utility functions
def calculate_slant_range_dme(thr_lat, thr_lon, dme_lat, dme_lon, altitude_ft):
    from geopy.distance import geodesic
    horizontal_nm = geodesic((thr_lat, thr_lon), (dme_lat, dme_lon)).nm
    vertical_nm = altitude_ft / 6076.12  # feet to NM
    slant_range = (horizontal_nm**2 + vertical_nm**2)**0.5
    return round(slant_range, 1)

def calculate_dme_table(tod_alt, mda, dme_thr, dme_mapt, sdf_list, gp_angle_deg, num_points=8):
    total_dist = dme_thr - dme_mapt
    min_dist = total_dist / (num_points - 1)

    dme_table = []
    dme_fix_positions = sorted([dme_thr - d for d in sdf_list] + [dme_thr - total_dist])

    for i in range(num_points):
        distance = dme_thr - i * min_dist
        vertical_diff_ft = (dme_thr - distance) * np.tan(np.radians(gp_angle_deg)) * 6076.12
        altitude = tod_alt - vertical_diff_ft
        altitude = max(altitude, mda)

        is_faf = (i == 0)
        is_mapt = (round(distance, 1) == round(dme_mapt, 1))
        is_sdf = any(abs(distance - sdf_d) < 0.05 for sdf_d in dme_fix_positions)

        dme_table.append({
            "DME (NM)": round(distance, 1),
            "Altitude (FT)": int(round(altitude)),
            "Fix": "FAF" if is_faf else ("MAPT" if is_mapt else ("SDF" if is_sdf else ""))
        })

    return dme_table

def calculate_rod_table(faf_mapt_dist, gp_angle_deg):
    ground_speeds = [80, 100, 120, 140, 160]
    table = []
    for gs in ground_speeds:
        vs_fpm = int(round(gs * 101.27 * np.tan(np.radians(gp_angle_deg))))
        time_sec = round(faf_mapt_dist / gs * 3600)
        table.append({
            "Ground Speed (KT)": gs,
            "ROD (FT/MIN)": vs_fpm,
            "Time FAF to MAPT (SEC)": time_sec
        })
    return table

def plot_descent_profile(dme_table, mda):
    fig, ax = plt.subplots()
    distances = [d["DME (NM)"] for d in dme_table]
    altitudes = [d["Altitude (FT)"] for d in dme_table]
    labels = [d["Fix"] for d in dme_table]

    ax.plot(distances, altitudes, marker='o', label="Descent Profile")
    ax.axhline(y=mda, color='red', linestyle='--', label="MDA")

    for x, y, label in zip(distances, altitudes, labels):
        if label:
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

    ax.set_xlabel("DME (NM)")
    ax.set_ylabel("Altitude (FT)")
    ax.set_title("CDFA Descent Profile")
    ax.grid(True)
    ax.invert_xaxis()
    ax.legend()
    return fig

def create_pdf(dme_table, rod_table):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "CDFA Descent Planner Output", ln=True, align="C")

    # DME Table
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "DME Table", ln=True)
    pdf.set_font("Arial", "", 10)
    for row in dme_table:
        pdf.cell(0, 8, f"DME: {row['DME (NM)']} NM | Altitude: {row['Altitude (FT)']} FT | Fix: {row['Fix']}", ln=True)

    # ROD Table
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "ROD Table", ln=True)
    pdf.set_font("Arial", "", 10)
    for row in rod_table:
        pdf.cell(0, 8, f"GS: {row['Ground Speed (KT)']} KT | ROD: {row['ROD (FT/MIN)']} | Time: {row['Time FAF to MAPT (SEC)']} sec", ln=True)

    return pdf.output(dest="S").encode("latin1", errors="replace")

# App UI
st.title("DME/CDFA Descent Planner")

col1, col2 = st.columns(2)

with col1:
    thr_lat = st.number_input("THR Latitude", value=0.0)
    thr_lon = st.number_input("THR Longitude", value=0.0)
    thr_elev = st.number_input("THR/TDZE Elevation (FT)", value=50)
    dme_lat = st.number_input("DME Station Latitude", value=0.0)
    dme_lon = st.number_input("DME Station Longitude", value=0.0)
    dme_thr = st.number_input("DME at THR (NM)", value=5.0)
    dme_mapt = st.number_input("DME at MAPT (NM)", value=0.5)
    mda = st.number_input("Minimum Descent Altitude (MDA, FT)", value=400)
    tod_alt = st.number_input("Top of Descent Altitude (FT)", value=2000)
    faf_to_mapt = st.number_input("FAF to MAPT Distance (NM)", value=4.5)
    gp_angle_deg = st.number_input("GP Angle (°)", value=3.0, help="Use real GP angle if known; otherwise use default")

with col2:
    sdf_list = []
    sdf_count = st.slider("Number of Step Down Fixes (SDF)", 0, 6, 2)
    for i in range(sdf_count):
        d = st.number_input(f"SDF {i+1} DME Distance from THR (NM)", value=2.0 - i * 0.5, key=f"sdf_d_{i}")
        sdf_list.append(d)

if st.button("Generate Descent Plan"):
    dme_table = calculate_dme_table(tod_alt, mda, dme_thr, dme_mapt, sdf_list, gp_angle_deg)
    rod_table = calculate_rod_table(faf_to_mapt, gp_angle_deg)

    st.success("Generated Successfully!")
    st.subheader("DME Table")
    st.dataframe(pd.DataFrame(dme_table))

    st.subheader("ROD Table")
    st.dataframe(pd.DataFrame(rod_table))

    fig = plot_descent_profile(dme_table, mda)
    st.pyplot(fig)

    csv_dme = pd.DataFrame(dme_table).to_csv(index=False).encode("utf-8")
    csv_rod = pd.DataFrame(rod_table).to_csv(index=False).encode("utf-8")
    st.download_button("Download DME Table (CSV)", csv_dme, "dme_table.csv", "text/csv")
    st.download_button("Download ROD Table (CSV)", csv_rod, "rod_table.csv", "text/csv")

    pdf_bytes = create_pdf(dme_table, rod_table)
    st.download_button("Download PDF Report", pdf_bytes, "descent_report.pdf", "application/pdf")

