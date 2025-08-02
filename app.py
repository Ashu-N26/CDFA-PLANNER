import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import io

# ---------- Title ----------
st.set_page_config(page_title="DME/CDFA Descent Planner Tool", layout="wide")
st.title("✈️ DME/CDFA Descent Planner Tool – Advanced Version")

# ---------- Input Fields ----------
st.header("🛬 Input Parameters")

col1, col2 = st.columns(2)
with col1:
    thr_lat = st.text_input("THR Latitude", "41.669")
    thr_lon = st.text_input("THR Longitude", "-70.280")
    thr_elev = st.number_input("THR/TDZE Elevation (FT)", min_value=0, value=55)
    mda = st.number_input("MDA (FT)", min_value=0, value=460)
    tod_alt = st.number_input("Top of Descent Altitude (FT)", min_value=0, value=1600)
    gp_angle = st.number_input("Glide Path Angle (°) – Optional", value=3.0)

with col2:
    dme_lat = st.text_input("DME Latitude", "41.701")
    dme_lon = st.text_input("DME Longitude", "-70.255")
    dme_thr = st.number_input("DME at Threshold (NM)", min_value=0.0, value=1.2)
    dme_mapt = st.number_input("DME at MAPt (NM)", min_value=0.0, value=0.6)
    faf_mapt_dist = st.number_input("FAF to MAPt Distance (NM)", min_value=0.1, value=4.0)

# ---------- Step-Down Fixes ----------
st.subheader("🔻 Optional Step-Down Fixes (SDF)")
sdf_count = st.slider("Number of SDFs", 0, 6, 2)
sdf_list = []
for i in range(sdf_count):
    col_sdf1, col_sdf2 = st.columns(2)
    with col_sdf1:
        sdf_dme = st.number_input(f"SDF {i+1} DME (NM)", key=f"sdf_dme_{i}")
    with col_sdf2:
        sdf_alt = st.number_input(f"SDF {i+1} Altitude (FT)", key=f"sdf_alt_{i}")
    sdf_list.append((sdf_dme, sdf_alt))

# ---------- Process Logic ----------
def generate_dme_table(tod_alt, mda, dme_thr, dme_mapt, gp_angle, sdf_list):
    total_distance = dme_thr - dme_mapt
    step = total_distance / 7
    dme_points = [round(dme_thr - step * i, 1) for i in range(8)]
    altitudes = []

    for d in dme_points:
        dist_from_thr = dme_thr - d
        altitude = tod_alt - (dist_from_thr * (100 * gp_angle))
        altitude = max(altitude, mda)
        altitudes.append(round(altitude))

    sdf_dict = {round(sdf[0], 1): int(sdf[1]) for sdf in sdf_list}
    labels = ["SDF" if round(d, 1) in sdf_dict else "" for d in dme_points]
    for i, d in enumerate(dme_points):
        if round(d, 1) in sdf_dict:
            altitudes[i] = max(altitudes[i], sdf_dict[round(d, 1)])
    df = pd.DataFrame({
        "DME (NM)": dme_points,
        "Altitude (FT)": altitudes,
        "Note": labels
    })
    return df

def generate_rod_table(faf_alt, mda, faf_mapt_dist, gp_angle):
    speeds = [80, 100, 120, 140, 160]
    rod = {}
    for spd in speeds:
        fpm = int(101.3 * spd * np.tan(np.radians(gp_angle)))
        time_min = faf_mapt_dist / spd
        time_sec = int(time_min * 60)
        rod[spd] = (fpm, time_sec)
    df = pd.DataFrame(rod).T
    df.columns = ["ROD (FPM)", "Time (sec)"]
    df.index.name = "GS (KT)"
    return df

# ---------- PDF Generator ----------
def generate_pdf(dme_df, rod_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "DME/CDFA Descent Planner Report", ln=1, align="C")

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "DME Table", ln=1)

    pdf.set_font("Arial", size=10)
    for index, row in dme_df.iterrows():
        note = f" ({row['Note']})" if row['Note'] else ""
        pdf.cell(0, 8, f"DME: {row['DME (NM)']} NM | Alt: {row['Altitude (FT)']} FT{note}", ln=1)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "ROD Table", ln=1)
    pdf.set_font("Arial", size=10)
    for idx, row in rod_df.iterrows():
        pdf.cell(0, 8, f"GS: {idx} KT | ROD: {row['ROD (FPM)']} FPM | Time: {row['Time (sec)']} sec", ln=1)

    return pdf.output(dest="S").encode("latin1", errors="ignore")

# ---------- Generate Descent Plan ----------
if st.button("📈 Generate Descent Plan"):
    dme_df = generate_dme_table(tod_alt, mda, dme_thr, dme_mapt, gp_angle, sdf_list)
    rod_df = generate_rod_table(tod_alt, mda, faf_mapt_dist, gp_angle)

    st.subheader("📋 DME Table")
    st.dataframe(dme_df)

    st.subheader("📋 ROD Table")
    st.dataframe(rod_df)

    # ---------- Plot ----------
    st.subheader("📊 Descent Profile")
    fig, ax = plt.subplots()
    ax.plot(dme_df["DME (NM)"], dme_df["Altitude (FT)"], marker="o", label="Glidepath")
    ax.axhline(y=mda, color="red", linestyle="--", label="MDA")
    for idx, row in dme_df.iterrows():
        if row["Note"]:
            ax.annotate("SDF", (row["DME (NM)"], row["Altitude (FT)"]), textcoords="offset points", xytext=(0,10), ha='center')
    ax.set_xlabel("DME (NM)")
    ax.set_ylabel("Altitude (FT)")
    ax.set_title("CDFA Descent Profile")
    ax.legend()
    st.pyplot(fig)

    # ---------- PDF & CSV ----------
    pdf_bytes = generate_pdf(dme_df, rod_df)
    st.download_button("📄 Download PDF", data=pdf_bytes, file_name="descent_report.pdf")

    dme_csv = dme_df.to_csv(index=False).encode("utf-8")
    rod_csv = rod_df.to_csv().encode("utf-8")
    st.download_button("⬇️ Download DME CSV", data=dme_csv, file_name="dme_table.csv")
    st.download_button("⬇️ Download ROD CSV", data=rod_csv, file_name="rod_table.csv")





