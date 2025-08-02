import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
import base64
import math
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from datetime import datetime

st.set_page_config(page_title="CDFA PLANNER", layout="wide")
st.title("🛬 CDFA PLANNER")

# ========================== PDF PARSER ==========================
def parse_pdf(pdf_file):
    try:
        images = convert_from_path(pdf_file.name, dpi=300)
        text = "\n".join(pytesseract.image_to_string(img) for img in images[:2])
        return extract_values_from_text(text)
    except Exception as e:
        st.warning("PDF Parsing failed. You can manually enter data.")
        return {}

def extract_values_from_text(text):
    import re
    data = {}

    gp_match = re.search(r"(?:GP ANGLE|GLIDE PATH ANGLE)\s*[:=]?\s*([0-9.]+)", text, re.I)
    if gp_match:
        data["gp_angle"] = float(gp_match.group(1))

    tod_match = re.search(r"(?:TOP OF DESCENT|TOD)\s*[:=]?\s*([0-9,]+)", text, re.I)
    if tod_match:
        data["tod_altitude"] = float(tod_match.group(1).replace(",", ""))

    mda_match = re.search(r"(?:MDA|MINIMUM DESCENT ALTITUDE)\s*[:=]?\s*([0-9,]+)", text, re.I)
    if mda_match:
        data["mda"] = float(mda_match.group(1).replace(",", ""))

    mapt_match = re.search(r"(?:DME AT MAPT)\s*[:=]?\s*([0-9.]+)", text, re.I)
    if mapt_match:
        data["dme_mapt"] = float(mapt_match.group(1))

    thr_match = re.search(r"(?:DME AT THR)\s*[:=]?\s*([0-9.]+)", text, re.I)
    if thr_match:
        data["dme_thr"] = float(thr_match.group(1))

    return data

# ======================== INPUT FIELDS ==========================
pdf_file = st.file_uploader("📎 Upload AIP/IAC PDF (optional)", type=["pdf"])

default_data = parse_pdf(pdf_file) if pdf_file else {}

col1, col2 = st.columns(2)
with col1:
    st.subheader("Runway Threshold (THR)")
    thr_lat = st.text_input("Latitude", "12.9716")
    thr_lon = st.text_input("Longitude", "77.5946")
    tdze = st.number_input("THR / TDZE Elevation (FT)", value=300)

    st.subheader("DME Station")
    dme_lat = st.text_input("DME Latitude", "12.9720")
    dme_lon = st.text_input("DME Longitude", "77.5950")

    dme_thr = st.number_input("DME at Threshold (NM)", value=default_data.get("dme_thr", 1.2))
    dme_mapt = st.number_input("DME at MAPT (NM)", value=default_data.get("dme_mapt", 3.2))

with col2:
    st.subheader("Descent Parameters")
    tod_alt = st.number_input("Top of Descent Altitude (TOD) (FT)", value=default_data.get("tod_altitude", 2200))
    mda = st.number_input("Minimum Descent Altitude (MDA) (FT)", value=default_data.get("mda", 920))
    gp_angle = st.number_input("Glide Path Angle (deg)", value=default_data.get("gp_angle", 3.0))

    faf_mapt = st.number_input("FAF–MAPT Distance (NM)", value=2.0)

# ========== Optional SDF Input ==========
st.subheader("Step-Down Fixes (SDFs)")
sdf_count = st.number_input("How many SDFs?", min_value=0, max_value=6, value=0, step=1)
sdfs = []
for i in range(int(sdf_count)):
    col_sdf1, col_sdf2 = st.columns(2)
    with col_sdf1:
        dist = st.number_input(f"SDF {i+1} Distance (NM)", key=f"dist{i}")
    with col_sdf2:
        alt = st.number_input(f"SDF {i+1} Altitude (FT)", key=f"alt{i}")
    sdfs.append((dist, alt))

# ================== DME + ROD LOGIC ==================
def generate_dme_table(tod_alt, mda, dme_thr, dme_mapt, gp_angle, sdfs):
    dme_points = np.linspace(dme_mapt, dme_thr, 8)
    table = []

    for dme in dme_points:
        dist_from_thr = dme_thr - dme
        height = tdze + dist_from_thr * 6076.12 * math.tan(math.radians(gp_angle)) / 100
        height = max(height, mda)
        sdf_label = ""
        for sdf_dist, sdf_alt in sdfs:
            if abs(sdf_dist - dme) < 0.1:
                height = max(height, sdf_alt)
                sdf_label = "SDF"
        table.append((round(dme, 1), round(height), sdf_label))
    return table

def generate_rod_table(faf_mapt, tod_alt, mda):
    dist_nm = faf_mapt
    height_diff = tod_alt - mda
    gradient = height_diff / (dist_nm * 6076.12)
    rodt = []
    for gs in [80, 100, 120, 140, 160]:
        vs = round(gs * gradient * 101.27)
        rodt.append((gs, vs))
    return rodt

# ================== Generate Tables ==================
if st.button("Generate CDFA Plan"):
    dme_table = generate_dme_table(tod_alt, mda, dme_thr, dme_mapt, gp_angle, sdfs)
    rod_table = generate_rod_table(faf_mapt, tod_alt, mda)

    st.success("✅ CDFA Plan Generated")
    dme_df = pd.DataFrame(dme_table, columns=["DME (NM)", "Altitude (FT)", "Remark"])
    rod_df = pd.DataFrame(rod_table, columns=["Ground Speed (KT)", "Rate of Descent (FPM)"])

    st.subheader("📊 DME Table")
    st.dataframe(dme_df)

    st.subheader("📉 ROD Table")
    st.dataframe(rod_df)

    # Graph
    fig, ax = plt.subplots()
    ax.plot(dme_df["DME (NM)"], dme_df["Altitude (FT)"], marker='o')
    ax.axhline(y=mda, color='red', linestyle='--', label="MDA")
    ax.set_title("CDFA Descent Profile")
    ax.set_xlabel("DME (NM)")
    ax.set_ylabel("Altitude (FT)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # PDF/CSV Export
    def generate_pdf(dme_df, rod_df):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="CDFA PLANNER OUTPUT", ln=True, align="C")
        pdf.ln(10)
        pdf.set_font("Arial", size=10)

        pdf.cell(200, 10, txt="DME Table", ln=True)
        for _, row in dme_df.iterrows():
            pdf.cell(200, 8, txt=f"{row[0]} NM - {row[1]} FT - {row[2]}", ln=True)

        pdf.ln(10)
        pdf.cell(200, 10, txt="ROD Table", ln=True)
        for _, row in rod_df.iterrows():
            pdf.cell(200, 8, txt=f"{row[0]} KT - {row[1]} FPM", ln=True)

        return pdf.output(dest="S").encode("latin1")

    pdf_bytes = generate_pdf(dme_df, rod_df)
    b64_pdf = base64.b64encode(pdf_bytes).decode()
    st.download_button("📥 Download PDF", data=pdf_bytes, file_name="cdfa_output.pdf")
    st.download_button("📥 Download CSV (DME)", dme_df.to_csv(index=False), "dme_table.csv")
    st.download_button("📥 Download CSV (ROD)", rod_df.to_csv(index=False), "rod_table.csv")






