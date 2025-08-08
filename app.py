import streamlit as st
import pandas as pd
import numpy as np
import math
from io import BytesIO
from fpdf import FPDF
import base64
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import os

st.set_page_config(page_title="CDFA-PLANNER", layout="centered")

st.title("🛬 CDFA-PLANNER")

st.markdown("""
Use this tool to plan a Continuous Descent Final Approach (CDFA) for non-precision approaches.  
Supports automated PDF parsing or manual input for accurate DME and ROD tables.
""")

# ------------------------- PDF PARSER -------------------------------------
def extract_text_from_pdf(uploaded_pdf):
    pdf_bytes = uploaded_pdf.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    text = "\n".join(pytesseract.image_to_string(img) for img in images[:2])
    return text

def parse_chart(text):
    parsed = {}
    try:
        for line in text.splitlines():
            if "GP ANGLE" in line.upper():
                parsed['gp_angle'] = float(next(filter(str.isdigit, line.split())))
            if "TOP OF DESCENT" in line.upper() or "TOD" in line:
                parsed['tod_alt'] = int(''.join(filter(str.isdigit, line)))
            if "MDA" in line and "CAT" in line:
                parsed['mda'] = int(''.join(filter(str.isdigit, line)))
    except:
        pass
    return parsed

# ------------------------ INPUT PANEL --------------------------------------

st.subheader("✈️ Inputs")

pdf_uploaded = st.file_uploader("Optional: Upload IAC PDF chart", type=["pdf"])

parsed_data = {}
if pdf_uploaded:
    try:
        text = extract_text_from_pdf(pdf_uploaded)
        parsed_data = parse_chart(text)
        st.success("✅ PDF parsed. You can adjust values below.")
    except Exception as e:
        st.warning("⚠️ PDF Parsing failed. You can manually enter data.")

# Threshold (THR)
st.markdown("#### Threshold (THR) Coordinates")
thr_lat = st.text_input("THR Latitude", "18.5822")
thr_lon = st.text_input("THR Longitude", "73.9197")

# TDZE
tdze = st.number_input("THR/TDZE Elevation (ft)", value=75)

# DME
st.markdown("#### DME Station Coordinates")
dme_lat = st.text_input("DME Latitude", "18.5994")
dme_lon = st.text_input("DME Longitude", "73.9125")

dme_thr = st.number_input("DME at THR (NM)", value=2.8)
dme_mapt = st.number_input("DME at MAPT (NM)", value=1.1)

# Glide Path Angle (optional)
gp_angle = st.number_input("Glide Path Angle (°)", value=parsed_data.get('gp_angle', 0.0), help="Optional. Will auto-calculate if blank.")

# Altitudes
tod_alt = st.number_input("Top of Descent Altitude (TOD, ft)", value=parsed_data.get('tod_alt', 3000))
mda = st.number_input("Minimum Descent Altitude (MDA, ft)", value=parsed_data.get('mda', 1000))

# SDF input (up to 6 optional)
st.markdown("#### Step-Down Fixes (Optional)")
sdf_count = st.number_input("Number of SDFs (max 6)", min_value=0, max_value=6, value=2)
sdf_inputs = []
for i in range(sdf_count):
    col1, col2 = st.columns(2)
    with col1:
        dist = st.number_input(f"SDF {i+1} Distance (NM)", value=6.0 - i, key=f"sdf_d_{i}")
    with col2:
        alt = st.number_input(f"SDF {i+1} Altitude (ft)", value=mda + 500 + (i * 100), key=f"sdf_a_{i}")
    sdf_inputs.append((round(dist, 1), alt))

# FAF to MAPT distance
faf_mapt = st.number_input("FAF to MAPT Distance (NM)", value=5.0)

# -------------------------- CDFA CALCULATION LOGIC -----------------------------

def compute_gp_angle(tod_alt, mda, tod_dist):
    try:
        angle_rad = math.atan((tod_alt - mda) / (tod_dist * 6076.12))
        return round(math.degrees(angle_rad), 2)
    except:
        return 3.0

def generate_dme_table(tod_alt, mda, dme_thr, dme_mapt, sdf_inputs):
    table = []
    total_distance = dme_thr - dme_mapt
    gp_angle_local = compute_gp_angle(tod_alt, mda, total_distance)

    dme_points = [round(dme_thr - i, 1) for i in range(8)]
    for d in dme_points:
        if d < dme_mapt:
            continue
        h = tod_alt - (dme_thr - d) * 6076.12 * math.tan(math.radians(gp_angle_local))
        h = max(h, mda)
        h = round(h)
        table.append((d, h))
    return table, gp_angle_local

def generate_rod_table(gp_angle, faf_dist):
    speeds = [80, 100, 120, 140, 160]
    rod_data = []
    for gs in speeds:
        fpm = round(gs * 101.27 * math.tan(math.radians(gp_angle)))
        time_sec = round((faf_dist / gs) * 3600)
        rod_data.append((gs, fpm, f"{time_sec//60}:{str(time_sec%60).zfill(2)}"))
    return rod_data

# ----------------------- TABLES + EXPORTS -------------------------------

if st.button("Generate CDFA Tables"):
    dme_data, final_gp = generate_dme_table(tod_alt, mda, dme_thr, dme_mapt, sdf_inputs)
    rod_data = generate_rod_table(final_gp, faf_mapt)

    st.success(f"✅ CDFA Slope: {final_gp}°")

    dme_df = pd.DataFrame(dme_data, columns=["DME (NM)", "Altitude (ft)"])
    sdf_labels = dict(sdf_inputs)
    for i, row in dme_df.iterrows():
        if round(row["DME (NM)"], 1) in sdf_labels:
            dme_df.at[i, "Label"] = f"SDF @ {sdf_labels[round(row['DME (NM)'], 1)]} ft"
        else:
            dme_df.at[i, "Label"] = ""

    rod_df = pd.DataFrame(rod_data, columns=["Ground Speed (kt)", "ROD (ft/min)", "Time (min:sec)"])

    st.subheader("📍 DME Table")
    st.dataframe(dme_df)

    st.subheader("📈 ROD Table")
    st.dataframe(rod_df)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(dme_df["DME (NM)"], dme_df["Altitude (ft)"], marker='o', label="Glide Path")
    ax.axhline(y=mda, color='r', linestyle='--', label="MDA")
    ax.set_xlabel("DME (NM)")
    ax.set_ylabel("Altitude (ft)")
    ax.set_title("CDFA Descent Slope")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # PDF Export
    def generate_pdf(dme_df, rod_df):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "CDFA Descent Planner", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "DME Table", ln=True)
        for _, row in dme_df.iterrows():
            pdf.set_font("Arial", "", 11)
            label = f" ({row['Label']})" if row['Label'] else ""
            pdf.cell(0, 8, f"{row['DME (NM)']} NM - {row['Altitude (ft)']} ft{label}", ln=True)

        pdf.ln(10)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "ROD Table", ln=True)
        for _, row in rod_df.iterrows():
            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 8, f"{row['Ground Speed (kt)']} kt - {row['ROD (ft/min)']} ft/min - {row['Time (min:sec)']}", ln=True)

        return pdf.output(dest='S').encode('latin1')

    pdf_bytes = generate_pdf(dme_df, rod_df)
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="cdfa_output.pdf">📄 Download PDF Report</a>'
    st.markdown(href, unsafe_allow_html=True)

    # CSV Export
    csv1 = dme_df.to_csv(index=False)
    b64_1 = base64.b64encode(csv1.encode()).decode()
    st.markdown(f'<a href="data:file/csv;base64,{b64_1}" download="DME_Table.csv">📁 Download DME CSV</a>', unsafe_allow_html=True)

    csv2 = rod_df.to_csv(index=False)
    b64_2 = base64.b64encode(csv2.encode()).decode()
    st.markdown(f'<a href="data:file/csv;base64,{b64_2}" download="ROD_Table.csv">📁 Download ROD CSV</a>', unsafe_allow_html=True)









