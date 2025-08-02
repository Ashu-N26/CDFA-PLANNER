import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
import base64
import math
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image

st.set_page_config(page_title="CDFA PLANNER", layout="wide")
st.title("🛬 CDFA PLANNER — Precision-Like Non-Precision Approach Generator")

# Function to extract text from PDF (OCR)
def extract_pdf_text(uploaded_pdf):
    try:
        images = convert_from_bytes(uploaded_pdf.read())
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img)
        return text
    except Exception as e:
        st.error(f"PDF parsing failed: {e}")
        return None

# Dummy parser logic (extend with regex later)
def parse_pdf_values(text):
    try:
        # Placeholder logic: you will need regex based on real chart format
        gp_angle = 3.0
        tod_alt = 2200
        mda = 1000
        dme_thr = 4.2
        dme_mapt = 0.6
        return {
            "gp_angle": gp_angle,
            "tod_alt": tod_alt,
            "mda": mda,
            "dme_thr": dme_thr,
            "dme_mapt": dme_mapt,
        }
    except:
        return None

# INPUT PANEL
st.sidebar.header("📥 Input Parameters")

pdf_file = st.sidebar.file_uploader("📄 Upload IAC Chart (optional)", type=["pdf"])
parsed = None

if pdf_file:
    st.sidebar.markdown("🔍 Parsing PDF...")
    text = extract_pdf_text(pdf_file)
    parsed = parse_pdf_values(text)
    if parsed:
        st.sidebar.success("✅ PDF parsed successfully")
    else:
        st.sidebar.warning("⚠️ PDF parsing failed. You can enter values manually.")

thr_lat = st.sidebar.text_input("THR Latitude", "")
thr_lon = st.sidebar.text_input("THR Longitude", "")
thr_elev = st.sidebar.number_input("THR / TDZE Elevation (ft)", value=0)

dme_lat = st.sidebar.text_input("DME Latitude", "")
dme_lon = st.sidebar.text_input("DME Longitude", "")
dme_thr = st.sidebar.number_input("DME at THR (NM)", value=parsed['dme_thr'] if parsed else 0.0)
dme_mapt = st.sidebar.number_input("DME at MAPt (NM)", value=parsed['dme_mapt'] if parsed else 0.0)

tod_alt = st.sidebar.number_input("TOD Altitude (ft)", value=parsed['tod_alt'] if parsed else 2000)
mda = st.sidebar.number_input("MDA (ft)", value=parsed['mda'] if parsed else 800)

faf_mapt = st.sidebar.number_input("FAF–MAPt Distance (NM)", value=parsed['dme_mapt'] if parsed else 2.0)
gp_angle = st.sidebar.number_input("Glide Path Angle (Optional)", value=parsed['gp_angle'] if parsed else 0.0)

# SDFs
st.sidebar.markdown("🔻 Step-Down Fixes (SDF)")
sdf_list = []
for i in range(6):
    sdf_nm = st.sidebar.text_input(f"SDF {i+1} Distance (NM)", "")
    sdf_alt = st.sidebar.text_input(f"SDF {i+1} Altitude (ft)", "")
    if sdf_nm and sdf_alt:
        sdf_list.append((float(sdf_nm), float(sdf_alt)))

# Button
generate = st.button("🛫 Generate DME & ROD Tables")

# CORE LOGIC
def generate_dme_table(tod_alt, mda, dme_thr, dme_mapt, sdf_list):
    dme_points = [dme_thr]
    distance = dme_thr
    while round(distance - dme_mapt, 1) > 1.1:
        distance -= 1.0
        dme_points.append(round(distance, 1))
    while round(distance - dme_mapt, 1) > 0.1:
        distance -= 0.2
        dme_points.append(round(distance, 1))
    dme_points.append(round(dme_mapt, 1))
    dme_points = sorted(set(dme_points), reverse=True)
    
    altitudes = np.linspace(tod_alt, mda, len(dme_points))
    table = pd.DataFrame({"DME (NM)": dme_points, "Altitude (ft)": altitudes.astype(int)})
    
    # Add SDF labels
    for sdf_nm, sdf_alt in sdf_list:
        sdf_nm_rounded = round(sdf_nm, 1)
        if sdf_nm_rounded in table["DME (NM)"].values:
            table.loc[table["DME (NM)"] == sdf_nm_rounded, "Note"] = f"SDF {sdf_alt} ft"
    
    return table

def generate_rod_table(faf_mapt, mda, tod_alt):
    descent_ft = tod_alt - mda
    gradient = descent_ft / (faf_mapt * 6076)  # in ft/ft
    speeds = [80, 100, 120, 140, 160]
    data = []
    for gs in speeds:
        vs = gradient * gs * 101.27  # ft/min
        time_sec = (faf_mapt * 6076) / (gs * 1.68781)
        data.append([gs, round(vs), f"{int(time_sec//60)}:{int(time_sec%60):02d}"])
    return pd.DataFrame(data, columns=["GS (kt)", "ROD (ft/min)", "Time (min:sec)"])

# PDF Export
def generate_pdf(dme_df, rod_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="CDFA PLANNER OUTPUT", ln=True, align="C")

    pdf.set_font("Arial", size=10)
    pdf.ln(10)
    pdf.cell(100, 10, "DME Table", ln=True)
    for idx, row in dme_df.iterrows():
        note = row.get("Note", "")
        pdf.cell(200, 8, f"{row['DME (NM)']} NM — {row['Altitude (ft)']} ft {note}", ln=True)

    pdf.ln(10)
    pdf.cell(100, 10, "ROD Table", ln=True)
    for idx, row in rod_df.iterrows():
        pdf.cell(200, 8, f"{row['GS (kt)']} kt — {row['ROD (ft/min)']} ft/min — Time: {row['Time (min:sec)']}", ln=True)

    return pdf.output(dest="S").encode("latin1")

# GENERATE OUTPUT
if generate:
    dme_df = generate_dme_table(tod_alt, mda, dme_thr, dme_mapt, sdf_list)
    rod_df = generate_rod_table(faf_mapt, mda, tod_alt)

    st.subheader("📊 DME Descent Table")
    st.dataframe(dme_df)

    st.subheader("📉 ROD Table")
    st.dataframe(rod_df)

    # Graph
    st.subheader("🧭 Descent Profile")
    fig, ax = plt.subplots()
    ax.plot(dme_df["DME (NM)"], dme_df["Altitude (ft)"], marker="o")
    ax.axhline(y=mda, color="red", linestyle="--", label="MDA")
    for idx, row in dme_df.iterrows():
        note = row.get("Note", "")
        if note:
            ax.annotate(note, (row["DME (NM)"], row["Altitude (ft)"] + 100))
    ax.set_xlabel("DME (NM)")
    ax.set_ylabel("Altitude (ft)")
    ax.set_title("CDFA Descent Profile")
    ax.grid(True)
    st.pyplot(fig)

    # PDF Export
    pdf_bytes = generate_pdf(dme_df, rod_df)
    st.download_button(label="📄 Download PDF", data=pdf_bytes, file_name="cdfa_output.pdf")

    # CSV Export
    csv_dme = dme_df.to_csv(index=False).encode('utf-8')
    csv_rod = rod_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download DME Table (CSV)", csv_dme, "dme_table.csv", "text/csv")
    st.download_button("⬇️ Download ROD Table (CSV)", csv_rod, "rod_table.csv", "text/csv")







