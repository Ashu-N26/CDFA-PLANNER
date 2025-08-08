import streamlit as st
import pandas as pd
import numpy as np
import math
from parser import parse_iac_pdf
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
import base64

st.set_page_config(page_title="CDFA-PLANNER", layout="wide")

st.title("CDFA-PLANNER ✈️")
st.markdown("Plan continuous, stabilized descents for **non-precision approaches** — with optional IAC PDF parsing.")

# ====== PDF Upload (Optional) =======
uploaded_file = st.file_uploader("📄 Upload IAC PDF (optional)", type=["pdf"])

autofill_data = {}
if uploaded_file:
    autofill_data = parse_iac_pdf(uploaded_file)

# ====== Inputs ======
st.header("🔧 Input Fields")

col1, col2 = st.columns(2)

with col1:
    thr_lat = st.text_input("THR Latitude", autofill_data.get("thr_lat", ""))
    thr_lon = st.text_input("THR Longitude", autofill_data.get("thr_lon", ""))
    thr_elev = st.number_input("THR / TDZE Elevation (FT)", value=float(autofill_data.get("thr_elev", 0)))
    dme_lat = st.text_input("DME Latitude", autofill_data.get("dme_lat", ""))
    dme_lon = st.text_input("DME Longitude", autofill_data.get("dme_lon", ""))
    dme_thr = st.number_input("DME at Threshold (NM)", value=float(autofill_data.get("dme_thr", 0)))
    dme_mapt = st.number_input("DME at MAPT (NM)", value=float(autofill_data.get("dme_mapt", 0)))

with col2:
    tod_alt = st.number_input("TOD Altitude (FT)", value=float(autofill_data.get("tod_alt", 0)))
    mda = st.number_input("MDA (FT)", value=float(autofill_data.get("mda", 0)))
    faf_mapt = st.number_input("FAF to MAPT Distance (NM)", value=float(autofill_data.get("faf_mapt", 0)))
    gp_angle = st.text_input("GP Angle (Optional)", autofill_data.get("gp_angle", ""))
    
    num_sdf = st.slider("Number of SDFs", min_value=0, max_value=6, value=int(autofill_data.get("sdf_count", 0)))
    sdf_list = []
    for i in range(num_sdf):
        dist = st.number_input(f"SDF {i+1} Distance (NM)", key=f"sdf_dist_{i}")
        alt = st.number_input(f"SDF {i+1} Altitude (FT)", key=f"sdf_alt_{i}")
        sdf_list.append((dist, alt))

# ====== Logic Functions ======
def generate_dme_table(tod_alt, mda, dme_thr, dme_mapt):
    num_points = 8
    dme_values = np.linspace(dme_thr, dme_mapt, num_points)
    altitudes = np.linspace(tod_alt, mda, num_points)
    df = pd.DataFrame({
        "DME (NM)": np.round(dme_values, 1),
        "Altitude (FT)": np.round(altitudes, 0)
    })
    return df

def generate_rod_table(gp_angle, faf_mapt):
    if not gp_angle:
        gp_angle = math.degrees(math.atan((tod_alt - mda) / (faf_mapt * 6076.12)))  # feet per NM
    else:
        gp_angle = float(gp_angle)

    groundspeeds = [80, 100, 120, 140, 160]
    rods = []
    for gs in groundspeeds:
        time_sec = (faf_mapt / gs) * 3600
        rod = gs * math.tan(math.radians(gp_angle)) * 101.27  # ft/min
        rods.append((gs, round(rod), int(time_sec)))

    df = pd.DataFrame(rods, columns=["GS (kt)", "ROD (ft/min)", "Time (sec)"])
    return df

# ====== Generate Tables ======
if st.button("📊 Generate DME & ROD Tables"):
    if dme_thr == dme_mapt or tod_alt == mda:
        st.error("Please ensure TOD altitude and DME distances are valid.")
    else:
        dme_df = generate_dme_table(tod_alt, mda, dme_thr, dme_mapt)
        rod_df = generate_rod_table(gp_angle, faf_mapt)

        st.subheader("📘 DME Table")
        st.dataframe(dme_df)

        st.subheader("📙 ROD Table")
        st.dataframe(rod_df)

        # ====== Export Buttons ======
        def generate_pdf(dme, rod):
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph("CDFA Planner – DME Table", styles["Title"]))
            dme_table = Table([dme.columns.tolist()] + dme.values.tolist())
            dme_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black)]))
            elements.append(dme_table)

            elements.append(Paragraph("Rate of Descent Table", styles["Title"]))
            rod_table = Table([rod.columns.tolist()] + rod.values.tolist())
            rod_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black)]))
            elements.append(rod_table)

            doc.build(elements)
            buffer.seek(0)
            return buffer

        pdf = generate_pdf(dme_df, rod_df)
        b64 = base64.b64encode(pdf.read()).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="CDFA_DME_ROD_Tables.pdf">📥 Download PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

        # CSV Export
        st.download_button("📥 Download DME Table (CSV)", dme_df.to_csv(index=False), file_name="dme_table.csv")
        st.download_button("📥 Download ROD Table (CSV)", rod_df.to_csv(index=False), file_name="rod_table.csv")










