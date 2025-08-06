import streamlit as st
import pandas as pd
import numpy as np
import math
from io import BytesIO
from fpdf import FPDF
import base64
from pdf_parser import parse_pdf_chart

st.set_page_config(layout="wide")
st.title("CDFA-PLANNER")

# -------------------- INPUTS --------------------

st.header("1️⃣ INPUT PANEL")

col1, col2 = st.columns(2)
with col1:
    thr_lat = st.text_input("THR Latitude", "12.982")
    thr_lon = st.text_input("THR Longitude", "77.607")
    thr_elev = st.number_input("THR / TDZE Elevation (ft)", value=300)

    dme_lat = st.text_input("DME Latitude", "12.990")
    dme_lon = st.text_input("DME Longitude", "77.600")

    dme_thr = st.number_input("DME at THR (NM)", value=0.5)
    dme_mapt = st.number_input("DME at MAPt (NM)", value=1.2)
    tod_alt = st.number_input("Top of Descent Altitude (FT)", value=3000)
    mda = st.number_input("Minimum Descent Altitude (MDA) (FT)", value=1000)

with col2:
    gp_angle = st.text_input("Glide Path Angle (deg) (optional)", "")
    faf_mapt = st.number_input("FAF to MAPt Distance (NM)", value=2.0)

    sdf_data = []
    sdf_count = st.number_input("Number of SDFs (Step-Down Fixes)", min_value=0, max_value=6, value=1)
    for i in range(int(sdf_count)):
        col_sdf1, col_sdf2 = st.columns(2)
        with col_sdf1:
            sdf_dist = st.number_input(f"SDF {i+1} Distance (NM)", key=f"dist_{i}", value=3.5 - i)
        with col_sdf2:
            sdf_alt = st.number_input(f"SDF {i+1} Altitude (FT)", key=f"alt_{i}", value=1500 - 200 * i)
        sdf_data.append((sdf_dist, sdf_alt))

st.markdown("#### 📎 Optional: Upload AIP/IAC Chart PDF")
uploaded_pdf = st.file_uploader("Upload Approach Chart (PDF)", type="pdf")

if uploaded_pdf:
    try:
        extracted = parse_pdf_chart(uploaded_pdf)
        st.success("✅ Parsed PDF Chart Data")
        st.write(extracted)
        # Auto-fill inputs (this assumes dictionary keys match)
        if extracted.get("gp_angle"):
            gp_angle = extracted["gp_angle"]
        if extracted.get("tod_alt"):
            tod_alt = extracted["tod_alt"]
        if extracted.get("mda"):
            mda = extracted["mda"]
    except Exception as e:
        st.warning("PDF Parsing failed. You can manually enter data.")

# -------------------- GP ANGLE GENERATION --------------------

def compute_gp_angle(tod_alt, mda, tod_dme, mda_dme):
    delta_alt = tod_alt - mda
    delta_dme = (tod_dme - mda_dme) * 1.852 * 1852  # NM to meters
    angle_rad = math.atan(delta_alt / delta_dme)
    return round(math.degrees(angle_rad), 2)

try:
    gp_angle_val = float(gp_angle)
except:
    gp_angle_val = compute_gp_angle(tod_alt, mda, sdf_data[0][0] if sdf_data else dme_mapt + 2.0, dme_mapt)
    st.info(f"Auto-calculated GP angle: {gp_angle_val}°")

# -------------------- DME + ALT TABLE GENERATION --------------------

def generate_dme_table(tod_alt, mda, tod_dme, mda_dme, sdf_data):
    dmes = [round(x, 1) for x in np.linspace(tod_dme, mda_dme, 8)]
    alts = []
    for d in dmes:
        alt = tod_alt - (tod_alt - mda) * ((tod_dme - d) / (tod_dme - mda_dme))
        # apply step-down fix restrictions
        for sdf_d, sdf_a in sdf_data:
            if d <= sdf_d:
                alt = max(alt, sdf_a)
        alts.append(int(round(alt, -1)))
    return pd.DataFrame({"DME (NM)": dmes, "Altitude (FT)": alts})

dme_df = generate_dme_table(tod_alt, mda, sdf_data[0][0] if sdf_data else dme_mapt + 2.0, dme_mapt, sdf_data)

# -------------------- ROD TABLE --------------------

def generate_rod_table(angle, faf_dist):
    speeds = [80, 100, 120, 140, 160]
    data = []
    for spd in speeds:
        vs = int(round(101.27 * spd * math.tan(math.radians(angle))))
        time = round((faf_dist / spd) * 60, 1)
        data.append((f"{spd} kt", f"{vs} fpm", f"{time} sec"))
    return pd.DataFrame(data, columns=["GS", "ROD", "Time"])

rod_df = generate_rod_table(gp_angle_val, faf_mapt)

# -------------------- DISPLAY --------------------

st.header("2️⃣ OUTPUT TABLES")

st.subheader("DME Descent Table")
st.dataframe(dme_df)

st.subheader("Rate of Descent Table")
st.dataframe(rod_df)

# -------------------- EXPORT --------------------

def generate_pdf(dme_df, rod_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "CDFA-PLANNER: Descent + ROD Table", ln=True)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "DME Table", ln=True)
    pdf.set_font("Arial", "", 11)
    for idx, row in dme_df.iterrows():
        pdf.cell(0, 10, f"{row['DME (NM)']} NM - {row['Altitude (FT)']} ft", ln=True)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Rate of Descent Table", ln=True)
    pdf.set_font("Arial", "", 11)
    for idx, row in rod_df.iterrows():
        pdf.cell(0, 10, f"{row['GS']} - {row['ROD']} - {row['Time']}", ln=True)

    return pdf.output(dest="S").encode("latin1")

st.subheader("📥 Export Options")

pdf_bytes = generate_pdf(dme_df, rod_df)
st.download_button("Download PDF", data=pdf_bytes, file_name="CDFA_output.pdf")

csv_dme = dme_df.to_csv(index=False).encode("utf-8")
csv_rod = rod_df.to_csv(index=False).encode("utf-8")

st.download_button("Download DME Table (CSV)", data=csv_dme, file_name="dme_table.csv")
st.download_button("Download ROD Table (CSV)", data=csv_rod, file_name="rod_table.csv")








