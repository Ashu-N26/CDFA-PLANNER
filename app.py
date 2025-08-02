import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from fpdf import FPDF
from io import BytesIO
import pytesseract
from pdf2image import convert_from_bytes

st.set_page_config(page_title="DME/CDFA Descent Planner", layout="centered")

st.title("🛫 Integrated DME‑CDFA Descent Planner")

# ---------------------------------------------------------------------
# Step 1: PDF Parser (optional upload)
pdf_file = st.file_uploader("Upload IAC Chart PDF (optional)", type=["pdf"])
parsed = {"gp_angle": None, "faf_dme": None, "mapt_dme": None, "mda": None}

if pdf_file:
    images = convert_from_bytes(pdf_file.read(), dpi=300)
    text = "\n".join(pytesseract.image_to_string(img) for img in images[:2])
    import re
    gp = re.search(r"GP[: ]*([0-9]\.[0-9])°", text)
    gd = re.search(r"MDA[: ]*([0-9]{3,4})", text)
    faf = re.search(r"FAF[ ]*DME[: ]*([0-9]\.[0-9])", text)
    mapt = re.search(r"MAPt[ ]*DME[: ]*([0-9]\.[0-9])", text)
    if gp:
        parsed["gp_angle"] = float(gp.group(1))
    if gd:
        parsed["mda"] = int(gd.group(1))
    if faf:
        parsed["faf_dme"] = float(faf.group(1))
    if mapt:
        parsed["mapt_dme"] = float(mapt.group(1))
    st.success("PDF chart parsed (some values may need manual adjustment)")

# ---------------------------------------------------------------------
# Step 2: Inputs (manual override)
col1, col2 = st.columns(2)
with col1:
    gp_angle = st.number_input("Glide Path (°)", min_value=2.0, max_value=4.0, step=0.01,
                               value=parsed["gp_angle"] or 3.00)
    mda = st.number_input("MDA (ft)", min_value=200, max_value=10000,
                          value=parsed["mda"] or 800)
    faf_dme = st.number_input("FAF DME (NM)", min_value=0.5, value=parsed["faf_dme"] or 6.7)
with col2:
    mapt_dme = st.number_input("MAPt DME (NM)", min_value=0.1, value=parsed["mapt_dme"] or 1.0)
    elevation = st.number_input("Threshold Elevation (ft)", value=100)
    faf_altitude = st.number_input("FAF Altitude (ft)", value=2000)

sdfs = []
for i in range(1,7):
    d = st.number_input(f"SDF {i} Distance (NM)", min_value=0.0, step=0.1, key=f"sdf_d{i}")
    a = st.number_input(f"SDF {i} Altitude (ft)", min_value=0, key=f"sdf_a{i}")
    if d>0 and a>0:
        sdfs.append({"Distance": d, "Altitude": a, "Label": f"SDF{i}"})

submitted = st.button("Generate Descent Plan")

# ---------------------------------------------------------------------
if submitted:
    distance_nm = round(faf_dme - mapt_dme, 2)
    height_diff = faf_altitude - mda
    angle_rad = math.radians(gp_angle)
    distances = np.linspace(faf_dme, mapt_dme, 8)
    altitudes = mda + (distances - mapt_dme)*6076.12*math.tan(angle_rad)
    altitudes = np.maximum(altitudes, mda)

    # Build DME table
    dme_rows = []
    for i, d in enumerate(distances):
        d_alt = int(round(altitudes[i]))
        lab = ""
        if abs(d - faf_dme) < 0.01: lab = "FAF"
        elif abs(d - mapt_dme) < 0.1: lab = "MAPt"
        for sdf in sdfs:
            if abs(d - sdf["Distance"])<0.15:
                d_alt = sdf["Altitude"]
                lab = sdf["Label"]
        dme_rows.append({"DME": round(d,2), "Altitude":d_alt, "Fix":lab})
    dme_df = pd.DataFrame(dme_rows)

    # ROD table
    gs_list = [80,100,120,140,160]
    rod_rows = []
    for gs in gs_list:
        gs_ft = gs*6076.12/60
        rod = round(gs_ft*math.tan(angle_rad)/10)*10
        t = distance_nm/gs*60
        rod_rows.append({"GS": gs, "ROD (fpm)": rod, "Time (min)": f"{int(t)}:{int((t%1)*60):02d}"})
    rod_df = pd.DataFrame(rod_rows)

    # Display tables
    st.subheader("DME Descent Table")
    st.table(dme_df)
    st.subheader("ROD Table (FAF → MAPt)")
    st.table(rod_df)
    st.markdown(f"**GP Angle:** {gp_angle:.2f}°   **Descent Gradient:** {round(height_diff/distance_nm):.0f} ft/NM")

    # Chart
    fig, ax = plt.subplots()
    ax.plot(dme_df["DME"], dme_df["Altitude"], marker='o', color='blue')
    ax.axhline(mda, color='red', linestyle='--', label='MDA')
    for _, row in dme_df.iterrows():
        ax.text(row["DME"], row["Altitude"]+50, row["Fix"], ha='center', fontsize=8)
    ax.set_xlabel("DME (NM)")
    ax.set_ylabel("Altitude (ft)")
    ax.set_title("Stabilized Descent Profile")
    ax.invert_xaxis()
    st.pyplot(fig)

    # PDF Export
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial","B",14)
    pdf.cell(0,8, "DME/CDFA Descent Planner Report",ln=1,align="C")
    for _, r in dme_df.iterrows():
        pdf.set_font("Arial","",10)
        pdf.cell(0,5, f"{r['DME']} NM — {r['Altitude']} ft — {r['Fix']}", ln=1)
    pdf.ln(4)
    for _, r in rod_df.iterrows():
        pdf.cell(0,5, f"{r['GS']} kt → ROD {r['ROD (fpm)']} fpm | {r['Time (min)']}", ln=1)
    buf = BytesIO(); pdf.output(buf); buf.seek(0)
    st.download_button("Download PDF", data=buf, file_name="DME_CDFAPLANNER.pdf", mime="application/pdf")

    st.download_button("Download DME CSV", data=dme_df.to_csv(index=False), file_name="dme.csv")
    st.download_button("Download ROD CSV", data=rod_df.to_csv(index=False), file_name="rod.csv")
