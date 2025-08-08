import streamlit as st
import pandas as pd
import numpy as np
import math
import io
from chart_parser import parse_iac_pdf
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
import threading

lock = threading.Lock()

st.set_page_config(layout="wide")
st.title("CDFA PLANNER")

# Sidebar
st.sidebar.header("📄 Upload IAC Chart (PDF)")
uploaded_pdf = st.sidebar.file_uploader("Upload IAC PDF", type=["pdf"])

if uploaded_pdf:
    parsed_data = parse_iac_pdf(uploaded_pdf)
else:
    parsed_data = {}

# Input Panel
st.header("Input Parameters")

thr_lat = st.text_input("1. THR Latitude", parsed_data.get("thr_lat", ""))
thr_lon = st.text_input("   THR Longitude", parsed_data.get("thr_lon", ""))
thr_elev = st.number_input("2. THR/TDZE Elevation (FT)", value=parsed_data.get("thr_elev", 0), step=1)

dme_lat = st.text_input("3. DME Latitude", parsed_data.get("dme_lat", ""))
dme_lon = st.text_input("   DME Longitude", parsed_data.get("dme_lon", ""))
dme_thr = st.number_input("4. DME at THR (NM)", value=parsed_data.get("dme_thr", 0.0), step=0.1)
dme_mapt = st.number_input("5. DME at MAPT (NM)", value=parsed_data.get("dme_mapt", 0.0), step=0.1)

tod_alt = st.number_input("6. TOD Altitude (FT)", value=parsed_data.get("tod_alt", 2500), step=50)
mda = st.number_input("7. MDA (FT)", value=parsed_data.get("mda", 1000), step=10)

# Optional GP Angle input
gp_angle = st.number_input("8. Glide Path Angle (°) (optional)", value=parsed_data.get("gp_angle", 0.0), step=0.1, format="%.1f")

# SDFs
st.subheader("Step-Down Fixes (Optional)")
sdfs = []
num_sdfs = st.number_input("Number of SDFs (max 6)", min_value=0, max_value=6, value=len(parsed_data.get("sdfs", [])))

for i in range(int(num_sdfs)):
    col1, col2 = st.columns(2)
    with col1:
        dme = st.number_input(f"SDF {i+1} - DME (NM)", value=parsed_data.get("sdfs", [{}]*6)[i].get("dme", 0.0), step=0.1)
    with col2:
        alt = st.number_input(f"SDF {i+1} - Altitude (FT)", value=parsed_data.get("sdfs", [{}]*6)[i].get("alt", 0), step=50)
    sdfs.append({"dme": dme, "alt": alt})

# FAF to MAPT
faf_mapt_dist = st.number_input("9. FAF to MAPT Distance (NM)", value=parsed_data.get("faf_mapt_dist", 5.0), step=0.1)

if st.button("Generate CDFA Profile"):
    # Compute Glide Path Angle if not given
    if gp_angle == 0.0:
        dist = dme_thr - dme_mapt
        height = tod_alt - mda
        if dist > 0:
            gp_angle = math.degrees(math.atan(height / (dist * 6076.12)))
        else:
            st.error("Invalid DME THR - MAPT distance to compute GP angle.")
            st.stop()

    # DME Table Generation
    dme_points = []
    dme_step = (dme_thr - dme_mapt) / 7
    for i in range(8):
        dme_dist = round(dme_thr - i * dme_step, 1)
        alt = tod_alt - (math.tan(math.radians(gp_angle)) * (dme_thr - dme_dist) * 6076.12)
        alt = max(mda, round(alt, 0))
        dme_points.append({"DME (NM)": dme_dist, "Altitude (FT)": alt})

    dme_df = pd.DataFrame(dme_points)

    # ROD Table Generation
    rod_points = []
    for gs in [80, 100, 120, 140, 160]:
        fpm = int(gs * 101.27 * math.tan(math.radians(gp_angle)))
        time_sec = int((faf_mapt_dist / gs) * 3600)
        rod_points.append({"GS (kt)": gs, "Time (sec)": time_sec, "ROD (ft/min)": fpm})
    rod_df = pd.DataFrame(rod_points)

    st.subheader("DME Table")
    st.dataframe(dme_df)

    st.subheader("ROD Table")
    st.dataframe(rod_df)

    # Plot
    fig, ax = plt.subplots()
    with lock:
        ax.plot(dme_df["DME (NM)"], dme_df["Altitude (FT)"], marker='o', label="CDFA Path")
        ax.axhline(y=mda, color='r', linestyle='--', label="MDA")
        for sdf in sdfs:
            ax.plot(sdf["dme"], sdf["alt"], marker='x', color='green')
            ax.text(sdf["dme"], sdf["alt"], f"SDF", fontsize=8)
        ax.set_xlabel("DME (NM)")
        ax.set_ylabel("Altitude (FT)")
        ax.set_title("CDFA Descent Profile")
        ax.invert_xaxis()
        ax.legend()
        st.pyplot(fig)

    # Export
    st.subheader("📥 Download Tables")
    csv_dme = dme_df.to_csv(index=False).encode()
    st.download_button("Download DME Table (CSV)", csv_dme, "dme_table.csv", "text/csv")

    csv_rod = rod_df.to_csv(index=False).encode()
    st.download_button("Download ROD Table (CSV)", csv_rod, "rod_table.csv", "text/csv")

    # PDF Export
    def generate_pdf(dme, rod):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = [Paragraph("CDFA Planner Output", styles["Title"]), Spacer(1, 12)]

        dme_data = [dme.columns.tolist()] + dme.values.tolist()
        rod_data = [rod.columns.tolist()] + rod.values.tolist()

        dme_table = Table(dme_data)
        dme_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.gray),
                                       ("GRID", (0, 0), (-1, -1), 1, colors.black)]))
        rod_table = Table(rod_data)
        rod_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.gray),
                                       ("GRID", (0, 0), (-1, -1), 1, colors.black)]))

        elements += [Paragraph("DME Table", styles["Heading2"]), dme_table, Spacer(1, 12),
                     Paragraph("ROD Table", styles["Heading2"]), rod_table]

        doc.build(elements)
        pdf = buffer.getvalue()
        buffer.close()
        return pdf

    pdf_bytes = generate_pdf(dme_df, rod_df)
    st.download_button("Download PDF", data=pdf_bytes, file_name="cdfa_output.pdf", mime="application/pdf")
