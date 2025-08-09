import streamlit as st
import pandas as pd
import numpy as np
import math
import datetime
from io import BytesIO
import os
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ------------------ CONFIG ------------------
RUN_AUTO_TESTS = False  # Toggle True in dev for auto-tests

DEFAULT_GP_MIN = 2.5
DEFAULT_GP_RAISE_TO = 3.0
DEFAULT_GP_MAX_WARN = 4.5
DEFAULT_GS_LIST = [80, 100, 120, 140, 160]

# ------------------ HELPERS ------------------
def parse_pdf_fields(pdf_bytes):
    parsed = {"mda_ft": None, "gp_deg": None, "dme_list": [], "coords": None}
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except:
        text = ""
    if not text.strip():
        images = convert_from_bytes(pdf_bytes)
        text = "\n".join(pytesseract.image_to_string(img) for img in images)

    import re
    mda_match = re.search(r"MDA[:\s]+(\d{3,4})", text)
    gp_match = re.search(r"GP[:\s]+(\d\.\d)", text)
    dme_matches = re.findall(r"(\d{1,2}\.\d)\s*NM", text)

    if mda_match: parsed["mda_ft"] = int(mda_match.group(1))
    if gp_match: parsed["gp_deg"] = float(gp_match.group(1))
    if dme_matches: parsed["dme_list"] = sorted(set(float(x) for x in dme_matches), reverse=True)
    return parsed

def compute_gp(tod_alt_ft, mda_ft, tod_dme_nm, mapt_dme_nm, gp_min, gp_raise_to, gp_max_warn):
    horiz_nm = abs(tod_dme_nm - mapt_dme_nm)
    if horiz_nm < 0.05:
        horiz_nm = 0.05
    derived_gp = math.degrees(math.atan2(tod_alt_ft - mda_ft, horiz_nm * 6076.12))
    gp_used = derived_gp
    warnings = []
    if gp_used < gp_min:
        gp_used = gp_raise_to
        warnings.append(f"GP raised from {derived_gp:.2f}° to {gp_raise_to:.2f}°")
    if gp_used > gp_max_warn:
        warnings.append(f"Steep GP {gp_used:.2f}° > {gp_max_warn:.2f}° — check chart")
    return gp_used, warnings

def generate_dme_table(tod_dme, tod_alt, mapt_dme, mda, gp_deg, sdf_list):
    points = []
    dist_total = abs(tod_dme - mapt_dme)
    sdf_map = {round(sdf_dme, 1): sdf_alt for sdf_alt, sdf_dme in sdf_list}

    # ensure ~1 NM spacing, at least 8 points
    step_nm = max(dist_total / 7, 1.0)  # ensures ≥ 8 incl start/end
    dme_values = list(np.arange(tod_dme, mapt_dme - 0.01, -step_nm))
    if mapt_dme not in dme_values:
        dme_values.append(mapt_dme)

    for dme in sorted(set(round(x, 1) for x in dme_values), reverse=True):
        if round(dme, 1) in sdf_map:
            alt = sdf_map[round(dme, 1)]
            sdf_mark = True
        else:
            horiz_nm = abs(tod_dme - dme)
            alt = tod_alt - math.tan(math.radians(gp_deg)) * horiz_nm * 6076.12
            sdf_mark = False
        points.append((round(dme, 1), int(round(alt)), sdf_mark))

    df = pd.DataFrame(points, columns=["DME (NM)", "Altitude (ft)", "SDF"])
    return df

def generate_rod_summary(faf_mapt_nm, faf_alt, mda_alt, gs_list):
    rod_rows = []
    alt_diff = faf_alt - mda_alt
    for gs in gs_list:
        time_min = (faf_mapt_nm / gs) * 60
        rod_fpm = alt_diff / time_min if time_min > 0 else 0
        mins = int(time_min)
        secs = int((time_min - mins) * 60)
        rod_rows.append([gs, round(rod_fpm), f"{mins}:{secs:02d}"])
    return pd.DataFrame(rod_rows, columns=["GS (kt)", "ROD (ft/min)", "Time FAF→MAPT (min:sec)"])

def generate_profile_plot(dme_df):
    fig, ax = plt.subplots()
    ax.plot(dme_df["DME (NM)"], dme_df["Altitude (ft)"], marker="o")
    for idx, row in dme_df.iterrows():
        ax.text(row["DME (NM)"], row["Altitude (ft)"]+50, f"{row['Altitude (ft)']}", fontsize=8)
    ax.set_xlabel("DME (NM)")
    ax.set_ylabel("Altitude (ft)")
    ax.set_title("Approach Profile")
    ax.grid(True)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

def generate_pdf_report(proc_id, gp_used, dme_df, rod_df, fig_buf, logo_path=None):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=30, rightMargin=30, topMargin=20, bottomMargin=20)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CenterBold', fontName='Helvetica-Bold', fontSize=16, alignment=1))
    styles.add(ParagraphStyle(name='Meta', fontName='Helvetica', fontSize=9, alignment=1))
    styles.add(ParagraphStyle(name='Footer', fontName='Helvetica-Oblique', fontSize=8, alignment=1))
    elements = []

    # Title row
    title = Paragraph("CDFA-Planner – Stabilized Approach Report", styles['CenterBold'])
    meta = Paragraph(f"Procedure: {proc_id} | GP Used: {gp_used:.2f}° | Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Meta'])
    if logo_path and os.path.exists(logo_path):
        elements.append(RLImage(logo_path, width=60, height=40))
    elements += [title, meta, Spacer(1, 8)]

    # Format DME table
    dme_data = [["DME (NM)", "Altitude (ft)", "SDF"]] + [
        [f"{d:.1f}", f"{a}", "✓" if s else ""] for d, a, s in dme_df.values
    ]
    dme_table = Table(dme_data, colWidths=[50, 70, 30])
    dme_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#D9D9D9")),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.25, colors.black),
    ]))

    # Format ROD summary table
    rod_data = [["GS (kt)", "ROD (ft/min)", "Time FAF→MAPT"]] + rod_df.values.tolist()
    rod_table = Table(rod_data, colWidths=[50, 70, 80])
    rod_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#D9D9D9")),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.25, colors.black),
    ]))

    # Place tables side-by-side
    combined = Table([[dme_table, rod_table]], colWidths=[200, 230])
    elements.append(combined)
    elements.append(Spacer(1, 12))

    # Profile plot
    img = RLImage(fig_buf, width=400, height=200)
    elements += [img, Spacer(1, 10)]

    # Footer
    footer = Paragraph("Verify against official AIP/Jeppesen/NavBlue charts before operational use.", styles['Footer'])
    elements.append(footer)

    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()

# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="CDFA-Planner", layout="wide")
st.title("CDFA-Planner – Stabilized Approach Generator")

col1, col2 = st.columns(2)
with col1:
    thr_lat = st.number_input("THR Latitude", format="%.6f")
    thr_lon = st.number_input("THR Longitude", format="%.6f")
    thr_elev = st.number_input("THR/TDZE Elev (ft)", value=0)
    dme_lat = st.number_input("DME Latitude", format="%.6f")
    dme_lon = st.number_input("DME Longitude", format="%.6f")
    dme_at_thr = st.number_input("DME at THR (NM)", value=0.0, format="%.1f")
    dme_at_mapt = st.number_input("DME at MAPT (NM)", value=0.0, format="%.1f")
with col2:
    tod_alt = st.number_input("TOD Altitude (ft)", value=0)
    mda_alt = st.number_input("MDA Altitude (ft)", value=0)
    faf_mapt_nm = st.number_input("FAF–MAPT Distance (NM)", value=0.0, format="%.1f")
    num_sdf = st.number_input("Number of SDFs", min_value=0, max_value=6, step=1)
    sdf_list = []
    for i in range(num_sdf):
        sdf_alt = st.number_input(f"SDF {i+1} Altitude (ft)", value=0)
        sdf_dme = st.number_input(f"SDF {i+1} DME (NM)", value=0.0, format="%.1f")
        sdf_list.append((sdf_alt, sdf_dme))

if st.button("Generate Tables"):
    gp_used, gp_warn = compute_gp(tod_alt, mda_alt, dme_at_thr, dme_at_mapt,
                                  DEFAULT_GP_MIN, DEFAULT_GP_RAISE_TO, DEFAULT_GP_MAX_WARN)
    if gp_warn:
        st.warning("\n".join(gp_warn))
    dme_df = generate_dme_table(dme_at_thr, tod_alt, dme_at_mapt, mda_alt, gp_used, sdf_list)
    rod_df = generate_rod_summary(faf_mapt_nm, tod_alt, mda_alt, DEFAULT_GS_LIST)
    fig_buf = generate_profile_plot(dme_df)

    st.subheader("DME Table")
    st.dataframe(dme_df)

    st.subheader("ROD Summary Table")
    st.dataframe(rod_df)

    csv_data = dme_df.to_csv(index=False).encode()
    st.download_button("Download DME Table (CSV)", csv_data, "dme_table.csv", "text/csv")

    pdf_bytes = generate_pdf_report("PROC-001", gp_used, dme_df, rod_df, fig_buf, logo_path="logo.png")
    st.download_button("Download PDF Report", pdf_bytes, "cdaf_report.pdf", "application/pdf")









