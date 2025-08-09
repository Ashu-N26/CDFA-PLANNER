import streamlit as st
import pandas as pd
import numpy as np
import math
import datetime
from io import BytesIO
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import re

# ------------------ CONFIG ------------------
RUN_AUTO_TESTS = False  # Toggle True in development to run built-in tests

DEFAULT_GP_MIN = 2.5
DEFAULT_GP_RAISE_TO = 3.0
DEFAULT_GP_MAX_WARN = 4.5
DEFAULT_GS_LIST = [80, 100, 120, 140, 160]

# ------------------ HELPERS ------------------
def haversine_nm(lat1, lon1, lat2, lon2):
    """Calculate horizontal great-circle distance in NM."""
    R_nm = 3440.065
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return R_nm * 2 * math.asin(math.sqrt(a))

def parse_pdf_fields(pdf_bytes):
    """Parse MDA, GP, DME, coords from PDF using pdfplumber + OCR fallback."""
    parsed = {"mda_ft": None, "gp_deg": None, "dme_list": [], "coords": None}
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception:
        text = ""
    if not text.strip():
        # OCR fallback for scanned PDF
        images = convert_from_bytes(pdf_bytes)
        text = "\n".join(pytesseract.image_to_string(img) for img in images)

    mda_match = re.search(r"MDA[:\s]+(\d{3,4})", text)
    gp_match = re.search(r"GP[:\s]+(\d\.\d)", text)
    dme_matches = re.findall(r"(\d{1,2}\.\d)\s*NM", text)
    coord_match = re.search(r"(\d{1,2}\.\d+)[, ]+(\d{1,3}\.\d+)", text)

    if mda_match:
        parsed["mda_ft"] = int(mda_match.group(1))
    if gp_match:
        parsed["gp_deg"] = float(gp_match.group(1))
    if dme_matches:
        parsed["dme_list"] = sorted(set(float(x) for x in dme_matches), reverse=True)
    if coord_match:
        parsed["coords"] = (float(coord_match.group(1)), float(coord_match.group(2)))
    return parsed

def compute_gp_and_points(tod_alt_ft, thr_elev_ft, tod_dme_nm, mda_ft, mda_dme_nm,
                          gp_min, gp_raise_to, gp_max_warn):
    """Derive GP, clamp if needed, return GP and warnings."""
    horiz_nm = abs(tod_dme_nm - mda_dme_nm)
    if horiz_nm < 0.05:
        horiz_nm = 0.05
    derived_gp = math.degrees(math.atan2(tod_alt_ft - mda_ft, horiz_nm * 6076.12))
    gp_used = derived_gp
    warnings = []
    if gp_used < gp_min:
        gp_used = gp_raise_to
        warnings.append(f"GP raised from {derived_gp:.2f}° to {gp_raise_to:.2f}° (min {gp_min:.2f}°)")
    if gp_used > gp_max_warn:
        warnings.append(f"Steep GP {gp_used:.2f}° > {gp_max_warn:.2f}° — verify chart.")
    return gp_used, warnings

def generate_dme_table(tod_dme_nm, tod_alt_ft, mda_dme_nm, mda_ft, gp_deg, sdf_list):
    """Generate DME vs altitude table from TOD to MDA including SDFs."""
    points = [(tod_dme_nm, tod_alt_ft)]
    for sdf_alt, sdf_dme in sdf_list:
        if mda_dme_nm <= sdf_dme <= tod_dme_nm:
            points.append((sdf_dme, sdf_alt))
    points.append((mda_dme_nm, mda_ft))
    points = sorted(points, key=lambda x: -x[0])
    df = pd.DataFrame(points, columns=["DME (NM)", "Altitude (ft)"])
    return df

def generate_rod_table(dme_df, gs_list):
    """Generate ROD table given DME vs altitude."""
    rod_rows = []
    for gs in gs_list:
        rods = []
        times = []
        for i in range(len(dme_df) - 1):
            d_nm = dme_df.iloc[i]["DME (NM)"] - dme_df.iloc[i+1]["DME (NM)"]
            d_ft = dme_df.iloc[i]["Altitude (ft)"] - dme_df.iloc[i+1]["Altitude (ft)"]
            if d_nm <= 0:
                rod, time = 0, 0
            else:
                time_min = (d_nm / gs) * 60
                rod = d_ft / time_min if time_min > 0 else 0
                time = time_min
            rods.append(round(rod))
            times.append(round(time, 2))
        rod_rows.append([gs] + rods + [sum(times)])
    cols = ["GS (kt)"] + [f"Seg{i+1} ROD" for i in range(len(dme_df)-1)] + ["Total Time (min)"]
    return pd.DataFrame(rod_rows, columns=cols)

def generate_pdf_report(proc_id, gp_used, dme_df, rod_df, fig_buf):
    """Generate NavBlue-style PDF report."""
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TableHeader', fontName='Helvetica-Bold', fontSize=9, alignment=1))
    styles.add(ParagraphStyle(name='TableCell', fontName='Helvetica', fontSize=9, alignment=1))
    elements = []
    title = Paragraph(f"<b>CDFA-PLANNER — Advanced DME & ROD Report</b>", styles['Title'])
    subtitle = Paragraph(f"Procedure: {proc_id} &nbsp;&nbsp;|&nbsp;&nbsp; GP Used: {gp_used:.2f}° &nbsp;&nbsp;|&nbsp;&nbsp; Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
    elements += [title, subtitle, Spacer(1, 8)]

    def make_table(data):
        t = Table(data, hAlign='CENTER')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#D9D9D9')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 9),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,1), (-1,-1), 9),
            ('GRID', (0,0), (-1,-1), 0.3, colors.black),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ('TOPPADDING', (0,0), (-1,-1), 3),
        ]))
        return t

    dme_data = [list(dme_df.columns)] + dme_df.round(2).values.tolist()
    elements += [Paragraph("<b>DME / Altitude Table</b>", styles['Heading3']), make_table(dme_data), Spacer(1, 10)]
    rod_data = [list(rod_df.columns)] + rod_df.round(0).values.tolist()
    elements += [Paragraph("<b>ROD Table</b>", styles['Heading3']), make_table(rod_data), Spacer(1, 10)]

    img = RLImage(fig_buf, width=420, height=200)
    elements += [Paragraph("<b>Approach Profile</b>", styles['Heading3']), img, Spacer(1, 10)]
    footer = Paragraph("Generated by CDFA-PLANNER — validate against official charts.", styles['Italic'])
    elements.append(footer)

    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()

# ------------------ STREAMLIT UI ------------------
st.title("✈️ Advanced CDFA / DME-ROD Planner")

with st.sidebar:
    st.header("Inputs")
    proc_id = st.text_input("Procedure ID / Name")
    thr_elev_ft = st.number_input("Threshold Elevation (ft)", value=0)
    tod_alt_ft = st.number_input("TOD Altitude (ft)", value=3000)
    tod_dme_nm = st.number_input("TOD DME (NM)", value=8.0)
    mda_ft = st.number_input("MDA (ft)", value=500)
    mda_dme_nm = st.number_input("MDA DME (NM)", value=0.5)
    gp_input = st.number_input("Glide Path (deg) (0=derive)", value=0.0)

    st.subheader("Step-Down Fixes (SDF)")
    num_sdf = st.number_input("Number of SDFs", min_value=0, max_value=5, value=0)
    sdf_list = []
    for i in range(num_sdf):
        sdf_alt = st.number_input(f"SDF {i+1} Altitude (ft)", value=1000, key=f"sdf_alt_{i}")
        sdf_dme = st.number_input(f"SDF {i+1} DME (NM)", value=3.0, key=f"sdf_dme_{i}")
        sdf_list.append((sdf_alt, sdf_dme))

    st.subheader("Advanced Settings")
    gp_min = st.number_input("GP Minimum (°)", value=DEFAULT_GP_MIN)
    gp_raise_to = st.number_input("GP Raise To (°)", value=DEFAULT_GP_RAISE_TO)
    gp_max_warn = st.number_input("GP Max Warning (°)", value=DEFAULT_GP_MAX_WARN)
    gs_list = st.text_input("Ground Speeds (comma)", value="80,100,120,140,160")
    gs_list = [int(x.strip()) for x in gs_list.split(",") if x.strip().isdigit()]

# Compute GP
gp_used, gp_warnings = compute_gp_and_points(
    tod_alt_ft, thr_elev_ft, tod_dme_nm, mda_ft, mda_dme_nm,
    gp_min, gp_raise_to, gp_max_warn
) if gp_input == 0 else (gp_input, [])

# Generate tables
dme_df = generate_dme_table(tod_dme_nm, tod_alt_ft, mda_dme_nm, mda_ft, gp_used, sdf_list)
rod_df = generate_rod_table(dme_df, gs_list)

# Plot
fig, ax = plt.subplots()
ax.plot(dme_df["DME (NM)"], dme_df["Altitude (ft)"], marker="o")
ax.set_xlabel("DME (NM)")
ax.set_ylabel("Altitude (ft)")
ax.invert_xaxis()
ax.grid(True)
st.pyplot(fig)

# Show warnings
for w in gp_warnings:
    st.warning(w)

# Show tables
st.subheader("DME Table")
st.dataframe(dme_df)
st.subheader("ROD Table")
st.dataframe(rod_df)

# Download PDF
img_buf = BytesIO()
fig.savefig(img_buf, format='png', dpi=200, bbox_inches='tight')
img_buf.seek(0)
pdf_bytes = generate_pdf_report(proc_id or "Procedure", gp_used, dme_df, rod_df, img_buf)
st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name=f"{proc_id or 'procedure'}_report.pdf", mime="application/pdf")

# Download CSV
csv_bytes = dme_df.to_csv(index=False).encode()
st.download_button("⬇️ Download DME CSV", data=csv_bytes, file_name="dme_table.csv", mime="text/csv")

# ------------------ AUTO-TEST ------------------
if RUN_AUTO_TESTS:
    st.header("Auto-Test Output")
    with open("test_chart.pdf", "rb") as f:
        pdf_bytes_test = f.read()
    parsed = parse_pdf_fields(pdf_bytes_test)
    st.write("Parsed PDF Fields:", parsed)








