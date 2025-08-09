"""
CDFA-PLANNER — Final Single-File Version for Render
Features:
- Minimum 8 DME rows (~1 NM spacing) from TOD → MAPT
- Inserts SDFs if provided
- ROD in ft/NM + FAF→MAPT time mm:ss
- Profile chart in UI & PDF
- NavBlue-style A4 landscape PDF (DME table left, ROD summary right, profile below)
- FAA/EASA GP guards
- CSV download for both tables
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
from io import BytesIO
import datetime
import os
import matplotlib.pyplot as plt
from typing import List, Tuple
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

NM_TO_FT = 6076.12
MIN_HORIZ_NM = 0.05
DEFAULT_GP_MIN = 2.5
DEFAULT_GP_RAISE_TO = 3.0
DEFAULT_GP_MAX_WARN = 4.5
DEFAULT_GS_PRESET = [80, 100, 120, 140, 160]
DEFAULT_APPROX_SPACING_NM = 1.0
DEFAULT_MIN_POINTS = 8

# ---------- helpers ----------
def round1(x: float) -> float:
    return round(x + 1e-9, 1)

def fmt_mmss_from_minutes(minutes: float) -> str:
    total_seconds = int(round(minutes * 60.0))
    mm = total_seconds // 60
    ss = total_seconds % 60
    return f"{mm:02d}:{ss:02d}"

# ---------- GP logic ----------
def derive_gp(tod_alt_ft: float, mda_ft: float, tod_dme_nm: float, mapt_dme_nm: float,
              gp_min: float, raise_to: float, gp_warn_max: float) -> Tuple[float, List[str]]:
    warnings = []
    horiz_nm = abs(tod_dme_nm - mapt_dme_nm)
    if horiz_nm < MIN_HORIZ_NM:
        horiz_nm = MIN_HORIZ_NM
        warnings.append(f"Small horizontal distance — using {MIN_HORIZ_NM} NM guard.")
    horiz_ft = horiz_nm * NM_TO_FT
    vert_drop = tod_alt_ft - mda_ft
    if vert_drop <= 0:
        warnings.append("TOD altitude <= MDA — using raise-to GP.")
        return raise_to, warnings
    gp_rad = math.atan2(vert_drop, horiz_ft)
    gp_deg = math.degrees(gp_rad)
    if gp_deg < gp_min:
        warnings.append(f"GP {gp_deg:.2f}° < min {gp_min}° — raising to {raise_to}°.")
        gp_deg = raise_to
    if gp_deg > gp_warn_max:
        warnings.append(f"GP {gp_deg:.2f}° > {gp_warn_max}° — verify obstacle clearance.")
    return round(gp_deg, 2), warnings

# ---------- DME points ----------
def make_dme_points(outer_dme: float, inner_dme: float, approx_spacing_nm: float = DEFAULT_APPROX_SPACING_NM, min_points: int = DEFAULT_MIN_POINTS) -> List[float]:
    if outer_dme < inner_dme:
        outer_dme, inner_dme = inner_dme, outer_dme
    total_nm = outer_dme - inner_dme
    segs_by_spacing = max(1, int(math.ceil(total_nm / approx_spacing_nm)))
    pts_by_spacing = segs_by_spacing + 1
    n_points = max(min_points, pts_by_spacing)
    arr = np.linspace(outer_dme, inner_dme, num=n_points)
    arr_rounded = [round1(x) for x in arr]
    uniq = []
    for x in arr_rounded:
        if not uniq or abs(x - uniq[-1]) > 1e-6:
            uniq.append(x)
    return uniq

def insert_sdfs(points: List[float], sdf_dmes: List[float]) -> List[Tuple[float, bool]]:
    pts = points.copy()
    sdf_list = sorted([round1(x) for x in sdf_dmes], reverse=True)
    for sdf in sdf_list:
        if sdf > pts[0] or sdf < pts[-1]:
            continue
        best_i = min(range(len(pts)), key=lambda i: abs(pts[i]-sdf))
        pts[best_i] = sdf
    sdf_set = set(sdf_list)
    return [(p, p in sdf_set) for p in pts]

# ---------- altitudes ----------
def compute_altitudes(points: List[float], gp_deg: float, mda_ft: float, mapt_dme: float) -> List[int]:
    gp_rad = math.radians(gp_deg)
    alts = []
    for p in points:
        dist_to_mapt_nm = max(0.0, p - mapt_dme)
        horiz_ft = dist_to_mapt_nm * NM_TO_FT
        alt = mda_ft + math.tan(gp_rad) * horiz_ft
        alts.append(int(round(alt)))
    return alts

# ---------- ROD summary ----------
def compute_rod_summary(faf_mapt_nm: float, tod_alt_ft: float, mda_ft: float, gs_list: List[int]) -> pd.DataFrame:
    rows = []
    total_alt = tod_alt_ft - mda_ft
    ft_per_nm = total_alt / faf_mapt_nm if faf_mapt_nm > 0 else 0
    for gs in gs_list:
        nm_per_min = gs / 60.0
        total_time_min = faf_mapt_nm / nm_per_min if nm_per_min > 0 else 0
        rows.append({"GS (kt)": gs, "ROD (ft/NM)": round(ft_per_nm, 1), "FAF->MAPT Time": fmt_mmss_from_minutes(total_time_min)})
    return pd.DataFrame(rows)

# ---------- PDF ----------
def generate_pdf_report(proc_id: str, gp_used: float, dme_df: pd.DataFrame, rod_df: pd.DataFrame, fig_buf: BytesIO, logo_path: str = "logo.png") -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=30, rightMargin=30, topMargin=15, bottomMargin=15)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleCenter', fontName='Helvetica-Bold', fontSize=16, alignment=1))
    styles.add(ParagraphStyle(name='Meta', fontName='Helvetica', fontSize=9, alignment=1))
    styles.add(ParagraphStyle(name='Footer', fontName='Helvetica-Oblique', fontSize=8, alignment=1))
    elements = []

    if logo_path and os.path.exists(logo_path):
        elements.append(RLImage(logo_path, width=60, height=40))
    elements.append(Paragraph("CDFA-PLANNER — Stabilized Approach Report", styles['TitleCenter']))
    elements.append(Paragraph(f"Procedure: {proc_id} | GP used: {gp_used:.2f}° | Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Meta']))
    elements.append(Spacer(1, 8))

    dme_rows = [["DME (NM)", "Dist to THR (NM)", "Altitude (ft)", "SDF"]] + \
               [[f"{row['DME (NM)']:.1f}", f"{row['Distance to THR (NM)']:.1f}", str(int(row['Altitude (ft)'])), "✓" if row.get("SDF") == "Yes" else ""] for _, row in dme_df.iterrows()]
    dme_table = Table(dme_rows, colWidths=[55, 85, 70, 30])
    dme_table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor("#D9D9D9")), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('GRID', (0,0), (-1,-1), 0.25, colors.black)]))

    rod_rows = [list(rod_df.columns)] + rod_df.values.tolist()
    rod_table = Table(rod_rows, colWidths=[60, 80, 120])
    rod_table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor("#D9D9D9")), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('GRID', (0,0), (-1,-1), 0.25, colors.black)]))

    combined = Table([[dme_table, rod_table]], colWidths=[260, 260])
    elements.append(combined)
    elements.append(Spacer(1, 10))

    img = RLImage(fig_buf, width=700, height=220)
    elements.append(img)
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("Verify against official AIP / Jeppesen / NavBlue charts before operational use.", styles['Footer']))
    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()

# ---------- UI ----------
st.set_page_config(page_title="CDFA DME & ROD Planner", layout="wide")
st.title("CDFA — DME & ROD Planner")

c1, c2 = st.columns(2)
with c1:
    thr_lat = st.text_input("THR Latitude (decimal)", value="")
    thr_lon = st.text_input("THR Longitude (decimal)", value="")
    thr_elev = st.number_input("THR / TDZE Elevation (ft)", value=0, step=1)
    dme_lat = st.text_input("DME Latitude (decimal)", value="")
    dme_lon = st.text_input("DME Longitude (decimal)", value="")
    dme_at_thr = st.number_input("DME at THR (NM)", value=1.0, step=0.1)

with c2:
    proc_id = st.text_input("Procedure ID / Name", value="PROC-001")
    tod_alt = st.number_input("TOD Altitude (ft)", value=3600, step=1)
    tod_dme = st.number_input("TOD DME (NM)", value=13.6, step=0.1)
    mapt_dme = st.number_input("DME at MAPT (NM)", value=1.8, step=0.1)
    mda_ft = st.number_input("MDA (ft)", value=1000, step=1)

st.markdown("---")
st.subheader("SDFs (Step-Down Fixes) — optional")
init_df = pd.DataFrame(columns=["alt_ft", "dme_nm"])
sdf_editor = st.data_editor(init_df, num_rows="dynamic", key="sdf_editor")
sdf_list = []
if isinstance(sdf_editor, pd.DataFrame):
    for _, r in sdf_editor.iterrows():
        try:
            a = float(r.get("alt_ft", 0))
            d = float(r.get("dme_nm", 0))
            if a > 0 and d > 0:
                sdf_list.append((a, d))
        except:
            continue

faf_mapt = st.number_input("FAF–MAPT Distance (NM)", value=5.0, step=0.1)

if st.button("Generate Tables & PDF"):
    gp_used, warnings = derive_gp(tod_alt, mda_ft, tod_dme, mapt_dme, DEFAULT_GP_MIN, DEFAULT_GP_RAISE_TO, DEFAULT_GP_MAX_WARN)
    dme_points = make_dme_points(tod_dme, mapt_dme)
    sdf_dmes = [d for _, d in sdf_list]
    dme_with_flags = insert_sdfs(dme_points, sdf_dmes)
    alts = compute_altitudes([p for p, _ in dme_with_flags], gp_used, mda_ft, mapt_dme)

    dme_df = pd.DataFrame({
        "DME (NM)": [p for p, _ in dme_with_flags],
        "Distance to THR (NM)": [round1(p - dme_at_thr) for p, _ in dme_with_flags],
        "Altitude (ft)": alts,
        "SDF": ["Yes" if is_sdf else "" for _, is_sdf in dme_with_flags]
    })

    rod_df = compute_rod_summary(faf_mapt, tod_alt, mda_ft, DEFAULT_GS_PRESET)

    st.subheader("DME Table")
    st.dataframe(dme_df)
    st.download_button("Download DME Table CSV", dme_df.to_csv(index=False), file_name=f"{proc_id}_DME_Table.csv", mime="text/csv")

    st.subheader("ROD Summary Table")
    st.dataframe(rod_df)
    st.download_button("Download ROD Summary CSV", rod_df.to_csv(index=False), file_name=f"{proc_id}_ROD_Summary.csv", mime="text/csv")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(dme_df["DME (NM)"], dme_df["Altitude (ft)"], marker="o")
    ax.set_xlabel("DME (NM)")
    ax.set_ylabel("Altitude (ft)")
    ax.set_title("Final Approach Profile")
    ax.grid(True)
    plt.gca().invert_xaxis()
    fig_buf = BytesIO()
    plt.savefig(fig_buf, format="png", bbox_inches="tight")
    fig_buf.seek(0)

    pdf_bytes = generate_pdf_report(proc_id, gp_used, dme_df, rod_df, fig_buf)
    st.download_button("Download PDF Report", pdf_bytes, file_name=f"{proc_id}_CDFA_Report.pdf", mime="application/pdf")

    for w in warnings:
        st.warning(w)












