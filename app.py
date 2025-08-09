"""
CDFA-PLANNER — corrected single-file app.py
Fixes applied:
- Always produce >= 8 DME rows (approx 1 NM spacing unless user changes)
- ROD summary uses ft/NM (primary) and shows FAF->MAPT Time in mm:ss
- Profile plot properly generated and displayed (same image used for PDF)
- SDF insertion into DME points, SDF column shows checkmark
- Distances rounded to 1 decimal
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

# ---------------- CONFIG ----------------
RUN_AUTO_TESTS = False  # set True locally for dev
NM_TO_FT = 6076.12
MIN_HORIZ_NM = 0.05

DEFAULT_GP_MIN = 2.5
DEFAULT_GP_RAISE_TO = 3.0
DEFAULT_GP_MAX_WARN = 4.5
DEFAULT_GS_PRESET = [80, 100, 120, 140, 160]
DEFAULT_APPROX_SPACING_NM = 1.0
DEFAULT_MIN_POINTS = 8

# ---------------- Helpers ----------------
def round1(x: float) -> float:
    return round(x + 1e-9, 1)

def fmt_mmss_from_minutes(minutes: float) -> str:
    total_seconds = int(round(minutes * 60.0))
    mm = total_seconds // 60
    ss = total_seconds % 60
    return f"{mm:02d}:{ss:02d}"

# ---------------- GP derivation ----------------
def derive_gp(tod_alt_ft: float, mda_ft: float, tod_dme_nm: float, mapt_dme_nm: float,
              gp_min: float, raise_to: float, gp_warn_max: float) -> Tuple[float, List[str]]:
    """
    Derive GP such that TOD altitude descends to MDA at MAPT.
    Apply guard for tiny horiz distances and raise if < gp_min.
    Returns (gp_deg, warnings_list).
    """
    warnings = []
    horiz_nm = abs(tod_dme_nm - mapt_dme_nm)
    if horiz_nm < MIN_HORIZ_NM:
        horiz_nm = MIN_HORIZ_NM
        warnings.append(f"Very small horizontal distance between TOD & MAPT; using guard {MIN_HORIZ_NM} NM.")
    horiz_ft = horiz_nm * NM_TO_FT
    vert_drop = tod_alt_ft - mda_ft
    if vert_drop <= 0:
        warnings.append("TOD altitude <= MDA — cannot derive GP. Using raise-to GP.")
        return raise_to, warnings
    gp_rad = math.atan2(vert_drop, horiz_ft)
    gp_deg = math.degrees(gp_rad)
    if gp_deg < gp_min:
        warnings.append(f"Derived GP {gp_deg:.2f}° < min {gp_min:.2f}° — raising to {raise_to:.2f}°.")
        gp_deg = raise_to
    if gp_deg > gp_warn_max:
        warnings.append(f"Derived GP {gp_deg:.2f}° > warn max {gp_warn_max:.2f}° — verify obstacle clearance.")
    return round(gp_deg, 2), warnings

# ---------------- DME point generation ----------------
def make_dme_points(outer_dme: float, inner_dme: float, approx_spacing_nm: float = DEFAULT_APPROX_SPACING_NM, min_points: int = DEFAULT_MIN_POINTS) -> List[float]:
    """
    Generate a list of DME points from outer (TOD) to inner (MAPT), inclusive.
    - Use approx spacing (~1 NM) when possible, but ensure at least min_points.
    - Return list outer->inner with 1-decimal rounding and unique values.
    """
    outer = float(outer_dme)
    inner = float(inner_dme)
    if outer < inner:
        outer, inner = inner, outer  # ensure outer > inner

    total_nm = outer - inner
    # number of segments by approx spacing
    if approx_spacing_nm <= 0:
        approx_spacing_nm = DEFAULT_APPROX_SPACING_NM
    segs_by_spacing = max(1, int(math.ceil(total_nm / approx_spacing_nm)))
    pts_by_spacing = segs_by_spacing + 1
    n_points = max(min_points, pts_by_spacing)
    # generate linspace with exactly n_points values
    if n_points <= 1:
        arr = [outer, inner] if outer != inner else [outer]
    else:
        arr = np.linspace(outer, inner, num=n_points)
    arr_rounded = [round1(x) for x in arr]
    # ensure uniqueness preserving order outer->inner
    uniq = []
    for x in arr_rounded:
        if not uniq or abs(x - uniq[-1]) > 1e-6:
            uniq.append(x)
    # if uniqueness reduced length below min_points (cases of zero total_nm), pad with inner
    while len(uniq) < min_points:
        # prepend slightly larger dme (outer + small increment) to avoid duplicates
        pad_val = round1(uniq[0] + 0.1)
        uniq.insert(0, pad_val)
    return uniq[:max(min_points, len(uniq))]

def insert_sdfs(points: List[float], sdf_dmes: List[float]) -> List[Tuple[float, bool]]:
    """
    Replace nearest interior points with SDF dme values.
    Returns list of tuples (dme, is_sdf) sorted outer->inner.
    """
    pts = points.copy()
    n = len(pts)
    interior_idx = list(range(1, n-1)) if n > 2 else []
    used_idx = set()
    sdf_list = sorted([round1(x) for x in sdf_dmes], reverse=True)
    for sdf in sdf_list:
        if sdf > pts[0] or sdf < pts[-1]:
            # outside range, skip
            continue
        # find nearest interior idx not used
        best_i = None
        best_dist = 1e9
        for i in interior_idx:
            if i in used_idx:
                continue
            d = abs(pts[i] - sdf)
            if d < best_dist:
                best_dist = d
                best_i = i
        if best_i is not None:
            pts[best_i] = sdf
            used_idx.add(best_i)
    # mark SDF membership
    sdf_set = set(sdf_list)
    out = [(round1(p), (round1(p) in sdf_set)) for p in pts]
    out_sorted = sorted(out, key=lambda x: -x[0])
    return out_sorted

# ---------------- Altitude computation ----------------
def compute_altitudes(points: List[float], gp_deg: float, mda_ft: float, mapt_dme: float) -> List[int]:
    """
    Compute altitude at each DME point such that altitude at MAPT equals MDA.
    alt = MDA + tan(GP) * horizontal_ft_from_MAPT
    """
    gp_rad = math.radians(gp_deg)
    alts = []
    for p in points:
        dist_to_mapt_nm = max(0.0, p - mapt_dme)
        horiz_ft = dist_to_mapt_nm * NM_TO_FT
        vert = math.tan(gp_rad) * horiz_ft
        alt = mda_ft + vert
        alts.append(int(round(alt)))
    return alts

# ---------------- ROD (ft/NM) and time (mm:ss) ----------------
def compute_rod_summary(faf_mapt_nm: float, tod_alt_ft: float, mda_ft: float, gs_list: List[int]) -> pd.DataFrame:
    """
    Primary ROD is reported as ft/NM = (TOD_alt - MDA) / FAF->MAPT_NM
    Also compute FAF->MAPT time for each GS and present as mm:ss.
    """
    rows = []
    total_alt = tod_alt_ft - mda_ft
    ft_per_nm = 0.0
    if faf_mapt_nm > 0 and total_alt > 0:
        ft_per_nm = total_alt / faf_mapt_nm
    for gs in gs_list:
        nm_per_min = gs / 60.0
        total_time_min = faf_mapt_nm / nm_per_min if nm_per_min > 0 else 0.0
        rows.append({
            "GS (kt)": gs,
            "ROD (ft/NM)": round(ft_per_nm, 1),
            "FAF->MAPT Time": fmt_mmss_from_minutes(total_time_min)
        })
    return pd.DataFrame(rows)

# ---------------- PDF Report (landscape NavBlue-like) ----------------
def generate_pdf_report(proc_id: str, gp_used: float, dme_df: pd.DataFrame, rod_df: pd.DataFrame, fig_buf: BytesIO, logo_path: str = "logo.png") -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=30, rightMargin=30, topMargin=15, bottomMargin=15)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleCenter', fontName='Helvetica-Bold', fontSize=16, alignment=1))
    styles.add(ParagraphStyle(name='Meta', fontName='Helvetica', fontSize=9, alignment=1))
    styles.add(ParagraphStyle(name='Footer', fontName='Helvetica-Oblique', fontSize=8, alignment=1))
    elements = []

    # Header
    if logo_path and os.path.exists(logo_path):
        elements.append(RLImage(logo_path, width=60, height=40))
    elements.append(Paragraph("CDFA-PLANNER — Stabilized Approach Report", styles['TitleCenter']))
    elements.append(Paragraph(f"Procedure: {proc_id or ''}  |  GP used: {gp_used:.2f}°  |  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Meta']))
    elements.append(Spacer(1, 8))

    # Build formatted DME table (left) and ROD table (right)
    dme_tbl_hdr = ["DME (NM)", "Distance to THR (NM)", "Altitude (ft)", "SDF"]
    dme_rows = [dme_tbl_hdr] + [
        [f"{row['DME (NM)']:.1f}", f"{row['Distance to THR (NM)']:.1f}", f"{int(row['Altitude (ft)'])}", "✓" if row.get("SDF", "") == "Yes" else ""] for idx, row in dme_df.iterrows()
    ]
    dme_table = Table(dme_rows, colWidths=[55, 85, 70, 30])
    dme_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#D9D9D9")),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.25, colors.black),
    ]))

    rod_tbl_hdr = list(rod_df.columns)
    rod_rows = [rod_tbl_hdr] + rod_df.values.tolist()
    rod_table = Table(rod_rows, colWidths=[60, 80, 120])
    rod_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#D9D9D9")),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.25, colors.black),
    ]))

    # Put side by side
    combined = Table([[dme_table, rod_table]], colWidths=[260, 260])
    elements.append(combined)
    elements.append(Spacer(1, 10))

    # Profile image
    img = RLImage(fig_buf, width=700, height=220)
    elements.append(img)
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("Verify against official AIP / Jeppesen / NavBlue charts before operational use.", styles['Footer']))

    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="CDFA DME & ROD Planner", layout="wide")
st.title("CDFA — DME & ROD Planner (TOD → MAPT)")

st.markdown("Fill inputs exactly. DME outputs will start at TOD and end at MAPT. SDFs optional (max 6).")

c1, c2 = st.columns(2)

with c1:
    st.subheader("Threshold (THR)")
    thr_lat = st.text_input("THR Latitude (decimal)", value="")
    thr_lon = st.text_input("THR Longitude (decimal)", value="")
    thr_elev = st.number_input("THR / TDZE Elevation (ft)", value=0, step=1)

    st.subheader("DME Beacon")
    dme_lat = st.text_input("DME Latitude (decimal)", value="")
    dme_lon = st.text_input("DME Longitude (decimal)", value="")
    dme_at_thr = st.number_input("DME at THR (NM)", value=1.0, step=0.1)

with c2:
    st.subheader("Approach / Minima")
    proc_id = st.text_input("Procedure ID / Name", value="PROC-001")
    tod_alt = st.number_input("TOD (Starting) Altitude (ft)", value=3600, step=1)
    tod_dme = st.number_input("TOD DME (NM) (outer)", value=13.6, step=0.1)
    mapt_dme = st.number_input("DME at MAPT (NM)", value=1.8, step=0.1)
    mda_ft = st.number_input("MDA (ft)", value=1000, step=1)

st.markdown("---")
st.subheader("SDFs (Step-Down Fixes) — optional (max 6)")
try:
    # interactive editor
    init_df = pd.DataFrame(columns=["alt_ft", "dme_nm"])
    sdf_editor = st.experimental_data_editor(init_df, num_rows="dynamic", key="sdf_editor")
    sdf_list = []
    if isinstance(sdf_editor, pd.DataFrame):
        for _, r in sdf_editor.iterrows():
            try:
                a = float(r.get("alt_ft", 0)); d = float(r.get("dme_nm", 0))
                if a>0 and d>0:
                    sdf_list.append((a, d))
            except:
                continue
except Exception:
    sdf_text = st.text_area("SDFs (alt_ft,dme_nm per line)", value="")
    sdf_list = []










