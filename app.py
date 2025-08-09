# app.py
import streamlit as st
import pandas as pd
import numpy as np
import math
import io
import datetime
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import fitz  # PyMuPDF used by parser
from chart_parser import parse_iac_pdf, haversine_nm

st.set_page_config(page_title="CDFA PLANNER", layout="wide")
st.title("CDFA PLANNER")

########################
# Helper functions
########################
NM_TO_FT = 6076.12

def angle_from_height_and_distance(ft_diff, dist_nm):
    """Compute angle in degrees from vertical diff (ft) and horizontal distance (nm)."""
    if dist_nm <= 0:
        return None
    angle_rad = math.atan(ft_diff / (dist_nm * NM_TO_FT))
    return math.degrees(angle_rad)

def ft_per_nm_from_angle_deg(angle_deg):
    """Vertical ft per NM for a given approach angle."""
    return math.tan(math.radians(angle_deg)) * NM_TO_FT

def rod_ft_per_min(angle_deg, gs_kt):
    """Rate of descent (ft/min) for given approach angle and groundspeed (kt)."""
    ft_per_nm = ft_per_nm_from_angle_deg(angle_deg)
    # ft/hr = gs_kt * ft_per_nm -> ft/min = ft/hr / 60
    return gs_kt * ft_per_nm / 60.0

def round_to_10_up(x):
    """Round up to next 10 ft per NAVBLUE formatting for timing tables."""
    return int(math.ceil(x / 10.0) * 10)

def format_nm(nm):
    return f"{nm:.1f}"

def format_ft(f):
    return f"{int(round(f))}"

########################
# Sidebar: PDF upload & parse
########################
st.sidebar.header("IAC PDF (optional)")
uploaded_pdf = st.sidebar.file_uploader("Upload IAC / Approach Chart PDF (optional)", type=["pdf"])
parsed = {}
if uploaded_pdf:
    try:
        # pass file bytes to parser
        uploaded_pdf.seek(0)
        parsed = parse_iac_pdf(uploaded_pdf.read())
        st.sidebar.success("PDF parsed (values loaded below). You can edit any field.")
    except Exception as e:
        st.sidebar.error(f"PDF parsing failed: {e}")
        parsed = {}

########################
# Inputs (Main)
########################
st.header("Input parameters")

c1, c2 = st.columns(2)
with c1:
    thr_lat = st.text_input("THR Latitude (deg)", parsed.get("thr_lat", ""))
    thr_lon = st.text_input("THR Longitude (deg)", parsed.get("thr_lon", ""))
    thr_elev = st.number_input("THR / TDZE Elevation (FT)", value=int(parsed.get("thr_elev", 0)), step=1)
    dme_lat = st.text_input("DME Latitude (deg)", parsed.get("dme_lat", ""))
    dme_lon = st.text_input("DME Longitude (deg)", parsed.get("dme_lon", ""))
with c2:
    dme_at_thr = st.number_input("DME at THR (NM)", value=float(parsed.get("dme_at_thr", 0.0)), format="%.2f", step=0.1)
    dme_at_mapt = st.number_input("DME at MAPT (NM)", value=float(parsed.get("dme_at_mapt", 0.0)), format="%.2f", step=0.1)
    tod_alt = st.number_input("TOD Altitude (FT)", value=int(parsed.get("tod_alt", 2500)), step=10)
    mda = st.number_input("MDA (FT)", value=int(parsed.get("mda", 1000)), step=10)

st.markdown("---")
# SDFs (dynamic)
st.subheader("Step-Down Fixes (SDF) — optional (max 6)")
sdf_count = st.number_input("Number of SDFs", min_value=0, max_value=6, value=int(len(parsed.get("sdfs", []))))
sdfs = []
for i in range(int(sdf_count)):
    col1, col2 = st.columns(2)
    with col1:
        d = st.number_input(f"SDF {i+1} DME (NM)", value=float(parsed.get("sdfs", [])[i].get("dme", 0.0) if parsed.get("sdfs") and i < len(parsed.get("sdfs")) else 0.0), format="%.2f", key=f"sdf_dme_{i}")
    with col2:
        a = st.number_input(f"SDF {i+1} Alt (FT)", value=int(parsed.get("sdfs", [])[i].get("alt", 0) if parsed.get("sdfs") and i < len(parsed.get("sdfs")) else 0), step=10, key=f"sdf_alt_{i}")
    sdfs.append({"dme": float(d), "alt": int(a)})

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    faf_mapt_dist = st.number_input("FAF → MAPt distance (NM) (for ROD timing)", value=float(parsed.get("faf_mapt_dist", 0.0)), format="%.2f", step=0.1)
with col2:
    gp_input = st.text_input("Glide Path angle (°) — optional (leave blank to auto compute)", value=str(parsed.get("gp_angle", "") if parsed.get("gp_angle", "") else ""))

calc_button = st.button("Generate CDFA profile")

########################
# Core calculation flow
########################
if calc_button:
    # Validate minimal fields
    if tod_alt <= mda:
        st.error("TOD altitude must be higher than MDA.")
    else:
        # Determine horizontal distance to use for GP calculation
        # Preferred: dme_at_thr - dme_at_mapt  (THR DME minus MAPT DME)
        # Fallback: faf_mapt_dist (FAF->MAPt)
        chosen_distance = None
        reason_for_choice = ""
        if dme_at_thr > 0 and dme_at_mapt > 0 and (dme_at_thr - dme_at_mapt) > 0.05:
            chosen_distance = dme_at_thr - dme_at_mapt
            reason_for_choice = "dme_at_thr - dme_at_mapt"
        elif faf_mapt_dist > 0.05:
            chosen_distance = faf_mapt_dist
            reason_for_choice = "faf_mapt_dist (fallback)"
        else:
            # try distance from coordinate lat/lon if available
            try:
                if thr_lat and thr_lon and dme_lat and dme_lon:
                    thr_lat_f = float(thr_lat); thr_lon_f = float(thr_lon)
                    dme_lat_f = float(dme_lat); dme_lon_f = float(dme_lon)
                    # distance DME->THR in NM using haversine
                    dme_to_thr_nm = haversine_nm(dme_lat_f, dme_lon_f, thr_lat_f, thr_lon_f)
                    if dme_at_mapt > 0:
                        chosen_distance = dme_to_thr_nm - dme_at_mapt
                    else:
                        chosen_distance = dme_to_thr_nm
                    reason_for_choice = "computed from lat/lon"
            except Exception:
                pass

        if not chosen_distance or chosen_distance <= 0:
            st.error("Cannot determine a valid horizontal distance (provide DME@THR & DME@MAPt or FAF->MAPt).")
        else:
            # compute initial GP angle
            # Vertical drop from TOD to threshold reference height (thr_elev + 50ft as navblue uses ~50ft above threshold)
            thr_ref = thr_elev + 50.0
            vertical_drop = tod_alt - thr_ref
            calc_angle = angle_from_height_and_distance(vertical_drop, chosen_distance)
            # NAVBLUE rule: if calculated angle < 2.5°, raise to 3.0° for advisory tables
            note_angle_original = calc_angle
            if calc_angle is None:
                st.error("Could not compute GP angle (invalid distances).")
            else:
                if calc_angle < 2.5:
                    adjusted_angle = 3.0
                    note = f"Calculated angle {calc_angle:.2f}° < 2.5°, NAVBLUE: raising to 3.0° for advisory table."
                    gp_angle = adjusted_angle
                else:
                    gp_angle = calc_angle
                    note = f"Calculated angle used: {gp_angle:.2f}°"
                # If user provided gp_input, prefer that (manual override)
                if gp_input.strip():
                    try:
                        gp_angle = float(gp_input)
                        note = f"Manual GP angle override used: {gp_angle:.2f}°"
                    except:
                        st.warning("Invalid manual GP angle ignored; using computed value.")

                st.success(f"Using GP angle: {gp_angle:.2f}° — ({reason_for_choice}). {note}")

                # Generate DME table: 8 entries from TOD -> MAPt
                # We will generate distances from starting DME (closest to TOD) to MAPt.
                # Determine start_dme and end_dme:
                # If we have dme_at_thr and chosen_distance was based on that, start_dme = dme_at_thr
                # else if chosen_distance was FAF->MAPt fallback, we treat start_dme as mapt + chosen_distance
                if reason_for_choice == "dme_at_thr - dme_at_mapt":
                    start_dme = dme_at_thr
                    end_dme = dme_at_mapt
                elif reason_for_choice == "faf_mapt_dist (fallback)":
                    # we only know FAF->MAPt; we treat start_dme = dme_at_mapt + faf_mapt_dist if dme_at_mapt known
                    if dme_at_mapt > 0:
                        start_dme = dme_at_mapt + faf_mapt_dist
                        end_dme = dme_at_mapt
                    else:
                        # assume start_dme equals chosen_distance (relative), set end at 0
                        start_dme = chosen_distance
                        end_dme = 0.0
                else:
                    # computed from lat/lon previously
                    if dme_at_mapt > 0:
                        start_dme = dme_at_mapt + chosen_distance
                        end_dme = dme_at_mapt
                    else:
                        start_dme = chosen_distance
                        end_dme = 0.0

                # Create 8 distance points with dynamic spacing:
                # Use larger spacing early and smaller closer to runway (log spacing).
                n_points = 8
                # produce fractional positions t in [0,1] skewed to produce denser points near 0 (MAPt side)
                t = np.linspace(0.0, 1.0, n_points)
                # apply cubic ease-in to bias near MAPt: s = t^3 (keeps small gaps near runway)
                s = t ** 1.8  # exponent between 1.5-2 gives desirable bias; tuned experimentally
                dme_points = start_dme - (start_dme - end_dme) * s  # descending from start_dme -> end_dme

                # Build DME table altitudes by projecting along GP angle from TOD reference
                # We'll compute altitude at each DME by: altitude = TOD_alt - vertical_rate_along_distance
                # Vertical drop per NM = tan(angle) * NM_TO_FT
                vertical_ft_per_nm = ft_per_nm_from_angle_deg(gp_angle)

                # Determine distance of start_dme -> each point in NM measured along horizontal axis
                # For altitude reference we need distance from start to point: ds = start_dme - point_dme
                dme_rows = []
                for point in dme_points:
                    ds_from_start = start_dme - point  # nm
                    altitude_at_point = tod_alt - vertical_ft_per_nm * ds_from_start
                    # Ensure not below MDA
                    if altitude_at_point < mda:
                        altitude_at_point = mda
                    # NAVBLUE rounding: publish altitude rounded up to next 10 ft
                    altitude_publish = round_to_10_up(altitude_at_point)
                    dme_rows.append({"DME (NM)": float(format_nm(point)), "Altitude (FT)": altitude_publish})

                # Guarantee last row altitude equals MDA (do not go below it)
                dme_rows[-1]["Altitude (FT)"] = round_to_10_up(max(mda, dme_rows[-1]["Altitude (FT)"]))

                dme_df = pd.DataFrame(dme_rows)

                st.subheader("DIST/ALT (DME) Table — 8 points (TOD → MAPt)")
                st.dataframe(dme_df)

                # ROD table: use FAF->MAPt distance for timing and ROD calculation
                st.subheader("ROD Table (FAF → MAPt) — 5 GS")
                # choose FAF->MAPt distance
                if faf_mapt_dist and faf_mapt_dist > 0:
                    rod_distance = faf_mapt_dist
                else:
                    # as fallback use chosen_distance
                    rod_distance = chosen_distance

                rod_rows = []
                for gs in [80, 100, 120, 140, 160]:
                    # ROD (ft/min) from angle & GS
                    rod = rod_ft_per_min(gp_angle, gs)  # ft/min
                    # Time to cover FAF->MAPt at GS: time_minutes = dist (nm) / GS (nm/hr) * 60
                    time_minutes = (rod_distance / gs) * 60.0
                    # Convert time to mm:ss
                    minutes_int = int(math.floor(time_minutes))
                    seconds_int = int(round((time_minutes - minutes_int) * 60))
                    if seconds_int == 60:
                        minutes_int += 1
                        seconds_int = 0
                    # ROD formatting: round to nearest whole ft/min and round to 10 for publication
                    rod_publish = int(round(rod))
                    rod_rows.append({"GS (kt)": gs, "ROD (ft/min)": rod_publish, "Time (MM:SS)": f"{minutes_int:02d}:{seconds_int:02d}"})

                rod_df = pd.DataFrame(rod_rows)
                st.dataframe(rod_df)

                # Plot profile
                st.subheader("Profile view")
                fig, ax = plt.subplots(figsize=(9,4))
                # Plot DME vs Altitude
                ax.plot(dme_df["DME (NM)"], dme_df["Altitude (FT)"], marker="o", label="CDFA profile")
                # Mark SDFs
                for sdf in sdfs:
                    if sdf["dme"] > 0 and sdf["alt"] > 0:
                        ax.scatter([sdf["dme"]], [sdf["alt"]], marker="x", color="red", label="SDF")
                        ax.text(sdf["dme"], sdf["alt"], f" SDF\n{format_nm(sdf['dme'])}NM\n{int(sdf['alt'])}ft", fontsize=8)
                # MDA line
                ax.axhline(y=mda, color="red", linestyle="--", label="MDA")
                ax.set_xlabel("DME (NM to/from THR)")
                ax.set_ylabel("Altitude (FT)")
                ax.set_title(f"CDFA Profile — GP {gp_angle:.2f}°")
                ax.invert_xaxis()  # conventional profile (distance decreases towards runway)
                ax.grid(True)
                ax.legend(loc="upper right")
                st.pyplot(fig)

                # CSV export
                csv_dme = dme_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download DME Table (CSV)", csv_dme, "dme_table.csv", "text/csv")
                csv_rod = rod_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download ROD Table (CSV)", csv_rod, "rod_table.csv", "text/csv")

                # PDF export using reportlab
                def create_pdf_bytes(dme_df, rod_df, gp_angle):
                    buf = io.BytesIO()
                    doc = SimpleDocTemplate(buf, pagesize=A4)
                    styles = getSampleStyleSheet()
                    elems = []

                    title = Paragraph("CDFA PLANNER — Output", styles["Title"])
                    elems.append(title)
                    elems.append(Paragraph(f"Generated: {datetime.datetime.utcnow().isoformat()} UTC", styles["Normal"]))
                    elems.append(Spacer(1, 8))
                    elems.append(Paragraph(f"GP angle used: {gp_angle:.2f}°", styles["Normal"]))
                    elems.append(Spacer(1, 8))

                    elems.append(Paragraph("DIST/ALT (DME) Table", styles["Heading2"]))
                    dme_data = [list(dme_df.columns)] + dme_df.values.tolist()
                    t = Table(dme_data, colWidths=[90, 90])
                    t.setStyle(TableStyle([
                        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
                        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey)
                    ]))
                    elems.append(t)
                    elems.append(Spacer(1, 12))

                    elems.append(Paragraph("ROD Table", styles["Heading2"]))
                    rod_data = [list(rod_df.columns)] + rod_df.values.tolist()
                    t2 = Table(rod_data, colWidths=[70, 90, 90])
                    t2.setStyle(TableStyle([
                        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
                        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey)
                    ]))
                    elems.append(t2)
                    elems.append(Spacer(1, 12))

                    doc.build(elems)
                    pdfbytes = buf.getvalue()
                    buf.close()
                    return pdfbytes

                pdf_bytes = create_pdf_bytes(dme_df, rod_df, gp_angle)
                st.download_button("Download PDF (CDFA output)", data=pdf_bytes, file_name="cdfa_output.pdf", mime="application/pdf")




