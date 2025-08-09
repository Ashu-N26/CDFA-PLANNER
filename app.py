"""
CDFA-PLANNER (single-file)
- PDF text extraction (pdfplumber) + OCR fallback (pdf2image + pytesseract)
- Slant-range correction (haversine) if coords supplied
- GP derivation with configurable clamps and NavBlue-style behavior
- 8-point DME table (quadratic spacing with tunable exponent)
- ROD table (per GS presets) with per-segment math and FAF->MAPt total
- Plot that matches table exactly
- CSV and PDF export
- Embedded auto-test harness (RUN_AUTO_TESTS = True) shows parsed fields and sanity checks for uploaded PDF
"""

import math
from io import BytesIO
from typing import List, Tuple, Optional, Dict
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import traceback

# PDF & OCR libs
try:
    import pdfplumber
except Exception:
    pdfplumber = None
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

# =========================
# Developer toggle / config
# =========================
RUN_AUTO_TESTS = True  # If True, uploading a PDF will auto-parse and run embedded tests
DEFAULT_GS_PRESET = [80, 100, 120, 140, 160]

# =========================
# Constants & helpers
# =========================
NM_TO_FT = 6076.12
EARTH_RADIUS_KM = 6371.0

def haversine_nm(lat1, lon1, lat2, lon2):
    """Great-circle distance in nautical miles."""
    try:
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2.0)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        km = EARTH_RADIUS_KM * c
        nm = km / 1.852
        return nm
    except Exception:
        return None

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# =========================
# Parser (pdfplumber + OCR fallback)
# =========================
class ChartParser:
    def __init__(self):
        # Patterns are intentionally conservative; parser returns candidates for user confirmation
        self.re_mda = re.compile(r'\bMDA\b[:\s]*([0-9]{3,5})\s*ft', re.IGNORECASE)
        self.re_gp = re.compile(r'([0-9]\.?[0-9])\s?°\s*(GP|GLIDE|GLIDESLOPE)?', re.IGNORECASE)
        self.re_dme_val = re.compile(r'([0-9]{1,3}\.[0-9])\s*NM', re.IGNORECASE)
        self.re_coords = re.compile(r'([-+]?\d+\.\d+)[,;\s]+([-+]?\d+\.\d+)')

    def extract_text(self, file_bytes: bytes) -> str:
        text_accum = []
        # Try pdfplumber first (text-based)
        if pdfplumber:
            try:
                with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                    for page in pdf.pages:
                        txt = page.extract_text()
                        if txt:
                            text_accum.append(txt)
            except Exception:
                text_accum = []
        text = "\n".join(text_accum).strip()
        if text:
            return text
        # Fallback to OCR
        try:
            images = convert_from_bytes(file_bytes, dpi=220)
            ocr_texts = []
            for im in images:
                ocr_texts.append(pytesseract.image_to_string(im))
            return "\n".join(ocr_texts)
        except Exception:
            return ""

    def parse(self, file_bytes: bytes) -> Dict:
        doc = self.extract_text(file_bytes)
        out = {}
        if not doc:
            return out
        # MDA
        m = self.re_mda.search(doc)
        if m:
            try:
                out['mda_ft'] = int(m.group(1))
            except:
                pass
        # GP angle candidates
        gp_matches = self.re_gp.findall(doc)
        if gp_matches:
            # prefer smaller pattern match (first)
            try:
                out['gp_angle'] = float(gp_matches[0][0])
            except:
                pass
        # DME decimals
        dmes = self.re_dme_val.findall(doc)
        if dmes:
            try:
                out['dme_candidates'] = [float(x) for x in dmes]
            except:
                out['dme_candidates'] = []
        # coords
        coords = []
        for c in self.re_coords.findall(doc):
            try:
                lat = float(c[0])
                lon = float(c[1])
                coords.append((lat, lon))
            except:
                continue
        if coords:
            out['coords'] = coords[0]
        # include a short snippet for debugging
        out['snippet'] = (doc[:1000] + "...") if len(doc) > 1000 else doc
        return out

# =========================
# CDFA algorithm class
# =========================
class CDFAGenerator:
    def __init__(self,
                 thr_elev_ft: float,
                 mda_ft: float,
                 dme_thr_nm: float,
                 tod_dme_nm: float,
                 mapt_dme_nm: Optional[float] = None,
                 faf_alt_ft: Optional[float] = None,
                 gp_angle_deg: Optional[float] = None,
                 sdf_list: Optional[List[Dict]] = None,
                 thr_coords: Optional[Tuple[float,float]] = None,
                 dme_coords: Optional[Tuple[float,float]] = None,
                 gp_clamp_min: float = 2.5,
                 gp_clamp_max_warn: float = 4.5,
                 raise_to: float = 3.0,
                 spacing_exponent: float = 2.0):
        self.thr_elev_ft = float(thr_elev_ft)
        self.mda_ft = float(mda_ft)
        self.dme_thr_nm = float(dme_thr_nm)
        self.tod_dme_nm = float(tod_dme_nm)
        self.mapt_dme_nm = float(mapt_dme_nm) if mapt_dme_nm is not None and mapt_dme_nm>0 else None
        self.faf_alt_ft = float(faf_alt_ft) if faf_alt_ft else None
        self.gp_angle_deg = float(gp_angle_deg) if gp_angle_deg and gp_angle_deg>0 else None
        self.sdf_list = sdf_list or []
        self.thr_coords = thr_coords
        self.dme_coords = dme_coords

        self.gp_clamp_min = gp_clamp_min
        self.gp_clamp_max_warn = gp_clamp_max_warn
        self.raise_to = raise_to
        self.spacing_exponent = spacing_exponent

        self.warn_messages: List[str] = []

    def derive_gp(self) -> Tuple[float, str]:
        """Derive or validate GP angle; returns (gp_deg, method_str)."""
        if self.gp_angle_deg is not None:
            gp = self.gp_angle_deg
            if gp < self.gp_clamp_min:
                self.warn_messages.append(f"Provided GP {gp:.2f}° < clamp {self.gp_clamp_min}°; raising to {self.raise_to}°.")
                return (self.raise_to, "user_clamped")
            if gp > self.gp_clamp_max_warn:
                self.warn_messages.append(f"Provided GP {gp:.2f}° > warning threshold {self.gp_clamp_max_warn}°; verify obstacle clearance.")
            return (gp, "user")
        # derive using FAF altitude if available
        if self.faf_alt_ft is not None and self.tod_dme_nm is not None:
            horiz_nm = max(0.05, self.tod_dme_nm - self.dme_thr_nm)
            horiz_ft = horiz_nm * NM_TO_FT
            vert_drop_ft = max(0.0, self.faf_alt_ft - self.thr_elev_ft)
            if horiz_ft <= 0.0 or vert_drop_ft <= 0.0:
                self.warn_messages.append("Insufficient vertical/horizontal info to derive GP; using fallback.")
                return (self.raise_to, "fallback")
            gp_rad = math.atan2(vert_drop_ft, horiz_ft)
            gp_deg = math.degrees(gp_rad)
            if gp_deg < self.gp_clamp_min:
                self.warn_messages.append(f"Derived GP {gp_deg:.2f}° < clamp {self.gp_clamp_min}°; raising to {self.raise_to}°.")
                return (self.raise_to, "derived_raised")
            if gp_deg > self.gp_clamp_max_warn:
                self.warn_messages.append(f"Derived GP {gp_deg:.2f}° exceeds warn threshold {self.gp_clamp_max_warn}°; use with caution.")
            return (round(gp_deg,2), "derived")
        # fallback default
        self.warn_messages.append("No FAF altitude or GP provided; using default 3.0°.")
        return (3.0, "default")

    def slant_to_horizontal_dme(self, point_dme_nm: float) -> float:
        """
        Convert a slant DME reading (point DME) into horizontal distance-to-threshold (NM).
        If THR & DME transmitter coords available, compute station-to-THR horizontal distance and
        use absolute difference approximation.
        """
        if self.thr_coords and self.dme_coords:
            try:
                station_to_thr_nm = haversine_nm(self.dme_coords[0], self.dme_coords[1],
                                                 self.thr_coords[0], self.thr_coords[1])
                if station_to_thr_nm is None:
                    return max(0.0, point_dme_nm - self.dme_thr_nm)
                horiz_to_thr = abs(station_to_thr_nm - point_dme_nm)
                return max(0.0, horiz_to_thr)
            except Exception:
                return max(0.0, point_dme_nm - self.dme_thr_nm)
        # otherwise assume DME values are referenced to same station and subtract threshold DME
        return max(0.0, point_dme_nm - self.dme_thr_nm)

    def pick_start_end(self) -> Tuple[float,float]:
        """Select start and end DME (outer->inner) for the 8-point table."""
        start_nm = max(self.tod_dme_nm, self.dme_thr_nm + 0.1)
        # consider outermost SDF if present
        if self.sdf_list:
            # take the SDF with largest DME if it's outside start
            sdf_sorted = sorted(self.sdf_list, key=lambda s: s['dme'], reverse=True)
            if sdf_sorted[0]['dme'] > start_nm:
                start_nm = sdf_sorted[0]['dme']
        # end prefers MAPt if given
        end_candidates = [self.dme_thr_nm + 0.05]
        if self.mapt_dme_nm:
            end_candidates.append(self.mapt_dme_nm)
        end_nm = max(end_candidates)
        if start_nm <= end_nm:
            # fix automatically by extending start outward
            start_nm = end_nm + 0.5
            self.warn_messages.append("Start DME <= end DME — adjusted start outward to create valid table.")
        return round(start_nm,4), round(end_nm,4)

    def generate_dme_points(self, start_nm: float, end_nm: float, num_points: int = 8) -> List[float]:
        t = np.linspace(0.0, 1.0, num_points)
        w = t ** float(self.spacing_exponent)
        dmes = start_nm + (end_nm - start_nm) * w
        # Ensure descending order outer->inner
        if dmes[0] < dmes[-1]:
            dmes = dmes[::-1]
        return [round(float(x), 1) for x in dmes]

    def compute_altitudes(self, dme_points: List[float], gp_deg: float) -> List[float]:
        gp_rad = math.radians(gp_deg)
        alts = []
        for d in dme_points:
            horiz_nm = self.slant_to_horizontal_dme(d)
            horizontal_ft = horiz_nm * NM_TO_FT
            vert_above_thr = math.tan(gp_rad) * horizontal_ft
            alt = self.thr_elev_ft + vert_above_thr
            alts.append(round(float(alt), 0))
        return alts

    def ensure_last_at_least_mda(self, dme_points: List[float], alts: List[float], gp_deg: float) -> Tuple[float,List[float]]:
        last_alt = alts[-1]
        if last_alt >= self.mda_ft:
            return gp_deg, alts
        # compute required GP to reach MDA at last point
        last_dme = dme_points[-1]
        last_horiz_nm = self.slant_to_horizontal_dme(last_dme)
        last_horiz_ft = last_horiz_nm * NM_TO_FT
        if last_horiz_ft <= 0:
            # fallback
            new_gp = self.raise_to
            self.warn_messages.append("Last horizontal distance trivial — used fallback GP to ensure MDA.")
            new_alts = self.compute_altitudes(dme_points, new_gp)
            return new_gp, new_alts
        required_gp_rad = math.atan2((self.mda_ft - self.thr_elev_ft), last_horiz_ft)
        required_gp_deg = math.degrees(required_gp_rad)
        # If required is small (<= warn max), use it; else clamp to warn max and warn
        if required_gp_deg <= self.gp_clamp_max_warn:
            self.warn_messages.append(f"Increasing GP to {required_gp_deg:.2f}° to ensure last point >= MDA.")
            new_alts = self.compute_altitudes(dme_points, required_gp_deg)
            return round(required_gp_deg,2), new_alts
        else:
            self.warn_messages.append(f"Required GP {required_gp_deg:.2f}° exceeds warning threshold {self.gp_clamp_max_warn}°. Using {self.gp_clamp_max_warn}° and flagging as steep.")
            new_alts = self.compute_altitudes(dme_points, self.gp_clamp_max_warn)
            return self.gp_clamp_max_warn, new_alts

    def build_dme_table(self) -> Tuple[pd.DataFrame, float, List[str]]:
        start, end = self.pick_start_end()
        dme_points = self.generate_dme_points(start, end, num_points=8)
        gp_deg, method = self.derive_gp()
        alts = self.compute_altitudes(dme_points, gp_deg)
        # ensure last meets MDA
        final_gp, final_alts = self.ensure_last_at_least_mda(dme_points, alts, gp_deg)
        rows = []
        for d, alt in zip(dme_points, final_alts):
            dist_to_thr = max(0.0, round(float(d - self.dme_thr_nm),1))
            rows.append({"DME (NM)": round(d,1), "Distance to THR (NM)": dist_to_thr, "Altitude (ft)": int(round(alt))})
        df = pd.DataFrame(rows)
        return df, final_gp, self.warn_messages

    def compute_rod_table(self, df: pd.DataFrame, gs_list: Optional[List[int]] = None) -> pd.DataFrame:
        if gs_list is None or len(gs_list) == 0:
            gs_list = DEFAULT_GS_PRESET
        dme = df["DME (NM)"].values.astype(float)
        alt = df["Altitude (ft)"].values.astype(float)
        seg_dist_nm = np.abs(np.diff(dme))
        seg_alt_ft = np.abs(np.diff(alt))
        rod_rows = []
        for gs in gs_list:
            nm_per_min = gs / 60.0
            # avoid division by zero
            times_min = seg_dist_nm / nm_per_min if np.sum(seg_dist_nm) > 0 else np.array([0.0]*len(seg_dist_nm))
            total_time_min = np.sum(times_min) if np.sum(times_min) > 0 else 0.0
            total_alt = np.sum(seg_alt_ft)
            rod = (total_alt / total_time_min) if total_time_min > 0 else 0.0
            total_seconds = int(round(total_time_min * 60.0))
            mm = total_seconds // 60
            ss = total_seconds % 60
            time_str = f"{mm:02d}:{ss:02d}"
            rod_rows.append({"GS (kt)": int(gs), "ROD (ft/min)": int(round(rod)), "FAF->MAPt Time": time_str})
        return pd.DataFrame(rod_rows)

# =========================
# Embedded unit-style tests (run in DEV mode)
# =========================
def run_simple_tests_on_generator(gen: CDFAGenerator, dme_df: pd.DataFrame, final_gp: float, mda_ft: float) -> List[Tuple[str,bool,str]]:
    results = []
    try:
        results.append(("8 DME points", len(dme_df) == 8, "Expect exactly 8 DME rows"))
        results.append(("GP reasonable", (final_gp >= 1.5 and final_gp <= 6.0), f"GP used = {final_gp:.2f}°"))
        results.append(("Last >= MDA", (dme_df.iloc[-1]["Altitude (ft)"] >= mda_ft), f"Last alt = {dme_df.iloc[-1]['Altitude (ft)']} ft, MDA = {mda_ft} ft"))
        results.append(("Monotonic DME", (dme_df["DME (NM)"].is_monotonic_decreasing if hasattr(dme_df["DME (NM)"], 'is_monotonic_decreasing') else True), "DME should be ordered outer->inner"))
    except Exception as e:
        results.append(("Tests error", False, str(e)))
    return results

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="CDFA-PLANNER (single-file)", layout="wide")
st.title("CDFA-PLANNER — Advanced DME & ROD Generator")

# Sidebar: upload & presets
st.sidebar.header("Upload & Presets")
uploaded = st.sidebar.file_uploader("Upload IAC / Approach Chart PDF (optional)", type=["pdf"])
country = st.sidebar.selectbox("Country preset (affects defaults)", ["DEFAULT","USA","EUROPE","JAPAN","ASIA","MIDDLE_EAST"])
gs_preset_input = st.sidebar.text_input("ROD GS preset (comma-separated kts)", value=",".join(str(x) for x in DEFAULT_GS_PRESET))
auto_test_toggle = st.sidebar.checkbox("Auto-run tests on upload (dev)", value=RUN_AUTO_TESTS)

# Show parser + auto-test area if file uploaded
parser = ChartParser()
parsed = {}
if uploaded is not None:
    file_bytes = uploaded.read()
    parsed = parser.parse(file_bytes)
    st.sidebar.success("Chart parsing attempted — verify parsed fields below")

# Main form
with st.form("main_form"):
    st.header("Approach Inputs (verify / edit parsed values)")
    col1, col2, col3 = st.columns(3)
    with col1:
        procedure_id = st.text_input("Procedure ID", value=parsed.get('procedure_id','') if parsed else "")
        thr_elev_ft = st.number_input("Threshold Elevation (ft)", value=float(parsed.get('thr_elev_ft', 100)) if parsed else 100.0, step=1.0)
        mda_ft = st.number_input("MDA / DA (ft)", value=float(parsed.get('mda_ft', 0)) if parsed and 'mda_ft' in parsed else 1000.0, step=1.0)
        gp_input = st.number_input("Glide Path Angle (°) - 0 = auto", value=float(parsed.get('gp_angle', 0.0)) if parsed and 'gp_angle' in parsed else 0.0, step=0.1)
    with col2:
        dme_thr_nm = st.number_input("DME at Threshold (NM)", value=float(parsed.get('dme_thr_nm', 0.0)) if parsed else 0.0, step=0.1)
        tod_dme_nm = st.number_input("TOD / FAF DME (NM)", value=(float(parsed.get('dme_candidates')[0]) if parsed and 'dme_candidates' in parsed and len(parsed['dme_candidates'])>0 else 7.0), step=0.1)
        mapt_dme_nm = st.number_input("MAPt DME (NM) (optional)", value=float(parsed.get('mapt_dme_nm', 0.0)) if parsed else 0.0, step=0.1)
        faf_alt_ft = st.number_input("FAF Altitude (ft) (optional)", value=float(parsed.get('faf_alt_ft', 0.0)) if parsed else 2500.0, step=1.0)
    with col3:
        thr_lat = st.text_input("THR Lat (decimal) (optional)", value=str(parsed.get('coords')[0]) if parsed and 'coords' in parsed else "")
        thr_lon = st.text_input("THR Lon (decimal) (optional)", value=str(parsed.get('coords')[1]) if parsed and 'coords' in parsed else "")
        dme_lat = st.text_input("DME Station Lat (decimal) (optional)", value="")
        dme_lon = st.text_input("DME Station Lon (decimal) (optional)", value="")
        antenna_height_ft = st.number_input("DME antenna height (ft) optional", value=0.0, step=1.0)

    st.markdown("**Step-Down Fixes / SDFs (optional)**")
    st.markdown("Enter up to 6 lines in the format: `alt_ft,dme_nm`")
    sdf_text = st.text_area("SDFs", value="")
    sdf_parsed = []
    for line in sdf_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parts = re.split('[,;\\s]+', line)
            alt = float(parts[0])
            dme = float(parts[1])
            sdf_parsed.append({'alt_ft': alt, 'dme': dme})
        except:
            continue

    st.markdown("**Advanced settings (UI-configurable)**")
    spacing_exponent = st.slider("DME spacing exponent (higher => denser near MAPt)", 1.0, 4.0, 2.0, 0.1)
    gp_clamp_min = st.number_input("GP clamp min (°) - values below will be raised", value=2.5, step=0.1)
    raise_to = st.number_input("GP raise-to (°) when below clamp", value=3.0, step=0.1)
    gp_clamp_max_warn = st.number_input("GP upper warn threshold (°) - warns if exceeded", value=4.5, step=0.1)
    gs_input = st.text_input("ROD GS list (kts comma-separated)", value=gs_preset_input)

    submit = st.form_submit_button("Generate CDFA profile")

# If auto-test is on and file uploaded, run parsing/test immediately
if uploaded is not None and auto_test_toggle:
    st.subheader("Parsed fields (auto-run)")
    st.json(parsed or {"note": "No fields parsed"})

    # Build a minimal generation run using parsed values (safe defaults if missing)
    try:
        # fill missing sensible defaults
        thr_elev_val = float(parsed.get('thr_elev_ft', thr_elev_ft if thr_elev_ft else 100.0))
        mda_val = float(parsed.get('mda_ft', mda_ft if mda_ft else 1000.0))
        gp_val = float(parsed.get('gp_angle', gp_input if gp_input else 0.0))
        tod_val = float(parsed.get('dme_candidates',[tod_dme_nm])[0]) if parsed.get('dme_candidates') else float(tod_dme_nm)
        dme_thr_val = float(dme_thr_nm)
        # coords from parsed if found
        thr_coords_val = None
        dme_coords_val = None
        if 'coords' in parsed:
            try:
                thr_coords_val = (float(parsed['coords'][0]), float(parsed['coords'][1]))
            except:
                thr_coords_val = None

        # Instantiate generator for test
        gen = CDFAGenerator(
            thr_elev_ft = thr_elev_val,
            mda_ft = mda_val,
            dme_thr_nm = dme_thr_val,
            tod_dme_nm = tod_val,
            mapt_dme_nm = None,
            faf_alt_ft = None,
            gp_angle_deg = gp_val if gp_val>0 else None,
            sdf_list = sdf_parsed,
            thr_coords = thr_coords_val,
            dme_coords = dme_coords_val,
            gp_clamp_min = gp_clamp_min,
            gp_clamp_max_warn = gp_clamp_max_warn,
            raise_to = raise_to,
            spacing_exponent = spacing_exponent
        )

        dme_df, gp_used, warnings = gen.build_dme_table()
        rod_df = gen.compute_rod_table(dme_df, gs_list=[int(x.strip()) for x in gs_input.split(",") if x.strip()])

        st.subheader("Auto-test warnings (if any)")
        if warnings:
            for w in warnings:
                st.warning(w)
        else:
            st.success("No warnings during auto-run")

        st.subheader("Auto-test results")
        test_results = run_simple_tests_on_generator(gen, dme_df, gp_used, mda_val)
        for name, passed, info in test_results:
            if passed:
                st.success(f"✅ {name} — {info}")
            else:
                st.error(f"❌ {name} — {info}")

        st.subheader("Auto-generated DME table (from parsed inputs)")
        st.dataframe(dme_df.style.format({"DME (NM)": "{:.1f}", "Distance to THR (NM)": "{:.1f}", "Altitude (ft)": "{:.0f}"}))

        st.subheader("Auto-generated ROD table")
        st.dataframe(rod_df)

        # Provide plot
        fig, ax = plt.subplots(figsize=(11,4))
        x = dme_df["Distance to THR (NM)"].values
        y = dme_df["Altitude (ft)"].values
        ax.plot(x, y, marker='o', linewidth=2)
        ax.invert_xaxis()
        ax.set_xlabel("Distance to THR (NM)")
        ax.set_ylabel("Altitude (ft)")
        ax.set_title(f"Auto CDFA Profile — GP used {gp_used:.2f}°")
        ax.axhline(mda_val, color='red', linestyle='--', label='MDA')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error("Auto-test failed: " + str(e))
        st.text(traceback.format_exc())

# Manual generation if user clicked Generate
if submit:
    try:
        # coords parsing
        thr_coords = None
        dme_coords = None
        try:
            thr_coords = (float(thr_lat), float(thr_lon)) if thr_lat and thr_lon else None
        except:
            thr_coords = None
        try:
            dme_coords = (float(dme_lat), float(dme_lon)) if dme_lat and dme_lon else None
        except:
            dme_coords = None

        gs_list = [int(x.strip()) for x in gs_input.split(",") if x.strip()]
        if not gs_list:
            gs_list = DEFAULT_GS_PRESET

        gen = CDFAGenerator(
            thr_elev_ft = float(thr_elev_ft),
            mda_ft = float(mda_ft),
            dme_thr_nm = float(dme_thr_nm),
            tod_dme_nm = float(tod_dme_nm),
            mapt_dme_nm = float(mapt_dme_nm) if mapt_dme_nm>0 else None,
            faf_alt_ft = float(faf_alt_ft) if faf_alt_ft>0 else None,
            gp_angle_deg = float(gp_input) if gp_input>0 else None,
            sdf_list = sdf_parsed,
            thr_coords = thr_coords,
            dme_coords = dme_coords,
            gp_clamp_min = float(gp_clamp_min),
            gp_clamp_max_warn = float(gp_clamp_max_warn),
            raise_to = float(raise_to),
            spacing_exponent = float(spacing_exponent)
        )

        dme_df, gp_used, warnings = gen.build_dme_table()
        rod_df = gen.compute_rod_table(dme_df, gs_list=gs_list)

        if warnings:
            for w in warnings:
                st.warning(w)

        st.subheader(f"DME / Alt Table (GP used: {gp_used:.2f}°)")
        st.dataframe(dme_df.style.format({"DME (NM)": "{:.1f}", "Distance to THR (NM)": "{:.1f}", "Altitude (ft)": "{:.0f}"}), height=300)

        st.subheader("ROD Table — FAF -> MAPt")
        st.dataframe(rod_df, height=180)

        fig, ax = plt.subplots(figsize=(11,4))
        x = dme_df["Distance to THR (NM)"].values
        y = dme_df["Altitude (ft)"].values
        ax.plot(x, y, marker='o', linewidth=2)
        ax.invert_xaxis()
        ax.set_xlabel("Distance to THR (NM)")
        ax.set_ylabel("Altitude (ft)")
        ax.set_title(f"CDFA Profile — GP {gp_used:.2f}°")
        ax.axhline(float(mda_ft), color='red', linestyle='--', label='MDA')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        st.pyplot(fig)

        # Exports
        csv_bytes = dme_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download DME CSV", data=csv_bytes, file_name=f"dme_table_{procedure_id or 'procedure'}.csv", mime='text/csv')

        pdf_buf = BytesIO()
        fig.savefig(pdf_buf, format='pdf', bbox_inches='tight')
        pdf_buf.seek(0)
        st.download_button("Download Profile PDF", data=pdf_buf.getvalue(), file_name=f"cdaf_profile_{procedure_id or 'procedure'}.pdf", mime='application/pdf')

        st.success("CDFA generation complete. Always validate against published charts and AIP.")
    except Exception as e:
        st.error("Generation error: " + str(e))
        st.text(traceback.format_exc())

# Footer
st.markdown("""
**Notes**
- This tool produces planner outputs. Always cross-check with official charts and AIP.
- OCR parsing is a best-effort fallback. Verify parsed fields and correct them before generating profiles.
- For OCR to work in Docker/Render, ensure `tesseract-ocr` and `poppler-utils` are installed (Dockerfile provided).
""")






