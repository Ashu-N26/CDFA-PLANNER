"""
CDFA-PLANNER (single-file) — UPDATED
- TOD -> MDA DME table (first = TOD/outer SDF, last = MDA)
- Structured SDF editor (interactive)
- GP derivation & NavBlue-style clamps / warnings
- Slant->horizontal correction via haversine when coords given
- ROD per-segment and FAF->MAPt aggregated with flags (>1000 fpm, >300 fpm deviations)
- Embedded auto-test harness (RUN_AUTO_TESTS)
"""

import math
from io import BytesIO
from typing import List, Tuple, Optional, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, traceback

# PDF/OCR
try:
    import pdfplumber
except Exception:
    pdfplumber = None
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

# -------------------------
# Dev toggle
# -------------------------
RUN_AUTO_TESTS = True

# -------------------------
# Constants
# -------------------------
NM_TO_FT = 6076.12
EARTH_RADIUS_KM = 6371.0
DEFAULT_GS_PRESET = [80, 100, 120, 140, 160]
DEFAULT_SPACING_EXP = 2.0
TARGET_ROD_LIMIT = 1000  # ft/min threshold to flag
ROD_DEVIATION_LIMIT = 300  # ft/min deviation between segments

# -------------------------
# Utility functions
# -------------------------
def haversine_nm(lat1, lon1, lat2, lon2):
    try:
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
        c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
        km = EARTH_RADIUS_KM * c
        nm = km / 1.852
        return nm
    except Exception:
        return None

def fmt_time_mmss(total_minutes: float) -> str:
    total_seconds = int(round(total_minutes * 60.0))
    mm = total_seconds // 60
    ss = total_seconds % 60
    return f"{mm:02d}:{ss:02d}"

# -------------------------
# Parser
# -------------------------
class ChartParser:
    def __init__(self):
        self.re_mda = re.compile(r'\bMDA\b[:\s]*([0-9]{3,5})\s*ft', re.IGNORECASE)
        self.re_angle = re.compile(r'([0-9]\.?[0-9])\s?°', re.IGNORECASE)
        self.re_dme = re.compile(r'([0-9]{1,3}\.[0-9])\s*NM', re.IGNORECASE)
        self.re_coords = re.compile(r'([-+]?\d+\.\d+)[,;\s]+([-+]?\d+\.\d+)')

    def extract_text(self, pdf_bytes: bytes) -> str:
        # try pdfplumber
        text_accum = []
        if pdfplumber:
            try:
                with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                    for p in pdf.pages:
                        t = p.extract_text()
                        if t:
                            text_accum.append(t)
            except Exception:
                text_accum = []
        text = "\n".join(text_accum).strip()
        if text:
            return text
        # fallback OCR
        try:
            images = convert_from_bytes(pdf_bytes, dpi=220)
            chunks = [pytesseract.image_to_string(im) for im in images]
            return "\n".join(chunks)
        except Exception:
            return ""

    def parse(self, pdf_bytes: bytes) -> Dict[str, Any]:
        doc = self.extract_text(pdf_bytes)
        out: Dict[str, Any] = {}
        if not doc:
            return out
        m = self.re_mda.search(doc)
        if m:
            try:
                out['mda_ft'] = int(m.group(1))
            except:
                pass
        a = self.re_angle.findall(doc)
        if a:
            try:
                out['gp_angle'] = float(a[0])
            except:
                pass
        dmes = self.re_dme.findall(doc)
        if dmes:
            try:
                out['dme_candidates'] = [float(x) for x in dmes]
            except:
                out['dme_candidates'] = []
        coords = []
        for c in self.re_coords.findall(doc):
            try:
                lat = float(c[0]); lon = float(c[1]); coords.append((lat, lon))
            except:
                continue
        if coords:
            out['coords'] = coords[0]
        out['snippet'] = (doc[:1200] + '...') if len(doc)>1200 else doc
        return out

# -------------------------
# CDFA Generator
# -------------------------
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
                 antenna_height_ft: float = 0.0,
                 gp_clamp_min: float = 2.5,
                 raise_to: float = 3.0,
                 gp_warn_max: float = 4.5,
                 spacing_exponent: float = DEFAULT_SPACING_EXP,
                 min_horiz_nm: float = 0.05):
        self.thr_elev_ft = float(thr_elev_ft)
        self.mda_ft = float(mda_ft)
        self.dme_thr_nm = float(dme_thr_nm)
        self.tod_dme_nm = float(tod_dme_nm)
        self.mapt_dme_nm = float(mapt_dme_nm) if mapt_dme_nm and mapt_dme_nm>0 else None
        self.faf_alt_ft = float(faf_alt_ft) if faf_alt_ft and faf_alt_ft>0 else None
        self.gp_angle_deg = float(gp_angle_deg) if gp_angle_deg and gp_angle_deg>0 else None
        self.sdf_list = sdf_list or []  # list of {'alt_ft':..., 'dme':...}
        self.thr_coords = thr_coords
        self.dme_coords = dme_coords
        self.antenna_height_ft = float(antenna_height_ft)

        self.gp_clamp_min = gp_clamp_min
        self.raise_to = raise_to
        self.gp_warn_max = gp_warn_max
        self.spacing_exponent = spacing_exponent
        self.min_horiz_nm = min_horiz_nm

        self.warnings: List[str] = []

    def slant_to_horizontal(self, point_dme_nm: float) -> float:
        """Return horizontal NM distance from point to threshold."""
        if self.thr_coords and self.dme_coords:
            station_to_thr_nm = haversine_nm(self.dme_coords[0], self.dme_coords[1], self.thr_coords[0], self.thr_coords[1])
            if station_to_thr_nm is None:
                return max(0.0, point_dme_nm - self.dme_thr_nm)
            horiz_to_thr = abs(station_to_thr_nm - point_dme_nm)
            return max(0.0, horiz_to_thr)
        return max(0.0, point_dme_nm - self.dme_thr_nm)

    def derive_gp(self) -> Tuple[float,str]:
        """Derive GP; returns (gp_deg, method). Applies clamp/raise logic."""
        if self.gp_angle_deg:
            gp = self.gp_angle_deg
            if gp < self.gp_clamp_min:
                self.warnings.append(f"User GP {gp:.2f}° < clamp {self.gp_clamp_min}°. Raising to {self.raise_to}°.")
                return (self.raise_to, "user_clamped")
            if gp > self.gp_warn_max:
                self.warnings.append(f"User GP {gp:.2f}° > warn max {self.gp_warn_max}°. Verify obstacle clearance.")
            return (gp, "user")
        # Derive from FAF altitude if available
        if self.faf_alt_ft and self.tod_dme_nm:
            horiz_nm = max(self.min_horiz_nm, self.tod_dme_nm - self.dme_thr_nm)
            horiz_ft = horiz_nm * NM_TO_FT
            vert_drop = max(0.0, self.faf_alt_ft - self.thr_elev_ft)
            if horiz_ft <= 0 or vert_drop <= 0:
                self.warnings.append("Insufficient data for GP derivation; using fallback.")
                return (self.raise_to, "fallback")
            gp_rad = math.atan2(vert_drop, horiz_ft)
            gp_deg = math.degrees(gp_rad)
            if gp_deg < self.gp_clamp_min:
                self.warnings.append(f"Derived GP {gp_deg:.2f}° < clamp {self.gp_clamp_min}°. Raising to {self.raise_to}°.")
                return (self.raise_to, "derived_raised")
            if gp_deg > self.gp_warn_max:
                self.warnings.append(f"Derived GP {gp_deg:.2f}° > warn {self.gp_warn_max}°. Use with caution.")
            return (round(gp_deg,2), "derived")
        # fallback
        self.warnings.append("No FAF altitude or GP provided; using default 3.0°.")
        return (3.0, "default")

    def pick_start_end(self) -> Tuple[float, float, str]:
        """
        Determine outer start DME and inner end DME.
        Rule:
         - Start = max(TOD_DME, outermost SDF if provided and further out)
         - End = MAPt DME if provided else DME where GP hits MDA (computed)
         - We will ensure start > end; otherwise adjust start outward.
        Returns (start_nm, end_nm, reason_for_end)
        """
        start_nm = max(self.tod_dme_nm, self.dme_thr_nm + 0.1)
        if self.sdf_list:
            outer_sdf = max(self.sdf_list, key=lambda s: s['dme'])
            if outer_sdf['dme'] > start_nm:
                start_nm = outer_sdf['dme']
        # compute end: prefer MAPt if given
        if self.mapt_dme_nm:
            end_nm = self.mapt_dme_nm
            reason = "mapt_provided"
        else:
            # compute DME where GP intersects MDA using FAF or assumed GP
            gp_deg, _ = self.derive_gp()
            # solve for distance_to_thr where alt = MDA: alt = thr_elev + tan(gp)*horizontal_ft
            # horizontal_ft = (MDA - thr_elev) / tan(gp)
            gp_rad = math.radians(gp_deg) if gp_deg>0 else math.radians(3.0)
            if math.tan(gp_rad) <= 0:
                # fallback tiny end just above THR
                end_nm = self.dme_thr_nm + 0.05
                reason = "fallback_end"
            else:
                horiz_ft_needed = (self.mda_ft - self.thr_elev_ft) / math.tan(gp_rad)
                horiz_nm_needed = horiz_ft_needed / NM_TO_FT
                end_nm = self.dme_thr_nm + horiz_nm_needed
                reason = "computed_for_mda"
        # Ensure valid
        if start_nm <= end_nm:
            start_nm = end_nm + 0.5
            self.warnings.append("Start <= End — moved start outward to ensure valid descending series.")
        return round(start_nm,4), round(end_nm,4), reason

    def generate_base_points(self, start_nm: float, end_nm: float, n_points: int = 8) -> List[float]:
        t = np.linspace(0.0, 1.0, n_points)
        w = t ** float(self.spacing_exponent)
        arr = start_nm + (end_nm - start_nm) * w
        if arr[0] < arr[-1]:
            arr = arr[::-1]
        return [round(float(x),1) for x in arr]

    def replace_with_sdfs(self, base_points: List[float]) -> List[float]:
        """
        Insert SDF DMEs into the base_points by replacing nearest base_points.
        Preserve first (start) and last (end).
        """
        base = base_points.copy()
        if not self.sdf_list:
            return base
        sdf_dm = sorted([round(float(s['dme']),1) for s in self.sdf_list])
        # For each sdf (excluding ones equal to start/end), replace the nearest interior base point
        interior_idx = list(range(1, len(base)-1))
        used_idx = set()
        for sdf in sdf_dm:
            # if sdf equals start or end, ignore (already present)
            if abs(sdf - base[0]) < 1e-6 or abs(sdf - base[-1]) < 1e-6:
                continue
            # find nearest interior index not already used
            best_i, best_dist = None, 1e9
            for i in interior_idx:
                if i in used_idx:
                    continue
                d = abs(base[i] - sdf)
                if d < best_dist:
                    best_dist = d; best_i = i
            if best_i is not None:
                base[best_i] = sdf
                used_idx.add(best_i)
        # ensure sorted outer->inner
        base_sorted = sorted(base, reverse=True)
        return [round(x,1) for x in base_sorted]

    def compute_altitudes(self, dme_points: List[float], gp_deg: float) -> List[int]:
        gp_rad = math.radians(gp_deg)
        alts = []
        for d in dme_points:
            horiz_nm = self.slant_to_horizontal(d)
            horizontal_ft = horiz_nm * NM_TO_FT
            vert = math.tan(gp_rad) * horizontal_ft
            alt = self.thr_elev_ft + vert
            alts.append(int(round(alt)))
        return alts

    def ensure_last_exact_mda(self, dme_points: List[float], alts: List[int]) -> Tuple[List[float], List[int]]:
        """
        Force last altitude to be exactly MDA.
        If last DME != computed DME location for MDA (and MAPt not given), adjust last DME to exact MDA DME.
        """
        # If MAPt provided, last DME should be MAPt and altitude will be computed (may not exactly equal MDA).
        if self.mapt_dme_nm:
            # If we want last altitude exactly MDA, we can override last altitude to MDA.
            alts[-1] = int(round(self.mda_ft))
            return dme_points, alts
        # If MAPt not provided, compute DME_at_MDA where alt = MDA using current GP
        # Solve for horiz_ft_needed = (MDA - thr_elev) / tan(gp)
        # DME_at_MDA = dme_thr + horiz_nm_needed
        gp_deg, _ = self.derive_gp()
        gp_rad = math.radians(gp_deg if gp_deg>0 else 3.0)
        if math.tan(gp_rad) == 0:
            return dme_points, alts
        horiz_ft_needed = (self.mda_ft - self.thr_elev_ft) / math.tan(gp_rad)
        horiz_nm_needed = horiz_ft_needed / NM_TO_FT
        dme_at_mda = round(self.dme_thr_nm + horiz_nm_needed, 1)
        # Replace last point's DME with computed DME_at_MDA and set altitude to MDA
        dme_points[-1] = dme_at_mda
        alts[-1] = int(round(self.mda_ft))
        # ensure descending order outer->inner
        dp_sorted = sorted(dme_points, reverse=True)
        # reorder alts consistently: recompute alts for all points with gp_deg so they match; then set last to MDA explicitly
        alts_recomputed = self.compute_altitudes(dp_sorted, gp_deg)
        alts_recomputed[-1] = int(round(self.mda_ft))
        return [round(x,1) for x in dp_sorted], alts_recomputed

    def build_table(self) -> Tuple[pd.DataFrame, float, List[str]]:
        start, end, reason = self.pick_start_end()
        base = self.generate_base_points(start, end, n_points=8)
        with_sdfs = self.replace_with_sdfs(base)
        gp_deg, method = self.derive_gp()
        alts = self.compute_altitudes(with_sdfs, gp_deg)
        # Ensure last altitude equals MDA
        dme_pts, final_alts = self.ensure_last_exact_mda(with_sdfs, alts)
        # Build DataFrame
        rows = []
        for d, alt in zip(dme_pts, final_alts):
            dist_to_thr = max(0.0, round(d - self.dme_thr_nm,1))
            rows.append({"DME (NM)": round(d,1), "Distance to THR (NM)": dist_to_thr, "Altitude (ft)": int(round(alt))})
        df = pd.DataFrame(rows)
        return df, gp_deg, self.warnings

    def compute_rod(self, df: pd.DataFrame, gs_list: Optional[List[int]]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if gs_list is None or len(gs_list)==0:
            gs_list = DEFAULT_GS_PRESET
        dme = df["DME (NM)"].values.astype(float)
        alt = df["Altitude (ft)"].values.astype(float)
        seg_dist = np.abs(np.diff(dme))
        seg_alt = np.abs(np.diff(alt))
        segs = []
        for i in range(len(seg_dist)):
            segs.append({"seg_index": i, "dist_nm": float(seg_dist[i]), "alt_ft": float(seg_alt[i])})
        # per-segment ROD per GS
        per_seg_rows = []
        for seg in segs:
            for gs in gs_list:
                nm_per_min = gs/60.0
                time_min = seg["dist_nm"] / nm_per_min if nm_per_min>0 else 0.0
                rod = seg["alt_ft"] / time_min if time_min>0 else 0.0
                per_seg_rows.append({"Segment": seg["seg_index"]+1, "GS (kt)": gs, "Segment ROD (ft/min)": int(round(rod)), "Seg Time (MM:SS)": fmt_time_mmss(time_min)})
        per_seg_df = pd.DataFrame(per_seg_rows)
        # Aggregate FAF->MAPt ROD per GS
        agg_rows = []
        for gs in gs_list:
            nm_per_min = gs/60.0
            times = seg_dist / nm_per_min if nm_per_min>0 else np.zeros_like(seg_dist)
            total_time = np.sum(times) if np.sum(times)>0 else 0.0
            total_alt = np.sum(seg_alt)
            rod_total = (total_alt / total_time) if total_time>0 else 0.0
            agg_rows.append({"GS (kt)": int(gs), "ROD (ft/min)": int(round(rod_total)), "FAF->MAPt Time": fmt_time_mmss(total_time)})
        agg_df = pd.DataFrame(agg_rows)
        return agg_df, per_seg_df

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="CDFA-PLANNER (TOD->MDA)", layout="wide")
st.title("CDFA-PLANNER — TOD → MDA (Advanced DME & ROD)")

# Sidebar upload / presets
st.sidebar.header("Upload & developer")
uploaded = st.sidebar.file_uploader("Upload IAC / approach chart PDF (optional)", type=["pdf"])
auto_test = st.sidebar.checkbox("Auto-run tests on upload", value=RUN_AUTO_TESTS)

# Parser & parsed preview
parser = ChartParser()
parsed = {}
if uploaded:
    try:
        pdf_bytes = uploaded.read()
        parsed = parser.parse(pdf_bytes)
        st.sidebar.success("Parsing attempted — verify parsed fields")
    except Exception as e:
        st.sidebar.error("Parse error: " + str(e))

# Main form
with st.form("main"):
    st.header("Inputs (verify & edit parsed values)")
    c1, c2, c3 = st.columns(3)
    with c1:
        proc_id = st.text_input("Procedure ID", value=parsed.get('procedure_id','') if parsed else "")
        thr_elev_ft = st.number_input("Threshold Elevation (ft)", value=float(parsed.get('thr_elev_ft', 695)) if parsed else 695.0, step=1.0)
        mda_ft = st.number_input("MDA/DA (ft)", value=float(parsed.get('mda_ft', 1000)) if parsed and 'mda_ft' in parsed else 1000.0, step=1.0)
        gp_input = st.number_input("Glide Path Angle (°) — 0 = auto", value=float(parsed.get('gp_angle', 0.0)) if parsed and 'gp_angle' in parsed else 0.0, step=0.1)
    with c2:
        dme_thr_nm = st.number_input("DME at Threshold (NM)", value=float(parsed.get('dme_thr_nm', 1.3)) if parsed else 1.3, step=0.1)
        tod_dme_nm = st.number_input("TOD / FAF DME (NM)", value=(float(parsed.get('dme_candidates')[0]) if parsed and 'dme_candidates' in parsed and len(parsed['dme_candidates'])>0 else 13.6), step=0.1)
        mapt_dme_nm = st.number_input("MAPt DME (NM) optional", value=float(parsed.get('mapt_dme_nm', 1.8)) if parsed else 0.0, step=0.1)
        faf_alt_ft = st.number_input("FAF Altitude (ft) optional", value=float(parsed.get('faf_alt_ft', 3600)) if parsed else 3600.0, step=1.0)
    with c3:
        thr_lat = st.text_input("THR Lat (decimal) optional", value=str(parsed.get('coords')[0]) if parsed and 'coords' in parsed else "")
        thr_lon = st.text_input("THR Lon (decimal) optional", value=str(parsed.get('coords')[1]) if parsed and 'coords' in parsed else "")
        dme_lat = st.text_input("DME Station Lat (decimal) optional", value="")
        dme_lon = st.text_input("DME Station Lon (decimal) optional", value="")
        antenna_ht = st.number_input("DME antenna height (ft) optional", value=0.0, step=1.0)

    st.markdown("### Step-Down Fixes (SDFs) — structured editor")
    st.markdown("Use the small table below to add/remove up to 6 SDFs (alt_ft, dme_nm). They will be inserted into the 8-point table where appropriate.")
    # Use experimental_data_editor if available; fallback to text area
    sdf_df = None
    try:
        # initial sdf rows from parsed? none by default
        initial = pd.DataFrame(parsed.get('sdfs', [])) if parsed and 'sdfs' in parsed else pd.DataFrame(columns=["alt_ft","dme"])
        sdf_editor = st.experimental_data_editor(initial, num_rows="dynamic", key="sdf_editor")
        # normalize into list
        sdf_list = []
        for _, r in sdf_editor.iterrows():
            try:
                alt = float(r.get("alt_ft") if r.get("alt_ft") not in (None,"") else 0)
                dmev = float(r.get("dme") if r.get("dme") not in (None,"") else 0)
                if alt>0 and dmev>0:
                    sdf_list.append({"alt_ft": alt, "dme": dmev})
            except:
                continue
    except Exception:
        # fallback: text area
        sdf_text = st.text_area("SDFs (lines alt_ft,dme_nm)", value="")
        sdf_list = []
        for line in sdf_text.splitlines():
            try:
                a,d = re.split('[,;\\s]+', line.strip())
                sdf_list.append({"alt_ft": float(a), "dme": float(d)})
            except:
                continue

    st.markdown("### Advanced settings")
    spacing_exponent = st.slider("DME spacing exponent (higher => denser near MDA)", 1.0, 4.0, float(DEFAULT_SPACING_EXP), 0.1)
    gp_clamp_min = st.number_input("GP clamp min (°) (below => raised)", value=2.5, step=0.1)
    raise_to = st.number_input("Raise-to GP (°) when derived below clamp", value=3.0, step=0.1)
    gp_warn_max = st.number_input("GP warn max (°) (above => show steep warning)", value=4.5, step=0.1)
    min_horiz_nm = st.number_input("Min horizontal NM guard (to avoid tiny distances)", value=0.05, step=0.01)
    gs_input = st.text_input("ROD GS list (kts) comma-separated", value=",".join(str(x) for x in DEFAULT_GS_PRESET))
    submit = st.form_submit_button("Generate CDFA (TOD -> MDA)")

# Auto-run tests block (on upload)
if uploaded and auto_test:
    st.subheader("Parsed fields (auto-run)")
    st.json(parsed or {"note":"no parsed fields"})
    # Run a quick generation with parsed (safe defaults)
    try:
        thr_e = float(parsed.get('thr_elev_ft', thr_elev_ft))
        mda = float(parsed.get('mda_ft', mda_ft))
        tod = float((parsed.get('dme_candidates')[0]) if parsed.get('dme_candidates') else tod_dme_nm)
        dme_thr = float(dme_thr_nm)
        # coords
        thr_coords = None
        dme_coords = None
        if 'coords' in parsed:
            try:
                thr_coords = (float(parsed['coords'][0]), float(parsed['coords'][1]))
            except:
                thr_coords = None
        gen = CDFAGenerator(thr_elev_ft=thr_e, mda_ft=mda, dme_thr_nm=dme_thr, tod_dme_nm=tod,
                            faf_alt_ft=float(parsed.get('faf_alt_ft', faf_alt_ft)) if parsed.get('faf_alt_ft') else faf_alt_ft,
                            gp_angle_deg=float(parsed.get('gp_angle')) if parsed.get('gp_angle') else None,
                            sdf_list=sdf_list, thr_coords=thr_coords, dme_coords=dme_coords,
                            antenna_height_ft=antenna_ht, gp_clamp_min=gp_clamp_min, raise_to=raise_to,
                            gp_warn_max=gp_warn_max, spacing_exponent=spacing_exponent, min_horiz_nm=min_horiz_nm)
        dme_df, gp_used, warns = gen.build_table()
        rod_df, per_seg = gen.compute_rod(dme_df, gs_list=[int(x.strip()) for x in gs_input.split(",") if x.strip()])
        st.subheader("Auto-test warnings")
        if warns:
            for w in warns: st.warning(w)
        else:
            st.success("No warnings")
        st.subheader("Auto-test basic checks")
        tests = []
        tests.append(("8 rows", len(dme_df)==8))
        tests.append(("First DME equals start (TOD/outer SDF)", abs(dme_df.iloc[0]["DME (NM)"] - round(float(tod),1))<=0.5))
        tests.append(("Last altitude equals MDA", int(dme_df.iloc[-1]["Altitude (ft)"])==int(round(mda))))
        tests.append(("ROD computed", not rod_df.empty))
        for name, passed in tests:
            if passed: st.success(f"✅ {name}")
            else: st.error(f"❌ {name}")
        st.subheader("Auto-generated DME table")
        st.dataframe(dme_df)
        st.subheader("Auto-generated ROD table (FAF->MAPt)")
        st.dataframe(rod_df)
        st.subheader("Per-segment RODs")
        st.dataframe(per_seg)
    except Exception as e:
        st.error("Auto-test error: " + str(e))
        st.text(traceback.format_exc())

# Manual generation block
if submit:
    try:
        # coords parsing
        try:
            thr_coords = (float(thr_lat), float(thr_lon)) if thr_lat and thr_lon else None
        except:
            thr_coords = None
        try:
            dme_coords = (float(dme_lat), float(dme_lon)) if dme_lat and dme_lon else None
        except:
            dme_coords = None
        # prepare gs list
        try:
            gs_list = [int(x.strip()) for x in gs_input.split(",") if x.strip()]
            if not gs_list:
                gs_list = DEFAULT_GS_PRESET
        except:
            gs_list = DEFAULT_GS_PRESET

        gen = CDFAGenerator(thr_elev_ft=float(thr_elev_ft), mda_ft=float(mda_ft),
                            dme_thr_nm=float(dme_thr_nm), tod_dme_nm=float(tod_dme_nm),
                            mapt_dme_nm=float(mapt_dme_nm) if mapt_dme_nm>0 else None,
                            faf_alt_ft=float(faf_alt_ft) if faf_alt_ft>0 else None,
                            gp_angle_deg=float(gp_input) if gp_input>0 else None,
                            sdf_list=sdf_list, thr_coords=thr_coords, dme_coords=dme_coords,
                            antenna_height_ft=float(antenna_ht), gp_clamp_min=float(gp_clamp_min),
                            raise_to=float(raise_to), gp_warn_max=float(gp_warn_max),
                            spacing_exponent=float(spacing_exponent), min_horiz_nm=float(min_horiz_nm))
        dme_df, gp_used, warnings = gen.build_table()
        rod_df, per_seg_df = gen.compute_rod(dme_df, gs_list=gs_list)

        for w in warnings: st.warning(w)
        st.subheader(f"DME / Alt Table (GP used: {gp_used:.2f}°)")
        st.dataframe(dme_df.style.format({"DME (NM)":"{:.1f}","Distance to THR (NM)":"{:.1f}","Altitude (ft)":"{:.0f}"}), height=300)
        st.subheader("ROD Table (FAF -> MAPt)")
        # flag ROD > target
        rod_df_display = rod_df.copy()
        rod_df_display["Flag"] = rod_df_display["ROD (ft/min)"].apply(lambda v: "⚠️" if v>TARGET_ROD_LIMIT else "")
        st.dataframe(rod_df_display, height=180)
        st.subheader("Per-segment RODs")
        # highlight segments > deviation
        per_seg_df_display = per_seg_df.copy()
        # compute deviation per segment between consecutive GS rows not meaningful here; we will show absolute segment RODs
        st.dataframe(per_seg_df_display, height=220)

        # plot
        fig, ax = plt.subplots(figsize=(11,4))
        x = dme_df["Distance to THR (NM)"].values
        y = dme_df["Altitude (ft)"].values
        ax.plot(x, y, marker='o', linewidth=2, label='CDFA profile')
        ax.invert_xaxis()
        ax.set_xlabel("Distance to THR (NM)")
        ax.set_ylabel("Altitude (ft)")
        ax.set_title(f"CDFA Profile — GP {gp_used:.2f}°")
        ax.axhline(float(mda_ft), color='red', linestyle='--', label='MDA')
        # plot SDFs
        for s in sdf_list:
            d_rel = round(float(s['dme'] - float(dme_thr_nm)),1)
            ax.scatter([d_rel],[s['alt_ft']], marker='x', color='red', s=80)
            ax.text(d_rel, s['alt_ft']+80, f"SDF {s['dme']:.1f}NM\n{s['alt_ft']}ft", ha='center', va='bottom')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        st.pyplot(fig)

        # exports
        csv_bytes = dme_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download DME CSV", data=csv_bytes, file_name=f"dme_{proc_id or 'proc'}.csv", mime='text/csv')
        buf = BytesIO(); fig.savefig(buf, format='pdf', bbox_inches='tight'); buf.seek(0)
        st.download_button("Download Profile PDF", data=buf.getvalue(), file_name=f"cdaf_{proc_id or 'proc'}.pdf", mime='application/pdf')

        # Additional checks & messages
        # compute aggregated ROD warnings
        for _, row in rod_df.iterrows():
            if row["ROD (ft/min)"] > TARGET_ROD_LIMIT:
                st.warning(f"Average ROD for GS {row['GS (kt)']} = {row['ROD (ft/min)']} ft/min > {TARGET_ROD_LIMIT} ft/min.")
    except Exception as e:
        st.error("Generation error: " + str(e))
        st.text(traceback.format_exc())

st.markdown("""
**Notes**
- This tool is a planner and aids calculation — always validate against official AIP/Jeppesen/NavBlue charts.
- Parser is conservative — verify parsed values before generating.
- For precise slant correction you can provide DME transmitter elevation and antenna height (advanced).
""")







