# app.py — Single-file CDFA-PLANNER with OCR fallback and GP safety clamp

import math
from io import BytesIO
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import streamlit as st
from PIL import Image

# PDF/text/ocr libs (may be optional on deployment)
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None
try:
    import pytesseract
except Exception:
    pytesseract = None

# ----- Constants -----
NM_TO_FT = 6076.12
KTS_TO_NM_PER_MIN = 1.0 / 60.0  # knot -> NM per min
KTS_TO_FT_PER_MIN = 101.268     # as used earlier

DEFAULT_GP_MIN = 2.5
DEFAULT_GP_MAX = 3.5
DEFAULT_GP_FALLBACK = 3.0

# ----------------------
# Utilities: PDF -> text/images
# ----------------------
def extract_text_from_pdf_bytes(data: bytes):
    """Try pdfplumber text extraction first; if that fails and OCR available, return None
    (caller may then attempt OCR)."""
    if pdfplumber:
        try:
            with pdfplumber.open(BytesIO(data)) as pdf:
                texts = []
                for p in pdf.pages:
                    try:
                        t = p.extract_text() or ""
                    except Exception:
                        t = ""
                    texts.append(t)
                joined = "\n".join(texts).strip()
                if joined:
                    return joined
        except Exception:
            return None
    return None

def pdf_to_images_bytes(data: bytes, dpi=200):
    """Convert PDF bytes to list of PIL images using pdf2image (requires poppler)."""
    if convert_from_bytes is None:
        raise RuntimeError("pdf2image.convert_from_bytes is not available.")
    imgs = convert_from_bytes(data, dpi=dpi)
    return [im.convert("RGB") for im in imgs]

def ocr_images_to_text(images):
    """Run pytesseract OCR over list of PIL images and return concatenated text."""
    if pytesseract is None:
        raise RuntimeError("pytesseract not available for OCR.")
    texts = []
    for im in images:
        try:
            texts.append(pytesseract.image_to_string(im))
        except Exception:
            texts.append("")
    return "\n".join(texts)

# ----------------------
# Chart parser heuristics (best-effort)
# ----------------------
class IACParserHeuristics:
    re_mda = re.compile(r"\bMDA\b[^0-9]{0,6}([0-9]{3,4})\s*ft", re.IGNORECASE)
    re_gp_explicit = re.compile(r"([23-6]\.?[0-9]?)\s*°\s*(gp|glide|glideslope|glide path)?", re.IGNORECASE)
    re_dme_value = re.compile(r"([0-9]{1,2}\.[0-9])\s*NM", re.IGNORECASE)
    re_coords = re.compile(r"(-?\d+\.\d+)[^\d-]{1,6}(-?\d+\.\d+)")
    # Search for useful numbers: MDA, FAF/TO DME, MAPt DME, DME@THR, FAF alt (ft)
    def parse(self, doc_text: str):
        out = {}
        if not doc_text:
            return out
        # MDA
        m = self.re_mda.search(doc_text)
        if m:
            try:
                out['mda_ft'] = int(m.group(1))
            except:
                pass
        # explicit GP
        m = self.re_gp_explicit.search(doc_text)
        if m:
            try:
                gp_val = float(m.group(1))
                # accept reasonable GP values (2.0 - 10.0) as hint
                if 1.5 <= gp_val <= 10.0:
                    out['gp_angle'] = gp_val
            except:
                pass
        # DME numbers (first few matches)
        dmes = self.re_dme_value.findall(doc_text)
        if dmes:
            # heuristics: first numeric NM often FAF/TOD; last small values maybe thresholds
            floats = [float(x) for x in dmes]
            out['dme_candidates'] = floats
            # set first as tod/faf if not present
            out.setdefault('tod_dme_nm', floats[0])
            # try to find very small near 0.1-1.5 as MAPt or DME@THR
            near_thr = [f for f in floats if f <= 2.0]
            if near_thr:
                out.setdefault('mapt_dme_nm', min(near_thr))
        # coords
        coords = self.re_coords.findall(doc_text)
        if coords:
            try:
                lat, lon = coords[0]
                out['thr_lat'] = float(lat)
                out['thr_lon'] = float(lon)
            except:
                pass
        # try to glean FAF altitude (e.g., 2500ft)
        m_alt = re.search(r"FAF[^0-9]{0,6}([0-9]{3,4})\s*ft", doc_text, re.IGNORECASE)
        if m_alt:
            try:
                out['faf_alt_ft'] = int(m_alt.group(1))
            except:
                pass
        # procedure id guess
        for line in doc_text.splitlines():
            s = line.strip()
            if s.isupper() and 3 < len(s) < 40 and any(ch.isalpha() for ch in s):
                out.setdefault('procedure_id', s)
                break
        return out

# ----------------------
# CDFA core planner
# ----------------------
class CDFAPlanner:
    def __init__(self,
                 thr_elev_ft: float,
                 mda_ft: float,
                 dme_thr_nm: float,
                 tod_dme_nm: float,
                 mapt_dme_nm: float = None,
                 gp_angle_user: float = 0.0,
                 faf_alt_ft: float = None,
                 sdf_list: list = None,
                 thr_coords: tuple = None,
                 dme_coords: tuple = None,
                 gp_min=DEFAULT_GP_MIN,
                 gp_max=DEFAULT_GP_MAX):
        self.thr_elev_ft = float(thr_elev_ft)
        self.mda_ft = float(mda_ft)
        self.dme_thr_nm = float(dme_thr_nm)
        self.tod_dme_nm = float(tod_dme_nm)
        self.mapt_dme_nm = float(mapt_dme_nm) if mapt_dme_nm is not None else None
        self.gp_angle_user = float(gp_angle_user) if gp_angle_user else 0.0
        self.faf_alt_ft = float(faf_alt_ft) if faf_alt_ft else None
        self.sdf_list = sdf_list or []
        self.thr_coords = thr_coords
        self.dme_coords = dme_coords
        self.gp_min = gp_min
        self.gp_max = gp_max

    def derive_gp_angle(self):
        """Derive GP angle (deg). Preference order:
        1) user-specified >0 and within allowed range (or override allowed)
        2) derive from FAF altitude -> threshold elevation over horizontal distance (TOD - DME@THR)
        3) fallback to DEFAULT_GP_FALLBACK
        Returns (gp_angle_used, was_clamped(boolean), raw_angle)
        """
        # if user forced a positive gp_angle_user, return it (caller must decide if allowed)
        if self.gp_angle_user and self.gp_angle_user > 0:
            raw = self.gp_angle_user
            clamped = False
            used = raw
            # clamp if outside allowed range
            if raw < self.gp_min:
                used = self.gp_min
                clamped = True
            elif raw > self.gp_max:
                used = self.gp_max
                clamped = True
            return used, clamped, raw

        # Build raw from FAF altitude (preferred) or from interpolated altitude at TOD
        raw_angle = None
        if self.faf_alt_ft is not None and self.tod_dme_nm is not None:
            vert_drop = self.faf_alt_ft - self.thr_elev_ft
            horiz_nm = max(0.001, self.tod_dme_nm - self.dme_thr_nm)  # horizontal from FAF to THR
            horiz_ft = horiz_nm * NM_TO_FT
            raw_angle = math.degrees(math.atan2(vert_drop, horiz_ft))
        # if raw_angle invalid, fallback
        if raw_angle is None or raw_angle <= 0 or raw_angle > 10:
            raw_angle = DEFAULT_GP_FALLBACK

        # clamp into allowed range (safe)
        clamped = False
        used = raw_angle
        if used < self.gp_min:
            used = self.gp_min
            clamped = True
        elif used > self.gp_max:
            used = self.gp_max
            clamped = True
        return round(used, 2), clamped, round(raw_angle, 2)

    def _slant_correction(self, dme_nm):
        """If coordinates provided, correct slant vs horizontal.
        For now returns horizontal-equivalent DME (placeholder for full slant calc).
        If both thr_coords and dme_coords present, we can compute horizontal distance between station and THR,
        then compute slant difference due to antenna height; here we leave simple (horizontal ~ DME)."""
        # TODO: implement full slant-range correction if required (using antenna heights)
        return max(0.0, dme_nm)

    def generate_dme_points(self):
        """Generate 8 DME points from TOD down to MAPt (or near threshold if MAPt missing).
        Points are DME (NM from DME station) — descending from outer to inner.
        We ensure first point equals TOD (or slightly adjusted) and last point near MAPt/DME@THR.
        Spacing: denser near inner end (MDA) to allow sub-1NM near runway.
        """
        start = float(self.tod_dme_nm)
        end = float(self.mapt_dme_nm) if self.mapt_dme_nm is not None else max(self.dme_thr_nm + 0.05, start - 0.05)

        if start <= end:
            # defensive: if start equals or inside end, ensure sensible start = end + 5 NM
            start = end + max(1.0, (start - end) * -1.0 + 5.0)

        # Create 8 points with denser spacing near end:
        lin = np.linspace(0.0, 1.0, 8)
        # use cubic-based easing to cluster near end: w = 1 - (1 - lin)^3  -> at lin=0 =>0, lin=1=>1
        weights = 1.0 - (1.0 - lin) ** 3
        # map weights 0..1 to dme from start..end
        dmes = start + (end - start) * weights  # note end < start, so works
        # ensure descending order (outer to inner)
        dmes = np.clip(dmes, end, start)
        # round to 1 decimal as requested
        dmes = np.round(dmes, 1)
        # ensure monotonic descending uniqueness (if duplicates due to small ranges, slightly adjust)
        for i in range(1, len(dmes)):
            if dmes[i] >= dmes[i - 1]:
                dmes[i] = max(end, dmes[i - 1] - 0.1)
        return dmes.tolist()

    def dme_table_and_alts(self, gp_angle_deg):
        """Return pandas DataFrame of 8 rows: DME(NM), Alt(ft), Distance to THR (NM)"""
        dmes = self.generate_dme_points()
        rows = []
        for d in dmes:
            d_corr = self._slant_correction(d)
            dist_to_thr = max(0.0, d_corr - self.dme_thr_nm)
            horiz_ft = dist_to_thr * NM_TO_FT
            alt_ft = self.thr_elev_ft + math.tan(math.radians(gp_angle_deg)) * horiz_ft
            rows.append({"DME (NM)": round(d_corr, 1),
                         "Distance to THR (NM)": round(dist_to_thr, 2),
                         "Altitude (ft)": round(alt_ft, 0)})
        # sort outer -> inner (descending DME)
        df = pd.DataFrame(rows).sort_values(by="DME (NM)", ascending=False).reset_index(drop=True)
        return df

    def generate_rod_table(self, dme_df, gs_list_kts=[80,100,120,140,160]):
        """Compute ROD (ft/min) per ground speed for total descent (FAF->MAPt).
        We calculate time per DME segment and ROD per segment, then average weighted by time."""
        dme = dme_df["DME (NM)"].values
        alt = dme_df["Altitude (ft)"].values
        seg_dist_nm = np.abs(np.diff(dme))  # between consecutive points
        seg_alt_ft = np.abs(np.diff(alt))
        results = []
        for gs in gs_list_kts:
            nm_per_min = gs * KTS_TO_NM_PER_MIN  # gs / 60
            # per segment time (min)
            times_min = np.where(seg_dist_nm > 0, seg_dist_nm / nm_per_min, 0.0)
            total_time_min = times_min.sum()
            total_alt_ft = seg_alt_ft.sum()
            avg_rod = (total_alt_ft / total_time_min) if total_time_min > 0 else 0.0
            # format MM:SS for total_time
            total_seconds = int(round(total_time_min * 60.0))
            mm = total_seconds // 60
            ss = total_seconds % 60
            results.append({"GS (kt)": int(gs), "ROD (ft/min)": int(round(avg_rod)), "Time (MM:SS)": f"{mm:02d}:{ss:02d}"})
        return pd.DataFrame(results)

    def plot_profile(self, dme_df, gp_angle_deg):
        # Plot Distance-to-THR on X (in NM), Altitude on Y (ft), MDA line, SDF markers
        x = dme_df["Distance to THR (NM)"].values  # NM to/from THR
        y = dme_df["Altitude (ft)"].values
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(x, y, marker='o', linestyle='-', linewidth=2, label='CDFA profile')
        ax.invert_xaxis()  # show outer (large) on left to inner on right
        ax.set_xlabel("DME (NM to/from THR)")
        ax.set_ylabel("Altitude (FT)")
        ax.set_title(f"CDFA Profile — GP {gp_angle_deg:.2f}°")





