# chart_parser.py
import fitz
import re
import math

def haversine_nm(lat1, lon1, lat2, lon2):
    """
    Haversine distance in NM between two lat/lon points (degrees).
    """
    R_km = 6371.0088
    def to_rad(x): return x * math.pi/180.0
    dlat = to_rad(lat2 - lat1)
    dlon = to_rad(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(to_rad(lat1)) * math.cos(to_rad(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    km = R_km * c
    nm = km * 0.539957
    return nm

def parse_iac_pdf(file_bytes):
    """
    Read PDF bytes (file-like) and attempt to extract:
      - gp_angle (°)
      - mda (ft)
      - tod_alt (ft) if present
      - dme_at_thr (nm)
      - dme_at_mapt (nm)
      - sdfs: list of {dme, alt}
      - faf_mapt_dist (nm)
      - thr_lat, thr_lon, dme_lat, dme_lon (if present)
    Returns dict (some keys may be missing).
    """
    data = {}
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for p in doc:
            try:
                text += p.get_text("text") + "\n"
            except:
                continue

        # Normalize whitespace
        t = re.sub(r"\s+", " ", text)

        # GP angle: common patterns "GP 3.00°", "Glide Path 3.00°", "GS 3.00°"
        m = re.search(r"(Glide Path|GP|GS)[\s:]*([0-9]\.[0-9])\s*°", t, re.IGNORECASE)
        if m:
            data["gp_angle"] = float(m.group(2))

        # MDA / MDH (numbers followed by 'MDA' or 'MDH')
        m2 = re.search(r"MDA[\s:=]*([0-9]{3,4})", t, re.IGNORECASE)
        if m2:
            data["mda"] = int(m2.group(1))
        else:
            # sometimes MDH or "MDA(H) 1000"
            m3 = re.search(r"MDA\(H\)[\s:=]*([0-9]{3,4})", t, re.IGNORECASE)
            if m3:
                data["mda"] = int(m3.group(1))

        # TOD alt sometimes printed like "TOD 2500'", or "Cross TOD at 2500 ft"
        m4 = re.search(r"\bTOD\b.*?([0-9]{3,4})\s*(?:ft|')", t, re.IGNORECASE)
        if m4:
            data["tod_alt"] = int(m4.group(1))
        else:
            # find first altitude-like pattern near FAF text
            m5 = re.search(r"FAF.*?([0-9]{3,4})\s*(?:ft|')", t, re.IGNORECASE)
            if m5:
                data["tod_alt"] = int(m5.group(1))

        # DME at THR / MAPt pattern: many charts include "NM to/from THR" scale or "MAPt D x.x"
        m6 = re.search(r"MAPT.*?([0-9]{1,2}\.[0-9])\s*NM", t, re.IGNORECASE)
        if m6:
            data["dme_at_mapt"] = float(m6.group(1))
        else:
            m6b = re.search(r"MAPt.*?DME\s*([\d\.]+)", t, re.IGNORECASE)
            if m6b:
                data["dme_at_mapt"] = float(m6b.group(1))

        m7 = re.search(r"to/from THR.*?([0-9]{1,2}\.[0-9])\s*NM", t, re.IGNORECASE)
        if m7:
            # ambiguous; store but not guaranteed
            data.setdefault("notes", []).append(("found scale to/from THR", m7.group(1)))

        # FAF to MAPt explicit text e.g. "FAF to MAP 5.7 NM" or "FAF to MAPt 5.7NM"
        m8 = re.search(r"FAF.*?MAPP?T?.*?([0-9]{1,2}\.[0-9])\s*NM", t, re.IGNORECASE)
        if m8:
            data["faf_mapt_dist"] = float(m8.group(1))

        # Step-down fixes: look for "D6 xxx (alt)" or "DME 6 - alt x"
        # generic regex to pick up pairs like "DME 6 (1000)" or "6.0 DME 1000"
        sdf_candidates = re.findall(r"([0-9]{1,2}\.[0-9])\s*NM.*?([0-9]{3,4})\s*(?:ft|')", t)
        if sdf_candidates:
            # Using first few as possible SDFs (best-effort)
            sdfs = []
            for dme_s, alt_s in sdf_candidates[:6]:
                sdfs.append({"dme": float(dme_s), "alt": int(alt_s)})
            data["sdfs"] = sdfs

        # Attempt to find lat/lon for THR and DME (simple numeric patterns)
        # Pattern e.g. "41°40'N -70°17'W" or decimal 41.123 -70.123
        latlon_matches = re.findall(r"([0-9]{1,2}\.[0-9]+)[°]?\s*[NnSs]?[,\s]+([\-0-9]{1,3}\.[0-9]+)[°]?\s*[EeWw]?", t)
        if latlon_matches:
            # Not always accurate; skip assigning unless confident
            # Provide first match in case user wants to override
            try:
                latf = float(latlon_matches[0][0])
                lonf = float(latlon_matches[0][1])
                data.setdefault("notes", []).append(("latlon_sample", (latf, lonf)))
            except:
                pass

    except Exception as exc:
        # Parser is best-effort; do not crash
        data["parse_error"] = str(exc)

    return data



