import pdfplumber
import re

def extract_number(text, label, default=0):
    try:
        pattern = rf"{label}\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    except:
        pass
    return default

def parse_iac_pdf(file):
    parsed = {}

    try:
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"

        # Attempt to extract key values
        parsed["gp_angle"] = extract_number(text, r"GP|Glide Path|Glide Slope", default=3.0)
        parsed["tod_alt"] = extract_number(text, r"TOD|FAF Altitude", default=2500)
        parsed["mda"] = extract_number(text, r"MDA|DA", default=1000)
        parsed["dme_thr"] = extract_number(text, r"DME at THR|Threshold", default=6.9)
        parsed["dme_mapt"] = extract_number(text, r"DME at MAPt|MAPT", default=1.3)
        parsed["faf_mapt"] = abs(parsed["dme_thr"] - parsed["dme_mapt"])

        # Optional SDF logic (reads up to 3)
        sdf_matches = re.findall(r"SDF\s*(\d?)\s*[:=]?\s*([0-9.]+)\s*NM.*?([0-9]{3,5})\s*FT", text, re.IGNORECASE)
        parsed["sdf_count"] = len(sdf_matches)
        for idx, (num, dist, alt) in enumerate(sdf_matches[:6]):
            parsed[f"sdf_dist_{idx}"] = float(dist)
            parsed[f"sdf_alt_{idx}"] = float(alt)

        return parsed

    except Exception as e:
        print(f"[PDF Parser Error] {e}")
        return {}  # fallback to manual input
