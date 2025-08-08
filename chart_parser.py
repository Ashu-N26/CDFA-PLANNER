import fitz  # PyMuPDF
import re

def parse_iac_pdf(file):
    data = {}
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()

        # GP angle (e.g. "GP 3.00°" or "Glide Path: 3.00°")
        gp_match = re.search(r"(GP|Glide Path)[^\d]*(\d+\.\d+)", text)
        if gp_match:
            data["gp_angle"] = float(gp_match.group(2))

        # TOD altitude (e.g. "FAF HESGI 6.9 DME 2500'")
        tod_match = re.search(r"FAF.*?(\d+\.\d+)\s*DME.*?(\d{3,4})'", text)
        if tod_match:
            data["dme_thr"] = float(tod_match.group(1))
            data["tod_alt"] = int(tod_match.group(2))

        # MDA (e.g. "MDA(H) 1000'")
        mda_match = re.search(r"MDA.*?(\d{3,4})'", text)
        if mda_match:
            data["mda"] = int(mda_match.group(1))

        # MAPt DME (e.g. "MAPt 1.1 DME")
        mapt_match = re.search(r"MAPt.*?(\d+\.\d+)\s*DME", text)
        if mapt_match:
            data["dme_mapt"] = float(mapt_match.group(1))

        # FAF–MAPt distance (optional if available)
        faf_mapt_match = re.search(r"FAF.*?MAPt.*?(\d+\.\d+)\s*NM", text)
        if faf_mapt_match:
            data["faf_mapt_dist"] = float(faf_mapt_match.group(1))
        else:
            data["faf_mapt_dist"] = round(data.get("dme_thr", 6.9) - data.get("dme_mapt", 1.1), 1)

        # Placeholder values for lat/lon (manual input expected)
        data["thr_lat"] = ""
        data["thr_lon"] = ""
        data["dme_lat"] = ""
        data["dme_lon"] = ""
        data["thr_elev"] = 0
        data["sdfs"] = []

    except Exception as e:
        print("PDF parsing failed:", e)
        data = {}

    return data
