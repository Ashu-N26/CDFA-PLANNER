import PyPDF2
import re

def parse_iac_pdf(pdf_path):
    """
    Parse ICAO Approach Chart (IAC) PDF to extract relevant data for the CDFA planner.
    
    :param pdf_path: Path to the IAC PDF file
    :return: Extracted values as a dictionary
    """
    data = {
        "gp_angle": None,
        "dme_thr": None,
        "dme_mapt": None,
        "tod_altitude": None,
        "mda": None,
        "sdf": [],
        "faf_mapt_dist": None
    }

    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in range(len(reader.pages)):
                text += reader.pages[page].extract_text()

            # Extract GP angle
            gp_match = re.search(r"Glide Path: ([\d.]+)°", text)
            if gp_match:
                data["gp_angle"] = float(gp_match.group(1))

            # Extract DME at THR (Threshold)
            dme_thr_match = re.search(r"DME at THR: (\d+\.\d+)", text)
            if dme_thr_match:
                data["dme_thr"] = float(dme_thr_match.group(1))

            # Extract DME at MAPT (Missed Approach Point)
            dme_mapt_match = re.search(r"DME at MAPT: (\d+\.\d+)", text)
            if dme_mapt_match:
                data["dme_mapt"] = float(dme_mapt_match.group(1))

            # Extract TOD (Top of Descent) Altitude
            tod_altitude_match = re.search(r"TOD Altitude: (\d+)", text)
            if tod_altitude_match:
                data["tod_altitude"] = int(tod_altitude_match.group(1))

            # Extract MDA (Minimum Descent Altitude)
            mda_match = re.search(r"MDA: (\d+)", text)
            if mda_match:
                data["mda"] = int(mda_match.group(1))

            # Extract Step-Down Fixes (SDF)
            sdf_match = re.findall(r"SDF (\d+): DME (\d+\.\d+) NM, Altitude (\d+)", text)
            for sdf in sdf_match:
                data["sdf"].append({
                    "dme": float(sdf[1]),
                    "altitude": int(sdf[2])
                })

            # Extract FAF-MAPT Distance
            faf_mapt_match = re.search(r"FAF to MAPT: (\d+\.\d+) NM", text)
            if faf_mapt_match:
                data["faf_mapt_dist"] = float(faf_mapt_match.group(1))

    except Exception as e:
        print(f"Error reading PDF: {e}")

    return data

def extract_values_from_pdf(pdf_path):
    """
    This function combines the PDF parser with the app's logic to extract the relevant
    values and return them in a format ready for use in CDFA calculations.
    
    :param pdf_path: Path to the IAC PDF file
    :return: Parsed values (dictionary)
    """
    parsed_data = parse_iac_pdf(pdf_path)

    return parsed_data


