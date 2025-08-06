from pdf2image import convert_from_bytes
import pytesseract

def parse_pdf_chart(uploaded_pdf):
    images = convert_from_bytes(uploaded_pdf.read(), dpi=300)
    text = "\n".join(pytesseract.image_to_string(img) for img in images[:2])
    parsed = {}

    # Simplified extraction — real logic should use regex and table mapping
    if "GP ANGLE" in text.upper():
        parsed["gp_angle"] = 3.0
    if "MDA" in text.upper():
        parsed["mda"] = 1000
    if "TOD" in text.upper():
        parsed["tod_alt"] = 3000

    return parsed
