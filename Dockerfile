# Base Python image
FROM python:3.11-slim

# Ensure system is up-to-date and install OS deps for OCR + PDF parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first (to force re-install if changed)
COPY requirements.txt .

# Force pip to reinstall all packages fresh (no cache, no reuse)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --force-reinstall -r requirements.txt

# Copy application code
COPY app.py .

# Streamlit settings
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose the port Render will use
EXPOSE 7860

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
