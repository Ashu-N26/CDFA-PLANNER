# Base Python image
FROM python:3.11-slim

# Install OS dependencies for Tesseract OCR, PDF processing, and plotting
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --force-reinstall -r requirements.txt

# Copy application code
COPY app.py .

# Streamlit configuration
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose port
EXPOSE 7860

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]

