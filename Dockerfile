FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy app
WORKDIR /app
COPY . .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Run app
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.enableCORS=false"]






