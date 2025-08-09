FROM python:3.11-slim

# Install system deps for pdf2image and pytesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/

ENV PORT=8501
EXPOSE ${PORT}

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
