FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y tesseract-ocr poppler-utils libgl1 ghostscript && \
    pip install --no-cache-dir --upgrade pip

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]





