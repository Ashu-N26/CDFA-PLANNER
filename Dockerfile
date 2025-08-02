FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx poppler-utils tesseract-ocr && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]




