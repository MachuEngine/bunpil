FROM python:3.11-slim

WORKDIR /app

# fitz(PyMuPDF) 의존성 + healthcheck용 curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch 먼저 설치 (EC2 t3.small — GPU 없음)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/ ./data/

# ChromaDB 영구 저장 위치 (EBS 볼륨 마운트 포인트)
VOLUME ["/data/chroma_db"]
ENV CHROMA_PERSIST_DIR=/data/chroma_db

RUN mkdir -p /tmp/gradio

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "app/ui.py"]
