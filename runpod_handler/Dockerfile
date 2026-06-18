# RunPod 서버리스 워커 이미지
# 베이스: CUDA 12.2 + Python 3.11 (RunPod 공식)
FROM runpod/base:0.6.2-cuda12.2.0

WORKDIR /app

RUN pip install --no-cache-dir \
    runpod==1.7.9 \
    vllm==0.6.6 \
    torch==2.5.1 \
    --extra-index-url https://download.pytorch.org/whl/cu122

COPY handler.py .

# 모델은 HuggingFace Hub에서 콜드스타트 시 자동 다운로드
# 빠른 웜업을 원하면 VOLUME에 캐시 마운트 후 --volume 지정
ENV HF_HOME=/hf_cache

CMD ["python", "-u", "handler.py"]
