# RunPod 서버리스 워커 이미지
# 베이스: CUDA 12.2 + Python 3.11 (RunPod 공식)
FROM vllm/vllm-openai:v0.6.6

WORKDIR /app

RUN pip install --no-cache-dir runpod==1.7.9

COPY handler.py .

ENV HF_HOME=/hf_cache

ENTRYPOINT []
CMD ["sh", "-c", "echo 'shell OK' && which python3 && python3 -u handler.py"]
