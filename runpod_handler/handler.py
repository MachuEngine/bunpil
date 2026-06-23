"""v7 진단용 — GPU 확인 후 vllm import 시도."""
import sys
import os
import traceback
import subprocess

print("=== handler.py 시작 ===", flush=True)
print(f"Python: {sys.version}", flush=True)

# GPU 가시성 확인
try:
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=10)
    print(f"nvidia-smi stdout: {r.stdout.strip()}", flush=True)
    print(f"nvidia-smi stderr: {r.stderr.strip()}", flush=True)
except Exception as e:
    print(f"nvidia-smi 실행 실패: {e}", flush=True)

print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}", flush=True)

try:
    import runpod
    print("runpod import OK", flush=True)
except Exception as e:
    print(f"runpod import 실패: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# torch로 CUDA 상태 확인 (vllm보다 가볍게)
try:
    import torch
    print(f"torch: {torch.__version__}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB", flush=True)
except Exception as e:
    print(f"torch CUDA 확인 실패: {e}", flush=True)
    traceback.print_exc()

# vllm import 시도
try:
    print("vllm import 시도...", flush=True)
    from vllm import LLM, SamplingParams
    print("vllm import OK", flush=True)
except Exception as e:
    print(f"vllm import 실패: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

MODEL = "Qwen/Qwen2.5-7B-Instruct"
print(f"모델 로드 시작: {MODEL}", flush=True)
try:
    llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.90)
    print("모델 로드 완료", flush=True)
except Exception as e:
    print(f"모델 로드 실패: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)


def _build_prompt(messages: list) -> str:
    prompt = ""
    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def handler(job: dict) -> dict:
    inp         = job.get("input", {})
    messages    = inp.get("messages", [])
    max_tokens  = int(inp.get("max_tokens", 512))
    temperature = float(inp.get("temperature", 0.7))

    if not messages:
        return {"error": "messages 필드가 비어 있습니다."}

    prompt  = _build_prompt(messages)
    params  = SamplingParams(temperature=temperature, max_tokens=max_tokens, stop=["<|im_end|>"])
    outputs = llm.generate([prompt], params)
    text    = outputs[0].outputs[0].text.strip()
    return {"response": text}


runpod.serverless.start({"handler": handler})
