"""RunPod 서버리스 핸들러 — Qwen2.5-7B-Instruct (vLLM).

RunPod 워커 컨테이너 안에서 실행된다.
입력 형식: {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}], "max_tokens": 512}
출력 형식: {"response": "..."}
"""
import runpod
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 컨테이너 시작 시 모델 로드 (콜드 스타트)
llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.90)


def _build_prompt(messages: list) -> str:
    """ChatML 포맷으로 변환 (Qwen2.5 Instruct 규격)."""
    prompt = ""
    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def handler(job: dict) -> dict:
    inp        = job.get("input", {})
    messages   = inp.get("messages", [])
    max_tokens = int(inp.get("max_tokens", 512))
    temperature = float(inp.get("temperature", 0.7))

    if not messages:
        return {"error": "messages 필드가 비어 있습니다."}

    prompt = _build_prompt(messages)
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens, stop=["<|im_end|>"])
    outputs = llm.generate([prompt], params)
    text = outputs[0].outputs[0].text.strip()

    return {"response": text}


runpod.serverless.start({"handler": handler})
