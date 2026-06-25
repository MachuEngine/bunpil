# 분필(Bunpil) 개발기 — 교사용 AI 어시스턴트를 실제로 배포하기까지

> 고등학교 사회 교사 지인의 "이런 거 있으면 좋겠다"는 말 한마디에서 시작된 프로젝트.
> ReAct 에이전트, RunPod 서버리스, EBS 영구 저장까지 — 삽질 기록을 남긴다.

---

## 프로젝트 개요

**분필**은 고등학교 사회 교사를 위한 AI 어시스턴트다. 두 가지 기능을 제공한다.

1. **출제 도우미**: 지문 PDF를 업로드하면 유형(객관식/서술형)·난이도·성취기준 조건에 맞는 문항 세트를 자동 생성한다.
2. **생기부 윤문**: 교사가 메모한 학생 관찰 내용을 학교생활기록부 문체에 맞게 다듬어 준다. PII 마스킹 후 모델에 전달하고, 없는 사실은 절대 추가하지 않는다.

포트폴리오 목적이지만 실제로 지인 교사가 쓴다. 그래서 "동작하는 것"에 집착했다.

---

## 기술 스택

| 구분 | 선택 |
|---|---|
| 에이전트 | LangGraph ReAct |
| 생기부 체인 | LangChain LCEL |
| 벡터스토어 | ChromaDB |
| 임베딩/리랭킹 | BGE-M3 + BGE-reranker (CPU) |
| LLM | Qwen2.5-7B-Instruct (RunPod 서버리스 vLLM) |
| UI | Gradio |
| 인프라 | AWS EC2 t3.medium + EBS + RunPod 서버리스 |

BGE 임베딩을 CPU로 돌리는 이유: EC2에 GPU를 붙이면 비용이 폭발한다. 임베딩은 추론보다 훨씬 가볍고, 실측 결과 EC2 t3.medium에서 573청크 임베딩에 약 25분 걸렸다. 한 번만 하면 EBS에 영구 저장되니 감내할 만하다.

---

## 아키텍처 결정: 왜 진짜 에이전트인가

출제 모듈을 단순 LLM 호출로 구현하는 것이 훨씬 쉬웠다. 그런데 그렇게 하지 않은 이유가 있다.

**교사의 요구는 본질적으로 다단계다.**

1. 지문에서 관련 내용을 검색한다.
2. 검색 결과를 보고 문항을 생성한다.
3. 생성된 문항의 품질을 평가한다.
4. 기출과 중복되지 않는지 확인한다.
5. 승인되지 않은 문항이 있으면 재시도한다.

이걸 하드코딩하면 "파이프라인"이지 "에이전트"가 아니다. LangGraph ReAct는 LLM이 스스로 도구 호출 순서를 결정하게 한다. `search_passages`를 먼저 쓸지, `generate_item`을 먼저 쓸지, 품질이 낮으면 `search_passages`를 다시 쓸지 — 이걸 LLM이 판단한다.

이것이 이 프로젝트에서 가장 기술적으로 어려운 부분이기도 했다.

---

## 가장 오래 고생한 버그: Tool Calling이 동작하지 않는다

### 증상

출제 에이전트를 처음 배포했을 때 결과가 이랬다:

```
검증 통과: ✗ | 생성: 0문항 | 승인: 0문항
⚠️ 문항이 생성되지 않았습니다. (LLM_BACKEND=runpod)
```

0문항. 에이전트가 도구를 한 번도 호출하지 않았다.

### 원인 추적

LangGraph ReAct 루프의 동작을 단계별로 따라갔다.

```
agent_node → llm.invoke(messages, tools=TOOLS)
           → AIMessage 반환
           → tool_calls가 있으면 → tool_node 실행
           → tool_calls가 없으면 → END (루프 종료)
```

문제는 `ChatRunPod._agenerate()`가 항상 `AIMessage(content=text)`를 반환하고 있었다는 것. `tool_calls` 필드가 없으니 ReAct 루프가 첫 번째 스텝에서 바로 종료됐다.

### RunPod + vLLM에서 Tool Calling을 구현하는 방법

OpenAI API처럼 `tools` 파라미터를 직접 지원하는 게 아니다. vLLM은 모델의 chat template을 통해 도구 정보를 프롬프트에 주입하고, 모델이 `<tool_call>` 태그로 출력하면 이를 파싱해야 한다.

**RunPod 핸들러 (`runpod_handler/handler.py`) 변경:**

```python
# 모델 로드 시 토크나이저도 함께 보관
tokenizer = llm.get_tokenizer()

# tools가 있으면 chat_template으로 프롬프트 구성
if tools:
    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True
    )
else:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

# 출력에서 <tool_call> 파싱
def _parse_tool_calls(text: str) -> list:
    results = []
    for m in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        try:
            results.append(json.loads(m.group(1)))
        except Exception:
            pass
    return results
```

**LangChain 어댑터 (`app/common/llm/backends/chat_runpod.py`) 변경:**

```python
def _build_ai_message(result: dict) -> AIMessage:
    raw_tool_calls = result.get("tool_calls") or []
    if raw_tool_calls:
        tool_calls = []
        for tc in raw_tool_calls:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except Exception:
                args = {}
            tool_calls.append({
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "args": args,
                "type": "tool_call"
            })
        return AIMessage(content=result.get("response") or "", tool_calls=tool_calls)
    return AIMessage(content=result.get("response") or "")
```

이 두 곳을 수정하고 나서야 에이전트가 제대로 도구를 호출하기 시작했다.

### 두 번째 버그: arguments 이중 인코딩

도구 호출이 동작하자 이번엔 Pydantic ValidationError가 터졌다:

```
pydantic_core.ValidationError: 1 validation error for AIMessage
tool_calls.0.args
  Input should be a valid dictionary [type=dict_type]
```

원인: Qwen 모델이 `arguments`를 JSON 문자열로 출력하면, 핸들러가 그것을 다시 `json.dumps()`로 감싸서 이중 인코딩이 발생했다. 어댑터에서 `json.loads()`를 하면 dict가 아니라 문자열이 나오는 것.

```python
# 핸들러에서 수정
args = tc.get("arguments", {})
arguments_str = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
```

### 세 번째 버그: 4문제 요청에 10문제 생성

`budget=2`로 설정하니 에이전트가 2번 루프를 돌면서 매번 요청한 문항 수를 전부 생성했다. 결과: 4문항 요청 → 10문항 생성.

재시도 시에는 이미 승인된 문항을 제외하고 부족한 수만큼만 생성하도록 수정했다:

```python
approved_counts = {}
for it in get_draft_items():
    if it.get("status") == "approved":
        t = it.get("item_type", "")
        approved_counts[t] = approved_counts.get(t, 0) + 1

items_to_generate = []
for itype, cnt in spec["type_dist"].items():
    deficit = cnt - approved_counts.get(itype, 0)
    for _ in range(deficit):
        items_to_generate.append(itype)
```

---

## 인프라: EBS 볼륨으로 ChromaDB 영구 저장

초기에는 컨테이너를 업데이트할 때마다 ChromaDB 데이터가 사라져서 재인덱싱을 해야 했다. 573청크 임베딩에 25분 걸리니, 배포할 때마다 25분을 기다리는 건 말이 안 된다.

해결책: AWS EBS 볼륨을 EC2에 붙이고, 컨테이너 볼륨으로 마운트한다.

```bash
# EBS 포맷 & 마운트 (최초 1회)
sudo mkfs.ext4 /dev/nvme1n1
sudo mkdir -p /data/chroma_db
echo '/dev/nvme1n1 /data/chroma_db ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab
sudo mount -a

# 컨테이너 실행
docker run -d --name bunpil \
  -v /data/chroma_db:/data/chroma_db \
  -v hf_cache:/root/.cache/huggingface \
  ...
```

`hf_cache` 볼륨은 BGE 모델 캐시용이다. 이것도 없으면 컨테이너 재시작마다 BGE-M3 모델을 다시 다운받는다.

---

## 인프라: RunPod 콜드스타트 문제

RunPod 서버리스는 기본적으로 요청이 없으면 워커를 0으로 줄인다. 콜드스타트 시 Qwen2.5-7B 모델 로딩에 30–60초가 걸린다. 교사가 처음 요청을 보냈을 때 1분을 기다리는 건 UX상 최악이다.

min workers를 1로 설정해서 항상 워커 하나를 켜두는 방식으로 해결했다. 비용은 조금 더 들지만, 응답 지연을 감수하는 것보다 낫다.

---

## 현재 성능

- 4문항(객관식 3 + 서술형 1) 생성: 약 120–180초 (RunPod RTX A5000)
- LLM 추론 자체: 문항 하나당 약 20–30초
- 나머지: 도구 호출 오버헤드, 폴링, 네트워크 레이턴시

추론 속도를 줄이려면:
1. `generate_item`의 `max_tokens`를 더 줄인다 (현재 400)
2. `judge_item`은 숫자 하나만 받으므로 `max_tokens=10` (이미 적용)
3. 병렬 생성을 시도할 수 있지만, ReAct 에이전트 특성상 도구 호출이 순차적으로 일어난다

---

## 보안 원칙 — 이건 타협 없이

생기부 기능은 학생 관련 정보를 다루기 때문에 다음 원칙을 코드 레벨에서 강제한다:

1. **PII 마스킹은 모델 호출 이전에**: 이름, 전화번호, 학교명, 이메일을 regex로 마스킹한 뒤 LLM에 전달한다.
2. **없는 사실 추가 금지**: 프롬프트에서 "메모에 있는 내용만 다듬어라"고 명시하고, 평가 스크립트로 사실추가율을 측정한다 (기준: 0).
3. **비저장**: 사용자가 입력한 메모와 지문은 세션 종료 시 폐기한다.
4. **교사 책임 고지**: 출력 하단에 항상 "최종 검토 및 책임은 교사에게 있습니다" 문구를 붙인다.

---

## 배운 것

**1. vLLM에서 tool calling은 직접 구현해야 한다.**
OpenAI 호환 API처럼 자동으로 되지 않는다. `apply_chat_template`으로 프롬프트를 구성하고, `<tool_call>` 출력을 파싱하는 코드를 직접 작성해야 한다.

**2. LangChain 어댑터의 반환 타입을 정확히 맞춰야 한다.**
`AIMessage.tool_calls`는 `list[dict]`이고 각 dict의 `args` 필드는 반드시 `dict`여야 한다. 문자열이 들어가면 Pydantic이 바로 에러를 낸다.

**3. EBS는 선택이 아니라 필수다.**
컨테이너 업데이트 때마다 재인덱싱을 하는 구조는 운영이 불가능하다. ChromaDB처럼 로컬 파일 기반 벡터스토어를 쓴다면 처음부터 영구 볼륨을 설계에 포함해야 한다.

**4. 에이전트를 에이전트답게 만드는 것이 어렵다.**
단순 파이프라인으로 구현하는 것은 쉽다. 하지만 LLM이 도구 호출 순서를 스스로 결정하게 하려면, LLM과 인프라 모두 그것을 지원하도록 맞춰야 한다. 중간에 어댑터 레이어를 "최적화"하겠다고 tool calling을 제거하면 에이전트가 아닌 파이프라인이 된다.

---

## 코드 / 저장소

- GitHub: https://github.com/MachuEngine/bunpil
- Docker Hub: `jongmin0826/bunpil-app`, `jongmin0826/bunpil-runpod`
