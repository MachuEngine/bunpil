# 분필(Bunpil) 개발기 — 교사용 AI 어시스턴트를 실제로 배포하기까지

> ⚠️ **초안 — 발행 전 전면 재검토 필요.** 아래 개요·기술 스택 절은 최신화했지만, "배운 것"
> 이하 트러블슈팅 서사는 각 사건 당시(작성 시점) 기준 그대로 남겨뒀다 — 사실관계 자체는
> 맞지만 "현재 상태" 설명은 아니다. 발행 전 README.md/MODEL_SELECTION.md와 다시 대조할 것.
> (2026-07-23 최종화: PDF 업로드→passage_text 붙여넣기 리디자인, Gradio→Next.js 전환,
> Qwen2.5-7B→14B 승격, 생성·Judge 모델 완전 분리를 모두 놓치고 있던 걸 뒤늦게 발견해 정정.)
>
> **2026-08-04 추가 정정**: 위 2026-07-23 최종화도 이미 그 뒤의 가장 큰 변경(생기부 윤문
> 모듈 2026-08-03 전체 제거, 사유는 CLAUDE.md·EVAL.md 14절 참고)을 놓치고 있었다. 아래
> "프로젝트 개요"·"보안 원칙" 절의 생기부 관련 서술은 **더 이상 현재 제품 설명이 아니다** —
> 발행 전 반드시 해당 절을 출제 단일 모듈 기준으로 다시 쓰거나 "과거엔 이런 기능도 있었다"는
> 회고 문단으로 바꿀 것.

> 고등학교 사회 교사 지인의 "이런 거 있으면 좋겠다"는 말 한마디에서 시작된 프로젝트.
> ReAct 에이전트, RunPod 서버리스, EBS 영구 저장까지 — 삽질 기록을 남긴다.

---

## 프로젝트 개요

**분필**은 고등학교 사회 교사를 위한 AI 어시스턴트다. 현재는 **출제 도우미** 단일 기능을 제공한다.

**출제 도우미**: 예시 문제 텍스트를 붙여넣으면(ChatGPT 쓰듯) 구성(개수·유형·난이도)이 비슷한 새 문항 세트를 자동 생성한다. (초기엔 PDF 업로드 + 유형/난이도/개수 드롭다운 방식이었으나, 실사용 교사 피드백을 받아 붙여넣기 방식으로 전면 재설계했다 — 이 리디자인 과정도 나중에 별도로 다룰 예정.)

> 초기에는 학생 관찰 메모를 학교생활기록부 문체로 다듬어주는 **생기부 윤문** 기능도 함께
> 제공했으나, 2026-08-03 전체 제거했다. 검증 규칙(종교·외모·추측 등 키워드)의 규정 근거를
> 추적한 결과 인덱싱된 교육부 기재요령에 해당 조항이 한 건도 없었고, 실제 학생 데이터를 쓸 수
> 없는 하드룰 때문에 검증 자체도 불가능해 범위에서 뺐다 — 이 조사·판단 과정도 별도로 다룰
> 가치가 있는 이야기라 남겨둔다.

포트폴리오 목적이지만 실제로 지인 교사가 쓴다. 그래서 "동작하는 것"에 집착했다.

---

## 기술 스택

| 구분 | 선택 |
|---|---|
| 에이전트 | LangGraph ReAct |
| 벡터스토어 | ChromaDB |
| 임베딩/리랭킹 | BGE-M3 + BGE-reranker (CPU) |
| 생성 LLM | Qwen2.5-14B (Ollama 로컬 / RunPod 서버리스 vLLM, AWQ 양자화) |
| Judge LLM | gpt-5.6-luna(OpenAI, 기본) — 생성 모델과 완전히 분리된 별도 백엔드 |
| UI | Next.js |
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

### 세 번째 버그: 4문제 요청에 10문제 생성 — 두 가지 원인이 겹쳤다

4문항(객관식 3 + 서술형 1)을 요청했는데 6문항이 생성됐다. 원인이 두 군데였다.

**원인 A: 시스템 프롬프트의 재시도 지시**

`agent_node`의 시스템 프롬프트에 이런 지시가 있었다:

```
"judge 점수가 3 미만이면 generate_item을 다시 호출하세요."
```

LLM이 문항 3을 생성하고 judge 점수가 0/5가 나오자, **같은 8-스텝 루프 안에서** generate_item을 한 번 더 호출해 문항 5를 만들었다. 이 재시도는 외부의 deficit 계산을 우회하기 때문에 수량 통제가 되지 않는다.

수정: 시스템 프롬프트에서 재시도 지시를 제거했다. 품질 재시도는 이미 바깥의 `budget` 루프가 담당한다.

```python
# 수정 전
"judge 점수가 3 미만이면 generate_item을 다시 호출하세요."

# 수정 후
"반드시 generate_item → judge_item → check_duplicate 순서로 도구를 정확히 한 번씩만 호출하세요."
```

**원인 B: 재시도 시 item_type만 추적, difficulty는 추적 안 함**

외부 retry 루프에서 deficit 계산이 `item_type`(객관식/서술형)만 보고 `difficulty`(상/중/하)를 보지 않았다. 객관식/하가 rejected 되면 재시도 시 또 객관식/상을 만들어 버리는 구조였다.

수정: `(item_type, difficulty)` 쌍 단위로 deficit을 추적하도록 변경했다.

```python
# 목표 (type, difficulty) 쌍 생성
target_pairs = _build_target_pairs(spec)

# 승인된 쌍을 차감
remaining = list(target_pairs)
for it in get_draft_items():
    if it.get("status") == "approved":
        pair = (it.get("item_type", ""), it.get("difficulty", ""))
        if pair in remaining:
            remaining.remove(pair)
```

### 네 번째 버그: EBS를 붙였는데 컨테이너 재시작마다 데이터가 사라진다

EBS 볼륨을 `/data/chroma_db`에 마운트하고 `docker run -v /data/chroma_db:/data/chroma_db`로 실행했다. 인덱싱도 완료됐다. 그런데 컨테이너를 업데이트할 때마다 3개 컬렉션이 모두 비어 있었다.

```
regulations 0
past_exams  0
standards   0
```

**원인: `.env` 파일이 Dockerfile의 `ENV`를 오버라이드**

Dockerfile에는 이렇게 돼 있었다:

```dockerfile
ENV CHROMA_PERSIST_DIR=/data/chroma_db
```

그런데 `.env` 파일에:

```
CHROMA_PERSIST_DIR=./chroma_db
```

`docker run --env-file .env`로 실행하면 `--env-file`이 Dockerfile `ENV`보다 우선순위가 높다. 그래서 실제로는 컨테이너 내부의 `/app/chroma_db`에 데이터가 쌓이고 있었고, EBS 볼륨은 텅 빈 채로 남아 있었다. 컨테이너가 교체되면 `/app/chroma_db`도 사라지니 매번 초기화된 것처럼 보였다.

**수정: `.env` 경로를 EBS 경로로 수정**

```
CHROMA_PERSIST_DIR=/data/chroma_db
```

이후 재인덱싱하면 EBS에 영구 저장되고, 컨테이너를 몇 번 교체해도 데이터가 유지된다.

**교훈**: `docker run --env-file`은 Dockerfile `ENV`를 조용히 덮어쓴다. 환경변수가 여러 곳에 정의될 수 있는 경우, 실제 컨테이너 안에서 `printenv`로 확인하는 습관이 필요하다.

```bash
docker exec bunpil printenv CHROMA_PERSIST_DIR
# ./chroma_db  ← 예상과 다름
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

RunPod 서버리스는 기본적으로 요청이 없으면 워커를 0으로 줄인다. 콜드스타트 시 모델 로딩에 30–60초가 걸린다(당시 배포 모델 Qwen2.5-7B 기준 실측 — 이후 14B로 승격되며 로딩 시간 재측정은 안 함). 교사가 처음 요청을 보냈을 때 1분을 기다리는 건 UX상 최악이다.

min workers를 1로 설정해서 항상 워커 하나를 켜두는 방식으로 해결했다. 비용은 조금 더 들지만, 응답 지연을 감수하는 것보다 낫다.

---

## 현재 성능

- 4문항(객관식 3 + 서술형 1) 생성: 약 **4–5분** (RunPod RTX A5000, min workers=1)
- LLM 호출: 문항당 약 7회 (search → generate → judge → duplicate → 라우팅 결정들)
- LLM 호출당: 약 8–20초 (generate_item이 가장 무거움, max_tokens=400)

**병렬화를 고려하는 이유**

4문항을 순차 생성하면 LLM 호출이 약 28회 발생한다. 4문항을 동시에 생성하면 7회로 줄어 시간이 ~1분대로 단축된다. 흥미로운 점은 **총 GPU 사용량(GPU-초)은 동일**하다는 것이다. RunPod 서버리스는 사용한 GPU 시간으로 과금하므로 병렬화가 비용을 올리지 않는다. 워커 수(max workers)만 늘리면 된다.

---

## 보안 원칙 — 이건 타협 없이

출제 입력(`passage_text`)도 교사가 붙여넣은 실제 지문이라 다음 원칙을 코드 레벨에서 강제한다:

1. **PII 마스킹은 모델 호출 이전에**: 이름, 전화번호, 학교명, 이메일을 regex로 마스킹한 뒤 LLM에 전달한다(`app/main.py`의 `_build_spec()`이 그래프 진입 전에 호출).
2. **실데이터 미사용**: 평가·실험은 전부 합성/공개 자료 기반 골든셋으로만 진행한다.
3. **비저장**: 사용자가 입력한 지문은 ChromaDB에 적재되지 않고 요청 처리 중 프롬프트에만 쓰인 뒤 폐기한다. (2026-07-24부터 PII 마스킹 후 프로덕션 LangSmith 트레이싱은 예외로 허용 — 관측성 목적, 사용자 승인.)

---

## 배운 것

**1. vLLM에서 tool calling은 직접 구현해야 한다.**
OpenAI 호환 API처럼 자동으로 되지 않는다. `apply_chat_template`으로 프롬프트를 구성하고, `<tool_call>` 출력을 파싱하는 코드를 직접 작성해야 한다.

**2. LangChain 어댑터의 반환 타입을 정확히 맞춰야 한다.**
`AIMessage.tool_calls`는 `list[dict]`이고 각 dict의 `args` 필드는 반드시 `dict`여야 한다. 문자열이 들어가면 Pydantic이 바로 에러를 낸다.

**3. EBS는 선택이 아니라 필수다.**
컨테이너 업데이트 때마다 재인덱싱을 하는 구조는 운영이 불가능하다. ChromaDB처럼 로컬 파일 기반 벡터스토어를 쓴다면 처음부터 영구 볼륨을 설계에 포함해야 한다.

**4. `--env-file`은 Dockerfile `ENV`보다 우선순위가 높다.**
EBS 볼륨을 마운트했는데 데이터가 계속 사라졌다. 원인은 `.env` 파일의 `CHROMA_PERSIST_DIR=./chroma_db`가 Dockerfile의 `/data/chroma_db`를 덮어쓴 것. `docker exec printenv`로 실제 컨테이너 환경변수를 확인하는 습관이 중요하다.

**5. 에이전트의 시스템 프롬프트가 수량 제어를 망친다.**
"judge 점수가 낮으면 재시도하라"는 프롬프트 한 줄이 외부의 deficit 계산을 우회해서 요청보다 많은 문항을 생성했다. LLM에게 재시도 판단을 맡기면 루프 단위 수량 제어가 깨진다. 품질 재시도는 외부 구조(budget 루프)로 처리하고, 내부 루프에서는 LLM이 정해진 순서대로만 도구를 호출하도록 제한하는 것이 안전하다.

**6. 에이전트를 에이전트답게 만드는 것이 어렵다.**
단순 파이프라인으로 구현하는 것은 쉽다. 하지만 LLM이 도구 호출 순서를 스스로 결정하게 하려면, LLM과 인프라 모두 그것을 지원하도록 맞춰야 한다. 중간에 어댑터 레이어를 "최적화"하겠다고 tool calling을 제거하면 에이전트가 아닌 파이프라인이 된다.

**7. 재시도(budget) 없이 1회차만 보면 tool-calling 성공률이 생각보다 낮다.**
STRUCTURE_GOLDEN용 골든셋을 만들려고 재시도를 끄고(budget=1) qwen2.5:7b(Ollama)에게 문항 세트를 시켜봤더니, 6개 중 5개가 문항 0개로 끝났다. 로그를 보니 `save_item`까지는 정상 호출하다가 다음 턴부터 실제 tool_call 대신 "이제 record_score를 호출하겠습니다"류의 일반 텍스트(중국어 섞임)로만 응답하고, 도구 호출이 없으면 루프를 끝내는 조건에 걸려 조기 종료된 것이었다. 코드 버그가 아니라 모델의 tool-calling 안정성 문제. 프로덕션은 budget=5 재시도로 이 실패를 대부분 가려주지만, 1회차 성공률만 보면 6개 중 1개(약 17%) 수준이었다 — 재시도 루프가 "왜 필요한지"를 수치로 확인한 셈. budget을 3으로 올려도 성공률은 35~40% 수준에서 크게 개선되지 않았다 — 재시도는 "가끔 실패"를 가려주는 안전망이지 "자주 실패"를 고쳐주진 못한다.

**8. eval 스크립트를 고쳤다고 해서 프로덕션 코드 변경이 검증되는 건 아니다.**
생성 프롬프트(agent_node)에 오답 매력도 지시를 추가한 뒤, 기존에 쓰던 `eval_exam.py`를 전/후로 재실행해서 효과를 확인하려 했다. 두 번 다 오답매력도 평균이 완전히 똑같이 나와서 "노이즈인가?" 하다가 원인을 보니, `eval_exam.py`의 문항 품질 평가는 스크립트에 미리 하드코딩해 둔 고정 30개 문항(`ITEM_GOLDEN`)을 채점하는 구조였다 — agent_node를 아예 호출하지 않는다. 즉 생성 프롬프트를 아무리 고쳐도 이 지표엔 원리적으로 반영될 수 없었다. Judge 프롬프트(`JUDGE_TPL`) 변경은 같은 방법으로 검증 가능했지만(채점 대상은 고정이어도 채점 기준이 바뀌니까), 생성 프롬프트 변경은 실제로 새 문항을 생성해서 채점하는 별도 스크립트가 필요했다. "eval 스크립트가 있다"는 것과 "내가 바꾼 부분이 그 eval이 실제로 exercise하는 경로에 있다"는 건 별개 확인이 필요하다.

**9. Ollama 기본 `num_ctx`(4096)가 모델의 네이티브 컨텍스트(32K)보다 훨씬 좁아서, 로컬 개발에서만 보이는 유령 버그를 만들었다.**
STRUCTURE_GOLDEN을 더 재생성하다가 성공률이 기존 35~40%대에서 갑자기 6%대로 폭락했다. 원인은 그날 새로 추가한 시스템 프롬프트 문구(오답매력도 지시 + num_items 지시) 두 줄이 아니라, `ollama ps`로 확인한 `CONTEXT 4096`이었다 — 멀티턴 ReAct 루프가 RAG 검색 결과(최대 300~400자×3개, 도구 2개면 최대 6개)를 계속 쌓다 보면 몇 턴 만에 4096을 넘기고, 컨텍스트가 잘리면서 모델이 시스템 프롬프트를 잃고 중국어·스페인어가 뒤섞인 응답을 냈다. 같은 passage로 `num_ctx=4096`(0/5문항)과 `num_ctx=16384`(5/5문항)를 직접 비교해 확인했다. 더 중요한 발견은 **`runpod_handler/handler.py`의 vLLM 초기화(`LLM(model=MODEL, ...)`)에는 `max_model_len`을 아예 지정하지 않아 모델 네이티브 값(32K)을 그대로 쓴다는 것** — 즉 이 버그는 로컬 Ollama 개발 환경에만 존재했고 프로덕션(RunPod)에는 원래 없었을 가능성이 크다. "재시도(budget)를 늘려도 성공률이 안 오른다"(7번 항목)는 관찰도 다시 보면 일부는 이 컨텍스트 문제가 섞여 있었던 것 — budget을 늘리는 것만으로는 근본 제약(컨텍스트 크기)을 못 넘는다는 걸 다시 확인한 셈이다. 로컬 dev 환경 설정이 프로덕션과 조용히 달라지는 것 자체가, 그 차이가 원인 규명을 몇 시간 지연시킬 수 있는 버그의 근원이었다.

---

## 코드 / 저장소

- GitHub: https://github.com/MachuEngine/bunpil
- Docker Hub: `jongmin0826/bunpil-app`, `jongmin0826/bunpil-runpod`
