<div align="center">

# 분필 (bunpil)

**고등학교 사회 교사를 위한 AI 어시스턴트 — 문항 출제 · 생활기록부 윤문**

![Skills](https://skillicons.dev/icons?i=python,fastapi,typescript,nextjs,tailwind,docker,react,aws)

![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-30363D?logo=langgraph&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1A5FB4)
![Ollama](https://img.shields.io/badge/Ollama-2B2B2B?logo=ollama&logoColor=white)
![RunPod](https://img.shields.io/badge/RunPod-5D29F0)
![LangSmith](https://img.shields.io/badge/LangSmith-1B2733)
![vLLM](https://img.shields.io/badge/vLLM-1B76C4?logo=vllm&logoColor=white)
![Caddy](https://img.shields.io/badge/Caddy-175F8C?logo=caddy&logoColor=white)

[개요](#개요) · [아키텍처](#아키텍처) · [설계 원칙](#설계-원칙) · [엔지니어링 하이라이트](#엔지니어링-하이라이트) · [품질 평가](#품질-평가) · [모델 비교](#모델-비교-실험) · [빠른 시작](#빠른-시작-로컬) · [배포](#배포-프로덕션)

</div>

---

## 개요

교사의 반복 업무 중 가장 시간이 많이 드는 두 가지 — **시험 문항 출제**와 **학교생활기록부 문구 작성** — 를 소형 오픈소스 LLM(Qwen2.5-7B)으로 보조하는 서비스입니다. 포트폴리오 프로젝트이자 현직 교사 1인이 실사용 중입니다.

| 모듈 | 입력 | 처리 | 출력 |
|---|---|---|---|
| 📝 **문항 출제** | 예시 문제 텍스트 붙여넣기 | LangGraph ReAct 에이전트가 교육과정·규정 RAG를 참조하며 생성 → 형식·언어·개수 검증 → 미달 시 부족분만 이어서 재시도 | 지정 개수의 새 문항 세트 (예시와 유사한 유형·난이도 구성) |
| ✍️ **생기부 윤문** | 교사 관찰 메모 | PII 마스킹(모델 호출 **전**) → 생기부 문체 교정 → 규정 위반 검증 | 교정된 문장 + 위반 플래그 + 교사 책임 고지 |

프로젝트의 특징 세 가지:

- **로컬 ↔ 프로덕션 전환 가능한 LLM 추상화** — 개발은 Ollama(로컬), 프로덕션은 RunPod 서버리스(vLLM). 환경변수 하나로 전환
- **"LLM이 판단하고, 코드가 결정한다"** — 품질·유사도 판단은 LLM에게, 통과/재시도/개수/언어 검증은 결정론적 코드에 ([설계 원칙](#설계-원칙))
- **평가 기반 개발** — 사람이 라벨링한 골든셋 6종으로 검색·생성·마스킹 품질을 수치로 추적 ([EVAL.md](./EVAL.md)), 삽질은 [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)에 기록

---

## 아키텍처

```mermaid
flowchart LR
    Browser["🌐 브라우저<br/>Next.js UI"]

    subgraph API["⚙️ FastAPI · app/main.py"]
        E1["POST /exam/stream<br/>SSE"]
        E2["POST /exam<br/>JSON"]
        E3["POST /record<br/>JSON"]
    end

    ExamMod["📝 출제 모듈<br/>LangGraph ReAct Agent"]
    RecordMod["✍️ 생기부 모듈<br/>Chain · 3단계"]
    RAG[("📚 ChromaDB<br/>BGE-M3 임베딩 · BGE 리랭커<br/>규정 510 + 성취기준 573 청크")]
    LLM["🧠 LLM 백엔드<br/>Ollama(개발) / RunPod vLLM(프로덕션)"]

    Browser --> API
    E1 --> ExamMod
    E2 --> ExamMod
    E3 --> RecordMod
    ExamMod --> RAG
    RecordMod --> RAG
    ExamMod --> LLM
    RecordMod --> LLM

    class Browser,E1,E2,E3,LLM,RAG neutral
    class ExamMod exam
    class RecordMod record
    style API fill:#FAFAF8,stroke:#C3C2B7,stroke-width:1px

    classDef neutral fill:#F5F4F1,stroke:#8A8880,stroke-width:1px,color:#1A1A1A
    classDef exam fill:#FEF3C7,stroke:#D97706,stroke-width:1.5px,color:#1A1A1A
    classDef record fill:#ECFEFF,stroke:#0891B2,stroke-width:1.5px,color:#1A1A1A
```

| 구분 | 기술 |
|---|---|
| 백엔드 | FastAPI (비동기) |
| 프론트엔드 | Next.js (`frontend/`) |
| 에이전트 | LangGraph (ReAct) |
| 생기부 체인 | LangChain (수동 루프) |
| RAG | ChromaDB + BGE-M3 임베딩 + BGE-reranker (모두 CPU) |
| LLM 서빙 | Ollama (개발) / RunPod 서버리스 vLLM (프로덕션) |
| 트레이싱 | LangSmith (합성 데이터 평가 스크립트에서만 선택) |
| 배포 | AWS EC2 t3.medium + EBS + RunPod 서버리스 + Caddy HTTPS |

### 출제 모듈 — ReAct 에이전트

에이전트(LLM)가 추론과 문항 생성을 **직접** 담당하고, 도구 7개는 검색·저장·검증의 **순수 계산**만 수행합니다(도구 내부 LLM 호출 없음 — LLM을 도구 안에 중첩하는 안티패턴 제거).

| 도구 | 역할 |
|---|---|
| `search_standards` / `search_regulations` | 성취기준·법령 RAG 검색 |
| `validate_item_format` | 선지 4개·①②③④ 형식 등 결정론적 형식 검증 (오류 시 수정 지침 반환 → 자기교정) |
| `save_item` | 문항 저장 — 한국어 검증 게이트 통과 시에만 저장 |
| `discard_item` | 승인 불가 문항을 ID로 폐기 |
| `record_score` | 문항 품질 자체 평가 기록 |
| `similarity_judge` | 세트 완성 후 예시 문제와의 구조 유사도 자체 평가 |

```mermaid
flowchart LR
    Start(["문항 시작"])

    subgraph Loop["🔁 문항마다 반복"]
        direction LR
        Search["🔍 search_standards<br/>search_regulations"]
        Validate{"validate_item_format"}
        Save["save_item<br/>+ 한국어 게이트"]
        Score["record_score"]

        Search --> Validate
        Validate -- "형식 오류" --> Validate
        Validate -- "통과" --> Save --> Score
        Score -.-> Search
    end

    Judge["🎯 similarity_judge<br/>세트 전체 구조 유사도"]
    End(["루프 종료"])

    Start --> Loop --> Judge --> End

    class Start,End,Validate neutral
    class Search,Save,Score,Judge exam
    style Loop fill:#FAFAF8,stroke:#C3C2B7,stroke-width:1px

    classDef neutral fill:#F5F4F1,stroke:#8A8880,stroke-width:1px,color:#1A1A1A
    classDef exam fill:#FEF3C7,stroke:#D97706,stroke-width:1.5px,color:#1A1A1A
```

세트 전체는 LangGraph 그래프가 관리합니다. `validate` 노드가 코드로 판정(문항 개수 일치 + `similarity_judge` 결과 threshold)하고, 미달 시 최대 5회까지 `agent`로 재시도합니다 — 이때 **이미 만든 문항은 유지하고 부족분만 이어서 작성**합니다(부분 진행 보존).

```mermaid
flowchart LR
    START(["START"])
    plan["plan<br/>세션 초기화 · 1회"]
    agent["agent<br/>문항 생성<br/>(재시도 시 부족분만)"]
    validate{"validate<br/>개수==num_items<br/>+ Judge threshold"}
    END(["END"])

    START --> plan --> agent --> validate
    validate -- "미달 & budget > 0" --> agent
    validate -- "통과 또는 소진" --> END

    class START,END,validate neutral
    class plan,agent exam

    classDef neutral fill:#F5F4F1,stroke:#8A8880,stroke-width:1px,color:#1A1A1A
    classDef exam fill:#FEF3C7,stroke:#D97706,stroke-width:1.5px,color:#1A1A1A
```

### 생기부 모듈 — 3단계 체인

순서가 고정된 파이프라인입니다. **PII 마스킹이 반드시 모델 호출보다 앞**에 있어, 원문 개인정보가 LLM에 도달하지 않습니다.

```mermaid
flowchart LR
    Memo["📄 교사 관찰 메모"]
    Mask["1️⃣ mask_pii<br/>정규식 · 모델 호출 전"]
    Polish["2️⃣ polish<br/>Few-shot 문체 교정"]
    Validate["3️⃣ validate<br/>규정 + 사실보존 검증"]
    Out["✅ 안전 출력 또는 출력 보류<br/>+ 사유 + 책임 고지"]

    Memo --> Mask --> Polish --> Validate --> Out

    class Memo,Out neutral
    class Mask,Polish,Validate record

    classDef neutral fill:#F5F4F1,stroke:#8A8880,stroke-width:1px,color:#1A1A1A
    classDef record fill:#ECFEFF,stroke:#0891B2,stroke-width:1.5px,color:#1A1A1A
```

### API와 스트리밍

- **`POST /exam/stream`** (SSE) — UI가 사용하는 기본 경로. `graph.stream(stream_mode="updates")`로 LangGraph 노드 완료 시점마다 진행 이벤트를 전송합니다. POST 요청이라 브라우저 네이티브 `EventSource`(GET 전용) 대신 프론트엔드가 `fetch` + `ReadableStream`을 수동 파싱합니다.
- **`POST /exam`** (JSON 단발) — 동일 로직의 대안 엔드포인트 (curl 등 비-브라우저 클라이언트용)
- **`POST /record`** (JSON 단발) — 생기부 윤문

```
data: {"status": "truncated", "msg": "입력이 길어 앞부분만 반영되었습니다."}  # 8,000자 초과 시만
data: {"status": "progress",  "msg": "준비 중..."}
data: {"status": "progress",  "msg": "AI가 문항을 생성하고 있습니다. 수 분 소요됩니다..."}
data: {"status": "progress",  "msg": "생성된 문항의 구조적 유사도를 검증하고 있습니다..."}
data: {"status": "progress",  "msg": "문항 세트를 다시 생성하고 있습니다 (2번째 시도)..."}  # 재시도(최대 5회)마다
data: {"status": "done",      "items": [...], "validation_passed": true, "truncated": false}
data: {"status": "error",     "msg": "요청을 처리하지 못했습니다."}  # 내부 상세는 노출하지 않음
```

<details>
<summary><b>동시성 설계 (펼치기)</b></summary>

- **요청 간 세션 격리**: 출제 요청별 컨텍스트를 `contextvars.ContextVar`로 분리. `asyncio.to_thread` + `contextvars.copy_context()`로 worker 스레드에 전파.
- **이벤트 루프 비블로킹**: `/exam`은 `asyncio.to_thread`로 LangGraph 실행. `/exam/stream`은 `graph.stream()`(동기 제너레이터)을 executor 스레드에서 돌리며 `asyncio.Queue`로 이벤트만 이벤트 루프에 전달. `/record`는 Chain 전체가 async이므로 `await chain.run()`으로 직접 호출.

</details>

### LLM 백엔드 — ChatRunPod ↔ RunPodBackend

LangGraph 에이전트는 BaseChatModel 인터페이스만 알면 되고, RunPod과의 실제 HTTP 통신은 별도 레이어가 전담합니다.

```mermaid
flowchart LR
    Agent["🤖 LangGraph ReAct 에이전트<br/>BaseChatModel 인터페이스만 필요"]
    Chat["🔌 ChatRunPod<br/>chat_runpod.py · 메시지 형식 변환"]
    Backend["📡 RunPodBackend<br/>runpod.py · HTTP 통신 전담"]
    GPU[("⚡ RunPod 서버리스 GPU<br/>vLLM + handler.py")]

    Agent --> Chat --> Backend --> GPU

    class Agent,GPU neutral
    class Chat,Backend llm

    classDef neutral fill:#F5F4F1,stroke:#8A8880,stroke-width:1px,color:#1A1A1A
    classDef llm fill:#EDE9FE,stroke:#7C3AED,stroke-width:1.5px,color:#1A1A1A
```

RunPodBackend는 비동기 `/run`으로 작업을 한 번만 제출한 뒤 동일한 `job_id`를 폴링해, 긴 생성(멀티턴 ReAct)도 중복 실행 없이 안전하게 기다립니다.

```mermaid
flowchart LR
    Submit["POST /run<br/>read timeout 35s"]
    ID["job_id 수신"]
    Submit --> ID

    subgraph Loop["🔁 최대 60회 · 5초 간격(최대 5분)"]
        direction LR
        Wait["5초 대기"]
        Status{"GET /status/job_id"}
        Wait --> Status
        Status -- "IN_QUEUE / IN_PROGRESS" --> Wait
    end

    ID --> Loop
    Status -- "COMPLETED" --> Done["output 반환"]
    Status -- "FAILED/CANCELLED" --> Fail["예외 발생"]
    Loop -. "60회 초과" .-> Timeout["TimeoutError"]
    Submit -. "제출 응답 타임아웃" .-> SubmitTimeout["TimeoutError<br/>재제출 안 함"]

    class Submit,ID,Status neutral
    class Done llm
    class Fail,Timeout,SubmitTimeout warn
    style Loop fill:#FAFAF8,stroke:#C3C2B7,stroke-width:1px

    classDef neutral fill:#F5F4F1,stroke:#8A8880,stroke-width:1px,color:#1A1A1A
    classDef llm fill:#EDE9FE,stroke:#7C3AED,stroke-width:1.5px,color:#1A1A1A
    classDef warn fill:#FEE2E2,stroke:#DC2626,stroke-width:1.5px,color:#1A1A1A
```

> `/run` 제출 응답을 받지 못하면 job 자체는 이미 실행 중일 수 있으므로, 무작정 재제출하지 않고 명확한 예외로 상위 로직에 알립니다.

---

## 설계 원칙

**1. LLM이 판단하고, 코드가 결정한다.**
LLM의 자기 평가는 "기록"까지만 — 그것으로 무엇을 할지는 전부 결정론적 코드가 정합니다. 판단 로직을 LLM에 맡겼다가 회수한 이력이 이 프로젝트의 핵심 학습 곡선입니다.

| 검증 대상 | 판단 주체 | 근거 |
|---|---|---|
| 구조 유사도 (유형·난이도) | LLM (`similarity_judge`) → threshold는 코드 | 정성 판단은 LLM이 낫고, 커트라인은 코드가 안정적 |
| 문항 개수 | 코드 (`len(items) == num_items`) | LLM Judge에 맡겼다가 설계 오류 발견 후 이관 |
| 언어 (한국어) | 코드 (`save_item` 한글 비율 게이트) | 중국어 오염 문항을 저장 전 차단 |
| 재시도 여부 | 코드 (budget 루프) | LLM에 재시도 판단을 맡기면 수량 제어가 깨짐 |

**2. 도구는 순수 계산만.**
ReAct 도구 내부에 LLM 호출이 없습니다. 추론·생성은 에이전트가 직접, 도구는 검색·저장·검증만.

**3. 보안 하드룰 (예외 없음).**
실제 학생 데이터 미사용(전부 합성/익명) · PII 마스킹은 모델 호출 **이전** · 사용자 입력(메모·예시 문제) 비저장(요청 처리 중에만 메모리에 존재) · API 런타임 트레이싱 비활성화 · 로그·캐시에 PII 금지 · Next.js→FastAPI 서버 간 API 키 인증 · 생기부는 메모에 없는 사실 추가 금지("생성"이 아닌 "다듬기") + 안전 검증 실패 시 출력 보류 + 교사 책임 고지.

**4. 평가 기반 개발.**
골든셋은 코드에 하드코딩하지 않고 전부 `data/golden/*.json`으로 관리하며(파일별 용도는 [data/golden/README.md](./data/golden/README.md)), 모델·프롬프트 변경마다 [EVAL.md](./EVAL.md)에 결과 이력을 남깁니다.

---

## 엔지니어링 하이라이트

소형 LLM(7B)으로 에이전트를 만들며 겪은 문제와 해결 과정입니다. 상세 진단 기록은 [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) 참고.

| 문제 | 진단 | 해결 |
|---|---|---|
| 골든셋 생성 성공률이 ~37%에서 갑자기 **6%로 급락**, 모델이 중국어·스페인어 섞인 응답 | 로컬 Ollama가 `num_ctx` 기본값 **4096**으로 실행 중이었음(모델은 32K 지원). 멀티턴 ReAct + RAG 검색 결과 누적이 몇 턴 만에 한도를 초과 → 컨텍스트가 잘리며 시스템 프롬프트 유실. 동일 케이스 재현: 4096에서 0/5문항 → 16384에서 5/5문항 | `num_ctx=16384` 명시. vLLM(프로덕션)은 모델 네이티브 값을 쓰므로 **로컬 개발 환경에만 있던 설정 격차**였음 — dev/prod parity의 실례 |
| 생성 문항의 **45%에 중국어 오염** — "한국어로만 응답" 지시에도 발생 | LangSmith 트레이스 100건 정량 분석: 오염 출력의 입력 크기 중앙값 11,263자 vs 정상 8,009자 — 컨텍스트가 길수록 오염 확률이 오르는 **확률적 드리프트**. 오염 문항이 재시도 프롬프트에 실려 다음 시도로 전파되는 캐스케이드 경로도 확인 | `save_item`에 결정론적 한국어 게이트(한글 부재 또는 한자 비율 ≥5% 시 저장 거부 + 재작성 피드백). 기존 오염 사례 9건 소급 판정에서 수동 분류와 100% 일치 |
| "생성 개수 = 예시 문제 개수" 전제로 만든 count_match 검증이 실제 요구사항과 불일치 | 개수는 예시와 무관하게 사용자가 지정하는 값(`num_items`)이어야 함 — **골든셋 라벨링 직전에 설계 전제 자체가 틀렸음을 발견** | count_match를 LLM Judge에서 제거하고 `len(items)==num_items` 코드 검증으로 이관. 골든셋 전면 재생성 |
| 생성 프롬프트를 개선했는데 eval 수치가 **전혀 안 변함** | eval의 문항 품질 평가는 하드코딩된 고정 30문항을 채점하는 구조 — 생성 코드를 아무리 바꿔도 이 지표에 반영될 수 없었음 | 실제로 문항을 새로 생성해 채점하는 별도 검증 스크립트 작성. "eval이 존재하는가"와 "내 변경이 eval이 실제로 exercise하는 경로에 있는가"는 별개 |
| 재시도마다 이전 시도의 문항까지 전부 폐기 → num_items가 클수록 성공률 급락 | 재시도 구조가 세트 전체 재생성 방식이었음 | **부분 진행 보존**: 재시도 시 저장된 문항은 유지하고 "나머지 N개만 작성" 프롬프트로 이어서 생성. 개수 기준으로 적용 전 14건 중 부족 실패 8건 → 적용 후 6건 전부 목표 근접 달성(통제 실험은 아닌 생성 이력 기반 비교) |

---

## 품질 평가

> 지표 전체 목록·골든셋 현황·결과 이력은 [EVAL.md](./EVAL.md)에서 계속 갱신합니다. 아래는 최신 스냅샷(2026-07-12 기준, 생성 모델 qwen2.5:7b / 구조 Judge qwen2.5:14b)입니다.

### 출제 모듈 — `scripts/eval_exam.py`

| 지표 | n | 기준 | 실측 |
|---|---|---|---|
| 검색 Recall@5 | 21 | ≥ 0.80 | **0.905** ✅ |
| 검색 MRR | 21 | 참고값 | 0.659 |
| LLM Judge 종합평균 | 30 | ≥ 4.0 / 5 | 3.79 ❌ |
| Judge 신뢰도 (Cohen's κ) | 30 | ≥ 0.4 | 0.328 ❌ |
| Judge 신뢰도 (±1 일치율) | 30 | ≥ 0.7 | **0.800** ✅ |
| 구조 유사도 Judge — diff κ(난이도) | 45 | ≥ 0.4 | **0.424**(3회 평균) ✅ |
| 구조 유사도 Judge — overall 이진 κ | 45 | ≥ 0.4 | 0.167(3회 평균) ❌ |

- 검색 수치는 LLM과 무관한 BGE-M3 + reranker 파이프라인 성능. past_exams 컬렉션 제거 후 0.679→0.905로 상승
- Judge 종합평균 미달의 주원인은 오답매력도 — 실제 생성 기준 재검증 시 2.500→2.846(n=8→13, 객관식만)까지 개선했으나 목표(4.0)에는 크게 못 미침
- 구조 유사도 골든셋은 20→45개로 확대·전량 라벨링 완료. 편향 원인 분석(Judge가 애매한 사례를 3점으로 회피 수렴하는 패턴 확인) 후 3점 앵커 few-shot 추가해 diff κ는 목표 달성, overall 이진 κ는 여전히 미달(EVAL.md 5·6절 참고)

### 생기부 모듈 — `scripts/eval_record.py`

| 지표 | n | 기준 | 실측 |
|---|---|---|---|
| PII 마스킹 FN율 | 20 | = 0 | **0.000** ✅ |
| 키워드 사실추가율 | 20 | = 0 | **0.000** ✅ |
| NLI 사실추가율 | 20 | = 0 | 0.000~0.050(3회 재측정) ❌ |
| 규정 위반 Recall | 50 | ≥ 0.95 | 0.920~1.000(3회 평균 0.927) ❌ |
| 규정 위반 F1 | 50 | 참고값 | 0.939~0.980(3회 평균 0.960) |

- PII 마스킹·키워드 검사는 규칙 기반이라 모델 크기와 무관하게 안정적
- NLI 사실추가율은 "문체 다듬기 vs 사실 추가" 경계 사례를 사람이 직접 검토해 판단 기준을 확정하고 Judge few-shot을 보강해 0.050~0.200(변동 큼) → 0.000~0.050(안정)로 개선
- 규정 위반 Recall은 규칙 기반 키워드 확장(가정환경·종교/정치·외모 등) + Judge 프롬프트에 위반 유형 8종 한정으로 0.840→0.927 개선. RAG 검색 자체의 약점(관련 규정이 검색 상위에 잘 안 잡힘)을 고친 게 아니라 규칙으로 우회한 것이라 골든셋에 없는 새 표현은 놓칠 수 있음(EVAL.md 5절 참고)

<details>
<summary><b>기능 검증 결과 — test_*.py (펼치기)</b></summary>

| 레이어 | 스크립트 | 목적 | 실행 시점 |
|---|---|---|---|
| 기능 검증 | `test_*.py` | 파이프라인이 에러 없이 동작하는가 | 개발 중 수시 |
| 품질 평가 | `eval_*.py` | 얼마나 잘 하는가 (수치 지표) | 모델·프롬프트 변경 시 |

| 테스트 | 항목 | 결과 |
|---|---|---|
| `test_rag.py` | PDF 파싱·청킹·임베딩·ChromaDB 저장/검색 | ✅ |
| `test_rag.py` | 검색 + BGE-reranker 재정렬 | ✅ |
| `test_llm.py` | Ollama 응답 수신 | ✅ |
| `test_llm.py` | local → RunPod 백엔드 전환 | ✅ |
| `test_exam.py` | passage_text → 에이전트 세트 생성 → similarity_judge 흐름 (그래프 무크래시, 도구 오류 자기수정) | ✅ |
| `test_record.py` | PII 마스킹 4케이스 (전화번호·주민번호·학교명·이메일) | ✅ |
| `test_record.py` | 관찰 메모 → 생기부 문체 교정 | ✅ |
| `test_record.py` | 교사 책임 고지 출력 | ✅ |

</details>

<details>
<summary><b>프로덕션 검증 결과 — RunPod Qwen2.5-7B, RTX A5000 (펼치기)</b></summary>

| 항목 | 결과 |
|---|---|
| 에이전트 tool calling (ChatRunPod → vLLM) | ✅ |
| 세트 출제 (save_item → record_score → similarity_judge) | 리디자인 후 RunPod 재검증 필요 |
| validate_item_format 자기교정 루프 | ✅ |
| RAG 인덱싱 (규정·성취기준 2개 컬렉션) | ✅ regulations 510 / standards 573 청크 |
| EBS 영구 저장 | ✅ 컨테이너 재시작 후 재인덱싱 불필요 |
| 업로드 PDF 인덱싱 제거 | ✅ passage_text 붙여넣기로 인덱싱 자체가 불필요해짐 |
| 추론 속도 (세트) | 리디자인 후 재측정 필요 (구 수치: ~2–3분/1문항, RTX A5000, min workers=1) |

</details>

---

## 모델 비교 실험

> 상세 방법론·전체 결과·raw 데이터는 [EVAL.md](./EVAL.md)의 "7. 모델 비교 실험" ·
> "7.1 budget=5 재검증" 절 참고.

Qwen2.5-7B(현재 프로덕션 기본값)를 정식 채택 모델로 확정하지 않고, 후보들을 동등 조건
(동일 passage_text 15개, 자기채점 편향을 피하기 위해 고정된 하나의 Judge로 통일)으로
비교했다. 1차는 재시도 없는 조건(`budget=1`)으로, 2차는 실제 프로덕션과 동일한 조건
(`budget=5`, 재시도 포함)으로 재검증했다.

| 모델 | 실패율(budget=1→5) | 평균 생성시간(budget=1→5) | 개수 충족률(budget=1→5) |
|---|---|---|---|
| Qwen2.5-7B | 40% → 6.7% | 50.3s → 258.8s | 0.15 → 0.65 |
| Qwen2.5-14B | 0% → 0% | 158.1s → **260.2s** | 0.76 → 1.04 |
| Llama3.1-8B | 60% → 26.7% | 59.4s → 289.4s | 0.11 → 0.34 |
| GPT-4o-mini | 0% → 13.3% | **18.1s → 66.5s** | 0.97 → 1.77 |
| Qwen3.5-9B | 26.7% → 0% | 171.1s → 621.2s | 0.55 → 1.40 |

budget=1에서는 "GPT-4o-mini가 전 지표 석권"으로 보였지만, 실제 재시도 조건(budget=5)에서
재검증하니 그림이 달라졌다. **가장 큰 발견은 Qwen2.5-7B와 14B의 속도 격차가 사실상
사라진 것**(258.8s vs 260.2s) — 7B는 재시도가 잦아 속도 이점을 스스로 까먹은 반면, 14B는
애초에 실패율이 0%라 손해가 없었다. 같은 속도에서 14B가 실패율·개수 충족률 모두 뚜렷이
우수해, **로컬 스택을 유지한다는 전제라면 이번 데이터는 7B→14B 승격을 지지**한다.
GPT-4o-mini는 속도는 여전히 압도적이지만 실패율이 오히려 늘고(0%→13.3%) 개수를 목표의
1.77배까지 과다생성하는 경향이 새로 나타나 "무조건 최선"이라 보기는 어려워졌다. 최종
채택은 비용·외부 API 의존성 트레이드오프를 포함해 검토 중이며, 프로덕션은 여전히
Qwen2.5-7B(로컬)를 사용 중이다.

---

## 빠른 시작 (로컬)

### 1. 환경 설정

```bash
git clone https://github.com/MachuEngine/bunpil.git
cd bunpil

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env   # 필요 시 값 수정
```

### 2. Ollama 모델 설치

```bash
# Ollama 설치: https://ollama.com

# 생성 전용 (OLLAMA_MODEL)
ollama pull qwen2.5:7b

# Judge 전용 (OLLAMA_JUDGE_MODEL) — 현재는 생성 모델과 동일한 7B 사용, 별도 pull 불필요
# (14B는 하드웨어 확보 후 Judge 분리 테스트 예정)

# 빠른 로직 테스트만 할 경우 (품질 낮음, 폴백 동작)
# ollama pull qwen2.5:1.5b
```

> **참고**: Ollama는 별도 설정이 없으면 `num_ctx`를 4096으로 제한합니다(모델 자체는 32K 지원). 멀티턴 ReAct 루프는 이를 몇 턴 만에 초과해 응답이 깨질 수 있어, `app/modules/exam/llm.py`에서 `num_ctx=16384`로 이미 올려뒀습니다 — 별도 조치 불필요. ([상세 기록](./TROUBLESHOOTING.md))

### 3. RAG 데이터 인덱싱

```bash
# data/ 경로에 PDF를 넣은 뒤 아래 순서대로 실행
.venv/bin/python scripts/index_regulations.py   # 생기부 기재요령·훈령
.venv/bin/python scripts/index_standards.py     # 사회과 교육과정 성취기준
```

> 이미 적재된 파일은 자동 스킵 (idempotent). 처음 한 번만 실행하면 됩니다.

### 4. 서버 실행

터미널 3개를 사용합니다. `app/main.py`는 정적 파일을 서빙하지 않으므로, UI를 보려면 프론트엔드(Next.js)도 별도로 띄워야 합니다.

```bash
# 터미널 1 — Ollama LLM 서버
ollama serve

# 터미널 2 — FastAPI (API 전용, 포트 8765)
# Windows
$env:BUNPIL_API_KEY="replace_with_a_long_random_value"; $env:LLM_BACKEND="local"; $env:OLLAMA_MODEL="qwen2.5:7b"
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8765

# macOS / Linux
BUNPIL_API_KEY=replace_with_a_long_random_value LLM_BACKEND=local OLLAMA_MODEL=qwen2.5:7b .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8765

# 터미널 3 — 프론트엔드 (Next.js, 포트 3000)
cd frontend
npm install    # 최초 1회
BUNPIL_API_KEY=replace_with_a_long_random_value BACKEND_URL=http://localhost:8765 npm run dev
```

브라우저에서 **http://localhost:3000** 접속(프론트엔드 포트 — `frontend/app/api/*/route.ts`가 `BACKEND_URL`로 FastAPI에 프록시). `BACKEND_URL` 미설정 시 기본값은 `http://localhost:8000`이라 위처럼 8765로 맞춰줘야 합니다.

---

## 배포 (프로덕션)

```mermaid
flowchart LR
    Browser["🌐 브라우저"]
    Caddy["🔒 Caddy<br/>HTTPS 프록시"]
    EC2["🖥️ EC2 t3.medium<br/>FastAPI + ChromaDB"]
    RunPod["⚡ RunPod 서버리스<br/>Qwen2.5-7B, vLLM"]
    EBS[("💾 EBS 10GB<br/>ChromaDB 저장")]

    Browser --> Caddy --> Frontend["▲ Next.js<br/>frontend (3000)"]
    Frontend --> EC2
    EC2["🖥️ EC2 t3.medium<br/>FastAPI + ChromaDB (8765)"] --> RunPod
    EC2 -.-> EBS

    class Browser,Caddy,Frontend,EC2,RunPod,EBS neutral

    classDef neutral fill:#F5F4F1,stroke:#8A8880,stroke-width:1px,color:#1A1A1A
```

> **프론트엔드 배포**: `docker-compose.yml`의 `frontend` 서비스(`frontend/Dockerfile`, Next.js `output: "standalone"` 빌드)가 3000 포트로 UI를 서빙하고, `frontend/app/api/*/route.ts`가 컨테이너 내부에서 `BACKEND_URL=http://app:8765`로 FastAPI에 프록시합니다. Caddy는 3000만 바라보면 됩니다(아래 Caddyfile 참고). `docker compose up -d --build`로 `app`+`frontend`가 함께 뜹니다.

<details>
<summary><b>대안: 외부 호스팅(Vercel 등)에 frontend만 분리 배포 (펼치기)</b></summary>

지금 채택한 방식은 EC2 안에서 `frontend` 컨테이너를 상시 프로세스로 돌리는 방식(기존 인프라로 해결, 신규 계정 불필요)입니다. 대신 Next.js를 Vercel 등 외부 호스팅에 올리고 `BACKEND_URL` 환경변수만 EC2 도메인(`https://your-domain.com`)으로 맞추는 방식도 가능합니다 — 이 경우 `frontend` 컨테이너는 필요 없고, Caddy도 다시 FastAPI(8765)를 직접 바라보도록 되돌려야 합니다. CDN 엣지 배포로 프론트 응답이 더 빨라지는 대신 신규 외부 계정이 필요해 이 프로젝트에서는 채택하지 않았습니다.

</details>

### RunPod 서버리스 설정

```bash
# 1. 핸들러 이미지 빌드 & 푸시
cd runpod_handler
docker build -t <your-dockerhub>/bunpil-runpod:latest .
docker push <your-dockerhub>/bunpil-runpod:latest

# 2. RunPod 콘솔 → Serverless → New Endpoint → 이미지 URL 입력
# 3. 워커 설정: min workers=1 (콜드스타트 방지), max workers=4 (병렬 출제 시)
# 4. 발급된 Endpoint ID를 .env에 입력
# LLM_BACKEND=runpod
# RUNPOD_API_KEY=...
# RUNPOD_ENDPOINT_ID=...
```

### EC2 배포 (Docker Hub 이미지 사용)

```bash
# EC2 (Ubuntu 22.04 t3.medium) 내부에서
docker pull jongmin0826/bunpil-app:latest
docker pull jongmin0826/bunpil-frontend:latest

# EBS 볼륨 마운트 (처음 한 번)
sudo mkfs.ext4 /dev/nvme1n1
sudo mkdir -p /data/chroma_db
echo '/dev/nvme1n1 /data/chroma_db ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab
sudo mount -a

# app과 frontend가 컨테이너 이름으로 서로 통신할 수 있도록 전용 네트워크 생성
docker network create bunpil-net

# 컨테이너 실행 (FastAPI)
docker run -d --name bunpil \
  --network bunpil-net \
  -p 8765:8765 \
  --env-file /home/ubuntu/.env \
  -v /data/chroma_db:/data/chroma_db \
  jongmin0826/bunpil-app:latest

# 컨테이너 실행 (Next.js — BACKEND_URL은 컨테이너 이름으로 접근)
docker run -d --name bunpil-frontend \
  --network bunpil-net \
  -p 3000:3000 \
  -e BUNPIL_API_KEY='<FastAPI와 동일한 값>' \
  -e BACKEND_URL=http://bunpil:8765 \
  jongmin0826/bunpil-frontend:latest

# RAG 인덱싱 (처음 한 번 — EBS에 영구 저장됨)
docker exec bunpil python scripts/index_regulations.py
docker exec bunpil python scripts/index_standards.py
```

> `docker-compose.yml`을 그대로 쓰는 경우(`docker compose up -d --build`)는 네트워크 생성이 자동이라 위 `docker network create`/`--network` 단계가 필요 없습니다. Docker Hub에 `bunpil-frontend` 이미지를 아직 안 올렸다면 `cd frontend && docker build -t jongmin0826/bunpil-frontend:latest . && docker push ...`로 먼저 푸시해야 합니다.

### 빌링 알람

```bash
bash deploy/billing_alarm.sh   # 월 $10 초과 시 이메일 알람
```

### 월 운영비 (1인 기준)

| 항목 | 비용 |
|---|---|
| EC2 t3.medium | ~$30 |
| RunPod 서버리스 (추론만 과금, min workers=1) | ~$5–15 |
| EBS 10GB | ~$1 |
| **합계** | **~$36–46** |

데모/개발 중에는 EC2를 필요할 때만 켜서 절감 가능. min workers=0 설정 시 RunPod 비용 대폭 절감 (단, 콜드스타트 30–60초 발생).

---

## 데이터

| 컬렉션 | 경로 | 출처 | 용도 |
|---|---|---|---|
| `regulations` | `data/regulations/` | 학교생활기록부 종합지원포털 | 생기부 규정 위반 검증 + 출제 시 교육과정 법령 참조 |
| `standards` | `data/standards/` | 국가교육과정정보센터(NCIC) | 출제 시 성취기준 원문 검색 (`search_standards` 도구) |

> `past_exams` 컬렉션(수능·모평 기출)은 리디자인으로 완전히 제거됨 — `check_duplicate` 폐기, 2028 수능 개편으로 과목별 구조 자체가 무의미해짐.

## 디렉토리 구조

```
bunpil/
├── app/
│   ├── common/
│   │   ├── llm/          # LLM 추상화 (OllamaBackend / RunPodBackend / ChatRunPod)
│   │   └── rag/          # PDF 파싱, 임베딩, 리랭킹, ChromaDB
│   ├── modules/
│   │   ├── exam/         # 출제 모듈 (LangGraph ReAct Agent, 7개 도구)
│   │   └── record/       # 생기부 모듈 (수동 루프 Chain)
│   └── main.py           # FastAPI (/exam/stream + /record)
├── frontend/             # Next.js UI
├── data/
│   ├── regulations/      # 생기부 기재요령, 작성·관리지침
│   ├── standards/        # 사회과 교육과정 PDF
│   └── golden/           # 골든셋 JSON — 정기 평가용 6개 + 실험 기록 5개
│                         # (파일별 용도·라벨 필드는 data/golden/README.md 참고)
├── scripts/
│   ├── index_*.py        # RAG 컬렉션 인덱싱
│   ├── test_*.py         # 기능 검증 (LLM·RAG·출제·생기부)
│   ├── eval_*.py         # 품질 평가 (골든셋 기반 수치 지표)
│   └── gen_*.py          # 골든셋 생성 도구
├── runpod_handler/       # RunPod 서버리스 핸들러 (Qwen2.5-7B vLLM)
├── deploy/               # EC2·Caddy·빌링알람 프로비저닝 스크립트
├── Dockerfile
├── docker-compose.yml
└── Caddyfile
```

## 환경변수

`.env.example` 참고. 시크릿은 `.env`에만 보관 — 커밋 금지.

| 변수 | 설명 | 기본값 |
|---|---|---|
| `BUNPIL_API_KEY` | Next.js → FastAPI 서버 간 인증 키(양쪽에 동일한 긴 무작위 값 설정) | 필수 |
| `LLM_BACKEND` | `local` 또는 `runpod` | `local` |
| `OLLAMA_MODEL` | 로컬 개발 생성 모델명 | `qwen2.5:7b` |
| `OLLAMA_JUDGE_MODEL` | 로컬 개발 Judge 모델명 (미설정 시 `OLLAMA_MODEL`로 폴백) | — (선택) |
| `OLLAMA_BASE_URL` | Ollama 서버 주소 | `http://localhost:11434` |
| `RUNPOD_API_KEY` | RunPod API 키 | — |
| `RUNPOD_ENDPOINT_ID` | RunPod 엔드포인트 ID | — |
| `CHROMA_PERSIST_DIR` | ChromaDB 저장 경로 | `/data/chroma_db` (EC2) / `./chroma_db` (로컬) |
| `BGE_EMBED_MODEL` | 임베딩 모델명 | `BAAI/bge-m3` |
| `BGE_RERANK_MODEL` | 리랭킹 모델명 | `BAAI/bge-reranker-base` |
| `LANGCHAIN_TRACING_V2` | 합성 데이터 평가 스크립트의 LangSmith 트레이싱 (`true` / `false`). API 서버에서는 강제 비활성화 | `false` |
| `LANGCHAIN_API_KEY` | LangSmith API 키 | — (선택) |
| `LANGCHAIN_PROJECT` | LangSmith 프로젝트 베이스명 — 기본값('bunpil') 유지 시 `LLM_BACKEND`에 따라 `-dev`(local, 순수 로컬 개발) 또는 `-prod`(runpod/openai 등 실제 서빙 백엔드) 접미사가 자동으로 붙음(`app/common/llm/tracing.py`). 로컬 개발 노이즈가 프로덕션 통계를 오염시키지 않도록 분리. 'bunpil'이 아닌 값을 직접 설정하면 그대로 override | `bunpil` → `bunpil-dev` / `bunpil-prod` |
